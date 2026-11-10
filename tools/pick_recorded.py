#!/usr/bin/env python3
"""Run one E71 Rhino pick through main's actions, recording five seconds of hold.

Requires an exclusive fresh simulator, motion and perception nodes, and the
canonical RGB-D pose service. Output is an execution record for video/physics
review. Hand.object_detected alone never sets physical_hold_confirmed.
"""
import argparse
import json
import math
from pathlib import Path
import threading
import time

import rclpy
from geometry_msgs.msg import Pose, WrenchStamped
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from hack_brain.blocking import BlockingClients, _wait
from hack_brain.recorded_pick import observed_pose, pick_recorded, transform
from hack_motion.recorded_grasp import RHINO


class Probe(Node):
    def __init__(self, stream, known):
        super().__init__('pick_recorded', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.stream, self.known = stream, known
        self.api = BlockingClients(self)
        self.pose_client = self.create_client(Trigger, 'perception/estimate_model_poses')
        self.tf = Buffer(cache_time=Duration(seconds=20.))
        self.listener = TransformListener(self.tf, self)
        self.last_stamp = 0
        self.epoch = None
        self.latest_hand = None
        self.record_lock = threading.Lock()
        self.create_subscription(JointState, '/inspire_hand/state/right_hand', self.on_hand, 10)
        self.create_subscription(WrenchStamped, '/arm_6dof_right', self.on_wrench, 10)

    def now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def record(self, phase, data):
        with self.record_lock:
            self.stream.write(json.dumps(dict(phase=phase, sim_s=self.now(), **data), allow_nan=False) + '\n')
            self.stream.flush()

    def on_hand(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.latest_hand = stamp, list(msg.position)
        values = list(msg.position) + list(msg.effort)
        if all(math.isfinite(v) for v in values):
            self.record('hand_state', dict(stamp_s=stamp, position=list(msg.position), effort=list(msg.effort)))

    def on_wrench(self, msg):
        wrench = msg.wrench
        values = [getattr(v, axis) for v in (wrench.force, wrench.torque) for axis in ('x', 'y', 'z')]
        if all(math.isfinite(v) for v in values):
            self.record('wrench', dict(stamp_s=msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
                                       frame_id=msg.header.frame_id, values=values))

    def observe(self):
        scene = self.api.perceive()
        if scene is None:
            raise RuntimeError('measured_collision_scene_unavailable')
        stamp = scene.header.stamp.sec + scene.header.stamp.nanosec / 1e9
        if scene.header.frame_id != 'pelvis' or not 0 <= self.now() - stamp <= 1.:
            raise RuntimeError('measured_collision_scene_stale')
        if self.known is not None:
            self.record('object_pose', dict(source='debug_known_pose', T_pelvis_object=self.known.tolist()))
            return self.known.copy()
        if not self.pose_client.wait_for_service(timeout_sec=5.):
            raise RuntimeError('model_pose_service_unavailable')
        deadline = time.monotonic() + 25.
        after = max(self.last_stamp, int(self.now() * 1e9))
        while time.monotonic() < deadline:
            future = self.pose_client.call_async(Trigger.Request())
            if not _wait(future, max(0., deadline - time.monotonic())):
                raise RuntimeError('model_pose_service_timeout')
            response = future.result()
            data = json.loads(response.message)
            if response.success and data.get('status') == 'ok':
                stamp = data.get('stamp_ns')
                if type(stamp) is int and stamp <= after:
                    time.sleep(.1)
                    continue
                value = observed_pose(data, self.now(), after)
                key = data['exposure_key']
                epoch = key['session_id'], key['epoch']
                if self.epoch is not None and epoch != self.epoch:
                    raise RuntimeError('scene_epoch_changed')
                self.epoch, self.last_stamp = epoch, data['stamp_ns']
                self.record('object_pose', dict(source='rgbd', response=data))
                return value
            if data.get('status') not in ('worker_loading', 'awaiting_fresh_pose'):
                raise RuntimeError('model_pose_unavailable:' + str(data.get('status')))
            time.sleep(.1)
        raise RuntimeError('model_pose_timeout')

    def tcp(self):
        item = self.tf.lookup_transform('pelvis', 'right_tcp_link', Time(), timeout=Duration(seconds=1.))
        stamp = item.header.stamp.sec + item.header.stamp.nanosec / 1e9
        if not 0 <= self.now() - stamp <= .5:
            raise RuntimeError('tcp_stale')
        t = item.transform
        self.record('tcp', dict(stamp_s=stamp, xyz=[t.translation.x, t.translation.y, t.translation.z],
                               xyzw=[t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]))
        result = Pose(orientation=t.rotation)
        result.position.x, result.position.y, result.position.z = t.translation.x, t.translation.y, t.translation.z
        return result

    def hold(self, duration):
        start = self.now()
        deadline = time.monotonic() + 60.
        while rclpy.ok():
            now = self.now()
            if now < start or time.monotonic() > deadline:
                raise RuntimeError('hold_clock_stalled_or_reset')
            sample = self.latest_hand
            if (sample is None or abs(now - sample[0]) > .5 or len(sample[1]) < 6
                    or not all(math.isfinite(v) for v in sample[1])):
                raise RuntimeError('hold_hand_state_stale')
            self.tcp()
            if now - start >= duration:
                return
            time.sleep(.05)
        raise RuntimeError('shutdown_during_hold')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path, help='new JSONL evidence file')
    parser.add_argument('--known-pose', type=Path, help='debug-only canonical pose JSON in pelvis')
    args = parser.parse_args()
    known = None
    if args.known_pose:
        data = json.loads(args.known_pose.read_text())
        if (data.get('frame_id') != 'pelvis' or data.get('object_name') != RHINO['object_name']
                or data.get('reference_frame') != 'canonical_rgbd_model'):
            parser.error('known pose must identify Rhino, pelvis and canonical_rgbd_model')
        known = transform(data['T_pelvis_object'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        rclpy.init()
        node = Probe(stream, known)
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        worker = threading.Thread(target=executor.spin, daemon=True)
        worker.start()
        result = dict(commands_completed=False, physical_hold_confirmed=None)
        try:
            node.record('start', dict(profile=RHINO, pose_source='known' if known is not None else 'rgbd'))
            if not node.api.wait_ready():
                raise RuntimeError('action_servers_unavailable')
            if not node.api.move_joints(['head_pitch_joint', 'head_yaw_joint'], [.4, 0.]):
                raise RuntimeError('initial_table_view_failed')
            # The probe runs in a fresh episode with both hands visibly empty.
            result = pick_recorded(node.api, node.observe, node.tcp, node.hold, node.record)
        except (Exception, KeyboardInterrupt) as exc:
            result['error'] = str(exc) or type(exc).__name__
        finally:
            node.record('result', result)
            executor.shutdown()
            worker.join(timeout=5.)
            node.destroy_node()
            rclpy.try_shutdown()
    print(json.dumps(result))
    return 0 if result['commands_completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
