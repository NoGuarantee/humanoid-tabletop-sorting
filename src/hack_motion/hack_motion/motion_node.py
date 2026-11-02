"""Motion server: actions MoveJoints, Hand, MoveTcp. Owner: motion.

MoveJoints: synchronised arrival of all joints, closed loop on /joint_states against the PD undershoot.
Hand:       named presets, reports whether the fingers stopped on an object.
MoveTcp:    needs ik.solve (milestone M1), answers not_implemented until then.
"""
import threading
import time

import rclpy
from bodyctrl_msgs.msg import CmdSetMotorPosition, SetMotorPosition
from hack_interfaces.action import Hand, MoveJoints, MoveTcp
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from . import ik
from .robot import ARM_JOINTS, HAND_PRESETS, JOINTS, MAX_SPEED, clip

PERIOD = 0.05  # s, real time between control checks
STALL_TIME = 0.5  # s of sim time without progress -> compensate the PD undershoot
MAX_COMPENSATION = 0.3  # rad added on top of the target at most


class MotionNode(Node):
    def __init__(self):
        super().__init__("hack_motion")
        self.lock = threading.Lock()
        self.q = {}
        self.hand_state = {"left": None, "right": None}
        cb = ReentrantCallbackGroup()
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10, callback_group=cb)
        for side in ("left", "right"):
            self.create_subscription(JointState, f"/inspire_hand/state/{side}_hand",
                                     lambda m, s=side: self.on_hand(s, m), 10, callback_group=cb)
        self.cmd = {g: self.create_publisher(CmdSetMotorPosition, f"/{g}/cmd_pos", 10)
                    for g in ("arm", "head", "waist")}
        self.hand_cmd = {s: self.create_publisher(JointState, f"/inspire_hand/ctrl/{s}_hand", 10)
                         for s in ("left", "right")}
        ActionServer(self, MoveJoints, "motion/move_joints", self.exec_move_joints, callback_group=cb)
        ActionServer(self, Hand, "motion/hand", self.exec_hand, callback_group=cb)
        ActionServer(self, MoveTcp, "motion/move_tcp", self.exec_move_tcp, callback_group=cb)
        self.get_logger().info("motion server ready")

    # ---------------------------------------------------------------- state
    def on_joints(self, msg):
        with self.lock:
            self.q.update(zip(msg.name, msg.position))

    def on_hand(self, side, msg):
        with self.lock:
            self.hand_state[side] = msg

    def joints(self, names):
        with self.lock:
            return [self.q.get(n) for n in names]

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def send(self, targets, speeds):
        by_group = {}
        for name, pos in targets.items():
            motor, group, _, _ = JOINTS[name]
            # spd 0 means max speed for the driver, so never send exactly 0
            by_group.setdefault(group, []).append(
                SetMotorPosition(name=motor, pos=float(pos), spd=float(max(speeds[name], 0.01)), cur=1.0))
        for group, cmds in by_group.items():
            msg = CmdSetMotorPosition()
            msg.cmds = cmds
            self.cmd[group].publish(msg)

    # ---------------------------------------------------------------- MoveJoints
    def move_joints(self, names, positions, speed=0.0, tol=0.0, max_time=0.0, feedback=None):
        """Blocking joint move. Returns (success, max_error, message)."""
        unknown = [n for n in names if n not in JOINTS]
        if unknown:
            return False, 0.0, f"unknown joints {unknown}"
        if len(names) != len(positions):
            return False, 0.0, "joints and positions differ in length"
        speed = min(speed or MAX_SPEED, MAX_SPEED)
        tol = tol or 0.02
        goal = {n: clip(n, p) for n, p in zip(names, positions)}
        start = self.joints(names)
        if any(v is None for v in start):
            return False, 0.0, "no /joint_states yet"
        dist = {n: abs(goal[n] - q0) for n, q0 in zip(names, start)}
        duration = max(max(dist.values()) / speed, 0.1)
        speeds = {n: dist[n] / duration for n in names}  # every joint arrives at the same time
        max_time = max_time or (2.0 * duration + 3.0)

        cmd = dict(goal)
        self.send(cmd, speeds)
        t0 = self.now()
        best, best_t = float("inf"), t0
        while rclpy.ok():
            time.sleep(PERIOD)
            q = dict(zip(names, self.joints(names)))
            err = {n: goal[n] - q[n] for n in names}
            max_err = max(abs(e) for e in err.values())
            if feedback:
                feedback(max_err)
            if max_err < tol:
                return True, max_err, "ok"
            t = self.now()
            if t - t0 > max_time:
                return False, max_err, "timeout"
            if max_err < best - 1e-3:
                best, best_t = max_err, t
            elif t - best_t > STALL_TIME:  # stopped short of the target: push the command past it
                for n in names:
                    if abs(err[n]) > tol:
                        extra = max(-MAX_COMPENSATION, min(MAX_COMPENSATION, cmd[n] - goal[n] + err[n]))
                        cmd[n] = clip(n, goal[n] + extra)
                self.send(cmd, {n: max(abs(err[n]) / STALL_TIME, 0.05) for n in names})
                best, best_t = max_err, t
        return False, 0.0, "shutdown"

    def exec_move_joints(self, handle):
        g = handle.request
        fb = MoveJoints.Feedback()

        def feedback(e):
            fb.max_error = float(e)
            handle.publish_feedback(fb)

        ok, err, msg = self.move_joints(list(g.joints), list(g.positions), g.speed, g.tol, g.max_time, feedback)
        handle.succeed()  # the action itself ran; whether the move reached the target is in the result
        return MoveJoints.Result(success=ok, max_error=float(err), message=msg)

    # ---------------------------------------------------------------- Hand
    def exec_hand(self, handle):
        res = self.hand(handle)
        handle.succeed()
        return res

    def hand(self, handle):
        g = handle.request
        res = Hand.Result()
        if g.hand not in self.hand_cmd or g.preset not in HAND_PRESETS:
            res.success = False
            return res
        if g.force > 0:
            # TODO(grasping): call /inspire_hand/set_force/{side}_hand (bodyctrl_msgs/srv/SetForce)
            self.get_logger().warn("Hand.force is not applied yet")
        target = HAND_PRESETS[g.preset]
        msg = JointState()
        msg.name = [str(i) for i in range(1, 7)]
        msg.position = [float(v) for v in target]
        self.hand_cmd[g.hand].publish(msg)

        t0, last, still_since = self.now(), None, None
        fb = Hand.Feedback()
        while rclpy.ok() and self.now() - t0 < 2.0:  # full stroke takes 0.8 s at default speed
            time.sleep(PERIOD)
            with self.lock:
                st = self.hand_state[g.hand]
            if st is None or len(st.position) < 6:
                continue
            pos = list(st.position[:6])
            fb.position = [float(p) for p in pos]
            handle.publish_feedback(fb)
            if last is not None and max(abs(a - b) for a, b in zip(pos, last)) < 0.005:
                still_since = still_since or self.now()
                if self.now() - still_since > 0.2:
                    break
            else:
                still_since = None
            last = pos
        with self.lock:
            st = self.hand_state[g.hand]
        if st is not None and len(st.position) >= 6:
            res.position = [float(p) for p in st.position[:6]]
            res.effort = [float(e) for e in st.effort[:6]] if len(st.effort) >= 6 else []
            closing = [i for i in range(6) if target[i] < 0.9]
            # a closing finger stopped well short of its target with noticeable effort -> something is in the hand
            res.object_detected = any(
                res.position[i] > target[i] + 0.05 and (not res.effort or res.effort[i] > 0.1) for i in closing)
        res.success = True
        return res

    # ---------------------------------------------------------------- MoveTcp
    def exec_move_tcp(self, handle):
        res = self.move_tcp(handle.request)
        handle.succeed()
        return res

    def move_tcp(self, g):
        names = ARM_JOINTS.get(g.arm)
        if names is None:
            return MoveTcp.Result(success=False, message=f"unknown arm {g.arm}")
        try:
            waist = self.joints(["body_yaw_joint"])[0] or 0.0
            q = ik.solve(g.arm, g.target, self.joints(names), waist)
        except NotImplementedError as e:
            return MoveTcp.Result(success=False, message=f"not_implemented: {e}")
        if q is None:
            return MoveTcp.Result(success=False, message="unreachable")
        # TODO(motion): mode "linear" = interpolate the Cartesian path and solve IK per waypoint
        ok, err, msg = self.move_joints(names, q, max_time=g.max_time)
        # TODO(motion): report the real TCP error from forward kinematics / tf, not the joint error
        return MoveTcp.Result(success=ok, pos_error=float(err), message=msg)


def main():
    rclpy.init()
    node = MotionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
