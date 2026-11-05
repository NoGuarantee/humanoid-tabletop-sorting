#!/usr/bin/env python3
"""Calls a running perception_node on the bench: head at a pose, then with the right arm in view.

Prints the service responses and saves rgb with the reported footprints drawn on it. Moves the head and, with --arm,
the right arm, so run it on a simulator nobody else uses, without the full launch:
    ros2 run hack_perception perception_node --ros-args -p use_sim_time:=true &
    python3 tools/perception_bench_check.py OUT_DIR [--pitch 0.4] [--yaw 0.0] [--arm]
"""
import argparse
import math
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from bodyctrl_msgs.msg import CmdSetMotorPosition, SetMotorPosition
from hack_interfaces.srv import Perceive
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, TransformListener

ARM_FRAMES = ("elbow_pitch_l_link", "wrist_pitch_l_link", "left_tcp_link",
              "elbow_pitch_r_link", "wrist_pitch_r_link", "right_tcp_link")
ARM_JOINTS = ("shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint", "elbow_pitch_r_joint",
              "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint")
# Neither pose was reached on the bench: the arm stops short with the hand in a corner of the frame.
ARM_POSES = {"arm raised": (0.0, -0.2, 0.0, -2.0, 0.0, 0.0, 0.0),
             "arm over the table": (-0.5, -0.2, 0.0, -1.1, 0.0, 0.0, 0.0)}


def matrix(tf):
    t, q = tf.transform.translation, tf.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), t.x],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), t.y],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), t.z],
                     [0, 0, 0, 1]])


class Check(Node):
    def __init__(self):
        super().__init__("perception_bench_check", parameter_overrides=[
            Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.joints, self.rgb, self.info = {}, None, None
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.head = self.create_publisher(CmdSetMotorPosition, "/head/cmd_pos", 10)
        self.arm = self.create_publisher(CmdSetMotorPosition, "/arm/cmd_pos", 10)
        self.perceive = self.create_client(Perceive, "/perception/perceive")
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)
        self.create_subscription(Image, "/cam_head/color/image_raw", self.on_rgb, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/cam_head/color/camera_info", self.on_info, qos_profile_sensor_data)

    def on_joints(self, m):
        self.joints.update(zip(m.name, m.position))

    def on_rgb(self, m):
        self.rgb = m

    def on_info(self, m):
        self.info = m

    def move(self, pub, motors, target, joints, wall_s=25.0):
        """Commands motors until /joint_states comes near: the drives fall short of the target, so it only warns."""
        msg = CmdSetMotorPosition(cmds=[SetMotorPosition(name=n, pos=float(p), spd=0.5, cur=1.0)
                                        for n, p in zip(motors, target)])
        end = time.monotonic() + wall_s
        while time.monotonic() < end:
            pub.publish(msg)
            time.sleep(0.5)
            if all(abs(self.joints.get(j, 1e3) - p) < 0.02 for j, p in zip(joints, target)):
                break
        print("joints:", " ".join(f"{j}={self.joints.get(j, math.nan):+.3f}" for j in joints))

    def call(self, camera):
        t0 = time.monotonic()
        fut = self.perceive.call_async(Perceive.Request(camera=camera, region="table"))
        while not fut.done():
            time.sleep(0.005)
        return fut.result(), (time.monotonic() - t0) * 1e3

    def report(self, label, camera, out=None):
        res, ms = self.call(camera)
        s = res.scene
        print(f"{label} [{camera}]: success={res.success} '{res.message}' {ms:.0f} ms, table_z {s.table_z:+.4f}, "
              f"objects {len(s.objects)}, trays {len(s.containers)}, "
              f"stamp {s.header.stamp.sec}.{s.header.stamp.nanosec:09d} in {s.header.frame_id}")
        for c in s.containers:
            p = c.pose.position
            print(f"  {c.kind} {c.id} {c.color} xyz ({p.x:+.3f} {p.y:+.3f} {p.z:+.3f}) "
                  f"size {c.size.x:.3f} {c.size.y:.3f} {c.size.z:.3f}")
        for o in s.objects:
            print(f"  {o.track_id} {o.category or '-'} {o.confidence:.2f} {o.name} "
                  f"xyz ({o.center.x:+.3f} {o.center.y:+.3f} {o.center.z:+.3f}) yaw {o.yaw:+.2f} "
                  f"size {o.size.x:.3f} {o.size.y:.3f} {o.size.z:.3f} top {o.top_z:+.3f}"
                  f"{' CUT' if o.partially_visible else ''}")
        if out:
            self.overlay(s, out)
        return s

    def overlay(self, scene, path):
        """Footprints at their top height, projected through tf: what the service said, over a separate colour frame."""
        rgb = np.frombuffer(self.rgb.data, np.uint8).reshape(self.rgb.height, self.rgb.width, 3)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        T = np.linalg.inv(matrix(self.tf.lookup_transform("pelvis", self.rgb.header.frame_id, Time(),
                                                          timeout=Duration(seconds=1.0))))
        K = np.array(self.info.k).reshape(3, 3)
        for o in scene.objects:
            c, s = math.cos(o.yaw), math.sin(o.yaw)
            corners = [(o.center.x + a * o.size.x / 2 * c - b * o.size.y / 2 * s,
                        o.center.y + a * o.size.x / 2 * s + b * o.size.y / 2 * c, o.top_z, 1.0)
                       for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
            cam = (T @ np.array(corners).T)[:3]
            uv = (K @ cam)[:2] / cam[2]
            cv2.polylines(img, [uv.T.astype(np.int32)], True, (0, 255, 0), 1)
            label = o.category or o.track_id
            cv2.putText(img, label, tuple(uv.T[0].astype(np.int32)), 0, 0.4, (0, 255, 255), 1)
        cv2.imwrite(path, img)
        print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--pitch", type=float, default=0.4)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--arm", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rclpy.init()
    node = Check()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    if not node.perceive.wait_for_service(timeout_sec=30.0):
        raise SystemExit("no /perception/perceive")
    while node.rgb is None or node.info is None or not node.joints:
        time.sleep(0.1)

    node.move(node.head, (2, 3), (args.pitch, args.yaw), ("head_pitch_joint", "head_yaw_joint"))
    time.sleep(2.0)
    for name in ARM_FRAMES:
        p = matrix(node.tf.lookup_transform("pelvis", name, Time(), timeout=Duration(seconds=2.0)))[:3, 3]
        print(f"tf {name}: {p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}")
    first = node.report("first", "head", os.path.join(args.out, "head.jpg"))
    again = [node.report(f"repeat {i}", "head") for i in range(3)]
    print("ids stable over repeats:", all([o.track_id for o in s.objects] == [o.track_id for o in first.objects]
                                          for s in again))
    node.report("torso", "torso")
    if args.arm:
        for i, (label, pose) in enumerate(ARM_POSES.items()):
            node.move(node.arm, range(21, 28), pose, ARM_JOINTS)
            time.sleep(2.0)
            node.report(label, "head", os.path.join(args.out, f"arm{i}.jpg"))
    rclpy.shutdown()


if __name__ == "__main__":
    main()
