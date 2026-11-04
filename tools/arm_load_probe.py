#!/usr/bin/env python3
"""What load do the arm drives really hold? No motion_node and no compensation: the raw PD against the world.

Debug only. Holds a pose, approaches every joint's target from both sides and reads where it settles. With kp known,
the mean offset is the static load and half the difference between the two sides is friction. Compare with the URDF
gravity torque to see whether the model behind the droop numbers is right. Needs only the simulator.

  python3 tools/arm_load_probe.py
"""
import threading
import time

import rclpy
import rclpy.parameter
from bodyctrl_msgs.msg import CmdSetMotorPosition, SetMotorPosition
from hack_motion.robot import ARM_JOINTS, JOINTS, raise_steps
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

KP = 80.0
STEP = 0.15  # rad, how far to the side each approach starts
SETTLE = 2.5  # sim s
NAMES = ARM_JOINTS["right"]
POSES = {
    "raised": [0.0, -0.2, 0.0, -2.0, 0.0, 0.0, 0.0],
    "forward horizontal": [-1.57, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    "pose A": [-0.2, -1.656, 0.686, -1.870, -2.774, 0.981, -0.794],
    "pose A pitch -0.5": [-0.5, -1.656, 0.686, -1.870, -2.774, 0.981, -0.794],
    "pose A roll -1.2": [-0.5, -1.2, 0.686, -1.870, -2.774, 0.981, -0.794],
    "pose B": [0.45, -1.618, 0.001, -2.097, -2.861, 0.377, -1.082],
    "pose C": [-0.5, -1.551, 1.128, -1.730, -0.706, 0.450, 0.849],
}


class Probe(Node):
    def __init__(self):
        super().__init__("arm_load_probe", parameter_overrides=[
            rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)])
        self.q = {}
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)
        self.pub = self.create_publisher(CmdSetMotorPosition, "/arm/cmd_pos", 10)

    def on_joints(self, msg):
        self.q.update(zip(msg.name, msg.position))

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def go(self, pose, wait=SETTLE):
        msg = CmdSetMotorPosition()
        msg.cmds = [SetMotorPosition(name=JOINTS[n][0], pos=float(p), spd=0.8, cur=1.0)
                    for n, p in zip(NAMES, pose)]
        self.pub.publish(msg)
        t_end = self.now() + wait
        while self.now() < t_end:
            time.sleep(0.02)
        return [self.q[n] for n in NAMES]


def main():
    rclpy.init()
    node = Probe()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        while len(node.q) < len(NAMES) or node.now() == 0.0:
            time.sleep(0.1)
        for step in raise_steps("right"):
            node.go(step, wait=4.0)
        for label, pose in POSES.items():
            node.go(pose, wait=5.0)
            # both sides at once: every joint starts STEP past the target one way, then the other
            above = node.go([p + STEP for p in pose], wait=1.5) and node.go(pose)
            below = node.go([p - STEP for p in pose], wait=1.5) and node.go(pose)
            print(label, flush=True)
            for i, n in enumerate(NAMES):
                # offset = target - position: the sign says which way the load pushes the joint
                a, b = pose[i] - above[i], pose[i] - below[i]
                print(f"  {n[:-8]:15s} target {pose[i]:+.3f}  from above {above[i]:+.3f} from below {below[i]:+.3f}  "
                      f"load {KP * (a + b) / 2:+6.1f} Nm  friction {KP * abs(a - b) / 2:5.1f} Nm", flush=True)
    finally:
        executor.shutdown()
        spin.join(timeout=5.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
