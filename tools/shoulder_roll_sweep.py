#!/usr/bin/env python3
"""Where does shoulder_pitch_r lose its target: a contact on one side of a roll, or the singularity at roll = -pi/2?

Debug only. Raw PD, no motion_node: holds a stuck pose, sweeps shoulder_roll_r through -pi/2 and reads the offset of
every joint. At roll = -pi/2 the shoulder_yaw axis lies on the shoulder_pitch axis, so a load peaking on both sides of
it points at the singularity and a load rising on one side only points at a contact. Needs only the simulator.

  python3 tools/shoulder_roll_sweep.py
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
SETTLE = 4.0  # sim s
NAMES = ARM_JOINTS["right"]
ROLLS = (-0.8, -1.0, -1.2, -1.3, -1.4, -1.5, -1.57, -1.65, -1.75, -1.9, -2.1, -2.3)
# (label, pitch, yaw, elbow, elbow_yaw, wrist_pitch, wrist_roll): stuck pose A, then the same with simple distal joints
SWEEPS = (
    ("pose A", -0.5, 0.686, -1.870, -2.774, 0.981, -0.794),
    ("distal zero", -0.5, 0.0, -1.0, 0.0, 0.0, 0.0),
)


class Probe(Node):
    def __init__(self):
        super().__init__("shoulder_roll_sweep", parameter_overrides=[
            rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)])
        self.q, self.v = {}, {}
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)
        self.pub = self.create_publisher(CmdSetMotorPosition, "/arm/cmd_pos", 10)

    def on_joints(self, msg):
        self.q.update(zip(msg.name, msg.position))
        self.v.update(zip(msg.name, msg.velocity))

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
        return [self.q[n] for n in NAMES], [self.v[n] for n in NAMES]


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
            node.go(step)
        short = [n[:-8] for n in NAMES]
        for label, pitch, *rest in SWEEPS:
            print(label, "load Nm:", " ".join(f"{s[:9]:>9s}" for s in short), " pitch vel", flush=True)
            for roll in ROLLS:
                pose = [pitch, roll, *rest]
                q, v = node.go(pose)
                loads = [KP * (p - x) for p, x in zip(pose, q)]
                print(f"  roll {roll:+.2f} pitch {q[0]:+.3f} " + " ".join(f"{x:+9.1f}" for x in loads)
                      + f"  {v[0]:+.3f}", flush=True)
    finally:
        executor.shutdown()
        spin.join(timeout=5.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
