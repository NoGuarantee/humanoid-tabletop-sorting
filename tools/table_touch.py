#!/usr/bin/env python3
"""Where is the table top for the fingers? Lowers a top-down hand in 1 cm steps until the wrist feels it. Debug tool.

The answer checks the table plane fitted from depth (z = -0.1792 + 0.07 x in pelvis) against contact. Park the objects
first (tools/motion_trace.py park) and record (tools/motion_trace.py record OUT) to get the mesh gap of each step.
Needs motion_node running and the bench held. Prints one line per step, sim time included to find it in the trace.
"""
import math
import sys
import threading
import time

import rclpy
from geometry_msgs.msg import WrenchStamped
from hack_brain.grasp_planner import pose, top_down
from move_tcp_check import Check
from rclpy.executors import MultiThreadedExecutor

# (arm, x, y): away from the trays (side trays reach y -+0.53 at x <= 0.45, center tray |y| <= 0.16 at x >= 0.42).
POINTS = [("right", 0.50, -0.33), ("right", 0.38, -0.33), ("right", 0.28, -0.33), ("left", 0.50, 0.33),
          ("left", 0.38, 0.33), ("left", 0.28, 0.33)]
START_Z = 0.13
END_Z = 0.00
STEP = 0.01
SETTLE = 1.0  # sim s before reading the wrench, the arm is still swinging when the goal returns
FORCE, TORQUE = 12.0, 1.2  # N, Nm: above these the wrist feels more than the tracking noise of free moves (5 N, 0.4 Nm)


def main():
    rclpy.init()
    node = Check()
    last = {}
    for arm in ("left", "right"):
        node.create_subscription(WrenchStamped, f"/arm_6dof_{arm}", lambda m, a=arm: last.__setitem__(a, m), 10)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    for arm, x, y in POINTS:
        target = pose(x, y, START_Z, top_down(0.0))
        res = node.move(arm, target, "direct", timeout=240.0)
        print(f"{arm} ({x},{y}): start {res.message if res else 'no result'} "
              f"{res.pos_error * 100 if res else math.nan:.2f} cm", flush=True)
        if res is None or not res.success:
            continue
        z = START_Z
        touched = 0
        while z >= END_Z - 1e-6 and touched < 2:
            z -= STEP
            target = pose(x, y, z, top_down(0.0))
            res = node.move(arm, target, "linear", timeout=60.0)
            t0 = node.get_clock().now().nanoseconds * 1e-9
            while node.get_clock().now().nanoseconds * 1e-9 - t0 < SETTLE:
                time.sleep(0.05)
            f = last[arm].wrench.force
            m = last[arm].wrench.torque
            fn, mn = math.hypot(f.x, f.y, f.z), math.hypot(m.x, m.y, m.z)
            at = node.measure(arm, target)
            touched += fn > FORCE or mn > TORQUE or at[0] > 0.015
            print(f"t {t0:8.2f} {arm} x {x} goal z {z:+.3f} tcp ({at[3][0]:+.3f},{at[3][1]:+.3f},{at[3][2]:+.3f}) "
                  f"err {at[0] * 100:.2f} cm |F| {fn:5.1f} |M| {mn:5.2f} {res.message if res else 'none'}", flush=True)
            if res is None or res.message == "unreachable":
                break
        res = node.move(arm, pose(x, y, START_Z, top_down(0.0)), "direct", timeout=120.0)
        print(f"  back up: {res.message if res else 'no result'}", flush=True)
    rclpy.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
