#!/usr/bin/env python3
"""Grid of top-down MoveTcp goals over the table, checked against tf.

The table top is 0.18 m below the pelvis (scene.yaml: table height 0.70, standing pelvis 0.88).
`top_down` is the quaternion grasp_planner will send: palm down, fingers across `yaw`.
For yaw 0 the TCP z axis in pelvis should be (0, 0, -1).

With the simulator running and only hack_motion up (the full launch exits when brain publishes done):

  bash tools/remote.sh dev --timeout 2400 -- 'bash setup.sh && source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install --build-base /tmp/b --install-base /tmp/i && source /tmp/i/setup.bash && \
    { ros2 run hack_motion motion_node --ros-args -p use_sim_time:=true & pid=$!; \
      for i in $(seq 60); do ros2 action list | grep -q motion/move_tcp && break; sleep 2; done; \
      python3 tools/move_tcp_check.py; status=$?; kill $pid; exit $status; }'
"""
import math
import sys
import threading
import time

import rclpy
from geometry_msgs.msg import Pose
from hack_brain.grasp_planner import pose, top_down
from hack_interfaces.action import MoveTcp
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from tf2_ros import Buffer, TransformListener

# (arm, x, y, z, yaw) in pelvis, top-down. Chest height and out to the side: the hanging arm
# crosses the thigh spheres on the way to a low pose over the near edge of the table.
GOALS = [
    ("right", 0.40, -0.35, 0.30, 0.0),
    ("right", 0.35, -0.40, 0.22, 0.0),
    ("right", 0.45, -0.30, 0.38, 0.0),
    ("left", 0.40, 0.35, 0.30, 0.0),
]
POS_LIMIT = 0.01  # m, milestone M1
ROT_LIMIT = 0.1  # rad, the MoveTcp default


def quat_angle(a, b):
    dot = min(1.0, abs(sum(x * y for x, y in zip(a, b))))
    return 2.0 * math.acos(dot)


def rot_z(xyzw):
    """Third column of the rotation matrix: where the TCP +z axis points."""
    x, y, z, w = xyzw
    return (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))


def wait(future, timeout):
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    return done.wait(timeout)


class Check(Node):
    def __init__(self):
        super().__init__("move_tcp_check", parameter_overrides=[
            Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.cli = ActionClient(self, MoveTcp, "motion/move_tcp")
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)

    def move(self, arm, target, mode, timeout):
        if not self.cli.wait_for_server(timeout_sec=30.0):
            return None
        fut = self.cli.send_goal_async(MoveTcp.Goal(
            arm=arm, target=target, mode=mode, pos_tol=POS_LIMIT, rot_tol=ROT_LIMIT))
        if not wait(fut, 10.0) or not fut.result().accepted:
            return None
        res = fut.result().get_result_async()
        if not wait(res, timeout):
            self.get_logger().error(f"no result in {timeout} s")
            return None
        return res.result().result

    def measure(self, arm, target):
        link = "left_tcp_link" if arm == "left" else "right_tcp_link"
        tf = None
        for _ in range(20):
            try:
                tf = self.tf.lookup_transform("pelvis", link, rclpy.time.Time())
                break
            except Exception:
                time.sleep(0.05)
        if tf is None:
            return None
        t = tf.transform.translation
        r = tf.transform.rotation
        p = target.position
        o = target.orientation
        pos = math.sqrt((t.x - p.x) ** 2 + (t.y - p.y) ** 2 + (t.z - p.z) ** 2)
        got = (r.x, r.y, r.z, r.w)
        want = (o.x, o.y, o.z, o.w)
        return pos, quat_angle(got, want), rot_z(got), (t.x, t.y, t.z)


def main():
    rclpy.init()
    node = Check()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    reached = 0
    bad = 0
    linear_done = False
    try:
        here = node.measure("right", pose(0, 0, 0, top_down(0)))
        if here is None:
            print("FAIL no tf for right_tcp_link", flush=True)
            sys.exit(1)
        _, _, axis, where = here
        print(f"right tcp now ({where[0]:.3f},{where[1]:.3f},{where[2]:.3f})  "
              f"z-axis ({axis[0]:+.2f},{axis[1]:+.2f},{axis[2]:+.2f})", flush=True)
        # The hanging pose has the forearms inside the thighs. The node steps out to the arms-out
        # pose on the first goal, and the goals below are planned from there. A sideways nudge of
        # that hang drives the wrist into the thigh, and cuRobo then refuses to start.
        for i, (arm, x, y, z, yaw) in enumerate(GOALS):
            target = pose(x, y, z, top_down(yaw))
            t0 = time.perf_counter()
            result = node.move(arm, target, "direct", timeout=240.0)
            wall = time.perf_counter() - t0
            if result is None:
                print(f"FAIL {arm} direct no result  wall {wall:.1f}s", flush=True)
                bad += 1
                continue
            measured = node.measure(arm, target) if result.success else None
            if not result.success:
                print(f"---- {arm} ({x:.2f},{y:.2f},{z:.2f}) yaw {yaw:.2f} direct  {result.message}"
                      f"  node {result.pos_error * 100:.1f} cm  wall {wall:.1f}s", flush=True)
                continue
            if measured is None:
                print(f"FAIL {arm} direct success but no tf", flush=True)
                bad += 1
                continue
            pos, rot, axis, where = measured
            ok = pos <= POS_LIMIT and rot <= ROT_LIMIT
            reached += 1
            bad += not ok
            print(f"{'OK  ' if ok else 'FAIL'} {arm} ({x:.2f},{y:.2f},{z:.2f}) yaw {yaw:.2f} direct  "
                  f"tf {pos * 100:.2f} cm  {math.degrees(rot):.1f} deg  "
                  f"tcp ({where[0]:.3f},{where[1]:.3f},{where[2]:.3f})  "
                  f"z-axis ({axis[0]:+.2f},{axis[1]:+.2f},{axis[2]:+.2f})  "
                  f"node {result.pos_error * 100:.2f} cm {math.degrees(result.rot_error):.1f} deg  "
                  f"wall {wall:.1f}s", flush=True)
            if ok and not linear_done:
                linear_done = True
                up = Pose()
                up.position.x, up.position.y, up.position.z = x, y, z + 0.05
                up.orientation = target.orientation
                t0 = time.perf_counter()
                result = node.move(arm, up, "linear", timeout=120.0)
                wall = time.perf_counter() - t0
                measured = node.measure(arm, up) if result is not None and result.success else None
                if result is None or measured is None:
                    print(f"FAIL {arm} linear {None if result is None else result.message}  wall {wall:.1f}s",
                          flush=True)
                    bad += 1
                else:
                    pos, rot, axis, where = measured
                    ok = pos <= POS_LIMIT and rot <= ROT_LIMIT
                    bad += not ok
                    print(f"{'OK  ' if ok else 'FAIL'} {arm} +5 cm linear  "
                          f"tf {pos * 100:.2f} cm  {math.degrees(rot):.1f} deg  "
                          f"z-axis ({axis[0]:+.2f},{axis[1]:+.2f},{axis[2]:+.2f})  "
                          f"{result.message}  wall {wall:.1f}s", flush=True)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    print(f"reached {reached}/{len(GOALS)} direct goals, failures {bad}", flush=True)
    if reached == 0 or bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
