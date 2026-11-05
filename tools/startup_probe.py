#!/usr/bin/env python3
"""Stands in for brain to time the start of the solution up to the first plan, on a clean container per layout.

Brain cannot measure this yet: without perception it publishes done before cuRobo has loaded. This does what the
mission does first (the head to the table, then MoveTcp), publishes done and exits; tools/startup_probe.launch.py is
the real launch with brain swapped for this. `eval --layouts` prints tools/startup_report.py after each layout:

  bash tools/remote.sh eval sort3 --pool 4 --layouts 2 --run 'ros2 launch tools/startup_probe.launch.py'

PROBE_DONE_AFTER=S instead publishes done S wall seconds after the probe started, with no goals: done while motion is
still loading, as brain does when it gives up early. `eval` prints how long run.sh took to end after that. $LAYOUT is
the layout number:

  --run 'PROBE_DONE_AFTER=$((LAYOUT*3)) ros2 launch tools/startup_probe.launch.py'
"""
import os
import threading
import time

import rclpy
from hack_brain.grasp_planner import pose, top_down
from hack_interfaces.action import MoveJoints, MoveTcp
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, String

LOOK_TABLE = {"head_pitch_joint": 0.4, "head_yaw_joint": 0.0, "body_yaw_joint": 0.0}  # brain_node.LOOK_TABLE
# Goals from tools/move_tcp_check.py, high above the objects. The second shows what a plan costs once warm.
GOALS = [("right", 0.40, -0.35, 0.30, 0.0), ("right", 0.30, -0.15, 0.25, 0.8)]
DONE_AFTER = os.environ.get("PROBE_DONE_AFTER")
TIMEOUT = 120.0  # s, wall clock: a load that long has failed anyway


class Probe(Node):
    def __init__(self):
        super().__init__("startup_probe", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.task = None
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, "/sim/task", lambda m: setattr(self, "task", m.data), latched)
        self.done_pub = self.create_publisher(Empty, "/sim/task/done", 10)
        self.joints = ActionClient(self, MoveJoints, "motion/move_joints")
        self.tcp = ActionClient(self, MoveTcp, "motion/move_tcp")

    def mark(self, what):
        self.get_logger().info(f"probe: {what}, sim {self.get_clock().now().nanoseconds * 1e-9:.2f}")

    def call(self, client, goal):
        done = threading.Event()
        client.wait_for_server(timeout_sec=TIMEOUT)
        future = client.send_goal_async(goal)
        future.add_done_callback(lambda _: done.set())
        done.wait(TIMEOUT)
        done.clear()
        result = future.result().get_result_async()
        result.add_done_callback(lambda _: done.set())
        done.wait(TIMEOUT)
        return result.result().result if result.done() else None

    def clock_samples(self):
        """Maps the wall clock to the sim clock in the log, for the moments only the wall clock is known."""
        while rclpy.ok():
            self.mark("clock")
            time.sleep(2.0)

    def mission(self):
        try:
            while rclpy.ok() and (self.task is None or self.get_clock().now().nanoseconds == 0):
                time.sleep(0.05)
            threading.Thread(target=self.clock_samples, daemon=True).start()
            self.mark("task and clock")
            if DONE_AFTER:
                time.sleep(float(DONE_AFTER))
                return
            res = self.call(self.joints, MoveJoints.Goal(joints=list(LOOK_TABLE), positions=list(LOOK_TABLE.values())))
            self.mark(f"head at the table {res.success if res else None}")
            for arm, x, y, z, yaw in GOALS:
                target = pose(x, y, z, top_down(yaw))
                res = self.call(self.tcp, MoveTcp.Goal(arm=arm, target=target, mode="direct", max_time=15.0))
                self.mark(f"move_tcp result {res.success if res else None} {res.message if res else 'none'}")
        except Exception as e:
            self.get_logger().error(f"probe failed: {e!r}")
        finally:
            for _ in range(3):
                self.done_pub.publish(Empty())
                time.sleep(0.2)
            self.get_logger().info("published /sim/task/done")
            rclpy.try_shutdown()


def main():
    rclpy.init()
    node = Probe()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=node.mission, daemon=True).start()
    try:
        executor.spin()
    except Exception:
        pass


if __name__ == "__main__":
    main()
