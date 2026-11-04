"""Task orchestrator. Owner: brain/infra.

INIT -> LOOK -> PERCEIVE -> SELECT -> PICK -> PLACE -> BUDGET -> (PERCEIVE | DONE), see docs/ARCHITECTURE.md.
The mission runs in its own thread with blocking calls; the node spins in a MultiThreadedExecutor.
/sim/task/done is published in every outcome, including exceptions.

Never use the debug topics here (/sim/task/check, /sim/objects/set, /sim/objects/randomize): they are off at the
final check and calls to them are logged. tools/check_rules.sh fails on them.
"""
import math
import os
import threading
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, String

from .clients import Clients
from .grasp_planner import load_db
from .skills import Skills
from .task import parse_task

LOOK_TABLE = {"head_pitch_joint": 0.4, "head_yaw_joint": 0.0, "body_yaw_joint": 0.0}


class Brain(Node):
    def __init__(self):
        super().__init__("hack_brain")
        self.declare_parameter("min_confidence", 0.6)
        self.declare_parameter("cycle_estimate", 40.0)  # sim seconds for one pick-and-place, used by the budget
        self.declare_parameter("time_margin", 10.0)  # sim seconds in reserve: our start-up and the end of the task
        self.declare_parameter("grasp_db", os.path.join(
            get_package_share_directory("hack_brain"), "config", "grasp_db.yaml"))
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.task_msg = None
        self.names_msg = None
        self.create_subscription(String, "/sim/task", lambda m: setattr(self, "task_msg", m.data), latched)
        self.create_subscription(String, "/sim/objects/names", lambda m: setattr(self, "names_msg", m.data), latched)
        self.done_pub = self.create_publisher(Empty, "/sim/task/done", 10)
        self.api = Clients(self)
        self.skills = Skills(self.api, load_db(self.get_parameter("grasp_db").value), self.get_logger())
        self.deadline = math.inf

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def time_left(self):
        return self.deadline - self.now()

    # ---------------------------------------------------------------- mission
    def mission(self):
        log = self.get_logger()
        try:
            while rclpy.ok() and (self.task_msg is None or self.now() == 0.0):
                time.sleep(0.1)  # wait for /sim/task and /clock
            spec = parse_task(self.task_msg)
            # The limit runs in sim time from the moment the layout is ready and run.sh starts, model loading included
            # (organisers' answer). The sim is not restarted between layouts, so its clock says nothing about that
            # moment: count from here, the launch start-up before this line is covered by time_margin.
            self.deadline = self.now() + spec.time_limit - self.get_parameter("time_margin").value
            log.info(f"task {spec.task} {spec.variant} | {spec.n_items} items | limit {spec.time_limit:.0f} s, "
                     f"{self.time_left():.0f} s left")
            if not self.api.wait_ready(timeout=60.0):
                log.error("perception or motion is not up")
                return
            if spec.kind == "sort":
                self.run_sort(spec)
            else:
                self.run_tubes(spec)
        except Exception as e:
            log.error(f"mission failed: {e!r}")
        finally:
            self.finish()

    def finish(self):
        try:
            self.api.move_joints(list(LOOK_TABLE), list(LOOK_TABLE.values()))
            # TODO(motion): arms to a safe home pose
        except Exception:
            pass
        for _ in range(3):
            self.done_pub.publish(Empty())
            time.sleep(0.2)
        self.get_logger().info("published /sim/task/done")

    # ---------------------------------------------------------------- sorting
    def run_sort(self, spec):
        log = self.get_logger()
        min_conf = self.get_parameter("min_confidence").value
        cycle = self.get_parameter("cycle_estimate").value
        # TODO(brain): split failures. Not lifted -> skip (0 points). Lifted and dropped on the table -> it is -10 now,
        #   retry it first (+20 if it reaches its tray); it may lie elsewhere, so match it by category and position.
        failed = set()
        while rclpy.ok() and self.time_left() > cycle:
            self.api.move_joints(list(LOOK_TABLE), list(LOOK_TABLE.values()))
            scene = self.api.perceive("head", "table")
            if scene is None:
                log.warn("no scene, stopping")
                return
            trays = {c.id: c for c in scene.containers if c.kind == "tray"}
            cands = [o for o in scene.objects
                     if o.category in spec.categories and spec.categories[o.category] in trays
                     and o.confidence >= min_conf and not o.partially_visible and key(o) not in failed]
            if not cands:
                log.info("nothing left to sort")
                return
            # nearest to its tray first: shortest transport
            cands.sort(key=lambda o: dist(o.center, trays[spec.categories[o.category]].pose.position))
            obj = cands[0]
            tray = trays[spec.categories[obj.category]]
            log.info(f"pick {obj.name or obj.category} ({obj.confidence:.2f}) -> {tray.id}, "
                     f"{self.time_left():.0f} s left")
            grasp = self.skills.pick(obj, scene)
            if grasp is None:
                failed.add(key(obj))
                continue
            if not self.skills.place(grasp, tray):
                failed.add(key(obj))

    # ---------------------------------------------------------------- tubes
    def run_tubes(self, spec):
        # TODO(M4): torso camera on the table racks -> pick -> waist +1.57 (in parallel with the lift) ->
        #           head camera on the hood racks -> slot by spec.hood_racks_for(type, cap) -> insert -> back
        self.get_logger().warn(f"tubes/{spec.variant} not implemented, skipping (0 points, no penalty)")


def key(obj):
    """Identity of an object between perception passes."""
    if obj.track_id:
        return obj.track_id
    return (round(obj.center.x / 0.03), round(obj.center.y / 0.03))


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def publish_done_fallback(err):
    """Brain() itself failed: still publish /sim/task/done, as the rules require in every outcome."""
    node = Node("hack_brain_fallback")
    node.get_logger().error(f"brain failed to start: {err!r}; publishing /sim/task/done")
    pub = node.create_publisher(Empty, "/sim/task/done", 10)
    for _ in range(10):  # give discovery time to match the simulator's subscriber
        pub.publish(Empty())
        time.sleep(0.5)
    node.get_logger().info("published /sim/task/done")  # tools/bench.sh waits for this line
    node.destroy_node()


def main():
    rclpy.init()
    try:
        node = Brain()
    except Exception as e:
        publish_done_fallback(e)
        rclpy.try_shutdown()
        raise
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=node.mission, daemon=True).start()
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
