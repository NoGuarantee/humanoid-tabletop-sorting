"""Single-owner mission event loop; callbacks enqueue, adapters execute effects."""
import json
import itertools
import os
import queue
import threading
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Empty, String

from .clients import Clients
from .context import Config, Context, Event, State
from .fsm import transition
from .grasp_planner import load_db
from .task import parse_task


class Brain(Node):
    def __init__(self):
        super().__init__('hack_brain')
        self.events = queue.PriorityQueue()
        self.sequence = itertools.count()
        self.clock_seen = threading.Event()
        self.done_attempts = 0
        values = {}
        for name in Config.__dataclass_fields__:
            self.declare_parameter(name, getattr(Config(), name))
            values[name] = self.get_parameter(name).value
        self.context = Context(config=Config(**values))
        self.declare_parameter('grasp_db', os.path.join(
            get_package_share_directory('hack_brain'), 'config', 'grasp_db.yaml'))
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, '/sim/task', lambda m: self.emit(Event('TASK', data=m.data)), latched)
        self.create_subscription(Clock, '/clock', lambda m: self.clock_seen.set(), 10)
        self.done_pub = self.create_publisher(Empty, '/sim/task/done', 10)
        self.clients = Clients(self, self.emit, load_db(self.get_parameter('grasp_db').value))
        self.get_logger().info('capabilities: ' + self.clients.unavailable)

    def emit(self, event):
        priority = 0 if event.kind in ('ERROR', 'CANCEL', 'CONNECTION_LOST') else 1
        self.events.put((priority, next(self.sequence), event))

    def tick(self):
        sim = self.get_clock().now().nanoseconds * 1e-9 if self.clock_seen.is_set() else None
        return Event('TICK', data=(sim, time.monotonic()))

    def process(self, event):
        old = self.context
        self.context, effects = transition(old, event)
        c = self.context
        if event.kind != 'TICK' or effects:
            try:
                self.get_logger().info(json.dumps({
                    'event': event.kind, 'operation_id': event.op_id,
                    'active_operation_id': c.active.id if c.active else None,
                    'from': old.state.value, 'to': c.state.value,
                    'target': c.target.id if c.target else None, 'hand': c.hand.value,
                    'reason': c.reason, 'blocked': c.blocked, 'sim': c.sim,
                    'remaining_sim': c.deadline - c.sim if c.sim is not None else None,
                    'attempt': c.history[c.target.id].attempts if c.target else 0,
                    'lift': c.history[c.target.id].lift.value if c.target else None,
                    'scene_stamp': c.scene.stamp if c.scene else None,
                    'operation_elapsed_sim': c.sim - old.active.sim_start if old.active and c.sim is not None else None,
                }))
            except Exception:
                pass  # Logging failure must not discard effects or done publication.

        for effect in effects:
            if effect.kind == 'DONE':
                self.publish_done()
            else:
                try:
                    self.clients.execute(effect, c)
                except Exception as exc:
                    self.emit(Event('ERROR', data='adapter_exception:' + repr(exc)))

    def publish_done(self):
        # Each attempt is independent; cleanup is never a prerequisite.
        for _ in range(3):
            self.done_attempts += 1
            try:
                self.done_pub.publish(Empty())
            except Exception as exc:
                try:
                    self.get_logger().error('done publication failed: ' + repr(exc))
                except Exception:
                    pass
            time.sleep(0.05)

    def mission(self):
        while self.context.state != State.DONE and rclpy.ok():
            try:
                self.process(self.tick())
                if self.context.state == State.DONE:
                    break
                try:
                    _, _, event = self.events.get(timeout=0.05)
                except queue.Empty:
                    continue
                if event.kind == 'TASK':
                    if self.context.spec is not None:
                        continue
                    event = Event('START', data=(parse_task(event.data), self.clients.capabilities))
                elif event.kind == 'ROS_SCENE':
                    if not self.context.active or event.op_id != self.context.active.id:
                        continue
                    event = self.clients.accept_scene(event)
                self.process(event)
            except Exception as exc:
                self.process(Event('ERROR', data='exception:' + repr(exc)))
        if self.context.state != State.DONE:
            self.process(Event('CONNECTION_LOST', data='ros_shutdown'))
        if not self.done_attempts:
            self.publish_done()


def main():
    rclpy.init()
    node = Brain()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    worker = threading.Thread(target=node.mission, daemon=True)
    worker.start()
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.emit(Event('CANCEL', data='shutdown'))
        # Keep callbacks alive for bounded cancellation when the ROS context permits.
        until = time.monotonic() + node.context.config.cancel_timeout + 0.5
        while worker.is_alive() and rclpy.ok() and time.monotonic() < until:
            executor.spin_once(timeout_sec=0.05)
    finally:
        if not node.done_attempts:
            node.publish_done()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
