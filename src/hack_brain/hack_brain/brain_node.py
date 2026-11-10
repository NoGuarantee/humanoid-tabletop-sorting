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
from .grasp_planner import confidence_floors, load_db, skip_names
from .task import parse_task


class Brain(Node):
    def __init__(self):
        super().__init__('hack_brain')
        self.events = queue.PriorityQueue()
        self.sequence = itertools.count()
        self.clock_seen = threading.Event()
        self.done_attempts = 0
        self.declare_parameter('grasp_db', os.path.join(
            get_package_share_directory('hack_brain'), 'config', 'grasp_db.yaml'))
        db = load_db(self.get_parameter('grasp_db').value)
        self.declare_parameter('recorded_grasps', False)
        self.declare_parameter('recorded_objects', 'rhino,brother,philips,nescafe')
        if self.get_parameter('recorded_grasps').value:
            from .recorded_grasps import evaluation_db
            db = evaluation_db(db, True, self.get_parameter('recorded_objects').value)
        from .recorded_grasps import postlift_names
        values = {'skip_names': skip_names(db), 'confidence_floors': confidence_floors(db),
                  'postlift_names': postlift_names(db)}
        for name in Config.__dataclass_fields__:
            if name in values:
                continue
            self.declare_parameter(name, getattr(Config(), name))
            values[name] = self.get_parameter(name).value
        self.mission_context = Context(config=Config(**values))
        if values['skip_names']:
            self.get_logger().info('left untouched: ' + ', '.join(sorted(values['skip_names'])))
        if values['confidence_floors']:
            self.get_logger().info('confidence floors: ' + ', '.join(f'{n} {v:.2f}' for n, v in values['confidence_floors']))
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, '/sim/task', lambda m: self.emit(Event('TASK', data=m.data)), latched)
        self.create_subscription(Clock, '/clock', lambda m: self.clock_seen.set(), 10)
        self.done_pub = self.create_publisher(Empty, '/sim/task/done', 10)
        self.adapters = Clients(self, self.emit, db)
        self.get_logger().info('capabilities: ' + self.adapters.unavailable)

    def emit(self, event):
        priority = 0 if event.kind in ('ERROR', 'CANCEL', 'CONNECTION_LOST') else 1
        self.events.put((priority, next(self.sequence), event))

    def tick(self):
        sim = self.get_clock().now().nanoseconds * 1e-9 if self.clock_seen.is_set() else None
        return Event('TICK', data=(sim, time.monotonic()))

    def process(self, event):
        old = self.mission_context
        self.mission_context, effects = transition(old, event)
        c = self.mission_context
        if event.kind != 'TICK' or effects:
            try:
                record = {}
                if old.state.value == 'OBSERVE' and getattr(event.data, 'objects', None) is not None:
                    # What select saw: an object never picked shows here why (confidence, CUT, missing).
                    record['scene_objects'] = [
                        f"{o.id} {o.name or '-'} {o.category or '-'} {o.confidence:.2f}{' CUT' if o.partial else ''}"
                        for o in event.data.objects]
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
                    'last_recovery': c.last_recovery,
                    'look_side': c.look_side,
                    'operation_elapsed_sim': c.sim - old.active.sim_start if old.active and c.sim is not None else None,
                    **record,
                }))
            except Exception:
                pass  # Logging failure must not discard effects or done publication.

        for effect in effects:
            if effect.kind == 'DONE':
                self.publish_done()
            else:
                try:
                    self.adapters.execute(effect, c)
                except Exception as exc:
                    self.emit(Event('ERROR', data='adapter_exception:' + repr(exc)))

    def publish_done(self, fallback=False):
        # Each attempt is independent; cleanup is never a prerequisite.
        for _ in range(10 if fallback else 3):
            self.done_attempts += 1
            try:
                self.done_pub.publish(Empty())
            except Exception as exc:
                try:
                    self.get_logger().error('done publication failed: ' + repr(exc))
                except Exception:
                    pass
            time.sleep(0.5 if fallback else 0.2)

        try:
            self.get_logger().info('published /sim/task/done')
        except Exception:
            pass

    def mission(self):
        while self.mission_context.state != State.DONE and rclpy.ok():
            try:
                self.process(self.tick())
                if self.mission_context.state != State.DONE:
                    self.adapters.poll(self.mission_context)
                if self.mission_context.state == State.DONE:
                    break
                try:
                    _, _, event = self.events.get(timeout=0.05)
                except queue.Empty:
                    continue
                if event.kind == 'TASK':
                    if self.mission_context.spec is not None:
                        continue
                    event = Event('START', data=(parse_task(event.data), self.adapters.capabilities))
                elif event.kind == 'ROS_SCENE':
                    if not self.mission_context.active or event.op_id != self.mission_context.active.id:
                        continue
                    event = self.adapters.accept_scene(event)
                self.process(event)
            except Exception as exc:
                self.process(Event('ERROR', data='exception:' + repr(exc)))
        if self.mission_context.state != State.DONE:
            self.process(Event('CONNECTION_LOST', data='ros_shutdown'))
        if not self.done_attempts:
            self.publish_done()


def publish_done_fallback(err):
    node = Node('hack_brain_fallback')
    try:
        node.done_attempts = 0
        node.done_pub = node.create_publisher(Empty, '/sim/task/done', 10)
        node.get_logger().error(f'brain failed to start: {err!r}')
        Brain.publish_done(node, fallback=True)
    finally:
        node.destroy_node()


def mission_then_exit(node):
    try:
        node.mission()
    finally:
        try:
            if not node.done_attempts:
                node.publish_done()
        finally:
            rclpy.try_shutdown()


def main():
    rclpy.init()
    try:
        node = Brain()
    except Exception as exc:
        try:
            publish_done_fallback(exc)
        finally:
            rclpy.try_shutdown()
        raise
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    worker = threading.Thread(target=mission_then_exit, args=(node,), daemon=True)
    worker.start()
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.emit(Event('CANCEL', data='shutdown'))
        until = time.monotonic() + node.mission_context.config.cancel_timeout + 6.0
        while worker.is_alive() and rclpy.ok() and time.monotonic() < until:
            executor.spin_once(timeout_sec=0.05)
    finally:
        worker.join(timeout=node.mission_context.config.cancel_timeout + 6.0)
        if not worker.is_alive():
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
