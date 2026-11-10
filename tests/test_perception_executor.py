"""Bound opt-in dispatch while keeping blocking legacy sensor waits live."""
import ast
from pathlib import Path
import threading
from types import SimpleNamespace as NS
import unittest

from hack_perception.rgbd_frames import CAMERAS, STREAMS, PairBuffer

try:
    import rclpy
except ImportError:
    rclpy = None


def factory(executor):
    path = Path(__file__).resolve().parents[1]/'src/hack_perception/hack_perception/perception_node.py'
    tree = ast.parse(path.read_text())
    function = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                    and item.name == 'perception_executor')
    namespace = {'MultiThreadedExecutor': executor}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['perception_executor']


class PerceptionExecutorTests(unittest.TestCase):
    def test_only_active_robot_filter_changes_legacy_dispatch_configuration(self):
        make = factory(lambda **kwargs: kwargs)
        context = object()
        for model in (None, NS(robot_self_filter=False)):
            self.assertEqual(make(NS(model_poses=model, context=context)),
                             dict(num_threads=8, context=context))
        self.assertEqual(make(NS(model_poses=NS(robot_self_filter=True), context=context)),
                         dict(num_threads=2, context=context))

    @unittest.skipIf(rclpy is None, 'real ROS unavailable on host')
    def test_optin_executor_delivers_fresh_pair_during_blocking_legacy_wait(self):
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
        from rclpy.context import Context
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        context = Context(); rclpy.init(context=context)
        node = Node('perception_executor_sensor_wait_test', context=context)
        node.model_poses = NS(robot_self_filter=True)
        executor = factory(MultiThreadedExecutor)(node)
        executor.add_node(node)
        condition = threading.Condition()
        entered, completed = threading.Event(), threading.Event()
        buffer, result = PairBuffer(), []
        stamp = 1_000_000_000

        def wait_for_frame():
            waiter.cancel()
            entered.set()
            with condition:
                result.append(condition.wait_for(
                    lambda: buffer.latest(stamp, 350_000_000) is not None, timeout=1.5))
            completed.set()

        def deliver_frame():
            if not entered.is_set():
                return
            source.cancel()
            frame = NS(header=NS(stamp=NS(sec=1, nanosec=0)))
            with condition:
                for camera in CAMERAS:
                    for kind in STREAMS:
                        buffer.add(camera, kind, frame)
                condition.notify_all()

        waiter = node.create_timer(.005, wait_for_frame,
                                   callback_group=MutuallyExclusiveCallbackGroup())
        source = node.create_timer(.01, deliver_frame,
                                   callback_group=MutuallyExclusiveCallbackGroup())
        thread = threading.Thread(target=executor.spin, daemon=True)
        try:
            self.assertEqual(executor._executor._max_workers, 2)
            thread.start()
            self.assertTrue(completed.wait(2.), 'blocking legacy wait starved sensor callbacks')
            self.assertEqual(result, [True])
            self.assertEqual(buffer.latest(stamp, 350_000_000)[0], stamp)
        finally:
            waiter.cancel(); source.cancel()
            with condition:
                condition.notify_all()
            executor.shutdown(timeout_sec=2.)
            thread.join(timeout=2.)
            node.destroy_node(); context.try_shutdown()


if __name__ == '__main__':
    unittest.main()
