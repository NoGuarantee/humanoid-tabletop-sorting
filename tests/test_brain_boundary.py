"""ROS-shaped stubs check owner/effect wiring, not actual middleware integration."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from hack_brain.context import Context, Event, State
from fakes import sort_spec


def load_brain():
    stubs = {
        'rclpy': NS(),
        'rclpy.executors': NS(MultiThreadedExecutor=object),
        'rclpy.node': NS(Node=object),
        'rclpy.qos': NS(DurabilityPolicy=object, QoSProfile=object, ReliabilityPolicy=object),
        'std_msgs.msg': NS(Empty=object, String=object),
        'rosgraph_msgs.msg': NS(Clock=object),
        'ament_index_python.packages': NS(get_package_share_directory=lambda _: ''),
        'hack_brain.clients': NS(Clients=object),
        'hack_brain.grasp_planner': NS(load_db=lambda _: {}),
    }
    path = Path(__file__).resolve().parents[1] / 'src/hack_brain/hack_brain/brain_node.py'
    spec = importlib.util.spec_from_file_location('hack_brain.boundary_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', stubs):
        spec.loader.exec_module(module)
    return module


class BoundaryTests(unittest.TestCase):
    def test_done_retries_even_if_publish_and_logger_fail(self):
        brain = load_brain().Brain
        node = NS(done_attempts=0, done_pub=NS(publish=Mock(side_effect=[RuntimeError()] + [None] * 9)),
                  get_logger=lambda: NS(error=Mock(side_effect=RuntimeError())))
        with patch('time.sleep'):
            brain.publish_done(node)
        self.assertEqual(node.done_attempts, 10)
        self.assertEqual(node.done_pub.publish.call_count, 10)

    def test_worker_shutdown_even_when_mission_raises(self):
        module = load_brain()
        node = NS(mission=Mock(side_effect=RuntimeError('failed')), done_attempts=0,
                  publish_done=Mock())
        module.rclpy.try_shutdown = Mock()
        with self.assertRaises(RuntimeError):
            module.mission_then_exit(node)
        node.publish_done.assert_called_once()
        module.rclpy.try_shutdown.assert_called_once()

    def test_constructor_failure_publishes_fallback_and_shuts_down(self):
        module = load_brain()
        module.rclpy.init = Mock()
        module.rclpy.try_shutdown = Mock()
        module.Brain = Mock(side_effect=ValueError('bad config'))
        module.publish_done_fallback = Mock()
        with self.assertRaises(ValueError):
            module.main()
        module.publish_done_fallback.assert_called_once()
        module.rclpy.try_shutdown.assert_called_once()

    def test_logging_exception_cannot_drop_done_effect(self):
        brain = load_brain().Brain
        node = NS(context=Context(), get_logger=lambda: NS(info=Mock(side_effect=RuntimeError())),
                  publish_done=Mock())
        brain.process(node, Event('ERROR', data='dependency_exception'))
        self.assertEqual(node.context.state, State.DONE)
        node.publish_done.assert_called_once()

    def test_capability_gate_real_entry_effect_is_done(self):
        brain = load_brain().Brain
        node = NS(context=Context(), get_logger=lambda: NS(info=Mock()), publish_done=Mock(),
                  clients=NS(execute=Mock()))
        brain.process(node, Event('START', data=(sort_spec(), frozenset())))
        brain.process(node, Event('TICK', data=(1, 100)))
        node.publish_done.assert_called_once()
        node.clients.execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
