"""Timing diagnostics preserve RGB-D scheduling, lock order and freshness."""
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def service_module():
    # Load this fixture independently so CPU tests do not replace a production
    # module or require an installed ROS distribution.
    modules = {
        'rclpy.callback_groups': SimpleNamespace(MutuallyExclusiveCallbackGroup=object),
        'rclpy.clock': SimpleNamespace(Clock=object, ClockType=object, JumpThreshold=object),
        'rclpy.duration': SimpleNamespace(Duration=object),
        'rclpy.time': SimpleNamespace(Time=object),
        'std_srvs.srv': SimpleNamespace(Trigger=object),
        'tf2_ros': SimpleNamespace(TransformException=RuntimeError),
    }
    path = Path(__file__).parents[1]/'src/hack_perception/hack_perception/rgbd_service.py'
    spec = importlib.util.spec_from_file_location('hack_perception._timing_fixture', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class TimedLock:
    def __init__(self, clock, delay=0.):
        self.lock, self.clock, self.delay = threading.RLock(), clock, delay
        self.acquisitions, self.held = 0, False

    def __enter__(self):
        self.clock[0] += self.delay
        self.lock.acquire()
        self.acquisitions += 1
        self.held = True

    def __exit__(self, *args):
        self.held = False
        self.lock.release()


class ServiceTimingTest(unittest.TestCase):
    def setUp(self):
        self.module = service_module()
        self.clock = [10.]
        self.patches = [patch.object(self.module, 'monotonic', side_effect=lambda: self.clock[0]),
                        patch.object(self.module, 'perf_counter', side_effect=lambda: self.clock[0]),
                        patch.object(self.module, 'wall_time', side_effect=lambda: 1000.+self.clock[0])]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.service = self.module.ModelPoseService.__new__(self.module.ModelPoseService)
        service = self.service
        service._init_diagnostics()
        service.lock = TimedLock(self.clock)
        self.events = []
        def log(message):
            self.assertFalse(service.lock.held, 'new timing logs must be outside the input lock')
            self.events.append(json.loads(message))
        service.node = SimpleNamespace(get_logger=lambda: SimpleNamespace(info=log),
            get_parameter=lambda name: SimpleNamespace(value=''))
        service.now_ns = lambda: 1_000_000_000
        service.pool_override, service.task, service.generation = 4, None, 2
        service.state, service.on_demand = 'ready', True
        service.pending = service.scene = service.pose = service.prepared_key = None
        service.last_scene_key = service.last_pose_key = None
        service.worker_deadline, service.next_scene_after = None, 0.
        service.pose_consumed, service.want_pose = False, False
        service.minimum_pose_stamp_ns = 0
        service.buffer = self.module.PairBuffer()
        service.worker = SimpleNamespace(poll=Mock(return_value=None), submit=Mock(return_value=True),
                                         process=SimpleNamespace(is_alive=lambda: True))

    def test_callback_reports_lock_wait_without_changing_consumption_or_payload(self):
        service = self.service
        service.lock.delay = .4
        payload = self.module.response('ok', stamp=900_000_000, pool=4,
                                       objects=[{'name': 'private_object_geometry'}])
        service.pose = ((2, 900_000_000), payload)
        service._diag_pump_completed = 8.
        result = service.on_estimate(None, SimpleNamespace())
        self.assertTrue(result.success)
        self.assertEqual(json.loads(result.message), payload)
        self.assertTrue(service.pose_consumed)
        self.assertEqual(service.lock.acquisitions, 1)
        event = self.events[-1]
        self.assertEqual(event['sequence'], 1)
        self.assertAlmostEqual(event['callback_wall_s']['estimate_lock_wait_s'], .4)
        self.assertAlmostEqual(event['last_pump_age_s'], 2.4)
        self.assertEqual(event['arrival_wall_s'], 1010.)
        self.assertNotIn('private_object_geometry', json.dumps(event))
        next_result = service.on_estimate(None, SimpleNamespace())
        self.assertEqual(json.loads(next_result.message)['status'], 'awaiting_fresh_pose')
        self.assertEqual(service.minimum_pose_stamp_ns, 1_000_000_000)
        self.assertEqual(self.events[-1]['sequence'], 2)

    def test_stale_pose_remains_rejected_and_logging_failure_does_not_break_response(self):
        service = self.service
        service.pose = ((1, 900_000_000), self.module.response('ok', stamp=900_000_000))
        service.node.get_logger = lambda: SimpleNamespace(info=Mock(side_effect=RuntimeError('closed log')))
        result = service.on_estimate(None, SimpleNamespace())
        self.assertFalse(result.success)
        self.assertEqual(json.loads(result.message)['status'], 'awaiting_fresh_pose')

    def test_waiting_pump_heartbeat_is_limited_and_exposes_pending_job_age(self):
        service = self.service
        service.pending = ('pose', (2, 900_000_000), None)
        service._diag_pending_started = 8.
        service.worker_deadline = 30.
        for index in range(100):
            self.clock[0] = 10.+index*.1
            service.pump()
        self.assertEqual(len(self.events), 2)
        self.assertEqual(self.events[0]['pending_kind'], 'pose')
        self.assertAlmostEqual(self.events[0]['pending_job_age_s'], 2.)
        self.assertEqual(self.events[-1]['last_pump_wall_s']['last_stage'], 'waiting_worker')
        service.worker.submit.assert_not_called()
        self.assertEqual(service.pending, ('pose', (2, 900_000_000), None))

    def test_decode_and_submit_timings_preserve_selected_exposure_and_job(self):
        service = self.service
        service.want_pose = True
        views, sensors = [object(), object()], ['unchanged_head', 'unchanged_torso']
        service.buffer.latest = Mock(return_value=(950_000_000, views))
        def decode(*args):
            self.clock[0] += .3
            return sensors
        with patch.object(self.module, 'decode_pair', side_effect=decode):
            service.pump()
        service.worker.submit.assert_called_once_with('scene', (2, 950_000_000), (sensors, None))
        self.assertEqual(service.pending, ('scene', (2, 950_000_000), views))
        self.assertEqual(service.worker_deadline, 30.)  # Existing monotonic deadline unchanged.
        event = self.events[-1]
        self.assertAlmostEqual(event['last_pump_wall_s']['decode_tf_s'], .3)
        self.assertAlmostEqual(event['latest_pair_age_sim_s'], .05)
        self.assertNotIn('unchanged_head', json.dumps(event))

    def test_frame_arrival_is_measured_before_waiting_for_shared_input_lock(self):
        service = self.service
        service.lock.delay = .7
        message = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=950_000_000)))
        service.store('cam_head', 'color', message)
        state = service._diagnostic_state(self.clock[0], service.now_ns())
        self.assertAlmostEqual(state['last_frame_received_age_s'], .7)
        self.assertAlmostEqual(state['latest_frame_age_sim_s'], .05)
        self.assertIs(service.buffer.frames['cam_head', 'color'][950_000_000], message)

    def test_one_pose_submission_per_exposure_even_after_failure_and_reset(self):
        service = self.service
        service.on_demand = False
        key = (2, 950_000_000)
        service.scene = (key, [None, None], (None, []))
        service.prepared_key = key
        service.want_pose = True
        service.pump()
        service.worker.submit.assert_called_once_with('pose', key, 4)
        service.worker.poll.return_value = ('job_failed', key, 'test failure', .1)
        service.node.get_logger = lambda: SimpleNamespace(info=lambda x: None, warning=lambda x: None)
        service.estimate()
        service.pump()
        self.assertEqual(service.worker.submit.call_count, 1)
        service.clear_baseline = Mock()
        service.reset()
        self.assertIsNone(service.scene)
        self.assertIsNone(service.pose)
        self.assertFalse(service.want_pose)
        self.assertEqual(service.generation, 3)

    def test_worker_shutdown_does_not_hold_input_lock(self):
        service = self.service
        service.worker_restarts = 2
        service.clear_baseline = Mock()
        service.node.get_logger = lambda: SimpleNamespace(error=lambda x: None)
        def close():
            self.assertFalse(service.lock.held)
            self.assertEqual(service.state, 'worker_unavailable')
        service.worker.close = close
        service.restart_worker('test', 10.)
        self.assertIsNone(service.pending)

    def test_expired_during_decode_is_not_submitted(self):
        service = self.service
        service.want_pose = True
        service.buffer.latest = Mock(return_value=(950_000_000, [object(), object()]))
        def decode(*args):
            service.now_ns = lambda: 2_000_000_000
            return ['head', 'torso']
        with patch.object(self.module, 'decode_pair', side_effect=decode):
            service.pump()
        service.worker.submit.assert_not_called()

    def test_spawn_failure_leaves_service_available_with_worker_unavailable_status(self):
        service = self.service
        service.worker_restarts = 0
        service.clear_baseline = Mock()
        service.worker.close = Mock()
        service.node.get_logger = lambda: SimpleNamespace(error=lambda x: None)
        with patch.object(self.module, 'PoseWorker', side_effect=RuntimeError('spawn denied')):
            service.restart_worker('test', 10.)
        self.assertEqual(service.estimate()['status'], 'worker_unavailable')
        self.assertFalse(service.lock.held)


if __name__ == '__main__':
    unittest.main()
