"""Recorded targets, frame errors and failure behavior through the action client."""
import importlib
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from hack_motion.recorded_grasp import RHINO, PRESHAPE, CLOSE, hand_steps


with patch.dict('sys.modules', {'geometry_msgs.msg': NS(Pose=NS, Point=NS, Quaternion=NS), 'yaml': NS()}):
    pick = importlib.import_module('hack_brain.recorded_pick')
    planner = importlib.import_module('hack_brain.grasp_planner')


def observation():
    key = dict(session_id='test', epoch=1, stamp_ns=10_000_000_000)
    return dict(schema='hack_perception.model_poses/v1', status='ok', frame_id='pelvis',
                stamp_ns=key['stamp_ns'], exposure_key=key,
                objects=[dict(name=RHINO['object_name'], pose_status='ok', exposure_key=key.copy(),
                              T_pelvis_object=np.eye(4).tolist())])


class RecordedPickTests(unittest.TestCase):
    def test_targets_rotate_and_translate_in_object_frame(self):
        world = np.array([[0., -1., 0., .4], [1., 0., 0., -.3], [0., 0., 1., -.1], [0., 0., 0., 1.]])
        result = pick.targets(world)
        for stage in result:
            expected = world @ np.asarray(RHINO['T_reference_' + stage]['right'])
            actual = result[stage]
            np.testing.assert_allclose([actual.position.x, actual.position.y, actual.position.z], expected[:3, 3])
            np.testing.assert_allclose(planner.matrix_from_quat(actual.orientation), expected[:3, :3], atol=1e-12)

    def test_pose_rejects_stale_ambiguous_reflected_and_mixed_exposure(self):
        np.testing.assert_array_equal(pick.observed_pose(observation(), 10.1), np.eye(4))
        invalid = []
        stale = observation()
        stale['stamp_ns'] = 1
        invalid.append(stale)
        mixed = observation()
        mixed['objects'][0]['exposure_key']['epoch'] = 2
        invalid.append(mixed)
        duplicate = observation()
        duplicate['objects'] *= 2
        invalid.append(duplicate)
        reflected = observation()
        reflected['objects'][0]['T_pelvis_object'][0][0] = -1
        invalid.append(reflected)
        unknown = observation()
        unknown['objects'][0]['pose_status'] = 'reacquire_uncertain_fit'
        invalid.append(unknown)
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                pick.observed_pose(data, 10.1)
        with self.assertRaises(ValueError):
            pick.observed_pose(observation(), 10.1, 10_000_000_000)

    def test_schedule_preserves_simultaneous_closure_and_timing(self):
        steps = hand_steps(CLOSE)
        self.assertEqual(len(steps), 13)
        self.assertAlmostEqual(sum(duration for _, duration in steps), 3.)
        start = np.asarray(RHINO['preshape']['right'])
        end = np.asarray(RHINO['closure']['positions']['right'])
        np.testing.assert_allclose(steps[0][0], start + (end - start) / 12)
        np.testing.assert_allclose(steps[11][0], end)
        self.assertEqual(hand_steps(PRESHAPE), [(list(start), .5)])

    def test_action_path_reacquires_and_lifts_from_measured_tcp(self):
        calls, observations, events, waits = [], [], [], []
        def move(side, target, mode, **kwargs):
            calls.append(('move', mode, deepcopy(target)))
            return True
        def hand(side, preset, force):
            calls.append(('hand', preset, force))
            # Detection is diagnostic; a missing detection must not replace the physical trial.
            return NS(success=True, position=[.4]*6, effort=[.2]*6, object_detected=False)
        client = NS(move_tcp=move, hand=hand)
        def observe():
            observations.append(len(calls))
            world = np.eye(4)
            world[0, 3] = len(observations) * .01
            return world
        measured = NS(position=NS(x=.3, y=-.2, z=.1), orientation=NS(x=0., y=0., z=0., w=1.))
        result = pick.pick_recorded(client, observe, lambda: measured, waits.append,
                                    lambda phase, data: events.append(phase))
        self.assertEqual(len(observations), 3)
        self.assertEqual([x[1] for x in calls if x[0]=='hand'], ['open', PRESHAPE, CLOSE])
        lift = calls[-1][2]
        self.assertAlmostEqual(lift.position.z, .147)
        self.assertEqual(measured.position.z, .1)
        self.assertEqual(waits, [5.])
        self.assertTrue(result['commands_completed'])
        self.assertIsNone(result['physical_hold_confirmed'])

    def test_failure_after_contact_never_opens_or_retries(self):
        calls = []
        def move(side, target, mode, **kwargs):
            return True
        def hand(side, preset, force):
            calls.append(preset)
            return NS(success=preset != CLOSE, position=[], effort=[], object_detected=False)
        with self.assertRaisesRegex(RuntimeError, 'close_hand_failed'):
            pick.pick_recorded(NS(move_tcp=move, hand=hand), lambda: np.eye(4),
                               lambda: None, lambda _: None, lambda *_: None)
        self.assertEqual(calls, ['open', PRESHAPE, CLOSE])


class RecordedHandTests(unittest.TestCase):
    """Run the actual action method with a deterministic clock and measured hand."""
    def run_hand(self, *, side='right', cancel_at=None, stale=False, force_ok=True, force=.6):
        import math
        import threading
        from hack_motion.recorded_grasp import HAND_TARGETS
        source = Path(__file__).resolve().parents[1] / 'src/hack_motion/hack_motion/motion_node.py'
        tree = ast.parse(source.read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                      and node.name == 'recorded_hand')
        clock = [10.]
        sent = []
        def sleep(dt):
            clock[0] += dt
        state = NS(position=[.4]*6, effort=[.2]*6)
        node = NS(lock=threading.Lock(), hand_state={'right':state},
                  now=lambda: clock[0], _set_hand_limit=lambda *_: force_ok,
                  _set_hand_speed=lambda *_: True,
                  _finger_sample=lambda _: (0. if stale else clock[0], list(state.position)),
                  _command_fingers=lambda _, values: sent.append(list(values)),
                  cancelled=lambda: cancel_at is not None and clock[0] >= cancel_at)
        namespace = dict(Hand=NS(Result=NS, Feedback=NS), RHINO=RHINO, HAND_TARGETS=HAND_TARGETS,
                         hand_steps=hand_steps, math=math, time=NS(monotonic=lambda:clock[0], sleep=sleep),
                         PERIOD=.05, rclpy=NS(ok=lambda:True))
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        goal = NS(request=NS(hand=side, preset=CLOSE, force=force), publish_feedback=lambda _:None)
        return namespace['recorded_hand'](node, goal, HAND_TARGETS[CLOSE]), sent, clock[0]

    def test_action_emits_all_steps_and_waits_for_settle(self):
        result, sent, end = self.run_hand()
        self.assertTrue(result.success)
        self.assertEqual(len(sent), 13)
        np.testing.assert_allclose(sent[-1], RHINO['closure']['positions']['right'])
        self.assertGreaterEqual(end, 13.)

    def test_left_stale_and_failed_force_send_no_finger_commands(self):
        for options in ({'side':'left'}, {'stale':True}, {'force_ok':False}, {'force':0.},
                        {'force':float('nan')}, {'force':1.}):
            with self.subTest(options=options):
                result, sent, _ = self.run_hand(**options)
                self.assertFalse(result.success)
                self.assertEqual(sent, [])

    def test_cancel_during_closure_stops_schedule(self):
        result, sent, _ = self.run_hand(cancel_at=10.3)
        self.assertFalse(result.success)
        self.assertGreater(len(sent), 0)
        self.assertLess(len(sent), 12)
