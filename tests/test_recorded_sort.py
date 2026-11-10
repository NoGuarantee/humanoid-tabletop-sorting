"""Recorded geometry and complete sorting mission wiring (no claim about physics)."""
import importlib
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from concurrent.futures import Future
import json

import numpy as np

from hack_motion.recorded_grasp import RHINO, PRESHAPE, CLOSE, hand_steps


with patch.dict('sys.modules', {'geometry_msgs.msg': NS(Pose=NS, Point=NS, Quaternion=NS), 'yaml': NS()}):
    pick = importlib.import_module('hack_brain.recorded_grasps')
    planner = importlib.import_module('hack_brain.grasp_planner')
    skills_module = importlib.import_module('hack_brain.skills')


def observation():
    key = dict(session_id='test', epoch=1, stamp_ns=10_000_000_000)
    return dict(schema='hack_perception.model_poses/v1', status='ok', frame_id='pelvis',
                stamp_ns=key['stamp_ns'], exposure_key=key,
                objects=[dict(name=RHINO['object_name'], pose_status='ok', exposure_key=key.copy(),
                              center_pelvis_m=[0., 0., 0.], size_model_m=[.18, .08, .10],
                              T_pelvis_object=np.eye(4).tolist())])


def tray(y=-.5, yaw=.4, size=(.32, .25)):
    return NS(id='blue', kind='tray', pose=planner.pose(.4, y, .08, planner.quat_from_rpy(0, 0, yaw)),
              size=NS(x=size[0], y=size[1], z=.07))


def detected():
    return NS(name=RHINO['object_name'], track_id='rhino', category='figure', center=NS(x=.4, y=-.2, z=.05))


class RecordedSortTests(unittest.TestCase):
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
        self.assertEqual(hand_steps(PRESHAPE), [([1.]*6, .5), (list(start), .5)])

    def test_whole_object_lands_inside_rotated_tray_with_tcp_offset(self):
        data = observation()
        data['objects'][0]['center_pelvis_m'] = [.03, -.02, .04]  # model origin is not its centre
        g = pick.candidate(data, 10.1, detected(), tray(), {}, 0)
        self.assertAlmostEqual(g.lift.position.z - g.grasp.position.z, .12)
        for heading in (-1., .4, 1.2):
            t = tray(yaw=heading)
            plan = skills_module.place_for(g, t, {'place': {'inset': .20}})
            r = np.asarray(planner.matrix_from_quat(g.grasp.orientation))
            tr = np.asarray(planner.matrix_from_quat(t.pose.orientation))
            origin = np.array([getattr(t.pose.position, k) for k in 'xyz'])
            for target, height in [(plan.approach, .20), (plan.release, .10), (plan.retreat, .20)]:
                tcp = np.array([getattr(target.position, k) for k in 'xyz'])
                center = tcp + r @ g.center_in_tcp
                corners = (center + np.asarray(g.corners_in_tcp) @ r.T - origin) @ tr
                self.assertLessEqual(abs(corners[:, 0]).max(), t.size.x/2 - .015 + 1e-10)
                self.assertLessEqual(abs(corners[:, 1]).max(), t.size.y/2 - .015 + 1e-10)
                self.assertAlmostEqual(corners[:, 2].min(), height)
                self.assertEqual(target.orientation, g.grasp.orientation)

    def test_rejects_left_arm_small_tray_reset_and_pose_jump_before_contact(self):
        for t in (tray(y=.5), tray(size=(.10, .10))):
            with self.assertRaises(ValueError):
                pick.candidate(observation(), 10.1, detected(), t, {}, 0)
        before = pick.candidate(observation(), 10.1, detected(), tray(), {}, 0)
        for reset, shift in [(True, 0.), (False, .04)]:
            data = observation()
            data['stamp_ns'] += 1
            data['exposure_key']['stamp_ns'] += 1
            data['exposure_key']['epoch'] += int(reset)
            data['objects'][0]['exposure_key'] = data['exposure_key'].copy()
            data['objects'][0]['T_pelvis_object'][0][3] += shift
            after = pick.candidate(data, 10.1, detected(), tray(), {}, 0)
            with self.assertRaises(ValueError):
                pick.refreshed(before, after)

    def test_default_skip_and_opt_in_use_same_task_routing(self):
        db = {'by_name': {RHINO['object_name']: {'skip': True}, 'other': {'skip': True}}}
        enabled = pick.evaluation_db(db, True)
        self.assertTrue(db['by_name'][RHINO['object_name']]['skip'])
        self.assertNotIn(RHINO['object_name'], planner.skip_names(enabled))
        self.assertIn('other', planner.skip_names(enabled))
        self.assertTrue(pick.uses_recorded(enabled, RHINO['object_name']))

    def test_normal_fsm_runs_through_task_destination_release_and_verification(self):
        from hack_brain.context import Object, State
        from hack_brain.fsm import VERIFY
        from fakes import Fake, sort_spec
        spec = sort_spec()
        spec.categories['figure'] = 'red'  # destination comes from the task, not a recorded blue/right tray ID
        obj = Object(RHINO['object_name'], 'figure', name=RHINO['object_name'], side='right')
        f = Fake(spec=spec, objects=[obj])
        skills = skills_module.Skills(pick.evaluation_db({}, True))
        destination = tray()
        destination.id = 'red'
        scene = NS(objects=[detected()], containers=[destination])
        goals, observed = [], []
        stubs = NS(Hand=NS(Goal=NS), MoveTcp=NS(Goal=NS), MoveJoints=NS(Goal=NS))
        with patch.dict('sys.modules', {'hack_interfaces.action': stubs}):
            for _ in range(100):
                if f.c.state == State.DONE:
                    break
                phase = f.c.state
                observed.append(phase)
                f.tick(sim=f.sim + .25, wall=f.wall + .25)
                effect = f.commands[-1]
                if phase not in VERIFY and phase not in (State.OBSERVE, State.DESTINATION):
                    if skills.needs_model_pose(effect):
                        data = observation()
                        data['stamp_ns'] = int((f.sim - .01) * 1e9)
                        data['exposure_key']['stamp_ns'] = data['stamp_ns']
                        data['objects'][0]['exposure_key'] = data['exposure_key'].copy()
                        skills.prepare_recorded(effect, scene, data, f.sim, int(f.c.active.sim_start*1e9))
                    name, goal = skills.command(effect, scene)
                    goals.append((phase, name, goal))
                # Explicit fake sensor evidence drives the FSM; this is not a physical validation.
                f.answer()
        self.assertEqual(f.c.state, State.DONE)
        self.assertTrue(f.c.history[obj.id].placed)
        transfer = next(g for p, _, g in goals if p == State.TRANSFER)
        self.assertEqual(transfer.mode, 'carry')
        self.assertTrue(all(e.destination == 'red' for e in f.commands if e.phase == State.TRANSFER))
        self.assertEqual(next(g.preset for p, _, g in goals if p == State.CLOSE), CLOSE)
        self.assertEqual(next(g.preset for p, _, g in goals if p == State.RELEASE), 'open')
        self.assertEqual([p for p in observed if p in (State.LIFT, State.TRANSFER, State.LOWER,
                         State.RELEASE, State.RETREAT, State.VERIFY_PLACE)],
                         [State.LIFT, State.TRANSFER, State.LOWER, State.RELEASE, State.RETREAT, State.VERIFY_PLACE])


def client_type():
    path = Path(__file__).resolve().parents[1] / 'src/hack_brain/hack_brain/clients.py'
    spec = importlib.util.spec_from_file_location('hack_brain.recorded_client_test', path)
    module = importlib.util.module_from_spec(spec)
    stubs = {'hack_interfaces.action': NS(Hand=NS, MoveJoints=NS, MoveTcp=NS),
             'hack_brain.skills': skills_module, 'hack_brain.grasp_planner': planner,
             'hack_brain.recorded_grasps': pick,
             'hack_interfaces.srv': NS(Perceive=NS),
             'rclpy.action': NS(ActionClient=object),
             'rclpy.callback_groups': NS(ReentrantCallbackGroup=object),
             'rclpy.time': NS(Time=object), 'std_srvs.srv': NS(Trigger=NS(Request=NS)),
             'sensor_msgs.msg': NS(JointState=object),
             'tf2_ros': NS(Buffer=object, TransformListener=object)}
    with patch.dict('sys.modules', stubs):
        spec.loader.exec_module(module)
    return module.Clients


class RecordedClientTests(unittest.TestCase):
    def setUp(self):
        from hack_brain.context import Context, Effect, Object, Operation, State
        self.State = State
        cls = client_type()
        self.client = cls.__new__(cls)
        self.events = []
        self.client.emit = self.events.append
        self.client.skills = Mock()
        self.client.skills.grasp.exposure = observation()['exposure_key']
        self.client.scenes = {'table': object()}
        self.client.model_cli = Mock()
        self.client.model_cli.service_is_ready.return_value = True
        self.client.execute_motion = Mock()
        self.client.motion_wait = None
        self.client.next_model_request = 0.
        self.effect = Effect('RUN', 7, State.OPEN, Object(RHINO['object_name'], 'figure', name=RHINO['object_name']), 'blue')
        self.context = Context(state=State.OPEN, sim=10.1, active=Operation(7, State.OPEN, 9.9, 0))
        self.client.model_pending = self.effect
        self.future = Future()
        self.client.model_future = self.future

    def response(self, data=None):
        self.future.set_result(NS(success=True, message=json.dumps(data or observation())))

    def test_fresh_pose_dispatches_once_on_owner_poll(self):
        self.response()
        self.client.execute_motion.assert_not_called()
        self.client.poll_model_pose(self.context)
        self.client.poll_model_pose(self.context)
        self.client.skills.prepare_recorded.assert_called_once()
        self.client.execute_motion.assert_called_once_with(self.effect, self.context)

    def test_cancel_and_old_operation_drop_late_response(self):
        from dataclasses import replace
        for cancel in (True, False):
            self.setUp()
            if cancel:
                self.client.execute(replace(self.effect, kind='CANCEL'), self.context)
                self.assertEqual([e.kind for e in self.events], ['CANCEL_ACK', 'RESULT'])
            else:
                self.context.active.id = 8
            self.response()
            self.client.poll_model_pose(self.context)
            self.client.execute_motion.assert_not_called()

    def test_old_frame_retries_then_times_out_without_motion(self):
        data = observation()
        data['stamp_ns'] = 9_000_000_000
        self.response(data)
        self.client.poll_model_pose(self.context)
        self.client.execute_motion.assert_not_called()
        self.context.sim = 16.
        self.client.poll_model_pose(self.context)
        self.assertEqual(self.events[-1].kind, 'REJECTED')
        self.assertEqual(self.events[-1].data, 'fresh_recorded_pose_unavailable')

    def test_invalid_geometry_rejects_before_motor_dispatch(self):
        self.client.skills.prepare_recorded.side_effect = ValueError('object does not fit')
        self.response()
        self.client.poll_model_pose(self.context)
        self.client.execute_motion.assert_not_called()
        self.assertEqual(self.events[-1].kind, 'REJECTED')

    def test_pose_expires_while_waiting_for_action_server(self):
        self.client.skills.needs_model_pose.return_value = True
        self.client.actions = {'hand': Mock()}
        self.client.send_motion(self.effect, 'hand', NS(hand='right'), 12.)
        self.client.actions['hand'].server_is_ready.assert_not_called()
        self.assertEqual(self.events[-1].data, 'recorded_pose_expired_before_motion')


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
