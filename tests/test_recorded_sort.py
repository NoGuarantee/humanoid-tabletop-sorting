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

from hack_motion.recorded_grasp import RHINO, PRESHAPE, CLOSE, PROFILES, hand_steps, profile_for_preset


with patch.dict('sys.modules', {'geometry_msgs.msg': NS(Pose=NS, Point=NS, Quaternion=NS), 'yaml': NS()}):
    pick = importlib.import_module('hack_brain.recorded_grasps')
    planner = importlib.import_module('hack_brain.grasp_planner')
    skills_module = importlib.import_module('hack_brain.skills')


def observation(name=RHINO['object_name']):
    key = dict(session_id='test', epoch=1, stamp_ns=10_000_000_000)
    return dict(schema='hack_perception.model_poses/v1', status='ok', frame_id='pelvis',
                stamp_ns=key['stamp_ns'], exposure_key=key,
                objects=[dict(name=name, pose_status='ok', exposure_key=key.copy(),
                              center_pelvis_m=[0., 0., 0.], size_model_m=[.18, .08, .10],
                              T_pelvis_object=np.eye(4).tolist())])


def tray(y=-.5, yaw=.4, size=(.32, .25)):
    return NS(id='blue', kind='tray', pose=planner.pose(.4, y, .08, planner.quat_from_rpy(0, 0, yaw)),
              size=NS(x=size[0], y=size[1], z=.07))


def detected(name=RHINO['object_name'], side='right'):
    return NS(name=name, track_id='object', category='figure', center=NS(x=.4, y=.2 if side == 'left' else -.2, z=.05))


DB = pick.evaluation_db({}, True)


class RecordedSortTests(unittest.TestCase):
    def test_recorded_head_look_compensates_waist_without_moving_pregrasp_arm(self):
        from hack_brain.context import State
        name = PROFILES['philips_right']['object_name']
        data = observation(name)
        data['objects'][0]['center_pelvis_m'] = [.414847, -.419326, -.094912]
        skills = skills_module.Skills(DB)
        skills.grasp = pick.candidate(data, 10.1, detected(name), tray(), DB, 0, profile_key='philips_right')
        effect = NS(phase=State.LOOK_GRASP)
        with patch.dict('sys.modules', {'hack_interfaces.action': NS(Hand=NS(Goal=NS), MoveTcp=NS(Goal=NS), MoveJoints=NS(Goal=NS))}):
            kind, goal = skills.command(effect, None, joint_positions={'body_yaw_joint': .576378})
            self.assertEqual(kind, 'joints')
            self.assertEqual(goal.joints, ['head_pitch_joint', 'head_yaw_joint'])
            self.assertAlmostEqual(goal.positions[1] + .576378, np.arctan2(-.419326, .414847))
            self.assertTrue(-1.57 <= goal.positions[1] <= 1.57)
            for q in (None, {}, {'body_yaw_joint': float('nan')}):
                with self.assertRaisesRegex(ValueError, 'body_yaw'):
                    skills.command(effect, None, joint_positions=q)
            _, clamped = skills.command(effect, None, joint_positions={'body_yaw_joint': 1.5})
            self.assertEqual(clamped.positions[1], -1.57)

    def test_descent_association_applies_existing_pose_drift_gate_before_ambiguity(self):
        name = PROFILES['philips_right']['object_name']
        raw = observation(name)
        previous = pick.candidate(raw, 10.1, detected(name), tray(), DB, 0, profile_key='philips_right')
        data = deepcopy(raw)
        data['stamp_ns'] = data['exposure_key']['stamp_ns'] = 11_000_000_000
        original = data['objects'][0]
        original['exposure_key'] = deepcopy(data['exposure_key'])
        flipped = deepcopy(original)
        flipped['T_pelvis_object'] = np.diag([-1.,-1.,1.,1.]).tolist()
        for rows in ([original,flipped],[flipped,original]):
            data['objects'] = deepcopy(rows)
            with self.assertRaisesRegex(ValueError,'ambiguous'):
                pick.associate_model_pose(data, [0.,0.,0.], 10_000_000_000, object_name=name)
            selected = pick.associate_model_pose(data, [0.,0.,0.], 10_000_000_000,
                                                  object_name=name, previous=previous)
            np.testing.assert_allclose(pick.observed_pose(selected,11.1,object_name=name),np.eye(4))
            self.assertEqual(sum(r['compatible'] for r in selected['target_association']['candidates']),1)
        # Two genuinely compatible poses remain ambiguous; no score/nearest tie-break.
        data['objects'] = [deepcopy(original),deepcopy(original)]
        with self.assertRaisesRegex(ValueError,'ambiguous'):
            pick.associate_model_pose(data,[0.,0.,0.],10_000_000_000,object_name=name,previous=previous)
        data['objects'] = [flipped]
        with self.assertRaisesRegex(ValueError,'unmatched'):
            pick.associate_model_pose(data,[0.,0.,0.],10_000_000_000,object_name=name,previous=previous)

    def test_philips_transfer_requests_native_inward_offset_from_observed_center(self):
        from hack_brain.context import State
        name = PROFILES['philips_right']['object_name']
        data = observation(name)
        data['objects'][0]['center_pelvis_m'] = [.34, .08, -.1]
        for i, v in enumerate((.34, .08, -.1)):
            data['objects'][0]['T_pelvis_object'][i][3] = v
        container = tray(y=0., yaw=0., size=(.32, .32))
        skills = skills_module.Skills(DB)
        skills.grasp = pick.candidate(data, 10.1, detected(name), container, DB, 0, profile_key='philips_right')
        with patch.dict('sys.modules', {'hack_interfaces.action': NS(Hand=NS(Goal=NS), MoveTcp=NS(Goal=NS), MoveJoints=NS(Goal=NS))}):
            _, goal = skills.command(NS(phase=State.TRANSFER, destination=container.id), NS(containers=[container]))
        np.testing.assert_allclose(goal.carry_inward_xy, -.08*np.array([.34,.08])/np.linalg.norm([.34,.08]))
        self.assertEqual(goal.mode, 'carry_fixed')

    def test_targets_rotate_and_translate_in_object_frame(self):
        world = np.array([[0., -1., 0., .4], [1., 0., 0., -.3], [0., 0., 1., -.1], [0., 0., 0., 1.]])
        result = pick.targets(world, RHINO)
        for stage in result:
            expected = world @ np.asarray(RHINO['T_reference_' + stage]['right'])
            actual = result[stage]
            np.testing.assert_allclose([actual.position.x, actual.position.y, actual.position.z], expected[:3, 3])
            np.testing.assert_allclose(planner.matrix_from_quat(actual.orientation), expected[:3, :3], atol=1e-12)

    def test_pose_rejects_stale_ambiguous_reflected_and_mixed_exposure(self):
        np.testing.assert_array_equal(pick.observed_pose(observation(), 10.1, object_name=RHINO['object_name']), np.eye(4))
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
                pick.observed_pose(data, 10.1, object_name=RHINO['object_name'])
        with self.assertRaises(ValueError):
            pick.observed_pose(observation(), 10.1, 10_000_000_000, object_name=RHINO['object_name'])

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
        g = pick.candidate(data, 10.1, detected(), tray(), DB, 0)
        self.assertAlmostEqual(g.lift.position.z - g.grasp.position.z, .12)
        for heading in (-1., .4, 1.2):
            t = tray(yaw=heading)
            plan = skills_module.place_for(g, t, {'place': {'inset': .20}})
            r = np.asarray(planner.matrix_from_quat(plan.release.orientation))
            tr = np.asarray(planner.matrix_from_quat(t.pose.orientation))
            origin = np.array([getattr(t.pose.position, k) for k in 'xyz'])
            for target in (plan.approach, plan.release, plan.retreat):
                tcp = np.array([getattr(target.position, k) for k in 'xyz'])
                center = tcp + r @ g.center_in_tcp
                corners = (center + np.asarray(g.corners_in_tcp) @ r.T - origin) @ tr
                self.assertLessEqual(abs(corners[:, 0]).max(), t.size.x/2 - .015 + 1e-10)
                self.assertLessEqual(abs(corners[:, 1]).max(), t.size.y/2 - .015 + 1e-10)
                if target is plan.release:
                    self.assertAlmostEqual(corners[:, 2].min(), .035)
                else:
                    self.assertGreaterEqual(corners[:, 2].min(), .075 - 1e-10)
                self.assertEqual(target.orientation, plan.release.orientation)
            from hack_motion.recorded_geometry import axis_angle
            expected = axis_angle(tr[:, 2], -1.47215)[:3, :3] @ planner.matrix_from_quat(g.grasp.orientation)
            np.testing.assert_allclose(r, expected, atol=1e-12)

    def test_rejects_disabled_hand_small_tray_reset_and_pose_jump_before_contact(self):
        with self.assertRaisesRegex(ValueError, 'no_recorded_profile'):
            pick.candidate(observation(), 10.1, detected(), tray(.5), DB, 0, disabled_arm='right')
        with self.assertRaisesRegex(ValueError, 'does_not_fit'):
            pick.candidate(observation(), 10.1, detected(), tray(size=(.10, .10)), DB, 0)
        before = pick.candidate(observation(), 10.1, detected(), tray(), DB, 0)
        for reset, shift in [(True, 0.), (False, .04)]:
            data = observation()
            data['stamp_ns'] += 1
            data['exposure_key']['stamp_ns'] += 1
            data['exposure_key']['epoch'] += int(reset)
            data['objects'][0]['exposure_key'] = data['exposure_key'].copy()
            data['objects'][0]['T_pelvis_object'][0][3] += shift
            after = pick.candidate(data, 10.1, detected(), tray(), DB, 0)
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
        for key in PROFILES:
            with self.subTest(profile=key):
                self.run_sort_profile(key)
                self.run_sort_profile(key, opposite_tray=True)

    def run_sort_profile(self, key, opposite_tray=False):
        from hack_brain.context import Config, Object, State
        from hack_brain.fsm import VERIFY
        from fakes import Fake, sort_spec
        profile = PROFILES[key]
        name, side = profile['object_name'], profile['hand']
        db = pick.evaluation_db({}, True)
        db['by_name'][name]['recorded_profiles'] = [key]
        spec = sort_spec()
        spec.categories['figure'] = 'red'  # destination comes from the task, not a recorded blue/right tray ID
        obj = Object(name, 'figure', name=name, side=side)
        f = Fake(spec=spec, objects=[obj], config=Config(postlift_names=pick.postlift_names(db),
                                                       recorded_arms=pick.recorded_arms(db)))
        skills = skills_module.Skills(db)
        destination = tray(y=.5 if side == 'left' else -.5)
        if opposite_tray:
            destination.pose.position.y *= -1
        destination.id = 'red'
        scene = NS(objects=[detected(name, side)], containers=[destination])
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
                        data = observation(name)
                        data['stamp_ns'] = int((f.sim - .01) * 1e9)
                        data['exposure_key']['stamp_ns'] = data['stamp_ns']
                        data['objects'][0]['exposure_key'] = data['exposure_key'].copy()
                        skills.prepare_recorded(effect, scene, data, f.sim, int(f.c.active.sim_start*1e9))
                    action, goal = skills.command(effect, scene, joint_positions={'body_yaw_joint': 0.})
                    if phase == State.DESCEND:
                        self.assertEqual(goal.contact_object, scene.objects[0].track_id)
                        self.assertEqual(goal.contact_center, list(skills.grasp.observed_center))
                        self.assertEqual(goal.contact_stamp_ns, skills.grasp.exposure['stamp_ns'])
                    else:
                        self.assertFalse(getattr(goal, 'contact_object', ''))
                    goals.append((phase, action, goal))
                # Explicit fake sensor evidence drives the FSM; this is not a physical validation.
                f.answer()
        self.assertEqual(f.c.state, State.DONE)
        self.assertTrue(f.c.history[obj.id].placed)
        transfer = next(g for p, _, g in goals if p == State.TRANSFER)
        self.assertEqual(transfer.mode, 'carry_fixed')
        self.assertEqual(transfer.arm, side)
        self.assertEqual(transfer.pos_tol, .01)
        for phase, action, goal in goals:
            if action == 'tcp' and phase not in (State.ARMS_READY, State.CLEAR):
                self.assertEqual(goal.pos_tol, .01 if phase in (State.APPROACH, State.TRANSFER) else .005)
                self.assertEqual(goal.rot_tol, .05)
        self.assertTrue(all(e.destination == 'red' for e in f.commands if e.phase == State.TRANSFER))
        self.assertEqual(next(g.preset for p, _, g in goals if p == State.CLOSE), key + '_close')
        self.assertEqual(next(g.preset for p, _, g in goals if p == State.RELEASE),
                         key + '_release' if 'release' in profile else 'open')
        if 'postlift' in profile:
            self.assertEqual(next(g.preset for p, _, g in goals if p == State.TIGHTEN), key + '_postlift')
            self.assertLess(observed.index(State.VERIFY_LIFT), observed.index(State.TIGHTEN))
            self.assertLess(observed.index(State.VERIFY_TIGHTEN), observed.index(State.TRANSFER))
        else:
            self.assertNotIn(State.TIGHTEN, observed)
        self.assertEqual([p for p in observed if p in (State.LIFT, State.TRANSFER, State.LOWER,
                         State.RELEASE, State.RETREAT, State.VERIFY_PLACE)],
                         [State.LIFT, State.TRANSFER, State.LOWER, State.RELEASE, State.RETREAT, State.VERIFY_PLACE])

    def test_selected_objects_only_and_original_db_unchanged(self):
        original = {'by_name': {p['object_name']: {'skip': True} for p in PROFILES.values()}}
        for family in ('rhino', 'brother', 'philips', 'nescafe'):
            db = pick.evaluation_db(original, True, family)
            for p in PROFILES.values():
                self.assertEqual(pick.uses_recorded(db, p['object_name']), p['family'] == family)
                self.assertTrue(original['by_name'][p['object_name']]['skip'])
        self.assertEqual(pick.evaluation_db(original, False), original)
        for selected in ('', 'mouse', 'brother,typo'):
            with self.assertRaises(ValueError):
                pick.evaluation_db(original, True, selected)

    def test_retry_uses_other_recorded_pose_and_descent_keeps_it(self):
        for key in ('brother_left_wide', 'philips_right', 'nescafe_right'):
            p = PROFILES[key]
            name, side = p['object_name'], p['hand']
            data = observation(name)
            obj, container = detected(name, side), tray(.5 if side == 'left' else -.5)
            first = pick.candidate(data, 10.1, obj, container, DB, 0, attempt=1)
            second = pick.candidate(data, 10.1, obj, container, DB, 0, attempt=2)
            self.assertNotEqual(first.profile_key, second.profile_key)
            same = pick.candidate(data, 10.1, obj, container, DB, 0, attempt=1, profile_key=second.profile_key)
            self.assertEqual(same.profile_key, second.profile_key)
            with self.assertRaises(ValueError):
                pick.refreshed(first, second)
        for key in ('philips_right', 'nescafe_right'):
            p = PROFILES[key]
            grasp = pick.candidate(observation(p['object_name']), 10.1, detected(p['object_name'], 'left'), tray(.5), DB, 0)
            self.assertEqual(grasp.arm, 'right')

    def test_model_response_selects_exact_object_not_first_pose(self):
        data = observation()
        name = PROFILES['nescafe_right']['object_name']
        other = observation(name)['objects'][0]
        other['T_pelvis_object'][0][3] = .33
        data['objects'].append(other)
        self.assertEqual(pick.observed_pose(data, 10.1, object_name=name)[0, 3], .33)
        with self.assertRaises(ValueError):
            pick.observed_pose(data, 10.1, object_name='missing')

    def test_saved_philips_duplicate_is_bound_to_selected_surface_without_score_tie_break(self):
        raw = json.loads(Path(__file__).with_name('fixtures').joinpath('philips_ambiguous_pose.json').read_text())
        raw['frame_id'] = 'pelvis'  # inspector saved a field subset; this endpoint's fixed output frame
        name = PROFILES['philips_right']['object_name']
        for chosen in (0, 1):
            # Either independently selected surface must bind to its own model,
            # regardless of ordering/confidence, while retaining the other region.
            point = raw['objects'][chosen]['center_pelvis_m']
            result = pick.associate_model_pose(raw, point, raw['stamp_ns']-100_000_000, object_name=name)
            self.assertEqual(result['target_association']['selected_index'], chosen)
            self.assertEqual(len(result['objects']), len(raw['objects']))
            self.assertEqual(result['objects'][1-chosen]['T_pelvis_object'], raw['objects'][1-chosen]['T_pelvis_object'])
            np.testing.assert_allclose(pick.observed_pose(result, raw['stamp_ns']/1e9+.1, object_name=name),
                                       raw['objects'][chosen]['T_pelvis_object'])
        self.assertEqual([o['name'] for o in raw['objects']], [name, name])
        duplicate=deepcopy(raw);duplicate['objects'][1]=deepcopy(duplicate['objects'][0])
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            pick.associate_model_pose(duplicate, raw['objects'][0]['center_pelvis_m'], 1, object_name=name)
        with self.assertRaisesRegex(ValueError, 'unmatched'):
            pick.associate_model_pose(raw, [2,2,2], 1, object_name=name)

    def test_selected_cluster_survives_duplicate_names(self):
        from hack_brain.context import Object
        name=PROFILES['philips_right']['object_name']
        a,b=detected(name),detected(name);a.track_id='partial';b.track_id='selected'
        target=Object(name,'box',name=name,track_id='selected')
        self.assertIs(pick.selected_object(NS(objects=[a,b]),target), b)
        with self.assertRaises(ValueError):
            pick.selected_object(NS(objects=[a]),target)

    def test_imported_geometry_reconstructs_source_and_six_actuator_commands(self):
        from scipy.spatial.transform import Rotation
        def matrix(p):
            t = np.eye(4)
            t[:3, :3] = Rotation.from_quat([*p[4:], p[3]]).as_matrix()
            t[:3, 3] = p[:3]
            return t
        sources = json.loads(Path(__file__).with_name('recorded_grasp_sources.json').read_text())
        self.assertEqual(len(sources), len(PROFILES) - 1)
        for row in sources:
            with self.subTest(profile=row['key']):
                profile = PROFILES[row['key']]
                side = row['arm_side']
                for phase, source in [('tcp', 'target'), ('pregrasp', 'pregrasp')]:
                    reconstructed = matrix(row['object_pose_world_wxyz']) @ profile['T_reference_' + phase][side]
                    np.testing.assert_allclose(reconstructed, matrix(row[source + '_tcp_pose_world_wxyz']), atol=1e-12)
                self.assertEqual(profile['preshape'][side], row['pregrasp_ratios'])
                self.assertEqual(profile['closure']['positions'][side], row['closed_ratios'])
                self.assertAlmostEqual(sum(dt for _, dt in hand_steps(row['key'] + '_close')),
                                       row['closure_duration_s'] + row['closure_settle_s'])
                if 'postlift' in profile:
                    self.assertEqual(profile['postlift']['positions'][side], row['postlift_closure_ratios'])


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
        from hack_brain.context import Context, Effect, History, Object, Operation, State
        self.State = State
        cls = client_type()
        self.client = cls.__new__(cls)
        self.client.node = Mock()
        self.events = []
        self.client.emit = self.events.append
        self.client.skills = Mock()
        self.client.skills.grasp.exposure = observation()['exposure_key']
        self.client.skills.grasp.profile_key = 'rhino_e71'
        self.client.skills.grasp.arm = 'right'
        obj=detected();obj.center=NS(x=0.,y=0.,z=0.)
        self.client.scenes = {'table': NS(objects=[obj], header=NS(frame_id='pelvis',stamp=NS(sec=9,nanosec=800_000_000)))}
        self.client.model_cli = Mock()
        self.client.model_cli.service_is_ready.return_value = True
        self.client.execute_motion = Mock()
        self.client.motion_wait = None
        self.client.next_model_request = 0.
        self.client.model_pose_error = ''
        self.effect = Effect('RUN', 7, State.OPEN, Object(RHINO['object_name'], 'figure', name=RHINO['object_name']), 'blue')
        self.context = Context(state=State.OPEN, sim=10.1, active=Operation(7, State.OPEN, 9.9, 0))
        self.context.history[self.effect.target.id] = History(attempts=1)
        self.client.model_pending = self.effect
        self.future = Future()
        self.client.model_future = self.future

    def response(self, data=None):
        self.future.set_result(NS(success=True, message=json.dumps(data or observation())))

    def test_head_look_uses_recent_joints_at_reducer_time_and_rejects_stale(self):
        from dataclasses import replace
        from threading import RLock
        effect = replace(self.effect, phase=self.State.LOOK_GRASP)
        self.client.send_motion = Mock()
        self.client.skills.command.return_value = ('joints', 'head_goal')
        self.client.measurements = NS(lock=RLock(), samples={'joints': [
            (10., {'body_yaw_joint': .576}, None),
            (10.2, {'body_yaw_joint': .58}, None)]})
        type(self.client).execute_motion(self.client, effect, self.context)
        self.assertEqual(self.client.skills.command.call_args.kwargs['joint_positions'], {'body_yaw_joint': .576})
        self.client.send_motion.assert_called_once_with(effect, 'joints', 'head_goal', 10.1)
        self.client.send_motion.reset_mock()
        self.context.sim = 11.
        type(self.client).execute_motion(self.client, effect, self.context)
        self.client.send_motion.assert_not_called()
        self.assertIn('stale_grasp_look_joints', self.events[-1].data)

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

    def test_cold_worker_can_finish_before_open_without_spending_a_grasp_attempt(self):
        self.future.set_result(NS(success=False, message='{"status":"worker_loading"}'))
        self.client.poll_model_pose(self.context)
        self.context.sim = 16.2  # > normal six-second scene budget
        self.client.poll_model_pose(self.context)
        self.assertEqual(self.events, [])
        self.client.execute_motion.assert_not_called()
        data = observation()
        data['stamp_ns'] = data['exposure_key']['stamp_ns'] = 16_200_000_000
        for row in data['objects']:
            row['exposure_key'] = dict(data['exposure_key'])
        self.client.model_future = Future()
        self.client.model_future.set_result(NS(success=True, message=json.dumps(data)))
        self.context.sim = 16.3
        self.client.poll_model_pose(self.context)
        self.client.execute_motion.assert_called_once_with(self.effect, self.context)

    def test_worker_initialization_budget_is_bounded_and_does_not_extend_descent(self):
        from dataclasses import replace
        for phase, elapsed in ((self.State.OPEN, 15.1), (self.State.DESCEND, 6.1)):
            self.setUp()
            self.client.model_pending = replace(self.effect, phase=phase)
            self.client.model_pose_error = 'worker_loading'
            self.context.sim = self.context.active.sim_start + elapsed
            self.client.poll_model_pose(self.context)
            self.assertEqual(self.events[-1].data, 'fresh_recorded_pose_unavailable')
            self.client.execute_motion.assert_not_called()

    def test_invalid_geometry_rejects_before_motor_dispatch(self):
        self.client.skills.prepare_recorded.side_effect = ValueError('object does not fit')
        self.response()
        self.client.poll_model_pose(self.context)
        self.client.execute_motion.assert_not_called()
        self.assertEqual(self.events[-1].kind, 'REJECTED')

    def test_pending_poll_does_not_erase_ambiguous_pose_failure(self):
        data = observation()
        data['objects'].append(deepcopy(data['objects'][0]))
        self.response(data)
        self.client.poll_model_pose(self.context)
        reason = self.client.model_pose_error
        self.assertIn('ambiguous', reason)
        self.client.model_future = Future()
        self.client.model_future.set_result(NS(success=False, message='{"status":"awaiting_fresh_pose"}'))
        self.client.poll_model_pose(self.context)
        self.assertEqual(self.client.model_pose_error, reason)
        self.client.execute_motion.assert_not_called()

    def test_pose_expires_while_waiting_for_action_server(self):
        self.client.skills.needs_model_pose.return_value = True
        self.client.actions = {'hand': Mock()}
        self.client.send_motion(self.effect, 'hand', NS(hand='right'), 12.)
        self.client.actions['hand'].server_is_ready.assert_not_called()
        self.assertEqual(self.events[-1].data, 'recorded_pose_expired_before_motion')

    def test_postlift_updates_contact_target_without_erasing_lift_evidence(self):
        from dataclasses import replace
        self.client.skills.needs_model_pose.return_value = False
        self.client.actions = {'hand': Mock()}
        self.client.actions['hand'].server_is_ready.return_value = True
        self.client.transport = Mock()
        self.client.verifier = NS(closure=None, closures={}, baseline=(9., (0., 0., .1)),
                                  contact_seen=True, lifted=True, contact_lost=False, unsure_since=10.)
        effect = replace(self.effect, phase=self.State.TIGHTEN)
        self.client.send_motion(effect, 'hand', NS(hand='right', preset='nescafe_right_postlift'), 10.1)
        v = self.client.verifier
        self.assertEqual(v.closure, tuple(PROFILES['nescafe_right']['postlift']['positions']['right']))
        self.assertEqual(v.closures['right'], v.closure)
        self.assertEqual(v.baseline, (9., (0., 0., .1)))
        self.assertTrue(v.contact_seen and v.lifted)
        self.assertFalse(v.contact_lost)
        self.assertIsNone(v.unsure_since)
        self.client.transport.send.assert_called_once()


class RecordedHandTests(unittest.TestCase):
    """Run the actual action method with a deterministic clock and measured hand."""
    def test_head_look_prepares_collision_model_before_returning_success(self):
        from threading import RLock
        source = Path(__file__).resolve().parents[1] / 'src/hack_motion/hack_motion/motion_node.py'
        method = next(n for n in ast.walk(ast.parse(source.read_text()))
                      if isinstance(n, ast.FunctionDef) and n.name == 'exec_move_joints')
        ns = dict(MoveJoints=NS(Result=NS, Feedback=NS))
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), ns)
        for names, ok, loaded, prepares in [(['head_yaw_joint'], True, True, True),
                                           (['head_yaw_joint'], False, True, False),
                                           (['head_yaw_joint', 'body_yaw_joint'], True, True, False),
                                           (['head_yaw_joint'], True, False, False)]:
            events = []
            planner = Mock()
            planner.load.side_effect = lambda q: events.append(('prepare', q.copy())) or 'ready'
            node = NS(_planner=planner if loaded else None, _planner_obj=lambda: planner,
                      lock=RLock(), _tcp_lock=RLock(), q={'head_yaw_joint': -.31}, scene_objects=(), carrying=False,
                      get_logger=Mock(), execute=lambda h,a,run: run(),
                      move_joints=lambda *a: events.append(('move', a[0])) or (ok, .01, 'ok'))
            handle = NS(request=NS(joints=names, positions=[0.]*len(names), speed=0., tol=0., max_time=0.),
                        publish_feedback=Mock())
            result = ns['exec_move_joints'](node, handle)
            events.append(('result', result.success))
            self.assertEqual([e[0] for e in events], ['move', 'prepare', 'result'] if prepares else ['move', 'result'])
            if prepares:
                planner.load.assert_called_once_with({'head_yaw_joint': -.31})

    def run_hand(self, *, side='right', preset=CLOSE, cancel_at=None, stale=False, force_ok=True, force=.6, sim_rate=1.):
        import math
        import threading
        from hack_motion.recorded_grasp import HAND_TARGETS
        source = Path(__file__).resolve().parents[1] / 'src/hack_motion/hack_motion/motion_node.py'
        tree = ast.parse(source.read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                      and node.name == 'recorded_hand')
        clock = [10.]
        wall = [0.]
        sent = []
        def sleep(dt):
            clock[0] += dt * sim_rate
            wall[0] += dt
        state = NS(position=[.4]*6, effort=[.2]*6)
        node = NS(lock=threading.Lock(), get_logger=Mock(), hand_state={'right':state, 'left':state}, hand_holds={'left':None, 'right':None},
                  now=lambda: clock[0], _set_hand_limit=lambda *_: force_ok,
                  _set_hand_speed=lambda *_: True,
                  _finger_sample=lambda _: (0. if stale else clock[0], list(state.position)),
                  _command_fingers=lambda _, values: sent.append(list(values)),
                  cancelled=lambda: cancel_at is not None and clock[0] >= cancel_at)
        namespace = dict(Hand=NS(Result=NS, Feedback=NS), profile_for_preset=profile_for_preset, HAND_TARGETS=HAND_TARGETS,
                         hand_steps=hand_steps, math=math, time=NS(monotonic=lambda:wall[0], sleep=sleep),
                         PERIOD=.05, CLOCK_STALL_WALL_S=15., rclpy=NS(ok=lambda:True))
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        goal = NS(request=NS(hand=side, preset=preset, force=force), publish_feedback=lambda _:None)
        return namespace['recorded_hand'](node, goal, HAND_TARGETS[preset]), sent, clock[0]

    def test_action_emits_all_steps_and_waits_for_settle(self):
        result, sent, end = self.run_hand()
        self.assertTrue(result.success)
        self.assertEqual(len(sent), 13)
        np.testing.assert_allclose(sent[-1], RHINO['closure']['positions']['right'])
        self.assertGreaterEqual(end, 13.)
        self.assertLess(end, 13.06)

    def test_ros_float32_force_is_accepted_at_profile_limit(self):
        result, sent, _ = self.run_hand(force=float(np.float32(.6)))
        self.assertTrue(result.success)
        self.assertTrue(sent)

    def test_slow_clock_finishes_but_frozen_clock_stops(self):
        result, _, end = self.run_hand(sim_rate=.05)  # >60 wall seconds, but valid 3 sim-second closure
        self.assertTrue(result.success)
        self.assertAlmostEqual(end, 13., delta=.01)
        result, sent, end = self.run_hand(sim_rate=0.)
        self.assertFalse(result.success)
        self.assertEqual(len(sent), 1)
        self.assertEqual(end, 10.)

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

    def test_brother_left_release_opens_flexion_before_thumb_rotation(self):
        key = 'brother_left_wide'
        steps = hand_steps(key + '_release')
        self.assertTrue(all(p[5] == PROFILES[key]['closure']['positions']['left'][5] for p, _ in steps[:30]))
        np.testing.assert_allclose(steps[29][0][:5], [1.]*5)
        np.testing.assert_allclose(steps[-1][0], [1.]*6)
        result, sent, _ = self.run_hand(side='left', preset=key + '_release')
        self.assertTrue(result.success)
        np.testing.assert_allclose(sent[-1], [1.]*6)
        result, sent, _ = self.run_hand(side='right', preset=key + '_release')
        self.assertFalse(result.success)
        self.assertEqual(sent, [])

    def test_postlift_reaches_recorded_target_and_is_cancellable(self):
        result, sent, _ = self.run_hand(preset='nescafe_right_postlift')
        self.assertTrue(result.success)
        np.testing.assert_allclose(sent[-1], [.44]*4 + [.59, .1])
        result, sent, _ = self.run_hand(preset='nescafe_right_postlift', cancel_at=10.3)
        self.assertFalse(result.success)
        self.assertLess(len(sent), 40)


class PostliftMissionTests(unittest.TestCase):
    def mission(self):
        from hack_brain.context import Config, Object
        from fakes import Fake
        name = PROFILES['nescafe_right']['object_name']
        return Fake(objects=[Object(name, 'figure', name=name)], config=Config(postlift_names=frozenset({name})))

    def test_failed_squeeze_cannot_proceed_to_carry(self):
        from hack_brain.context import State
        for phase in (State.TIGHTEN, State.VERIFY_TIGHTEN):
            f = self.mission()
            f.until(phase)
            f.answer('failed', ok=False)
            self.assertEqual(f.c.state, State.DONE)
            self.assertEqual(f.c.reason, 'postlift_closure_failed')
            self.assertFalse(any(e.phase == State.TRANSFER for e in f.commands))

    def test_unconfirmed_squeeze_cannot_proceed_to_carry(self):
        from hack_brain.context import Evidence, Hand, State
        for hand, lifted, stale in [(Hand.EMPTY, True, False), (Hand.UNKNOWN, True, False),
                                     (Hand.HOLDING, False, False), (Hand.HOLDING, True, True)]:
            f = self.mission()
            f.until(State.VERIFY_TIGHTEN)
            f.answer(Evidence(f.sim - int(stale)*5, hand, f.c.target.id, lifted=lifted, source='fake'))
            self.assertEqual(f.c.reason, 'postlift_closure_unconfirmed')
            self.assertFalse(any(e.phase == State.TRANSFER for e in f.commands))

    def test_recover_after_lift_tightens_once_before_carry(self):
        from hack_brain.context import Evidence, Hand, State
        f = self.mission()
        f.until(State.LIFT)
        f.answer('timeout', ok=False)
        self.assertEqual(f.c.state, State.RECOVER)
        f.answer(Evidence(f.sim, Hand.HOLDING, f.c.target.id, lifted=True, source='fake'))
        self.assertEqual(f.c.state, State.TIGHTEN)
        f.until(State.TRANSFER)
        f.answer('unreachable', ok=False)
        self.assertEqual(f.c.state, State.RECOVER)
        f.answer(Evidence(f.sim, Hand.HOLDING, f.c.target.id, lifted=True, source='fake'))
        self.assertEqual(f.c.state, State.DESTINATION)
        f.run()
        self.assertEqual(sum(e.phase == State.TIGHTEN for e in f.commands), 1)


class RecordedCarryTests(unittest.TestCase):
    def test_carry_propagates_orientation_requirement_to_both_legs_and_fallback(self):
        source = Path(__file__).resolve().parents[1] / 'src/hack_motion/hack_motion/motion_node.py'
        tree = ast.parse(source.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_carry_legs')
        namespace = dict(MoveTcp=NS(Goal=NS, Result=NS), copy=NS(deepcopy=deepcopy))
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        for mode in ('carry',):
            for failed_leg in (None, 1, 2):
                sent = []
                def move(goal):
                    sent.append(goal)
                    return NS(success=len(sent) != failed_leg, message='test')
                node = NS(move_tcp=move, cancelled=lambda:False, get_logger=Mock())
                goal = NS(arm='right', target=planner.pose(.5, -.4, .5, planner.quat_from_rpy(0., 0., .3)),
                          mode=mode, pos_tol=.005, rot_tol=.05, max_time=0.)
                tf = NS(transform=NS(translation=NS(x=.3, y=-.2, z=.1)))
                result = namespace['_carry_legs'](node, goal, tf)
                self.assertTrue(result.success)
                suffix = '_fixed' if mode == 'carry_fixed' else ''
                expected = ['linear' + suffix]*2 if failed_leg is None else ['linear' + suffix]*failed_leg + ['direct' + suffix]
                self.assertEqual([g.mode for g in sent], expected)
                self.assertTrue(all(g.target.orientation == goal.target.orientation for g in sent))

    def plan_carry(self, *, strict, level_turn=None, fallback_turn=.1):
        import math
        source = Path(__file__).resolve().parents[1] / 'src/hack_motion/hack_motion/curobo_planner.py'
        tree = ast.parse(source.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_plan_carry')
        class Unreachable(RuntimeError):
            pass
        calls = []
        def level(arm, target, state, free_yaw):
            calls.append(free_yaw)
            if level_turn is None:
                raise Unreachable('no level path')
            return 'level', level_turn
        planner = NS(free_yaw=False, scene_note='', _plan_level=level,
                     _plan_direct=lambda *_:'direct', _goal=lambda *_:(None, None),
                     _carry_deviation=lambda arm, plan, *args:level_turn if plan == 'level' else fallback_turn,
                     _max_turn=lambda *_:fallback_turn)
        namespace = dict(Unreachable=Unreachable, tilt=lambda _:0., _xyzw=lambda _:None, VERTICAL=.1, math=math)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        try:
            result = namespace['_plan_carry'](planner, 'right', None, {}, .005, .05, keep_orientation=strict)
        except Unreachable:
            result = 'rejected'
        return result, calls

    def test_recorded_carry_disallows_free_yaw_and_rotating_fallback(self):
        self.assertEqual(self.plan_carry(strict=True), ('rejected', [False]))
        self.assertEqual(self.plan_carry(strict=True, level_turn=.02), ('level', [False]))
        self.assertEqual(self.plan_carry(strict=True, level_turn=.08), ('rejected', [False]))
        self.assertEqual(self.plan_carry(strict=True, fallback_turn=.02), ('direct', [False]))
        self.assertEqual(self.plan_carry(strict=False), ('direct', [False, True]))
