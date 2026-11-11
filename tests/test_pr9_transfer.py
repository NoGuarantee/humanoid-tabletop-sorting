"""PR9 E71/E68 transfer regressions; source evidence, not a new physical trial."""
import ast
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from contextlib import nullcontext
from unittest.mock import Mock

import numpy as np

import test_recorded_sort as base
from hack_brain.context import Config, Context, Container, Object, Scene
from hack_brain.fsm import _side_to_look
from hack_brain.policies import select
from hack_motion.recorded_carry import carry_waypoints, pose_matrix, with_matrix, _minimum_rotating_support
from hack_motion.recorded_geometry import axis_angle, interpolate, rotation_error
from fakes import sort_spec

ROOT = Path(__file__).resolve().parents[1]


def method(file, name, **namespace):
    source = ROOT / 'src/hack_motion/hack_motion' / file
    tree = ast.parse(source.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[name]


class PR9TransferTests(unittest.TestCase):
    def test_philips_orders_existing_profiles_by_pick_and_place_reach(self):
        name = base.PROFILES['philips_right']['object_name']
        for yaw in (0., np.pi):
            data = base.observation(name)
            world = np.eye(4)
            world[:3, :3] = axis_angle([0., 0., 1.], yaw)[:3, :3]
            world[:3, 3] = [.34, .08, -.16]
            data['objects'][0].update(T_pelvis_object=world.tolist(), center_pelvis_m=[.34, .08, -.10])
            container = base.tray(y=0., yaw=0., size=(.32, .32))
            choices = []
            for key in ('philips_right', 'philips_right_turn180'):
                g = base.pick.candidate(data, 10.1, base.detected(name), container, base.DB, 0, profile_key=key)
                p = base.pick.place(g, container, base.DB)
                choices.append((max(np.hypot(t.position.x, t.position.y) for t in (g.grasp, p.approach)), key))
            expected = sorted(choices)
            for attempt in (1, 2):
                g = base.pick.candidate(data, 10.1, base.detected(name), container, base.DB, 0, attempt=attempt)
                self.assertEqual(g.profile_key, expected[attempt-1][1])
            # An explicit profile survives the OPEN -> DESCEND refresh.
            far = base.pick.candidate(data, 10.1, base.detected(name), container, base.DB, 0,
                                      profile_key=expected[1][1])
            self.assertEqual(far.profile_key, expected[1][1])

    def test_philips_carry_reuses_native_rim_clearance_without_changing_release(self):
        for key in ('philips_right', 'philips_right_turn180'):
            name = base.PROFILES[key]['object_name']
            container = base.tray(y=0., yaw=0.)
            g = base.pick.candidate(base.observation(name), 10.1, base.detected(name),
                                    container, base.DB, 0, profile_key=key)
            p = base.pick.place(g, container, {'place': {'approach_height': .20, 'release_height': .10}})
            points = np.asarray(g.corners_in_tcp) + g.center_in_tcp
            for target, clearance in ((p.approach, .15), (p.release, .10)):
                matrix = pose_matrix(target)
                low = (points @ matrix[:3, :3].T + matrix[:3, 3])[:, 2].min()
                self.assertAlmostEqual(low-container.pose.position.z, clearance)

    def test_failed_fixed_carry_cannot_repark_other_arm_and_swing_shared_torso(self):
        class Unreachable(RuntimeError): pass
        execute = method('motion_node.py', 'move_tcp', Unreachable=Unreachable,
                         ARM_JOINTS={'right': ['r'], 'left': ['l']}, json=json,
                         contact_object=lambda *a: '', traceback=__import__('traceback'))
        def blocked(*args, **kwargs): raise Unreachable('unreachable carry height')
        for fixed, carrying in ((True, True), (True, False), (False, True)):
            repark = Mock(return_value=False)
            planner = NS(table=[0, 0, 1, 0], trays=[], load=lambda _: 'curobo already loaded',
                         sync_world=lambda _: None, plan=blocked, start_report=lambda _: 'clear', scene_note='')
            node = NS(get_parameter=lambda _: NS(value='curobo'), lock=nullcontext(), _tcp_lock=nullcontext(),
                      _wait_joints=lambda: True, _planner_obj=lambda: planner, q={}, scene_objects=[],
                      carrying=carrying, _leave_colliding_home=lambda *a: None,
                      _world=lambda: ([0, 0, 1, 0], []), now=lambda: 10., _other_arm_to_ready=repark,
                      _refused=lambda *args: NS(success=False), get_logger=Mock())
            goal = NS(arm='right', mode='direct_fixed' if fixed else 'direct', pos_tol=.005, rot_tol=.05,
                      target=base.planner.pose(.4, .1, .378, base.planner.quat_from_rpy(0, 0, 0)))
            self.assertFalse(execute(node, goal).success)
            self.assertEqual(repark.call_count, 0 if fixed and carrying else 1)

    def test_e71_real_pose_uses_right_hand_left_tray_and_recorded_heading(self):
        f = json.loads((ROOT / 'tests/fixtures/rhino_e71_transfer.json').read_text())
        profile, pick = base.RHINO, base.pick
        self.assertEqual(profile['id'], f['request']['profile_id'])
        self.assertEqual(profile['source']['request_sha256'], f['source_sha256']['request.json'])
        self.assertEqual(profile['effort_limit'], f['request']['hand_effort_limit_ratio'])
        self.assertTrue(f['outcome']['flags']['carry_completed'])
        self.assertFalse(f['outcome']['success'])
        self.assertIsNone(f['outcome']['flags']['released'])
        t = f['destination']
        opening = np.array(t['T_pelvis_opening'])
        container = NS(id=t['id'], kind='tray', size=NS(**dict(zip('xyz', t['size_m']))),
                       pose=with_matrix(base.tray().pose, opening))
        data = f['observation']
        grasp = pick.candidate(data, data['stamp_ns']/1e9 + .1, base.detected(), container,
                               base.DB, 0, disabled_arm='left')
        self.assertEqual(grasp.arm, 'right')
        plan = pick.place(grasp, container, base.DB)
        want = axis_angle(opening[:3, 2], f['request']['placement_yaw_preference_rad'])[:3, :3]
        np.testing.assert_allclose(pose_matrix(plan.release)[:3, :3],
                                   want @ pose_matrix(grasp.grasp)[:3, :3], atol=1e-12)
        points = np.array(grasp.corners_in_tcp) + grasp.center_in_tcp
        release = pose_matrix(plan.release)
        placed = (points @ release[:3, :3].T + release[:3, 3] - opening[:3, 3]) @ opening[:3, :3]
        self.assertTrue(np.all(abs(placed[:, :2]).max(0) <= np.array(t['size_m'][:2])/2 - .015 + 1e-12))
        self.assertAlmostEqual(placed[:, 2].min(), .035)
        # Sweep includes the measured tray tilt and a nonzero object/TCP offset.
        start = pose_matrix(grasp.lift)
        route = carry_waypoints(start, pose_matrix(plan.approach))
        floor = (points @ start[:3, :3].T + start[:3, 3])[:, 2].min()
        previous = start
        for waypoint in route:
            for alpha in np.linspace(0, 1, 21):
                pose = interpolate(previous, waypoint, alpha)
                self.assertGreaterEqual((points @ pose[:3, :3].T + pose[:3, 3])[:, 2].min(), floor - 1e-10)
            previous = waypoint

    def test_disabled_arm_policy_and_head_look_keep_cross_table_route(self):
        spec = sort_spec(); spec.categories['figure'] = 'tray_left'
        obj = Object('rhino', 'figure', name=base.RHINO['object_name'], side='right')
        c = Context(spec=spec, config=Config(recorded_arms=base.pick.recorded_arms(base.DB)), disabled_arm='left')
        scene = Scene(10., (obj,), (Container('tray_left', 'tray', side='left'),))
        self.assertEqual(select(c, scene), obj)
        self.assertEqual(_side_to_look(c, Scene(10.)), 'left')
        c.disabled_arm = 'right'
        self.assertIsNone(select(c, scene))
        # Legacy profiles still obey their original side/disabled-arm policy.
        c.config = Config(); c.disabled_arm = 'left'
        self.assertIsNone(select(c, scene))
        self.assertEqual(_side_to_look(c, Scene(10.)), '')
        name = base.PROFILES['brother_right']['object_name']
        grasp = base.pick.candidate(base.observation(name), 10.1, base.detected(name), base.tray(.5),
                                    base.DB, 0, disabled_arm='left')
        self.assertEqual(grasp.profile_key, 'brother_right')

    def test_raise_does_not_turn_and_turn_segments_remain_bounded(self):
        start = axis_angle([1., 0., 0.], .4); start[:3, 3] = [.3, -.3, .1]
        target = axis_angle([0., 0., 1.], -1.47215) @ start; target[:3, 3] = [.4, .6, .4]
        route = carry_waypoints(start, target)
        np.testing.assert_allclose(route[0][:3, :3], start[:3, :3])
        np.testing.assert_allclose(route[0][:2, 3], start[:2, 3])
        self.assertEqual(route[0][2, 3], .4)
        for a, b in zip(route, route[1:]):
            self.assertLessEqual(rotation_error(a, b), .04 + 1e-10)
            self.assertLessEqual(np.linalg.norm(a[:3, 3] - b[:3, 3]), .08 + 1e-10)
        np.testing.assert_allclose(route[-1], target, atol=1e-12)

    def test_recorded_action_retries_only_failed_segment_and_stops_on_cancel(self):
        execute = method('motion_node.py', '_recorded_carry_legs', MoveTcp=NS(Goal=NS, Result=NS),
                         copy=NS(deepcopy=deepcopy), carry_waypoints=carry_waypoints,
                         pose_matrix=pose_matrix, with_matrix=with_matrix)
        target = base.planner.pose(.4, .6, .4, base.planner.quat_from_rpy(.4, 0, -1.47215))
        g = NS(arm='right', target=target, pos_tol=.005, rot_tol=.05, max_time=0.)
        tf = NS(header=NS(stamp=NS(sec=10, nanosec=0)),
                transform=NS(translation=NS(x=.3, y=-.3, z=.1), rotation=base.planner.quat_from_rpy(.4, 0, 0)))
        for fail, cancel in ((True, False), (False, True)):
            sent = []
            def move(goal):
                sent.append(goal)
                return NS(success=not fail, message='blocked')
            node = NS(now=lambda:10.1, move_tcp=move, cancelled=lambda:cancel and len(sent) == 1)
            result = execute(node, g, tf)
            self.assertFalse(result.success)
            self.assertEqual(len(sent), 1 if cancel else 2)
            self.assertEqual(sent[0].mode, 'linear_fixed')
            self.assertEqual(sent[0].target.orientation, tf.transform.rotation)
            self.assertNotEqual(sent[0].target.position.y, target.position.y)
            if fail:
                self.assertEqual(sent[1].mode, 'direct_fixed')
                self.assertEqual(sent[1].target, sent[0].target)
        node = NS(now=lambda:11., move_tcp=lambda _:self.fail('stale TF moved'), cancelled=lambda:False)
        self.assertFalse(execute(node, g, tf).success)
        self.assertFalse(execute(node, g, None).success)

    def test_strict_carry_linear_path_cannot_skip_orientation_bound(self):
        class Unreachable(RuntimeError): pass
        plan = method('curobo_planner.py', 'plan', Unreachable=Unreachable)
        for held in ([], ['held_object']):
            node = NS(mp=object(), carrying=True, sync_world=lambda _:None, _held=lambda _:held,
                      _without=lambda _, fn:fn(), _plan_in_world=lambda *_:'path',
                      _goal=lambda *_:('goal', 'state'), _carry_deviation=lambda *_:.08, scene_note='')
            with self.assertRaises(Unreachable):
                plan(node, 'right', None, {}, 'linear', rot_tol=.05, keep_orientation=True)
            self.assertEqual(plan(node, 'right', None, {}, 'linear', rot_tol=.05), 'path')

    # Adapted directly from PR9 test_grasp_carry_route.py; retains its seed and cases.
    def test_exact_swept_support_matches_dense_rotations_including_interior_minimum(self):
        rng = np.random.default_rng(5917)
        points = rng.uniform(-.25, .25, (8, 3))
        cases = [(np.eye(4), axis_angle([0, 1, 0], np.pi))]
        for _ in range(14):
            start = axis_angle(rng.normal(size=3), rng.uniform(-np.pi, np.pi))
            finish = start @ axis_angle(rng.normal(size=3), rng.uniform(0., np.pi))
            cases.append((start, finish))
        for start, finish in cases:
            up = rng.normal(size=3); up /= np.linalg.norm(up)
            exact = _minimum_rotating_support(points, start[:3, :3], finish[:3, :3], up)
            dense = min(float(np.min(points @ (interpolate(start, finish, a)[:3, :3].T @ up)))
                        for a in np.linspace(0, 1, 1001))
            self.assertLessEqual(exact, dense+1e-12)
            self.assertLess(dense-exact, 2e-6)
        points = np.asarray(list(product([-.01, .01], [-.01, .01], [-.11, -.09])))
        start = axis_angle([0, 1, 0], -np.pi/3)
        finish = axis_angle([0, 1, 0], np.pi/3)
        minimum = _minimum_rotating_support(points, start[:3, :3], finish[:3, :3], np.array([0., 0., 1.]))
        self.assertLess(minimum, -.11)
        self.assertGreater(np.min(points @ start[:3, :3].T @ [0., 0., 1.]), -.07)
