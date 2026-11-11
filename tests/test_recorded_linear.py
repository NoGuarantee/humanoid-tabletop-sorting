"""Recorded Philips approach geometry; GPU collision acceptance is tested live."""
import ast
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires Torch (available in deployment image)')
class RecordedLinearTests(unittest.TestCase):
    def test_refreshed_contact_aligns_lateral_error_before_descending(self):
        import torch
        source = ROOT / 'src/hack_motion/hack_motion/curobo_planner.py'
        method = next(n for n in ast.walk(ast.parse(source.read_text()))
                      if isinstance(n, ast.FunctionDef) and n.name == '_plan_linear')
        namespace = dict(math=math, TCP_LINK={'right': 'right_tcp'}, LEAN_STEP=.12)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        # Measured seed7 pregrasp and fresh contact target, 72e27c3.
        start, end = [.39301, .12001, .16837], [.40863, .12031, .07016]
        class Goal(NS):
            def clone(self): return deepcopy(self)
        goal = Goal(tool_frames=['right_tcp'],
                    position=torch.tensor(end).reshape(1,1,1,1,3),
                    quaternion=torch.tensor([1.,0.,0.,0.]).reshape(1,1,1,1,4))
        for tolerance in (None, .005):
            calls = []
            def linear(link, target, state, axis, *, goal_frame):
                point = target.position[0,0,0,0].tolist()
                calls.append((axis, goal_frame, point))
                return ['point'], [point], None
            planner = NS(mp=NS(compute_kinematics=lambda state: NS(tool_poses=NS(
                position=torch.tensor(state.p).reshape(1,1,1,3)))),
                _linear_axis=linear, _chain_state=lambda names, point, prev: NS(p=point))
            namespace['_plan_linear'](planner,'right',goal,NS(p=start),world_axes=True,
                                      axis_tolerance=tolerance)
            if tolerance is None:
                self.assertEqual([axis for axis,_,_ in calls], ['z'])
            else:
                self.assertEqual([axis for axis,_,_ in calls], ['x','y','z'])
                self.assertTrue(all(not frame for _,frame,_ in calls))
                self.assertTrue(all(abs(point[2]-start[2])<1e-6 for _,_,point in calls[:-1]))
                np.testing.assert_allclose(calls[-1][2], end, atol=1e-6)

    def test_recorded_philips_descent_keeps_world_vertical_path(self):
        import torch
        source = ROOT / 'src/hack_motion/hack_motion/curobo_planner.py'
        method = next(n for n in ast.walk(ast.parse(source.read_text()))
                      if isinstance(n, ast.FunctionDef) and n.name == '_plan_linear')
        namespace = dict(math=math, TCP_LINK={'right': 'right_tcp'}, LEAN_STEP=.12)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        profile = next(p for p in json.loads((ROOT / 'src/hack_motion/hack_motion/profiles/boxes.json').read_text())['profiles']
                       if p['key'] == 'philips_right')
        start = np.array(profile['T_reference_pregrasp']['right'])
        end = np.array(profile['T_reference_tcp']['right'])
        xyzw = Rotation.from_matrix(end[:3, :3]).as_quat()
        class Goal(NS):
            def clone(self): return deepcopy(self)
        goal = Goal(tool_frames=['right_tcp'],
                    position=torch.tensor(end[:3, 3]).reshape(1, 1, 1, 1, 3),
                    quaternion=torch.tensor(xyzw[[3, 0, 1, 2]]).reshape(1, 1, 1, 1, 4))
        for fixed in (False, True):
            calls = []
            def linear(link, target, state, axis, *, goal_frame):
                p = target.position[0, 0, 0, 0].tolist()
                calls.append((axis, goal_frame, p))
                return ['point'], [p], None
            planner = NS(mp=NS(compute_kinematics=lambda state: NS(tool_poses=NS(
                position=torch.tensor(state.p).reshape(1, 1, 1, 3)))),
                _linear_axis=linear, _chain_state=lambda names, point, prev: NS(p=point))
            namespace['_plan_linear'](planner, 'right', goal, NS(p=start[:3, 3].tolist()), world_axes=fixed)
            if fixed:
                self.assertEqual([(axis, frame) for axis, frame, _ in calls], [('z', False)])
                np.testing.assert_allclose(calls[0][2], end[:3, 3])
            else:
                # Actual source profile exposes the old lateral dogleg, rather
                # than a synthetic upright hand which would pass either way.
                self.assertGreater(len(calls), 1)
                self.assertGreater(max(np.linalg.norm(np.array(p)[:2]-start[:2, 3]) for _, _, p in calls), .02)

