"""Contact permissions remain scoped to one measured object and one descent."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from hack_motion.recorded_contact import contact_object, measured_contact


def setup():
    goal = NS(mode='linear_fixed', contact_object='selected', contact_stamp_ns=10_000_000_000,
              contact_center=[.34, .085, -.10], target=NS(position=NS(x=.409, y=.121, z=.069)))
    objects = [('selected', .342, .085, -.102, 1.389, .104, .066, .107, -.048),
               ('neighbour', .34, -.157, -.12, 1., .096, .097, .071, -.084)]
    tcp = NS(header=NS(stamp=NS(sec=10, nanosec=100_000_000)),
             transform=NS(translation=NS(x=.405, y=.117, z=.170)))
    return goal, objects, tcp


class RecordedContactTests(unittest.TestCase):
    def test_selected_visible_cluster_accepts_fresh_full_model_center(self):
        goal, objects, tcp = setup()
        self.assertEqual(contact_object(goal, objects, 10.2, tcp), 'selected')
        self.assertEqual(contact_object(NS(), objects, 10.2, None), '')

    def test_fresh_tf_one_tick_ahead_waits_for_clock_without_relaxing_gate(self):
        goal, objects, tcp = setup()
        with self.assertRaisesRegex(ValueError, 'stale_recorded_contact_tcp'):
            contact_object(goal, objects, 10.09, tcp)
        read_tcp = Mock(return_value=tcp)
        read_now = Mock(side_effect=[10.09, 10.10])
        with patch('hack_motion.recorded_contact.time.sleep') as sleep:
            self.assertEqual(measured_contact(goal, objects, read_now, read_tcp), 'selected')
        self.assertEqual(read_tcp.call_count, 1)
        sleep.assert_called_once_with(.01)

    def test_contact_refresh_wait_is_bounded_and_preserves_cancel_and_pose_expiry(self):
        goal, objects, tcp = setup()
        old = deepcopy(tcp); old.header.stamp.sec = 9
        read_tcp = Mock(side_effect=[old, tcp])
        with patch('hack_motion.recorded_contact.time.sleep'):
            self.assertEqual(measured_contact(goal, objects, lambda: 10.2, read_tcp), 'selected')
        for sample in (old, None):
            with patch('hack_motion.recorded_contact.time.monotonic', side_effect=[0., .51]):
                with self.assertRaises(ValueError):
                    measured_contact(goal, objects, lambda: 10.2, lambda: sample)
        with self.assertRaisesRegex(ValueError, 'cancelled'):
            measured_contact(goal, objects, lambda: 10.2, lambda: tcp, cancelled=lambda: True)
        with self.assertRaisesRegex(ValueError, 'stale_recorded_contact$'):
            measured_contact(goal, objects, lambda: 12., lambda: tcp)

    def test_stale_future_moved_missing_and_duplicate_targets_reject(self):
        goal, objects, tcp = setup()
        for stamp in (8_000_000_000, 11_000_000_000):
            bad = deepcopy(goal); bad.contact_stamp_ns = stamp
            with self.assertRaises(ValueError): contact_object(bad, objects, 10.2, tcp)
        for rows in (objects[1:], objects + [objects[0]], [('selected', *objects[1][1:])]):
            with self.assertRaises(ValueError): contact_object(goal, rows, 10.2, tcp)
        bad = deepcopy(goal); bad.contact_center[0] += .20
        with self.assertRaises(ValueError): contact_object(bad, objects, 10.2, tcp)

    def test_contact_cannot_authorize_carry_direct_lift_or_long_move(self):
        goal, objects, tcp = setup()
        with self.assertRaises(ValueError): contact_object(goal, objects, 10.2, tcp, carrying=True)
        with self.assertRaises(ValueError): contact_object(goal, objects, 10.2, None)
        for mode, z in [('direct_fixed', .069), ('linear', .069), ('linear_fixed', .20), ('linear_fixed', -.20)]:
            bad = deepcopy(goal); bad.mode = mode; bad.target.position.z = z
            with self.assertRaises(ValueError): contact_object(bad, objects, 10.2, tcp)
        old = deepcopy(tcp); old.header.stamp.sec = 9
        with self.assertRaises(ValueError): contact_object(goal, objects, 10.2, old)

    def test_explicit_contact_preserves_neighbour_and_restores_collision_world(self):
        source = Path(__file__).resolve().parents[1] / 'src/hack_motion/hack_motion/curobo_planner.py'
        tree = ast.parse(source.read_text())
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                     and n.name in ('plan', '_without', '_plan_in_world')]
        class Unreachable(RuntimeError): pass
        ns = dict(Unreachable=Unreachable)
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), ns)
        for fail in (False, True):
            enabled = {'selected': True, 'neighbour': True, 'table': True}
            worlds = []
            def inner(*args, **kwargs):
                self.assertTrue(kwargs['explicit_contact'])
                self.assertEqual(enabled, {'selected': False, 'neighbour': True, 'table': True})
                if fail: raise Unreachable('blocked by neighbour')
                return 'path'
            planner = NS(mp=NS(scene_collision_checker=NS(enable_obstacle=lambda name, value: enabled.__setitem__(name, value)),
                               graph_planner=NS(reset_buffer=lambda: None)),
                         carrying=False, sync_world=worlds.append, _object_cuboids=lambda: {'selected': {}, 'neighbour': {}},
                         _plan_in_world=inner, scene_note='')
            planner._without = lambda names, run: ns['_without'](planner, names, run)
            if fail:
                with self.assertRaises(Unreachable):
                    ns['plan'](planner, 'right', None, {}, 'linear', keep_orientation=True, contact_object='selected')
            else:
                self.assertEqual(ns['plan'](planner, 'right', None, {}, 'linear', keep_orientation=True,
                                           contact_object='selected'), 'path')
            self.assertEqual(worlds, [None])
            self.assertTrue(all(enabled.values()))
        planner = NS(_goal=lambda *_: (None, None),
                     _plan_linear=lambda *a, **kw: (_ for _ in ()).throw(Unreachable('neighbour')),
                     _touching=lambda *_: self.fail('contact must not widen'))
        with self.assertRaises(Unreachable):
            ns['_plan_in_world'](planner, 'right', None, {}, 'linear', .005, .05, True, explicit_contact=True)
