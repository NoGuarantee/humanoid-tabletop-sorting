"""Robot exclusion uses measured 3D depth, never just a projected silhouette."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src/hack_perception'), str(ROOT/'src/hack_motion')]

from hack_perception.robot_self_mask import mask_robot_depth


class Spheres:
    required = ('measured_joint',)

    def __init__(self, directory, spheres):
        self.values = spheres
        for name in ('urdf_path', 'cfg_path', 'hand_path'):
            path = Path(directory)/name
            path.write_text(name)
            setattr(self, name, path)

    def spheres(self, q):
        return self.values


def brute(depth, k, transform, spheres, margin):
    y, x = np.indices(depth.shape)
    rays = np.stack((x, y, np.ones_like(x)), axis=-1) @ np.linalg.inv(k).T
    points = rays*(depth/rays[..., 2])[..., None]
    points = points @ transform[:3, :3].T+transform[:3, 3]
    result = np.zeros(depth.shape, bool)
    for _, center, radius in spheres:
        result |= np.sum((points-center)**2, axis=-1) <= (radius+margin)**2
    return result & np.isfinite(depth) & (depth > 0)


class RobotSelfMaskTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.geometry = Spheres(self.directory.name, [('robot', np.array([0., 0., 1.]), .125)])
        self.q = {'measured_joint': .4}
        self.k = np.array([[100., 0., 1.], [0., 100., 0.], [0., 0., 1.]])
        self.transform = np.eye(4)

    def apply(self, depth, **kwargs):
        return mask_robot_depth(depth, kwargs.pop('K', self.k), kwargs.pop('T', self.transform),
                                kwargs.pop('q', self.q), kwargs.pop('geometry', self.geometry), **kwargs)

    def test_front_and_behind_survive_projected_silhouette(self):
        result = self.apply(np.array([[.8, 1., 1.2]]), margin_m=0.)
        np.testing.assert_array_equal(result['mask'], [[False, True, False]])

    def test_explicit_margin_and_boundary(self):
        k = np.eye(3)
        np.testing.assert_array_equal(self.apply(np.array([[1.125]]), K=k, margin_m=0.)['mask'], [[True]])
        self.assertFalse(self.apply(np.array([[1.127]]), K=k, margin_m=0.)['mask'][0, 0])
        self.assertTrue(self.apply(np.array([[1.127]]), K=k)['mask'][0, 0])
        self.assertFalse(self.apply(np.array([[1.129]]), K=k)['mask'][0, 0])

    def test_rotated_and_translated_camera(self):
        transform = np.array([[0., 0., 1., .2], [0., 1., 0., -.1], [-1., 0., 0., .3], [0., 0., 0., 1.]])
        point = transform[:3, :3] @ np.array([0., 0., 1.])+transform[:3, 3]
        self.geometry.values = [('hand', point, .125)]
        result = self.apply(np.array([[.8, 1., 1.2]]), T=transform, margin_m=0.)
        np.testing.assert_array_equal(result['mask'], [[False, True, False]])

    def test_roi_matches_independent_full_image_reference(self):
        rng = np.random.default_rng(20261108)
        depth = rng.uniform(.01, 2., (37, 51))
        for index in range(30):
            angle = rng.uniform(-np.pi, np.pi)
            c, s = np.cos(angle), np.sin(angle)
            transform = np.eye(4)
            transform[:3, :3] = [[c, 0., s], [0., 1., 0.], [-s, 0., c]]
            transform[:3, 3] = rng.uniform(-.2, .2, 3)
            # Skew and off-diagonal calibration exercise conservative rectangle bounds.
            k = np.array([[35., 2., 25.], [.5, 33., 18.], [0., 0., 1.]])
            camera_centers = rng.uniform([-.8, -.8, -.4], [.8, .8, 2.], (8, 3))
            centers = camera_centers @ transform[:3, :3].T+transform[:3, 3]
            self.geometry.values = [(str(i), p, float(r)) for i, (p, r) in enumerate(zip(centers, rng.uniform(.03, .5, 8)))]
            result = self.apply(depth, K=k, T=transform)['mask']
            np.testing.assert_array_equal(result, brute(depth, k, transform, self.geometry.values, .003), err_msg=str(index))

    def test_sphere_crossing_camera_plane_and_sphere_behind(self):
        self.geometry.values = [('crossing', np.array([0., 0., .02]), .2),
                                ('behind', np.array([0., 0., -1.]), .1)]
        depth = np.full((3, 3), .1)
        np.testing.assert_array_equal(self.apply(depth)['mask'], brute(depth, self.k, self.transform, self.geometry.values, .003))
        self.geometry.values = self.geometry.values[1:]
        self.assertFalse(self.apply(depth)['mask'].any())

    def test_unknown_depth_is_not_robot(self):
        result = self.apply(np.array([[0., np.nan, np.inf, -1.]], dtype=np.float32))
        self.assertFalse(result['mask'].any())
        self.assertEqual(result['provenance']['valid_depth_pixels'], 0)

    def test_projection_limits_work_to_local_sphere_roi(self):
        self.geometry.values = [('finger', np.array([0., 0., 1.]), .02)]
        k = np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]])
        result = self.apply(np.ones((100, 100)), K=k)
        self.assertEqual(result['provenance']['projected_sphere_count'], 1)
        self.assertLess(result['provenance']['candidate_roi_pixels'], 1000)
        self.assertLess(result['provenance']['tested_pixels'], 1000)

    def test_missing_nonfinite_and_boolean_measured_joints_rejected(self):
        for q in ({}, {'measured_joint': np.nan}, {'measured_joint': np.inf},
                  {'measured_joint': True}, {'measured_joint': '0'}, {'measured_joint': .4, 'extra': np.nan}):
            with self.subTest(q=q), self.assertRaises(ValueError):
                self.apply(np.ones((1, 3)), q=q)

    def test_invalid_units_shape_calibration_transform_and_margin_rejected(self):
        for depth in (np.ones((2, 2), np.uint16), np.ones((2, 2), bool), np.ones(3), np.ones((0, 2))):
            with self.subTest(depth=str(depth.shape)), self.assertRaises(ValueError):
                self.apply(depth)
        bad_k = self.k.copy(); bad_k[0, 0] = np.nan
        reflected = np.eye(4); reflected[0, 0] = -1
        for kwargs in ({'K': bad_k}, {'K': np.zeros((3, 3))}, {'T': reflected},
                       {'T': np.ones((4, 4))}, {'margin_m': True}, {'margin_m': np.nan},
                       {'margin_m': -.001}, {'margin_m': .021}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.apply(np.ones((1, 3)), **kwargs)

    def test_inputs_immutable_and_hashes_reproducible(self):
        depth = np.array([[.8, 1., 1.2]], dtype=np.float32)
        original = [x.copy() for x in (depth, self.k, self.transform)]
        q = copy.deepcopy(self.q)
        for array in (depth, self.k, self.transform):
            array.setflags(write=False)
        first, second = self.apply(depth), self.apply(depth)
        for actual, before in zip((depth, self.k, self.transform), original):
            np.testing.assert_array_equal(actual, before)
        self.assertEqual(self.q, q)
        self.assertEqual(first['provenance'], second['provenance'])
        self.assertEqual(first['mask'].dtype, np.bool_)
        first['mask'][:] = False
        self.assertTrue(second['mask'][0, 1])
        self.assertEqual(set(second['provenance']['robot_files_sha256']), {'urdf', 'curobo_spheres', 'native_hand_spheres'})
        json.dumps(second['provenance'], allow_nan=False)

    def test_real_robot_includes_articulated_fingers_and_full_body(self):
        from hack_motion.perception_robot_geometry import RobotGeometry
        root = ROOT/'src/hack_motion/robot'
        geometry = RobotGeometry(root/'tiangong2pro_with_hands.urdf', root/'tienkung_curobo.yml', root/'native_hand_collision.json')
        q = dict.fromkeys(geometry.required, 0.)
        original = q.copy()
        first = self.apply(np.ones((2, 3)), q=q, geometry=geometry)
        self.assertEqual(q, original)
        self.assertEqual(first['provenance']['robot_sphere_count'], 706)
        names = [name for name, _, _ in geometry.spheres(q)]
        self.assertTrue(any(name.startswith('left/hand/') for name in names))
        self.assertTrue(any(name.startswith('right/hand/') for name in names))
        self.assertTrue(any('knee' in name for name in names))
        q['right_thumb_1_joint'] = -.2
        second = self.apply(np.ones((2, 3)), q=q, geometry=geometry)
        self.assertNotEqual(first['provenance']['sphere_geometry_sha256'], second['provenance']['sphere_geometry_sha256'])


if __name__ == '__main__':
    unittest.main()
