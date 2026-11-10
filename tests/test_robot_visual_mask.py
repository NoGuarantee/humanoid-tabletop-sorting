"""Native robot FK, exact 3mm hull distance and immutable measured RGB-D."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import ConvexHull

from hack_motion.perception_robot_geometry import RobotGeometry
from hack_perception.robot_self_mask import mask_robot_depth
from hack_perception.robot_visual_mask import (ASSET_SHA256, NativeBodyVisualHulls,
                                               hull_contains, point_triangle_distance)

ROOT = Path(__file__).resolve().parents[1]
ROBOT = ROOT/'src/hack_motion/robot'


class TriangleDistanceTests(unittest.TestCase):
    def test_triangle_faces_edges_corners_and_degenerate_edges(self):
        triangle = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
        for point, expected in (([.2, .2, .1], .1), ([.5, -.1, 0.], .1),
                                ([-.1, -.1, 0.], np.sqrt(.02)), ([0., 0., 0.], 0.)):
            self.assertAlmostEqual(point_triangle_distance(point, triangle), expected, places=14)
        degenerate = np.array([[[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]]])
        self.assertAlmostEqual(point_triangle_distance([.5, .2, 0.], degenerate), .2)
        with self.assertRaises(ValueError):
            point_triangle_distance([0., 0., 0.], triangle, chunk_size=0)

    def test_cube_corner_is_not_inflated_halfspace_shortcut(self):
        vertices = np.array([[x, y, z] for x in (0., 1.) for y in (0., 1.) for z in (0., 1.)])
        convex = ConvexHull(vertices)
        points = np.array([[.5, .5, .5], [1.003, .5, .5], [1.00301, .5, .5],
                           [1.0025, 1.0025, .5], [1.001, 1.001, .5], [.5, .5, -1.], [.5, .5, 2.]])
        # Each corner plane is within3mm for this point, but its Euclidean
        # distance is3.536mm. Only a triangle/edge distance may accept it.
        self.assertLess(np.max(convex.equations[:, :3]@points[3]+convex.equations[:, 3]), .003)
        mask, _ = hull_contains(points, vertices, convex.simplices, convex.equations, .003)
        np.testing.assert_array_equal(mask, [True, True, False, False, True, False, False])
        repeated = np.repeat(vertices[convex.simplices], 100, axis=0)
        self.assertAlmostEqual(point_triangle_distance(points[3], repeated, chunk_size=7), np.sqrt(2)*.0025, places=14)


class NativeVisualMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = NativeBodyVisualHulls.load(ROBOT/'native_body_visual_hulls.json', ROBOT/'native_hand_collision.json')
        cls.data = json.loads((ROBOT/'native_body_visual_hulls.json').read_text())
        cls.fixture = json.loads((ROOT/'tests/fixtures/native_body_measured_points135.json').read_text())

    def test_real_measured_fixture_matches_all135_native_decisions(self):
        fixture = self.fixture
        # This recorded fixture predates the added shoulders; its decisions stay unchanged.
        self.assertEqual(fixture['model_asset_sha256'], '5ba0a9027f7b099eadcd9956ef002d59d8ced956e32fc3a15add5c006837700d')
        original = copy.deepcopy(fixture['q'])
        mask, stats = self.model.cover_points(fixture['points_pelvis_m'], fixture['q'])
        np.testing.assert_array_equal(mask, fixture['expected_native_body_mask'])
        self.assertEqual(int(mask.sum()), 21)
        self.assertGreater(stats['triangle_tests'], 0)
        self.assertEqual(fixture['q'], original)
        # Every newly removed native fringe point is OUTSIDE the body volume;
        # inside-only masking cannot fix this measured failure.
        self.assertTrue(all(.0003 < row['distance_m'] < .0017 for row in fixture['native_body_results'] if row['covered']))

    def test_e68_shoulder_fringe_is_masked_at_exposure_without_erasing_rhino(self):
        fixture=json.loads((ROOT/'tests/fixtures/native_shoulder_e68.json').read_text())
        geometry=RobotGeometry(ROBOT/'tiangong2pro_with_hands.urdf',ROBOT/'tienkung_curobo.yml',ROBOT/'native_hand_collision.json')
        for case in fixture['cases']:
            q=case['q'];point=np.asarray(case['point']);k=np.asarray(case['K']);t=np.asarray(case['T'])
            # Original sphere margin cannot identify this visual shoulder fringe.
            self.assertGreater(min(np.linalg.norm(point-c)-r for _,c,r in geometry.spheres(q)),.003)
            covered,_=self.model.cover_points([point],q)
            self.assertTrue(covered[0])
            target=np.asarray(case['target_surface_points'])
            self.assertGreater(len(target),500)
            erased,_=self.model.cover_points(target,q)
            self.assertFalse(erased.any())
            camera=(point-t[:3,3])@t[:3,:3];pixel=k@camera;pixel=pixel[:2]/pixel[2]
            np.testing.assert_allclose(pixel,np.round(pixel),atol=1e-7)
            self.assertAlmostEqual(camera[2]*1000,round(camera[2]*1000),places=6)
            small_k=k.copy();small_k[:2,2]+=2-np.round(pixel)
            measured=np.zeros((5,5));measured[2,2]=camera[2]
            geometry.body_visual_hulls=self.model
            mask=mask_robot_depth(measured,small_k,t,q,geometry)['mask']
            self.assertTrue(mask[2,2]);self.assertEqual(int(mask.sum()),1)

    def test_native_fk_not_urdf_fk_and_no_missing_or_default_joint(self):
        fixture = self.fixture
        native = self.model.native_fk(fixture['q'])
        geometry = RobotGeometry(ROBOT/'tiangong2pro_with_hands.urdf', ROBOT/'tienkung_curobo.yml', ROBOT/'native_hand_collision.json')
        urdf = geometry.fk(fixture['q'])
        self.assertGreater(np.linalg.norm(native['wrist_pitch_r_link'][:3, 3]-urdf['wrist_pitch_r_link'][:3, 3]), .002)
        old_mask = np.zeros(135, bool)
        points = np.asarray(fixture['points_pelvis_m'])
        for name, model in self.model.models.items():
            local = (points-urdf[name][:3, 3])@urdf[name][:3, :3]
            hit, _ = hull_contains(local, model['vertices'], model['triangles'], model['planes'], .003)
            old_mask |= hit
        np.testing.assert_array_equal(old_mask, fixture['expected_urdf_fk_mask'])
        self.assertEqual(int(old_mask.sum()), 20)
        for bad in (None, np.nan, True, '0'):
            q = dict(fixture['q'])
            if bad is None: q.pop(self.model.required[0])
            else: q[self.model.required[0]] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError): self.model.native_fk(q)
        poses = self.model.native_fk(dict.fromkeys(self.model.required, 0.))
        self.assertIn('body_yaw_link', poses)
        self.assertNotIn('waist_yaw_link', poses)

    def test_pinned_asset_hand_and_semantic_metadata_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bad.json'; path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                NativeBodyVisualHulls.load(path, ROBOT/'native_hand_collision.json')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                NativeBodyVisualHulls.load(ROBOT/'native_body_visual_hulls.json', path)
            with self.assertRaises(FileNotFoundError):
                NativeBodyVisualHulls.load(path.with_name('missing.json'), ROBOT/'native_hand_collision.json')
        for field in ('source_robot_sha256', 'axis', 'body1', 'planes', 'triangles', 'missing_link'):
            data = copy.deepcopy(self.data)
            if field == 'source_robot_sha256': data[field] = '0'*64
            elif field == 'axis': data['body_joints'][0]['axis'] = 'X'
            elif field == 'body1': data['body_joints'][0]['body1'] = ['/World/humanoid/waist_yaw_link']
            elif field == 'planes': data['links']['elbow_pitch_l_link']['planes'][0][0] += .1
            elif field == 'triangles': data['links']['elbow_pitch_l_link']['triangles'][0][0] = -1
            else: data['links'].pop('elbow_pitch_l_link')
            with self.subTest(field=field), self.assertRaises(ValueError): NativeBodyVisualHulls(data)

    def test_roi_matches_full_cloud_body_predicate_and_preserves_invalid_depth(self):
        q = dict(self.fixture['q'])
        k = np.array([[80., 0., 25.], [0., 80., 18.], [0., 0., 1.]])
        t = np.array([[0., 0., 1., 0.], [-1., 0., 0., -.24], [0., -1., 0., .22], [0., 0., 0., 1.]])
        rng = np.random.default_rng(20261109)
        depth = rng.uniform(.15, .5, (37, 51));depth[0, :4] = [0., np.nan, np.inf, -1.]
        before = depth.copy()
        geometry = NS(required=self.model.required, body_visual_hulls=self.model,
            urdf_path=ROBOT/'tiangong2pro_with_hands.urdf', cfg_path=ROBOT/'tienkung_curobo.yml', hand_path=ROBOT/'native_hand_collision.json',
            spheres=lambda q: [('far', np.array([0., 0., -10.]), .001)])
        result = mask_robot_depth(depth, k, t, q, geometry)
        y, x = np.indices(depth.shape);valid = np.isfinite(depth)&(depth > 0)
        rays = np.stack((x, y, np.ones_like(x)), -1)@np.linalg.inv(k).T
        points = rays[valid]*depth[valid, None]
        points = points@t[:3, :3].T+t[:3, 3]
        covered, _ = self.model.cover_points(points, q)
        expected = np.zeros(depth.shape, bool);expected[valid] = covered
        np.testing.assert_array_equal(result['mask'], expected)
        self.assertGreater(int(expected.sum()), 0)
        np.testing.assert_array_equal(depth, before)
        self.assertFalse(result['mask'][0, :4].any())
        provenance = result['provenance']['body_visual_hulls']
        self.assertEqual(provenance['asset_sha256'], ASSET_SHA256)
        self.assertFalse(provenance['control_collision_model_changed'])
        self.assertEqual(provenance['added_masked_pixels'], int(expected.sum()))
        self.assertEqual(result['provenance']['robot_files_sha256']['native_body_visual_hulls'], ASSET_SHA256)

    def test_perception_loader_requires_installed_visual_asset_without_changing_control_geometry(self):
        from hack_perception.rgbd_robot_filter import load_robot_geometry
        modules = {'ament_index_python': NS(), 'ament_index_python.packages': NS(get_package_share_directory=lambda _: str(ROBOT.parent))}
        with patch.dict(sys.modules, modules): geometry = load_robot_geometry()
        self.assertIsInstance(geometry.body_visual_hulls, NativeBodyVisualHulls)
        self.assertEqual(len(geometry.spheres(self.fixture['q'])), 706)
        with patch.dict(sys.modules, modules), patch.object(NativeBodyVisualHulls, 'load', side_effect=ValueError('missing visual model')):
            with self.assertRaisesRegex(ValueError, 'missing visual model'): load_robot_geometry()


if __name__ == '__main__': unittest.main()
