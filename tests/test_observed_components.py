"""Split occluded masks while retaining unidentified obstacle geometry.

The first two point-cloud regressions adapt the grasp-worktree b88a154 tests.
The remaining tests guard the multi-object scene's conservative ID contract.
"""
import unittest
from unittest.mock import patch

import numpy as np

def same_instance(a, b):
    return any(b["instances"].get(camera) == ident for camera, ident in a["instances"].items())
from hack_perception.rgbd_response import bind_exposure, infer_response, observed_components, observed_regions
from types import SimpleNamespace


PLANE = (np.array([0., 0., 1.]), 0.)


def cloud(height=.05):
    x, y, z = np.meshgrid(np.arange(0., .05, .004), np.arange(0., .05, .004),
                          np.arange(.01, height, .004))
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def regions_for(points, plane=PLANE):
    sensor = dict(depth=None, K=None, T=None)
    segment = (np.array([1]), [dict(instance_id=1, class_id=1, confidence=.9)], None)
    with patch('hack_perception.rgbd_response.depth_mask_to_pelvis', return_value=points):
        return observed_regions([sensor, None], [segment, None], plane, {1})


class ObservedComponentTests(unittest.TestCase):
    def test_large_raised_arm_does_not_extend_table_object_bounds(self):
        object_points = cloud()
        arm = object_points + [0., 0., .20]
        points = np.vstack([object_points, arm, arm + [.005, 0., 0.]])
        groups = observed_components(points, PLANE)
        self.assertLess(groups[0][:, 2].max(), .06)
        self.assertEqual(len(groups[0]), len(object_points))
        # An apparent arm is not proven robot geometry and must remain.
        self.assertEqual(sum(map(len, groups)), len(points))
        self.assertEqual(len(groups), 2)

    def test_continuous_tall_object_keeps_its_full_height(self):
        points = cloud(.18)
        groups = observed_components(points, PLANE)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), len(points))
        self.assertGreater(groups[0][:, 2].max(), .17)

    def test_two_real_objects_under_one_mask_keep_second_obstacle_on_target_exclusion(self):
        first = cloud()
        second = first + [.20, 0., 0.]
        regions = regions_for(np.vstack([first, second]))
        self.assertEqual(len(regions), 2)
        target = dict(instances={'cam_head': 1})
        self.assertTrue(same_instance(target, regions[0]))
        remaining = [region for region in regions if not same_instance(target, region)]
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]['source_instances'], target['instances'])
        self.assertNotEqual(remaining[0]['instances'], target['instances'])
        self.assertGreater(remaining[0]['low_pelvis_m'][0], .19)
        self.assertLess(regions[0]['high_pelvis_m'][0], .06)

    def test_high_unknown_surfaces_are_retained_without_robot_identification(self):
        points = cloud()
        self.assertEqual(len(regions_for(points + [0., 0., .25])), 1)
        regions = regions_for(np.vstack([points, points + [0., 0., .25]]))
        self.assertEqual(len(regions), 2)
        self.assertGreater(regions[1]['low_pelvis_m'][2], .25)

    def test_component_ids_are_unique_and_do_not_alias_other_source_masks(self):
        points = np.vstack([cloud(), cloud() + [.20, 0., 0.]])
        sensor = dict(depth=None, K=None, T=None)
        # Reserve even a negative source ID rather than assuming the segmenter
        # can never supply one; separate masks need distinct component IDs.
        proposals = [dict(instance_id=ident, class_id=1, confidence=.9) for ident in (1, -1)]
        segment = (np.array([1, -1]), proposals, None)
        with patch('hack_perception.rgbd_response.depth_mask_to_pelvis', return_value=points):
            regions = observed_regions([sensor, None], [segment, None], PLANE, {1})
        identifiers = [region['instances']['cam_head'] for region in regions]
        self.assertEqual(identifiers, [1, -2, -1, -3])

    def test_tiny_fragments_and_fully_sparse_masks_are_retained(self):
        points = cloud()
        residual = np.array([[.20, .20, .03], [.30, .20, .03]])
        groups = observed_components(np.vstack([points, residual]), PLANE)
        self.assertEqual(sum(map(len, groups)), len(points) + len(residual))
        self.assertEqual(len(regions_for(np.vstack([points, residual]))), 3)
        sparse = np.column_stack([np.arange(40) * .02, np.zeros(40), np.full(40, .03)])
        self.assertEqual(len(observed_components(sparse, PLANE)), 40)
        regions = regions_for(sparse)
        self.assertEqual(len(regions), 40)
        self.assertEqual(sum(region['depth_points'] for region in regions), 40)
        self.assertTrue(all(region['category'] == 'unknown' for region in regions))
        self.assertTrue(all(region['instances']['cam_head'] < 0 for region in regions))
        self.assertTrue(all(region['source_instances'] == {'cam_head': 1} for region in regions))

    def test_no_plane_does_not_remove_disconnected_geometry(self):
        points = np.vstack([cloud(), cloud() + [0., 0., .25]])
        groups = observed_components(points, None)
        self.assertEqual(len(groups), 2)
        self.assertEqual(sum(map(len, groups)), len(points))

    def test_pose_filter_never_filters_neighbours_or_turns_off_regions(self):
        points = cloud()
        sensor = dict(depth=None, K=None, T=None)
        proposals = [dict(instance_id=ident, class_id=category, confidence=.4)
                     for ident, category in ((1, 1), (2, 4))]
        segment = (np.array([1, 2]), proposals, None)
        backend = SimpleNamespace(estimate=lambda *a, **kw: [], pool_categories=lambda pool: {1})
        prepared = ([segment, None], PLANE, [])
        with patch('hack_perception.rgbd_response.depth_mask_to_pelvis', return_value=points):
            for names in (None, ['Schleich_African_Black_Rhino']):
                payload = infer_response(backend, [sensor, None], 100, 4, prepared=prepared, names=names)
                self.assertEqual(len(payload['observed_regions']), 2)
                self.assertEqual(payload['observed_regions'][1]['category'], 'electronics')

    def test_exposure_scopes_every_id_and_does_not_alias_reset_or_worker_restart(self):
        payload = dict(objects=[{'instances': {'cam_head': 1}}], containers=[{}],
                       observed_regions=[{'instances': {'cam_head': 1}}])
        bind_exposure(payload, (2, 100), 'session-a', ['cam_head'])
        key = payload['exposure_key']
        for item in payload['objects'] + payload['containers'] + payload['observed_regions']:
            self.assertEqual(item['exposure_key'], key)
        self.assertNotEqual(key, dict(key, epoch=3))
        self.assertNotEqual(key, dict(key, session_id='session-b'))

    def test_visible_bounds_preserve_extreme_points_and_sub_30_point_masks(self):
        points = np.vstack([cloud(), [[.6, .2, .03]]])
        regions = regions_for(points)
        self.assertEqual(max(region['high_pelvis_m'][0] for region in regions), .6)
        regions = regions_for(np.array([[.2, .2, .03]]))
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]['depth_points'], 1)

    def test_sparse_connected_surface_keeps_points_without_inheriting_identity(self):
        points = np.array([[.20, .20, .03], [.21, .20, .03], [.22, .20, .03]])
        groups = observed_components(points, PLANE)
        self.assertEqual(len(groups), 1)
        np.testing.assert_array_equal(groups[0], points)
        regions = regions_for(points)
        self.assertEqual([r['depth_points'] for r in regions], [1, 1, 1])
        self.assertEqual([r['instances']['cam_head'] for r in regions], [-1, -2, -3])
        self.assertTrue(all(r['category'] == 'unknown' and r['source_instances'] == {'cam_head': 1}
                            and r['low_pelvis_m'] == r['high_pelvis_m'] for r in regions))
        np.testing.assert_array_equal([r['low_pelvis_m'] for r in regions], points)

    def test_sparse_duplicate_points_are_all_retained_beside_unchanged_dense_components(self):
        dense = cloud()
        other_dense = dense + [.4, 0., .1]
        sparse = np.array([[.20, .20, .03], [.20, .20, .03], [.21, .20, .03]])
        original = regions_for(np.vstack([dense, other_dense]))
        regions = regions_for(np.vstack([dense, other_dense, sparse]))
        self.assertEqual(regions[:2], original)
        remaining = regions[2:]
        self.assertEqual(len(remaining), 3)
        np.testing.assert_array_equal([r['low_pelvis_m'] for r in remaining], sparse)
        self.assertTrue(all(r['depth_points'] == 1 and r['category'] == 'unknown'
                            and r['low_pelvis_m'] == r['high_pelvis_m'] for r in remaining))
        identifiers = [r['instances']['cam_head'] for r in regions]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(sum(r['depth_points'] for r in regions), len(dense)+len(other_dense)+len(sparse))

    def test_four_point_wrist_table_bridge_is_split_without_dropping_geometry(self):
        # Three exact extrema measured in E11trial02scene001region8, plus a
        # fourth nearby sample. Sparse geometry must not form a 457mm bridge.
        wrist = np.array([[.26976719121212805, .27801989209999195, .22797597388922447],
                          [.2688459049214351, .2791707488106065, .22728560144271698]])
        table = np.array([[.4075392143924511, .5305908858793256, -.12737537692813483]])
        points = np.vstack([wrist, table, wrist[0]+[.0001, .0001, 0.]])
        groups = observed_components(points, None)
        self.assertEqual(sorted(map(len, groups)), [1, 3])
        self.assertEqual(sorted(tuple(p) for group in groups for p in group), sorted(map(tuple, points)))
        self.assertLess(max(np.linalg.norm(group.max(0)-group.min(0)) for group in groups), .003)


if __name__ == '__main__':
    unittest.main()
