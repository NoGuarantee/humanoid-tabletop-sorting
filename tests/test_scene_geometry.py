"""Exercise table and tray output with depth rendered in a known optical frame."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from hack_perception.rgbd.geometry import table_z_at
from hack_perception.rgbd.scene_geometry import TRAY_CLASS, TRAY_SIZE_M, estimate_scene_geometry


def scene(*, visible_walls=4, color=(218, 238, 249)):
    height, width = 160, 200
    k = np.array([[300., 0., width / 2], [0., 300., height / 2], [0., 0., 1.]])
    camera = np.eye(4)
    camera[:3, :3] = np.diag([1., -1., -1.])
    camera[:3, 3] = [.5, .06, .8]
    v, u = np.indices((height, width))
    rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(k).T @ camera[:3, :3].T
    table_z = -.19
    yaw = np.deg2rad(23)
    center = np.array([.56, -.01, table_z + TRAY_SIZE_M[2]])
    xy = (camera[:3, 3] + rays * (camera[2, 3] - center[2]) - center)
    local = (xy @ Rotation.from_euler("z", yaw).as_matrix())[..., :2]
    half = np.array(TRAY_SIZE_M[:2]) / 2
    footprint = (np.abs(local) <= half).all(axis=2)
    rim = footprint & (np.abs(local) >= half - .008).any(axis=2)
    z = np.full((height, width), table_z)
    z[footprint] += .008
    z[rim] = center[2]
    depth = np.rint(1000 * (camera[2, 3] - z)).astype(np.uint16)
    rgb = np.full((height, width, 3), 145, np.uint8)
    rgb[footprint] = color
    if visible_walls < 4:
        footprint &= local[..., 0] < -.02
    if visible_walls < 3:
        footprint &= local[..., 1] < .06
    depth[~footprint & (z > table_z)] = 0
    raster = footprint.astype(np.uint16)
    proposals = [dict(class_id=TRAY_CLASS, instance_id=1, confidence=.9)]
    return dict(rgb=rgb, depth=depth, K=k, T=camera), (raster, proposals, {}), table_z, center, yaw


class SceneGeometryTest(unittest.TestCase):
    def test_measure_table_and_blue_tray_in_pelvis(self):
        sensor, segment, z, center, yaw = scene()
        plane, trays = estimate_scene_geometry([None, sensor], [None, segment])
        self.assertAlmostEqual(table_z_at(plane), z, places=3)
        self.assertEqual(len(trays), 1)
        tray = trays[0]
        self.assertEqual((tray.id, tray.color), ("tray_right", "blue"))
        np.testing.assert_allclose(tray.center, center, atol=.005)
        self.assertAlmostEqual(tray.yaw, yaw, delta=np.deg2rad(2))

    def test_two_cameras_emit_one_tray_and_use_task_id(self):
        sensor, segment, _, _, _ = scene(color=(75, 170, 82))
        _, trays = estimate_scene_geometry([sensor, sensor], [segment, segment], {"green": "destination"})
        self.assertEqual([tray.id for tray in trays], ["destination"])

    def test_three_walls_support_a_clipped_tray(self):
        sensor, segment, _, center, _ = scene(visible_walls=3)
        _, trays = estimate_scene_geometry([sensor, None], [segment, None])
        self.assertEqual(len(trays), 1)
        np.testing.assert_allclose(trays[0].center, center, atol=.008)

    def test_two_clipped_walls_leave_opening_unavailable(self):
        sensor, segment, z, _, _ = scene(visible_walls=2, color=(221, 83, 78))
        plane, trays = estimate_scene_geometry([sensor, None], [segment, None])
        self.assertAlmostEqual(table_z_at(plane), z, places=3)
        self.assertEqual(trays, [])

    def test_missing_depth_leaves_table_and_trays_unavailable(self):
        sensor, segment, _, _, _ = scene()
        sensor["depth"][:] = 0
        self.assertEqual(estimate_scene_geometry([sensor, None], [segment, None]), (None, []))

    def test_table_height_uses_the_requested_pelvis_xy(self):
        normal = np.array([-.07, .01, 1.])
        self.assertAlmostEqual(table_z_at((normal, .18), .38, 0.), -.1534)


if __name__ == "__main__":
    unittest.main()
