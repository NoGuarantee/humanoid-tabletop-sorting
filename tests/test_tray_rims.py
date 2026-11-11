"""Color/depth fallback must measure a supported rim, including clipped trays."""
import unittest
from types import SimpleNamespace as NS
import numpy as np
from test_scene_geometry import scene
from hack_perception import table_clusters as tc
from hack_perception.tray_rims import color_openings
from hack_perception import classify
from hack_perception.rgbd.scene_geometry import Tray


def cloud_of(sensor):
    z = sensor['depth'].astype(float)*.001
    v,u = np.indices(z.shape);K=sensor['K'];T=sensor['T']
    camera = np.stack(((u-K[0,2])*z/K[0,0], (v-K[1,2])*z/K[1,1], z),axis=-1)
    cloud = camera@T[:3,:3].T+T[:3,3]
    cloud[z==0]=np.nan
    return cloud


class ColorRimTest(unittest.TestCase):
    def test_measured_rim_wins_over_tray_merged_with_placed_brother(self):
        # Actual seed-7 cluster after placing Brother; its top includes the
        # contents. Passive RGB-D rim fits before/after placement both stayed
        # at z=-0.0704, yaw=0 and the measured 0.22 x 0.32 opening.
        cluster = NS(rejected='tray', size=(.314804, .254881, .096578),
                     center=(.549564, -.032376, -.0925), top_z=-.044212,
                     yaw=1.177363, n_points=100, pixels=np.arange(100))
        rgb = np.tile(np.array(classify.TRAY_CHROMA['green'])*255, (10, 10, 1)).astype(np.uint8)
        mapping = {'green':'tray_center'}
        baseline = classify.merge_trays([], [cluster], rgb, mapping)
        self.assertAlmostEqual(baseline[0][2][2], -.044212)
        measured = Tray('tray_center', 'green', np.array([.535934, .001052, -.070497]), 0.)
        corrected = classify.merge_trays([], [cluster], rgb, mapping, measured=[measured])
        np.testing.assert_allclose(corrected[0][2], measured.center)
        self.assertEqual(corrected[0][4], (.22, .32, .07))
        # Without a supported rim, retain the existing conservative fallback.
        self.assertEqual(classify.merge_trays([], [cluster], rgb, mapping, measured=[]), baseline)

    def test_model_supported_opening_keeps_priority_over_color_fit(self):
        model = Tray('tray_center', 'green', np.array([.535, 0., -.07]), 0.)
        color = Tray('tray_center', 'green', np.array([.536, .001, -.0705]), .01)
        merged = classify.merge_trays([model], [], None, {'green':'tray_center'}, measured=[color])
        np.testing.assert_allclose(merged[0][2], model.center)

    def test_four_or_three_walls_measure_pose_and_use_only_task_color_id(self):
        for walls in (3,4):
            sensor,_,_,center,_ = scene(visible_walls=walls)
            cloud=cloud_of(sensor)
            plane=tc.fit_table_plane(cloud.reshape(-1,3)[np.isfinite(cloud.reshape(-1,3)[:,2])])
            trays=color_openings(sensor,cloud,plane,{'blue':'destination'})
            self.assertEqual([t.id for t in trays],['destination'])
            np.testing.assert_allclose(trays[0].center,center,atol=.008)

    def test_two_walls_or_missing_depth_never_invent_an_opening(self):
        sensor,_,_,_,_ = scene(visible_walls=2)
        cloud=cloud_of(sensor)
        plane=tc.fit_table_plane(cloud.reshape(-1,3)[np.isfinite(cloud.reshape(-1,3)[:,2])])
        self.assertEqual(color_openings(sensor,cloud,plane,{'blue':'destination'}),[])
        self.assertEqual(color_openings(sensor,cloud,None,{'blue':'destination'}),[])
