"""Color/depth fallback must measure a supported rim, including clipped trays."""
import unittest
import numpy as np
from test_scene_geometry import scene
from hack_perception import table_clusters as tc
from hack_perception.tray_rims import color_openings


def cloud_of(sensor):
    z = sensor['depth'].astype(float)*.001
    v,u = np.indices(z.shape);K=sensor['K'];T=sensor['T']
    camera = np.stack(((u-K[0,2])*z/K[0,0], (v-K[1,2])*z/K[1,1], z),axis=-1)
    cloud = camera@T[:3,:3].T+T[:3,3]
    cloud[z==0]=np.nan
    return cloud


class ColorRimTest(unittest.TestCase):
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
