import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
import math

class GeometryTests(unittest.TestCase):
    def test_mount_goal_keeps_fingertips_above_local_table(self):
        path=Path(__file__).resolve().parents[1]/'src/hack_brain/hack_brain/grasp_planner.py'
        spec=importlib.util.spec_from_file_location('grasp_coordinate_test',path)
        module=importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules',{'geometry_msgs.msg':NS(Point=NS,Pose=NS,Quaternion=NS),
                                      'yaml':NS(),spec.name:module}):
            spec.loader.exec_module(module)
        obj=NS(name='a',category='figure',center=NS(x=.4,y=.2),top_z=.06,size=NS(z=.03),yaw=.7)
        candidates=module.plan_grasps(obj,NS(table_z=0),{})
        g=candidates[0]
        self.assertGreaterEqual(g.grasp.position.z-module.FINGERTIP,.045-1e-9)
        self.assertAlmostEqual(g.grasp.orientation.z,math.sin(.7/2))
        self.assertAlmostEqual(g.lift.position.z-g.grasp.position.z,.15)
        q=module.quat_from_rpy(math.pi/2,0,0)
        tcp=module.tcp_at(0,0,0,q)
        self.assertAlmostEqual(tcp.position.y,-.21)
        self.assertAlmostEqual(tcp.position.z,0)
