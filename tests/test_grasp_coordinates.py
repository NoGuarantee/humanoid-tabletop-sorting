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


def _planner():
    path=Path(__file__).resolve().parents[1]/'src/hack_brain/hack_brain/grasp_planner.py'
    spec=importlib.util.spec_from_file_location('grasp_place_test',path)
    module=importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules',{'geometry_msgs.msg':NS(Point=NS,Pose=NS,Quaternion=NS),
                                  'yaml':NS(),spec.name:module}):
        spec.loader.exec_module(module)
    return module


def _tray(x,y,yaw=0.0,size=(.32,.22)):
    return NS(pose=NS(position=NS(x=x,y=y,z=.08),orientation=NS(x=0,y=0,z=math.sin(yaw/2),w=math.cos(yaw/2))),
              size=NS(x=size[0],y=size[1],z=.07))


class PlaceTests(unittest.TestCase):
    def setUp(self):
        self.m=_planner()

    def test_side_tray_drop_point_moves_towards_the_robot(self):
        x,y=self.m.place_point(_tray(.34,.69),.06,.06)
        self.assertLess(math.hypot(x,y),math.hypot(.34,.69)-.04)
        self.assertLess(y,.69-.04)
        self.assertLess(x,.34)

    def test_drop_point_stays_inside_the_tray(self):
        for yaw in (0.0,.4,math.pi/2,-1.2):
            with self.subTest(yaw=yaw):
                t=_tray(.34,-.69,yaw)
                x,y=self.m.place_point(t,.5,.06)
                dx,dy=x-.34,y+.69
                lx,ly=math.cos(yaw)*dx+math.sin(yaw)*dy,-math.sin(yaw)*dx+math.cos(yaw)*dy
                self.assertLessEqual(abs(lx),.16-.06+1e-9)
                self.assertLessEqual(abs(ly),.11-.06+1e-9)

    def test_unknown_size_or_no_inset_keeps_the_centre(self):
        self.assertEqual(self.m.place_point(_tray(.34,.69,size=(0,0)),.06,.06),(.34,.69))
        self.assertEqual(self.m.place_point(_tray(.53,0),0.0,.06),(.53,0))

    def test_plan_place_uses_the_drop_point(self):
        plan=self.m.plan_place('left',_tray(.53,0),{'place':{'inset':.06,'rim_margin':.06}})
        for p in (plan.approach,plan.release,plan.retreat):
            self.assertAlmostEqual(p.position.x,.47)
            self.assertAlmostEqual(p.position.y,0)
        self.assertGreater(plan.approach.position.z,plan.release.position.z)
