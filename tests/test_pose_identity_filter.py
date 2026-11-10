"""A requested identity must not remove its geometric competitors."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import numpy as np


class IdentityFilterTests(unittest.TestCase):
    def estimate(self, hypotheses, winner, names):
        path=Path(__file__).resolve().parents[1]/'src/hack_perception/hack_perception/rgbd/pose_backend.py'
        tree=ast.parse(path.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PoseBackend')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='estimate')
        scope=dict(np=np, POOL_IDENTITIES={4:{3:[0,1]}}, OBJECTS=['Philips','Brother'],
                   descriptors=lambda *a:[], association=lambda *a:[],
                   features=lambda *a:(np.zeros(1),np.zeros(1),np.zeros(3)),
                   pair_features=lambda *a:np.zeros(1),depth_view=lambda s:None,
                   fit=NS(measure=lambda *a:np.ones((30,3))))
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
        backend=NS(channels=1,calibrated=None,clipped=lambda mask:False,
                   guesses=lambda tasks:[hypotheses],fit_pose=Mock(return_value=NS(name=winner)))
        sensor=dict(rgb=None,depth=None,K=None,T=None)
        segmentation=[(np.ones((2,2)),[dict(class_id=3,instance_id=1)],{}),None]
        result=scope['estimate'](backend,[sensor,None],4,segmentation,None,names=names)
        return result,backend.fit_pose

    def test_requested_philips_does_not_force_brother_into_philips(self):
        guesses=[('Philips',np.eye(4),.55),('Brother',np.eye(4),.45)]
        result,fit=self.estimate(guesses,'Brother',['Philips'])
        self.assertEqual(result,[])
        self.assertEqual(len(fit.call_args.args[1]),2)

    def test_second_ranked_philips_can_win_geometric_fit(self):
        guesses=[('Brother',np.eye(4),.55),('Philips',np.eye(4),.45)]
        result,fit=self.estimate(guesses,'Philips',['Philips'])
        self.assertEqual([d.name for d in result],['Philips'])
        self.assertEqual(len(fit.call_args.args[1]),2)

    def test_unrequested_identity_skips_expensive_pose_fit(self):
        result,fit=self.estimate([('Brother',np.eye(4),.99)],'Brother',['Philips'])
        self.assertEqual(result,[])
        fit.assert_not_called()
