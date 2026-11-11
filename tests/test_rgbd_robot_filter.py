"""Strict joint/exposure binding, raw preservation and zero/three-point proposals."""
import ast
import copy
from pathlib import Path
from queue import Queue
import sys
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from hack_perception.rgbd_robot_filter import JointStateBuffer,filter_sensor_packets,filter_provenance
from hack_perception.rgbd.geometry import depth_mask_to_pelvis
from hack_perception.rgbd.rgbd_free_space import depth_view
from hack_perception.rgbd.tube_nn_features import make_features
from hack_perception.rgbd.detection import Detection
from hack_perception.rgbd_response import infer_response
from hack_perception.rgbd_worker import _serve

try:
    import rclpy as real_rclpy
except ImportError:
    real_rclpy=None


STAMP=1_000_000_000
BROTHER='Brother_Ink_Cartridge_Magenta_LC75M'


def sensor():
    states=JointStateBuffer();states.add(STAMP,['joint'],[0.])
    witness=states.at(STAMP);witness['generation']=2
    return dict(rgb=np.full((12,12,3),127,np.uint8),depth=np.full((12,12),1000,np.uint16),
                K=np.array([[100.,0,6.],[0,100.,6.],[0,0,1.]]),T=np.eye(4),
                camera='cam_head',stamp_ns=STAMP,robot_joint_state=witness)


class JointExposureTests(unittest.TestCase):
    def test_exact_and_interpolated_full_measured_values_are_bound_to_exposure(self):
        states=JointStateBuffer()
        states.add(STAMP-50_000_000,['a','b'],[0.,1.])
        states.add(STAMP+50_000_000,['b','a'],[3.,2.])
        witness=states.at(STAMP)
        self.assertEqual(witness['q'],{'a':1.,'b':2.})
        self.assertEqual(witness['max_alignment_ns'],50_000_000)
        self.assertEqual(witness['method'],'linear_interpolation')
        self.assertEqual(states.at(STAMP+50_000_000)['method'],'exact')

    def test_missing_future_bracket_and_old_10hz_packet_never_use_nearest_or_default_q(self):
        states=JointStateBuffer();states.add(STAMP-10_000_000,['a'],[0.])
        with self.assertRaisesRegex(ValueError,'alignment_unavailable'):states.at(STAMP)
        states.clear()
        states.add(STAMP-60_000_000,['a'],[0.]);states.add(STAMP+50_000_000,['a'],[0.])
        with self.assertRaisesRegex(ValueError,'alignment_unavailable'):states.at(STAMP)

    def test_invalid_conflicting_missing_or_nonfinite_samples_fail_closed(self):
        for names,values in [(['a'],[float('nan')]),(['a'],[True]),(['a','a'],[0.,1.]),(['a'],[])]:
            states=JointStateBuffer();states.add(STAMP,names,values)
            with self.assertRaisesRegex(ValueError,'invalid'):states.at(STAMP)
        states=JointStateBuffer();states.add(STAMP,['a'],[0.]);states.add(STAMP,['a'],[1.])
        with self.assertRaisesRegex(ValueError,'invalid'):states.at(STAMP)
        states.clear();states.add(STAMP-1,['a'],[0.]);states.add(STAMP+1,['b'],[0.])
        with self.assertRaisesRegex(ValueError,'invalid_bracket'):states.at(STAMP)

    def test_reset_or_bounded_eviction_cannot_reuse_old_epoch(self):
        states=JointStateBuffer(capacity=2)
        for stamp in (STAMP,STAMP+1,STAMP+2):states.add(stamp,['a'],[0.])
        self.assertEqual(len(states.samples),2)
        states.add(STAMP-100_000_000,['a'],[0.])
        self.assertEqual(list(states.samples),[STAMP-100_000_000])
        states.clear()
        with self.assertRaisesRegex(ValueError,'unavailable'):states.at(STAMP)


class ServiceBindingTests(unittest.TestCase):
    @unittest.skipIf(real_rclpy is None,'real ROS unavailable on host')
    def test_real_ros_optin_registers_joint_reader_and_accepts_typed_measured_sample(self):
        from rclpy.context import Context
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from hack_perception.rgbd_service import ModelPoseService
        from hack_perception.rgbd_frames import PairBuffer
        context=Context();real_rclpy.init(context=context)
        node=Node('robot_self_filter_api_test',context=context);node.camera_buffer=PairBuffer()
        service=None
        try:
            with patch('hack_perception.rgbd_service.PoseWorker',return_value=NS(close=lambda:None)):
                service=ModelPoseService(node,NS(),'/unused',4,False,2,
                    lock=threading.RLock(),clear_baseline=lambda:None,robot_self_filter=True,on_demand=True)
            self.assertIn('/joint_states',[subscription.topic_name for subscription in node.subscriptions])
            sample=JointState();sample.header.stamp.sec=1;sample.name=['joint'];sample.position=[.25]
            service.store_joint_state(sample)
            self.assertEqual(service.robot_joints.at(STAMP)['q'],{'joint':.25})
        finally:
            if service is not None:service.close()
            node.destroy_node();context.try_shutdown()

    def test_service_binds_one_strict_witness_to_pair_without_touching_raw_views(self):
        from test_rgbd_service_timing import service_module
        cls=service_module().ModelPoseService
        service=cls.__new__(cls);service.lock=threading.RLock();service.generation=4
        service.robot_joints=JointStateBuffer()
        service.robot_joints.add(STAMP-10_000_000,['joint'],[0.])
        service.robot_joints.add(STAMP+10_000_000,['joint'],[.02])
        views=[dict(depth=NS(header=NS(frame_id=frame))) for frame in ('head_optical','torso_optical')]
        packets=[dict(rgb=np.zeros((1,1,3),np.uint8)),dict(rgb=np.zeros((1,1,3),np.uint8))]
        result=service.bind_robot_state(packets,views,(4,STAMP))
        self.assertEqual(result[0]['robot_joint_state']['q'],{'joint':.01})
        self.assertEqual(result[0]['robot_joint_state']['q_sha256'],result[1]['robot_joint_state']['q_sha256'])
        self.assertEqual([p['optical_frame_id'] for p in result],['head_optical','torso_optical'])
        self.assertEqual(result[0]['robot_joint_state']['generation'],4)
        self.assertEqual(set(views[0]),{'depth'})
        with self.assertRaisesRegex(ValueError,'epoch_changed'):service.bind_robot_state(packets,views,(3,STAMP))

    def test_pending_future_encoder_does_not_submit_a_nearest_pose(self):
        from test_rgbd_service_timing import service_module
        cls=service_module().ModelPoseService
        service=cls.__new__(cls);service.lock=threading.RLock();service.generation=4
        service.robot_joints=JointStateBuffer();service.robot_joints.add(STAMP-10_000_000,['joint'],[0.])
        with self.assertRaisesRegex(ValueError,'alignment_unavailable'):
            service.bind_robot_state([{},None],[dict(depth=NS(header=NS(frame_id='head'))),None],(4,STAMP))


class FilterTests(unittest.TestCase):
    def test_copies_preserve_raw_and_masked_depth_is_unknown_in_existing_free_space_logic(self):
        raw=sensor();before=copy.deepcopy(raw)
        def mask(depth,K,T,q,geometry,margin_m):
            self.assertEqual(depth.dtype,np.float64);self.assertEqual(float(depth[0,0]),1.)
            self.assertEqual(margin_m,.003)
            pixels=np.zeros(depth.shape,bool);pixels[0,0]=True
            return dict(mask=pixels,provenance=dict(margin_m=margin_m))
        filtered=filter_sensor_packets([raw,None],NS(required=['joint']),mask)[0]
        np.testing.assert_array_equal(raw['rgb'],before['rgb']);np.testing.assert_array_equal(raw['depth'],before['depth'])
        self.assertFalse(np.shares_memory(filtered['depth'],raw['depth']))
        self.assertFalse(np.shares_memory(filtered['rgb'],raw['rgb']))
        self.assertEqual(filtered['depth'][0,0],0)
        np.testing.assert_array_equal(filtered['rgb'],raw['rgb'])
        self.assertFalse(depth_view(filtered)[2][0,0])
        self.assertTrue(depth_view(filtered)[2][0,1])
        self.assertEqual(filtered['robot_self_filter']['masked_pixels'],1)
        self.assertEqual(filtered['robot_self_filter']['rgb_policy'],'byte_identical_copy')
        self.assertEqual(filtered['robot_self_filter']['raw_rgb_sha256'],filtered['robot_self_filter']['filtered_rgb_sha256'])
        self.assertEqual(filter_provenance([filtered,None])['views'][0]['joint_state'],raw['robot_joint_state'])

    def test_bad_alignment_epoch_q_completeness_or_hash_cannot_be_masked(self):
        for change in ('stamp','missing','hash','stale'):
            raw=sensor()
            if change=='stamp':raw['stamp_ns']+=1
            elif change=='missing':raw['robot_joint_state']['q']={}
            elif change=='hash':raw['robot_joint_state']['q_sha256']='0'*64
            else:raw['robot_joint_state']['source_stamps_ns']=[STAMP-50_000_001]
            with self.subTest(change=change),self.assertRaises(ValueError):
                filter_sensor_packets([raw],NS(required=['joint']),lambda *args,**kw:self.fail('bad witness reached model'))

    def test_worker_filters_once_for_prepared_exposure_and_pose_reuses_filtered_copies(self):
        requests,results=Queue(),Queue();raw=sensor();key=(2,STAMP)
        requests.put(('scene',key,([raw,None],None)));requests.put(('pose',key,4));requests.put(None)
        def mask(depth,*args,**kwargs):return dict(mask=np.eye(12,dtype=bool),provenance={})
        def filtering(sensors,geometry):return filter_sensor_packets(sensors,geometry,mask)
        seen=[]
        def prepare(backend,sensors,mapping):
            seen.append(sensors);return 'segmentation',None,[]
        def pose(backend,sensors,*args,**kwargs):
            self.assertIs(sensors,seen[0]);self.assertEqual(sensors[0]['depth'][0,0],0)
            return dict(objects=[],containers=[],observed_regions=[])
        with patch.dict(sys.modules,{'torch':NS(set_num_threads=lambda n:None),
                'hack_perception.rgbd.pose_backend':NS(PoseBackend=lambda *args,**kw:object())}),\
                patch('hack_perception.rgbd_robot_filter.load_robot_geometry',return_value=NS(required=['joint'])),\
                patch('hack_perception.rgbd_robot_filter.filter_sensor_packets',side_effect=filtering) as filtered,\
                patch('hack_perception.rgbd_response.prepare_scene',side_effect=prepare),\
                patch('hack_perception.rgbd_response.infer_response',side_effect=pose):
            _serve(requests,results,'/bundle',True,2,robot_self_filter=True)
        self.assertEqual(results.get_nowait()[0],'ready');self.assertEqual(results.get_nowait()[0],'scene')
        kind,_,value,_=results.get_nowait();self.assertEqual(kind,'pose')
        self.assertEqual(filtered.call_count,1);self.assertTrue(value['robot_self_filter']['enabled'])
        self.assertEqual(raw['depth'][0,0],1000)

    def test_worker_does_not_infer_when_joint_epoch_does_not_match_selected_exposure(self):
        requests,results=Queue(),Queue();raw=sensor();raw['robot_joint_state']['generation']=99
        requests.put(('scene',(2,STAMP),([raw,None],None)));requests.put(None)
        with patch.dict(sys.modules,{'torch':NS(set_num_threads=lambda n:None),
                'hack_perception.rgbd.pose_backend':NS(PoseBackend=lambda *args,**kw:object())}),\
                patch('hack_perception.rgbd_robot_filter.load_robot_geometry',return_value=NS(required=['joint'])),\
                patch('hack_perception.rgbd_response.prepare_scene') as prepare:
            _serve(requests,results,'/bundle',True,2,robot_self_filter=True)
        self.assertEqual(results.get_nowait()[0],'ready');self.assertEqual(results.get_nowait()[0],'job_failed')
        prepare.assert_not_called()


class SuppressedProposalTests(unittest.TestCase):
    def test_existing_estimate_does_not_name_zero_or_three_point_robot_proposal(self):
        # Execute the production estimate body and actual depth feature support
        # gate without Torch/weights. Identity/pose networks run only for the
        # untouched true component; no detector threshold or selection changes.
        source=Path(__file__).resolve().parents[1]/'src/hack_perception/hack_perception/rgbd/pose_backend.py'
        tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PoseBackend')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='estimate')
        namespace=dict(np=np,POOL_IDENTITIES={4:{3:[0]}},OBJECTS=[BROTHER],
            descriptors=lambda *args:[],association=lambda *args:[],
            features=lambda obs,model:make_features(obs,5),pair_features=lambda *args:None,
            fit=NS(measure=lambda depth,mask,K,T,calibrated:depth_mask_to_pelvis(depth,mask,K,T)),depth_view=depth_view)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source),'exec'),namespace)
        for remaining in (0,3,30):
            raw=sensor();raster=np.zeros((12,12),int);raster[:6,:6]=8;raster[6:,6:]=14
            mask=raster==14
            if remaining:mask.flat[np.flatnonzero(mask)[-remaining:]]=False
            filtered=filter_sensor_packets([raw,None],NS(required=['joint']),
                lambda *args,**kw:dict(mask=mask,provenance={}))[0]
            backend=NS(channels=1,calibrated=False,clipped=lambda mask:False,pool_categories=lambda pool:{3})
            backend.estimate=lambda *args,**kw:namespace['estimate'](backend,*args,**kw)
            backend.guesses=lambda tasks:[[ (BROTHER,np.eye(4),1.) ] for _ in tasks]
            backend.fit_pose=lambda task,*args:Detection(BROTHER,3,np.eye(4),np.zeros(3),np.ones(3),
                max(ref['confidence'] for ref in task[1]),.001,1,'c2',True,False,len(task[9]),task[0])
            proposals=[dict(class_id=3,instance_id=8,confidence=.977),dict(class_id=3,instance_id=14,confidence=.277)]
            payload=infer_response(backend,[filtered,None],STAMP,4,prepared=([(raster,proposals,{}),None],None,[]))
            named=[o['instances'] for o in payload['objects'] if o['name']==BROTHER]
            other=next(o for o in payload['objects'] if o['instances']=={'cam_head':14})
            if remaining<30:
                self.assertEqual(named,[{'cam_head':8}]);self.assertIsNone(other['name'])
            else:
                # Negative control: supported ambiguity is preserved. The
                # integration must not silently select the higher confidence.
                self.assertEqual(named,[{'cam_head':8},{'cam_head':14}])
            self.assertEqual(other['detector_score'],.277)
            fragments=[r for r in payload['observed_regions'] if r['instances']=={'cam_head':14}
                       or r.get('source_instances')=={'cam_head':14}]
            self.assertEqual(sum(r['depth_points'] for r in fragments),remaining)
            if remaining<30:
                self.assertEqual(len(fragments),remaining)
                self.assertTrue(all(r['category']=='unknown' and r['instances']['cam_head']<0
                                    and r['source_instances']=={'cam_head':14} and r['depth_points']==1
                                    and r['low_pelvis_m']==r['high_pelvis_m'] for r in fragments))
            self.assertEqual(sum(r['depth_points'] for r in payload['observed_regions'] if r['instances']=={'cam_head':8}),36)


if __name__=='__main__':unittest.main()
