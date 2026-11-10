"""Exposure-aligned measured robot state and opt-in RGB-D input copies."""
from collections import OrderedDict
import copy
import hashlib
import json
from numbers import Real
from pathlib import Path

import numpy as np

ALIGNMENT_NS=50_000_000


def q_checksum(q):
    return hashlib.sha256(json.dumps(q,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class JointStateBuffer:
    def __init__(self,capacity=256):
        self.capacity=capacity
        self.samples=OrderedDict()

    def clear(self):
        self.samples.clear()

    def add(self,stamp,names,positions):
        if not isinstance(stamp,int) or isinstance(stamp,bool) or stamp<=0:
            return False
        try:
            if (len(names)!=len(positions) or not names or len(set(names))!=len(names)
                    or any(not isinstance(n,str) or not n for n in names)):
                raise ValueError('invalid_joint_names')
            if any(isinstance(v,(bool,np.bool_)) or not np.isfinite(float(v)) for v in positions):
                raise ValueError('nonfinite_joint_positions')
            q={str(n):float(v) for n,v in zip(names,positions)}
        except (TypeError,ValueError,OverflowError):
            q=None
        if self.samples and stamp<max(self.samples)-ALIGNMENT_NS:
            self.clear()
        if stamp in self.samples and self.samples[stamp]!=q:
            q=None  # Conflicting publishers/exposure samples cannot establish a pose.
        self.samples[stamp]=q
        self.samples.move_to_end(stamp)
        while len(self.samples)>self.capacity:self.samples.popitem(last=False)
        return q is not None

    def at(self,stamp):
        items=sorted(self.samples.items())
        if not items:raise ValueError('robot_joint_state_unavailable')
        exact=self.samples.get(stamp)
        if stamp in self.samples:
            if exact is None:raise ValueError('robot_joint_state_invalid')
            return self._witness(stamp,'exact',exact,[stamp])
        before=[item for item in items if item[0]<stamp]
        after=[item for item in items if item[0]>stamp]
        if before and after:
            a,qa=before[-1];b,qb=after[0]
            if stamp-a<=ALIGNMENT_NS and b-stamp<=ALIGNMENT_NS:
                if qa is None or qb is None or qa.keys()!=qb.keys():
                    raise ValueError('robot_joint_state_invalid_bracket')
                alpha=(stamp-a)/(b-a)
                q={n:qa[n]+alpha*(qb[n]-qa[n]) for n in qa}
                return self._witness(stamp,'linear_interpolation',q,[a,b])
        raise ValueError('robot_joint_state_alignment_unavailable')

    @staticmethod
    def _witness(stamp,method,q,sources):
        q=dict(q)
        return dict(schema='hack_perception.robot_joint_state/v1',method=method,
            source='/joint_states',exposure_stamp_ns=stamp,source_stamps_ns=sources,
            alignment_offsets_ns=[s-stamp for s in sources],max_alignment_ns=max(abs(s-stamp) for s in sources),
            q=q,q_sha256=q_checksum(q))


def load_robot_geometry():
    from ament_index_python.packages import get_package_share_directory
    from hack_motion.perception_robot_geometry import RobotGeometry
    root=Path(get_package_share_directory('hack_motion'))/'robot'
    from .robot_visual_mask import NativeBodyVisualHulls
    geometry=RobotGeometry(root/'tiangong2pro_with_hands.urdf',root/'tienkung_curobo.yml',root/'native_hand_collision.json')
    # Only perception/capture attach this model. Control collision geometry
    # remains the original robot spheres and measured articulated fingers.
    geometry.body_visual_hulls=NativeBodyVisualHulls.load(root/'native_body_visual_hulls.json',root/'native_hand_collision.json')
    return geometry


def filter_sensor_packets(sensors,geometry,masker=None):
    """Filter only copied worker inputs; zero depth is invalid/unknown, not free."""
    if masker is None:
        from .robot_self_mask import mask_robot_depth
        masker=mask_robot_depth
    result=[]
    exposure_q={}
    for sensor in sensors:
        if sensor is None:
            result.append(None);continue
        witness=sensor.get('robot_joint_state',{})
        stamp=sensor.get('stamp_ns')
        sources=witness.get('source_stamps_ns',[])
        if (witness.get('schema')!='hack_perception.robot_joint_state/v1'
                or witness.get('method') not in ('exact','linear_interpolation')
                or not isinstance(stamp,int) or isinstance(stamp,bool) or stamp<=0
                or witness.get('exposure_stamp_ns')!=stamp or not sources
                or any(not isinstance(s,int) or isinstance(s,bool) or abs(s-stamp)>ALIGNMENT_NS for s in sources)
                or witness.get('max_alignment_ns')!=max(abs(s-stamp) for s in sources)):
            raise ValueError('robot_joint_state_witness_invalid')
        if ((witness['method']=='exact' and sources!=[stamp])
                or (witness['method']=='linear_interpolation'
                    and (len(sources)!=2 or not sources[0]<stamp<sources[1]))):
            raise ValueError('robot_joint_state_witness_invalid')
        q=witness.get('q',{})
        if (not isinstance(q,dict) or q_checksum(q)!=witness.get('q_sha256')
                or not set(geometry.required)<=q.keys()
                or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Real)
                       or not np.isfinite(float(v)) for v in q.values())):
            raise ValueError('robot_joint_state_incomplete_or_modified')
        if stamp in exposure_q and exposure_q[stamp]!=witness['q_sha256']:
            raise ValueError('robot_joint_state_pair_mismatch')
        exposure_q[stamp]=witness['q_sha256']
        rgb,depth=np.asarray(sensor['rgb']),np.asarray(sensor['depth'])
        if depth.dtype!=np.uint16 or depth.ndim!=2 or rgb.dtype!=np.uint8 or rgb.shape!=(*depth.shape,3):
            raise ValueError('robot_filter_requires_registered_rgb8_depth_mm')
        masked=masker(depth.astype(float)*.001,sensor['K'],sensor['T'],q,geometry,margin_m=.003)
        mask=np.asarray(masked['mask'])
        if mask.dtype!=np.bool_ or mask.shape!=depth.shape:raise ValueError('invalid_robot_pixel_mask')
        filtered=dict(sensor,rgb=rgb.copy(),depth=depth.copy(),visibility_depth_mm=depth.copy())
        # Preserve detector/features' original image context. Only known robot
        # depth becomes invalid; black RGB can alter other objects' identities.
        filtered['depth'][mask]=0
        checksum=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
        filtered['robot_self_filter']=dict(schema='hack_perception.robot_self_filter/v1',
            camera=sensor.get('camera'),optical_frame_id=sensor.get('optical_frame_id'),
            stamp_ns=stamp,enabled=True,pixel_semantics='masked_depth_is_unknown_not_free',
            rgb_policy='byte_identical_copy',depth_policy='robot_points_invalidated',
            hash_convention='sha256_contiguous_array_bytes',
            K=np.asarray(sensor['K'],dtype=float).tolist(),
            T_pelvis_camera=np.asarray(sensor['T'],dtype=float).tolist(),
            joint_state=copy.deepcopy(witness),model=copy.deepcopy(masked['provenance']),
            masked_pixels=int(np.count_nonzero(mask)),mask_sha256=checksum(mask),
            raw_rgb_sha256=checksum(rgb),raw_depth_sha256=checksum(depth),
            filtered_rgb_sha256=checksum(filtered['rgb']),filtered_depth_sha256=checksum(filtered['depth']),
            transform_method='tf2_lookup_at_exposure',transform_stamp_ns=stamp)
        result.append(filtered)
    return result


def filter_provenance(sensors):
    return dict(schema='hack_perception.robot_self_filter/v1',enabled=True,
                views=[copy.deepcopy(s['robot_self_filter']) for s in sensors if s is not None])
