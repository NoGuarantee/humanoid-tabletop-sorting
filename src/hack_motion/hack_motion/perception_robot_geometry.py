"""Robot-only FK and measured hand spheres, extracted from PR9 e499c89.

Used by perception only; the main motion planner remains unchanged.
"""
import math
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import yaml
from .robot import ARM_JOINTS, TCP_LINK
ORDER = ARM_JOINTS['left'] + ARM_JOINTS['right'] + ['body_yaw_joint']
SIDES = ('left', 'right')

def finite(value, shape=None, name='value'):
    a = np.asarray(value, dtype=float)
    if (shape is not None and a.shape != shape) or not np.isfinite(a).all():
        raise ValueError(f'invalid {name}')
    return a


def matrix(value):
    a = finite(value, (4, 4), 'rigid transform')
    if (not np.allclose(a[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=2e-5)
            or not np.isclose(np.linalg.det(a[:3, :3]), 1., atol=2e-5)):
        raise ValueError('transform is not SE(3)')
    return a.copy()


def axis_rotation(axis, angle):
    v = finite(axis, (3,), 'rotation axis')
    v = v / np.linalg.norm(v)
    x, y, z = v
    k = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle) * k + (1-math.cos(angle)) * (k @ k)


def xyz_rpy(xyz, rpy):
    t = np.eye(4)
    r, p, y = rpy
    t[:3, :3] = (axis_rotation([0, 0, 1], y) @ axis_rotation([0, 1, 0], p)
                  @ axis_rotation([1, 0, 0], r))
    t[:3, 3] = xyz
    return t


def quaternion(rot):
    """Rotation matrix -> normalized wxyz, including rotations near pi."""
    r = finite(rot, (3, 3))
    vals, vecs = np.linalg.eigh(np.array([
        [r[0, 0]-r[1, 1]-r[2, 2], r[1, 0]+r[0, 1], r[2, 0]+r[0, 2], r[2, 1]-r[1, 2]],
        [r[1, 0]+r[0, 1], r[1, 1]-r[0, 0]-r[2, 2], r[2, 1]+r[1, 2], r[0, 2]-r[2, 0]],
        [r[2, 0]+r[0, 2], r[2, 1]+r[1, 2], r[2, 2]-r[0, 0]-r[1, 1], r[1, 0]-r[0, 1]],
        [r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1], np.trace(r)]]) / 3.)
    q = vecs[:, np.argmax(vals)][[3, 0, 1, 2]]
    return q if q[0] >= 0 else -q


def rotation_from_quaternion(q):
    w, x, y, z = finite(q, (4,)) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def _sphere_groups(mapping,prefix=''):
    """Fixed robot-model data only; no measured transforms are cached."""
    return [(prefix+name,np.array([sphere['center'] for sphere in spheres],dtype=float).reshape(-1,3),
             [float(sphere['radius']) for sphere in spheres]) for name,spheres in mapping.items() if spheres]


class RobotGeometry:
    def __init__(self, urdf_path, cfg_path, hand_path):
        self.urdf_path, self.cfg_path, self.hand_path = str(urdf_path), str(cfg_path), str(hand_path)
        self.cfg = yaml.safe_load(Path(cfg_path).read_text())['kinematics']
        self.hand_data = json.loads(Path(hand_path).read_text())
        if self.hand_data.get('schema') != 'native_hand_collision/v1':
            raise ValueError('invalid robot hand model')
        root = ET.parse(urdf_path).getroot()
        self.joints, self.children = {}, {}
        for j in root.findall('joint'):
            origin, axis, limit, mimic = j.find('origin'), j.find('axis'), j.find('limit'), j.find('mimic')
            nums = lambda x: list(map(float, x.split()))
            xyz = nums(origin.get('xyz', '0 0 0')) if origin is not None else [0, 0, 0]
            rpy = nums(origin.get('rpy', '0 0 0')) if origin is not None else [0, 0, 0]
            self.joints[j.get('name')] = dict(parent=j.find('parent').get('link'), child=j.find('child').get('link'),
                origin=xyz_rpy(xyz, rpy), axis=nums(axis.get('xyz', '1 0 0')) if axis is not None else [1, 0, 0],
                kind=j.get('type'), mimic=None if mimic is None else (mimic.get('joint'), float(mimic.get('multiplier', 1)), float(mimic.get('offset', 0))))
            self.children.setdefault(j.find('parent').get('link'), []).append(j.get('name'))
        self.hand_joints = tuple(j['name'] for h in self.hand_data['hands'].values() for j in h['joints'])
        self._body_sphere_groups=[group for group in _sphere_groups(self.cfg['collision_spheres'])
                                  if group[0] not in ('L_base_link','R_base_link')]
        self._hand_fixed={}
        for side,hand in self.hand_data['hands'].items():
            joints=[(joint,np.array(joint['parent_to_joint']),np.linalg.inv(joint['child_to_joint']),
                     np.eye(3)['XYZ'.index(joint['axis'])]) for joint in hand['joints']]
            self._hand_fixed[side]=(hand['root'],matrix(hand['tcp_to_root']),joints,
                                    _sphere_groups(hand['spheres']))
        # One exact measured configuration per hand is enough for an arm CCD
        # segment: its finger angles stay fixed while the arm/waist changes.
        self._hand_local_cache={}
        self.required = tuple(dict.fromkeys([*ORDER, *self.cfg.get('lock_joints', {}), *self.hand_joints]))
        self.ignore = {frozenset((a, b)) for a, bs in self.cfg.get('self_collision_ignore', {}).items() for b in bs}
        self._pair_indices = None
        self._pair_names = None
        by_child={j['child']:(n,j) for n,j in self.joints.items()}
        self.signatures={}
        for link in set(by_child)|{'pelvis'}:
            walk,signature=link,[]
            while walk in by_child:
                n,j=by_child[walk]
                if n in ORDER:
                    signature.append(n)
                walk=j['parent']
            self.signatures[link]=tuple(signature)
        self.locked_ancestors={}
        for link in set(by_child)|{'pelvis'}:
            walk,locked=link,[]
            while walk in by_child:
                name,joint=by_child[walk]
                if name not in ORDER and joint['kind']!='fixed':locked.append(name)
                walk=joint['parent']
            self.locked_ancestors[link]=tuple(locked)


    def fk(self, positions, require=True):
        missing = [n for n in self.required if n not in positions]
        if require and missing:
            raise ValueError('missing_measured_joints: '+','.join(missing))
        q = dict(positions)
        poses = {'pelvis': np.eye(4)}
        todo = ['pelvis']
        while todo:
            parent = todo.pop()
            for name in self.children.get(parent, []):
                j = self.joints[name]
                v = q.get(name)
                if v is None and j['mimic']:
                    master, scale, offset = j['mimic']
                    v = q.get(master, 0)*scale+offset
                if v is None:
                    v = 0.
                if not math.isfinite(float(v)):
                    raise ValueError('nonfinite joint')
                t = poses[parent] @ j['origin']
                if j['kind'] in ('revolute', 'continuous'):
                    r = np.eye(4)
                    r[:3, :3] = axis_rotation(j['axis'], float(v))
                    t = t @ r
                poses[j['child']] = t
                todo.append(j['child'])
        return poses


    def tcp(self, positions):
        poses = self.fk(positions)
        return {side: poses[TCP_LINK[side]] for side in SIDES}


    def hand_local(self, side, positions):
        root,root_transform,joints,groups=self._hand_fixed[side]
        values=[]
        for joint,_,_,_ in joints:
            name=joint['name']
            if name not in positions:
                raise ValueError('missing measured finger '+name)
            value=float(positions[name])
            if not math.isfinite(value):
                raise ValueError('nonfinite measured finger '+name)
            values.append(value)
        key=tuple(values)
        cached=self._hand_local_cache.get(side)
        if cached is not None and cached[0]==key:
            return list(zip(cached[1],cached[2].copy(),cached[3]))
        transforms = {root:root_transform}
        pending = list(joints)
        while pending:
            rest = []
            for j,parent_to_joint,child_inverse,axis in pending:
                if j['parent'] not in transforms:
                    rest.append((j,parent_to_joint,child_inverse,axis))
                    continue
                if j['name'] not in positions:
                    raise ValueError('missing measured finger '+j['name'])
                t = np.eye(4)
                t[:3, :3] = axis_rotation(axis, positions[j['name']])
                transforms[j['child']] = transforms[j['parent']] @ parent_to_joint @ t @ child_inverse
            if len(rest) == len(pending):
                raise ValueError('disconnected hand model')
            pending = rest
        out = []
        for link, points,radii in groups:
            t = transforms[link]
            centres=points @ t[:3,:3].T+t[:3,3]
            out.extend(zip([side+'/hand/'+link]*len(radii),centres,radii))
        names=tuple(sphere[0] for sphere in out)
        centres=np.array([sphere[1] for sphere in out]).reshape(-1,3)
        radii=np.array([sphere[2] for sphere in out])
        centres.setflags(write=False)
        radii.setflags(write=False)
        self._hand_local_cache[side]=(key,names,centres,radii)
        # Callers receive their own coordinates, never a writable cache view.
        return list(zip(names,centres.copy(),radii))


    def finger_commands(self, values):
        result = {}
        for side, ratios in values.items():
            names = [side+'_'+finger+'_1_joint' for finger in ('little','ring','middle','index')]
            names += [side+'_thumb_2_joint',side+'_thumb_1_joint']
            limits = [-1.333]*4+([-.48,1.246165] if side=='left' else [.48,-1.246165])
            positions = dict(zip(names, [(1-float(v))*lim for v,lim in zip(ratios,limits)]))
            for finger in ('little','ring','middle','index'):
                positions[side+'_'+finger+'_2_joint'] = 1.1169*positions[side+'_'+finger+'_1_joint']
            positions[side+'_thumb_3_joint'] = (1.1425 if side=='left' else -1.1425)*positions[side+'_thumb_2_joint']
            positions[side+'_thumb_4_joint'] = .7508*positions[side+'_thumb_3_joint']
            for joint in self.hand_data['hands'][side]['joints']:
                positions[joint['name']] = float(np.clip(positions[joint['name']], joint['lower'], joint['upper']))
            result.update(positions)
        return result


    def spheres(self, positions):
        poses = self.fk(positions)
        out = []
        for link,points,radii in self._body_sphere_groups:
            if link not in poses:
                continue
            t = poses[link]
            centres=points @ t[:3,:3].T+t[:3,3]
            out.extend(zip([link]*len(radii),centres,radii))
        for side in SIDES:
            t = poses[TCP_LINK[side]]
            local=self.hand_local(side,positions)
            centres=np.array([sphere[1] for sphere in local]) @ t[:3,:3].T+t[:3,3]
            out.extend((sphere[0],point,sphere[2]) for sphere,point in zip(local,centres))
        return out
