"""Robot-only native visual hulls; measured-joint FK and exact Euclidean margin.

This module does not change the control collision model. No runtime USD, object
CAD, scene layout, TF lookup or default joint position is used.
"""
from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np

from hack_motion.perception_robot_geometry import axis_rotation, rotation_from_quaternion

ASSET_SHA256 = '9cbb5f1c8340b89fc179cf765619938cafb2d2d0dc6de2bbdb574b0fe1ab4ebe'
ROBOT_SHA256 = '0e99d7d34ed7b4bd6aef0a0246c5d6f0e8d68b99a567a3e836f13cbdef1331e5'
HAND_SHA256 = '69a783d66b27f93625f921b07895fc44f5589486d039bdd4acff22f14a8280df'
TOLERANCE_M = 1e-12
HULL_LINKS = tuple(name+'_'+side+'_link' for side in ('l', 'r')
                   for name in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw',
                                'elbow_pitch', 'elbow_yaw', 'wrist_pitch', 'wrist_roll'))


def _hash(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(json.dumps([value.dtype.str, value.shape]).encode()+value.tobytes()).hexdigest()


def _finite(value, shape, label):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError('invalid native body '+label)
    return array


def _rigid(value):
    t = _finite(value, (4, 4), 'frame')
    if (not np.allclose(t[3], [0, 0, 0, 1], rtol=0, atol=1e-9)
            or not np.allclose(t[:3, :3].T@t[:3, :3], np.eye(3), rtol=0, atol=1e-6)
            or abs(np.linalg.det(t[:3, :3])-1) > 1e-6):
        raise ValueError('invalid native body rigid frame')
    return t


def point_triangle_distance(point, triangles, *, chunk_size=512):
    """Exact Euclidean point-to-triangle minimum, with bounded temporaries.

    Interior orthogonal projections and all three clamped segments participate;
    degenerate triangles reduce to their edges. Plane offsets alone never accept
    an outside point, including around hull edges and corners.
    """
    point = _finite(point, (3,), 'query point')
    triangles = np.asarray(triangles, dtype=float)
    if (triangles.ndim != 3 or triangles.shape[1:] != (3, 3)
            or not np.isfinite(triangles).all() or type(chunk_size) is not int or chunk_size <= 0):
        raise ValueError('invalid native body triangle query')
    best = float('inf')
    for start in range(0, len(triangles), chunk_size):
        a, b, c = np.moveaxis(triangles[start:start+chunk_size], 1, 0)
        ab, ac, ap = b-a, c-a, point-a
        normal = np.cross(ab, ac)
        normal2 = np.einsum('ij,ij->i', normal, normal)
        signed = np.einsum('ij,ij->i', ap, normal)
        projection = ap-normal*(signed/np.maximum(normal2, 1e-300))[:, None]
        aa = np.einsum('ij,ij->i', ab, ab)
        bb = np.einsum('ij,ij->i', ac, ac)
        cc = np.einsum('ij,ij->i', ab, ac)
        v = np.einsum('ij,ij->i', projection, ab)
        w = np.einsum('ij,ij->i', projection, ac)
        denominator = aa*bb-cc*cc
        u = (bb*v-cc*w)/np.maximum(denominator, 1e-300)
        z = (aa*w-cc*v)/np.maximum(denominator, 1e-300)
        interior = (denominator > 1e-28) & (u >= 0) & (z >= 0) & (u+z <= 1)
        distance2 = np.full(len(a), np.inf)
        distance2[interior] = signed[interior]**2/normal2[interior]
        for left, right in ((a, b), (b, c), (c, a)):
            edge, delta = right-left, point-left
            t = np.clip(np.einsum('ij,ij->i', delta, edge)/np.maximum(
                np.einsum('ij,ij->i', edge, edge), 1e-300), 0, 1)
            residual = delta-t[:, None]*edge
            distance2 = np.minimum(distance2, np.einsum('ij,ij->i', residual, residual))
        best = min(best, float(distance2.min()))
    return math.sqrt(best)


def hull_contains(points, vertices, triangles, planes, margin_m, *, chunk_size=64, triangle_data=None):
    """Inside volume or within the Euclidean margin; return mask and work count."""
    result = np.zeros(len(points), bool)
    triangle_tests = 0
    limit = margin_m+TOLERANCE_M
    low, high = vertices.min(0), vertices.max(0)
    delta = np.maximum(np.maximum(low-points, points-high), 0.)
    candidates = np.flatnonzero(np.einsum('ij,ij->i', delta, delta) <= limit*limit)
    if triangle_data is None:
        tri = vertices[triangles]
        tri_low, tri_high = tri.min(1), tri.max(1)
    else:
        tri, tri_low, tri_high = triangle_data
    for offset in range(0, len(candidates), chunk_size):
        indices = candidates[offset:offset+chunk_size]
        signed = np.einsum('ij,kj->ik', points[indices], planes[:, :3])+planes[:, 3]
        maximum = signed.max(1)
        result[indices[maximum <= TOLERANCE_M]] = True
        boundary = indices[(maximum > TOLERANCE_M) & (maximum <= limit)]
        for index in boundary:
            delta = np.maximum(np.maximum(tri_low-points[index], points[index]-tri_high), 0.)
            near = np.einsum('ij,ij->i', delta, delta) <= limit*limit
            triangle_tests += int(near.sum())
            if near.any() and point_triangle_distance(points[index], tri[near]) <= limit:
                result[index] = True
    return result, triangle_tests


def _topology():
    result = {'body_yaw_joint': ('pelvis', 'body_yaw_link', 'Z')}
    for side in ('l', 'r'):
        parent = 'body_yaw_link'
        for name, axis in (('shoulder_pitch', 'Y'), ('shoulder_roll', 'X'), ('shoulder_yaw', 'Z'),
                           ('elbow_pitch', 'Y'), ('elbow_yaw', 'Z'), ('wrist_pitch', 'Y'), ('wrist_roll', 'X')):
            child = name+'_'+side+'_link'
            result[name+'_'+side+'_joint'] = (parent, child, axis)
            parent = child
    return result


class NativeBodyVisualHulls:
    @classmethod
    def load(cls, asset_path, hand_path):
        asset_path, hand_path = Path(asset_path), Path(hand_path)
        raw, hand_raw = asset_path.read_bytes(), hand_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != ASSET_SHA256:
            raise ValueError('native body visual asset checksum mismatch')
        if hashlib.sha256(hand_raw).hexdigest() != HAND_SHA256:
            raise ValueError('native hand source checksum mismatch')
        data, hand = json.loads(raw), json.loads(hand_raw)
        if data.get('source_robot_sha256') != ROBOT_SHA256 or hand.get('source_usd_sha256') != ROBOT_SHA256:
            raise ValueError('native body/hand robot source mismatch')
        return cls(data)

    def __init__(self, data):
        if (not isinstance(data, dict) or data.get('schema') != 'hack_motion.robot_visual_hulls/v1'
                or data.get('source_robot_sha256') != ROBOT_SHA256 or data.get('margin_m') != .003
                or data.get('urdf_frame_equivalence') is not False
                or data.get('numeric_containment_tolerance_m') != 1e-10
                or set(data.get('links', {})) != set(HULL_LINKS)):
            raise ValueError('invalid native body visual model metadata')
        topology = _topology()
        joints = data.get('body_joints', [])
        if len(joints) != len(topology) or {j.get('name') for j in joints} != set(topology):
            raise ValueError('invalid native body joint set')
        self.joints = []
        for joint in joints:
            parent, child, axis = topology[joint['name']]
            if (joint.get('body0') != ['/World/humanoid/'+parent]
                    or joint.get('body1') != ['/World/humanoid/'+child] or joint.get('axis') != axis):
                raise ValueError('invalid native body joint topology')
            frames = []
            for i in (0, 1):
                rotation = _finite(joint['local_rot'+str(i)+'_wxyz'], (4,), 'joint quaternion')
                if abs(np.linalg.norm(rotation)-1) > 1e-6:
                    raise ValueError('invalid native body joint quaternion')
                frame = np.eye(4)
                frame[:3, :3] = rotation_from_quaternion(rotation)
                frame[:3, 3] = _finite(joint['local_pos'+str(i)+'_m'], (3,), 'joint origin')
                frames.append(frame)
            self.joints.append((joint['name'], parent, child, 'XYZ'.index(axis), frames[0], np.linalg.inv(frames[1])))
        self.required = tuple(topology)
        self.models = {}
        for name in HULL_LINKS:
            item = data['links'][name]
            vertices = np.asarray(item['vertices'], dtype=float)
            indices = np.asarray(item['triangles'])
            planes = np.asarray(item['planes'], dtype=float)
            if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 4
                    or not np.isfinite(vertices).all() or indices.dtype.kind not in 'iu'
                    or indices.ndim != 2 or indices.shape[1:] != (3,) or len(indices) < 4
                    or np.min(indices) < 0 or np.max(indices) >= len(vertices)
                    or planes.shape != (len(indices), 4) or not np.isfinite(planes).all()
                    or not np.allclose(np.linalg.norm(planes[:, :3], axis=1), 1, rtol=0, atol=1e-8)):
                raise ValueError('invalid native body hull geometry')
            tri = vertices[indices]
            if (np.min(np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)) <= 1e-16
                    or np.max(np.abs(np.einsum('ijk,ik->ij', tri, planes[:, :3])+planes[:, 3, None])) > 1e-9
                    or item.get('proof_source_triangles_contained_by_convexity') is not True):
                raise ValueError('invalid native body hull facets')
            for start in range(0, len(vertices), 64):
                if np.max(np.einsum('ij,kj->ik', vertices[start:start+64], planes[:, :3])+planes[:, 3]) > 1e-9:
                    raise ValueError('native body hull is not convex/outward')
            meshes = item.get('source_meshes', [])
            if (not meshes or sum(m.get('vertex_count', 0) for m in meshes) != item.get('source_vertex_count')
                    or any(not isinstance(m.get('path'), str) or not m['path'].startswith('/World/humanoid/'+name+'/')
                           or m.get('point_time_samples') != [] or m.get('has_collision_api') is not False
                           or type(m.get('vertex_count')) is not int or m['vertex_count'] <= 0
                           or type(m.get('face_count')) is not int or m['face_count'] <= 0
                           or type(m.get('face_index_count')) is not int or m['face_index_count'] <= 0
                           or len(m.get('face_indices_sha256', '')) != 64
                           or m.get('visibility') not in ('inherited', 'visible') or m.get('purpose') != 'default'
                           for m in meshes)):
                raise ValueError('invalid native body visual provenance')
            for mesh in meshes:
                _rigid(mesh['T_link_mesh'])
            violation = item.get('proof_all_vertices_max_plane_violation_m')
            if not isinstance(violation, Real) or not math.isfinite(violation) or abs(violation) > 1e-9:
                raise ValueError('invalid native body containment provenance')
            low, high = vertices.min(0), vertices.max(0)
            centre = (low+high)/2
            self.models[name] = dict(vertices=vertices, triangles=indices, planes=planes,
                triangle_data=(tri, tri.min(1), tri.max(1)),
                centre=centre, radius=float(np.linalg.norm(vertices-centre, axis=1).max()),
                default_frame=_rigid(item['T_pelvis_default_link']))
        zero = self.native_fk(dict.fromkeys(self.required, 0.))
        if any(np.max(np.abs(zero[name]-model['default_frame'])) > 1e-7 for name, model in self.models.items()):
            raise ValueError('native body default frame validation failed')

    def native_fk(self, q):
        if not isinstance(q, Mapping) or any(name not in q or isinstance(q[name], (bool, np.bool_))
                or not isinstance(q[name], Real) or not math.isfinite(float(q[name])) for name in self.required):
            raise ValueError('missing or invalid measured native body joints')
        poses = {'pelvis': np.eye(4)}
        pending = list(self.joints)
        while pending:
            rest = []
            for name, parent, child, axis, before, after in pending:
                if parent not in poses:
                    rest.append((name, parent, child, axis, before, after)); continue
                rotation = np.eye(4)
                rotation[:3, :3] = axis_rotation(np.eye(3)[axis], float(q[name]))
                poses[child] = poses[parent]@before@rotation@after
            if len(rest) == len(pending):
                raise ValueError('unreachable native body joint tree')
            pending = rest
        if not set(HULL_LINKS) <= poses.keys():
            raise ValueError('missing native body hull frame')
        return poses

    def cover_points(self, points, q, margin_m=.003):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
            raise ValueError('invalid native body measured points')
        if isinstance(margin_m, (bool, np.bool_)) or not isinstance(margin_m, Real) or not 0 <= margin_m <= .02:
            raise ValueError('invalid native body margin')
        poses, result, tests = self.native_fk(q), np.zeros(len(points), bool), 0
        for name, model in self.models.items():
            pending = np.flatnonzero(~result)
            local = (points[pending]-poses[name][:3, 3])@poses[name][:3, :3]
            hit, work = hull_contains(local, model['vertices'], model['triangles'], model['planes'], margin_m,
                                      triangle_data=model['triangle_data'])
            result[pending[hit]] = True; tests += work
        return result, dict(triangle_tests=tests)

    def mask_optical(self, optical, K, T, q, pending, margin_m, project_roi):
        poses = self.native_fk(q)
        result = np.zeros(pending.shape, bool)
        roi_pixels = candidates = triangle_tests = 0
        for name, model in self.models.items():
            pose = poses[name]
            centre = pose[:3, :3]@model['centre']+pose[:3, 3]
            camera_centre = (centre-T[:3, 3])@T[:3, :3]
            roi = project_roi(camera_centre, model['radius']+margin_m, K, *pending.shape)
            if roi is None: continue
            y0, y1, x0, x1 = roi; roi_pixels += (y1-y0)*(x1-x0)
            iy, ix = np.where(pending[y0:y1, x0:x1] & ~result[y0:y1, x0:x1])
            camera_points = optical[y0:y1, x0:x1][iy, ix]
            pelvis = camera_points@T[:3, :3].T+T[:3, 3]
            local = (pelvis-pose[:3, 3])@pose[:3, :3]
            hit, work = hull_contains(local, model['vertices'], model['triangles'], model['planes'], margin_m,
                                      triangle_data=model['triangle_data'])
            candidates += len(local); triangle_tests += work
            result[y0+iy[hit], x0+ix[hit]] = True
        joint_values = {name: float(q[name]) for name in self.required}
        provenance = dict(schema='hack_perception.native_body_visual_mask/v1',
            method='native_joint_frames_exact_euclidean_hull_margin/v1',frame_id='pelvis',
            asset_sha256=ASSET_SHA256, source_robot_sha256=ROBOT_SHA256,source_hand_collision_sha256=HAND_SHA256,
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            margin_m=margin_m,numeric_tolerance_m=TOLERANCE_M,hull_count=len(self.models),
            q_sha256=hashlib.sha256(json.dumps(joint_values,sort_keys=True,allow_nan=False).encode()).hexdigest(),
            native_frames_sha256=_hash(np.array([poses[name] for name in self.models])),
            candidate_roi_pixels=roi_pixels,tested_pixels=candidates,triangle_tests=triangle_tests,
            added_masked_pixels=int(result.sum()),mask_sha256=_hash(result),control_collision_model_changed=False,
            limitations=['Convex hulls approximate selected robot body visuals; overlapping objects may lose pixels.',
                         'Exclusion does not establish object identity or free space.'])
        return result, provenance
