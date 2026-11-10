"""Recorded grasp geometry for the normal sorting mission, from canonical RGB-D poses."""
from copy import deepcopy
from dataclasses import dataclass
from itertools import product

import numpy as np
from geometry_msgs.msg import Quaternion

from hack_motion.recorded_grasp import PROFILES
from hack_motion.robot import _quat_wxyz
from .grasp_planner import GraspCandidate, PlacePlan, choose_arm, matrix_from_quat, pose, tcp_from_local


def uses_recorded(db, name):
    return bool(db.get('by_name', {}).get(name, {}).get('recorded_profiles'))


def evaluation_db(db, enabled, selection='rhino,brother,philips,nescafe'):
    """Opt in to a full mission trial; default skip stays until bench qualification."""
    result = deepcopy(db)
    if enabled:
        families = {s.strip() for s in selection.split(',') if s.strip()}
        known = {p['family'] for p in PROFILES.values()}
        if not families or families - known:
            raise ValueError('unknown or empty recorded_objects selection')
        for name in {p['object_name'] for p in PROFILES.values() if p['family'] in families}:
            keys = [key for key, p in PROFILES.items() if p['object_name'] == name]
            result.setdefault('by_name', {}).setdefault(name, {}).update(skip=False, recorded_profiles=keys)
    return result


def postlift_names(db):
    return frozenset(name for name, entry in db.get('by_name', {}).items()
                     if entry and any('postlift' in PROFILES[key] for key in entry.get('recorded_profiles', [])))


@dataclass
class RecordedGrasp(GraspCandidate):
    center_in_tcp: tuple = ()
    corners_in_tcp: tuple = ()  # offsets from the object's centre, expressed in TCP axes
    exposure: dict = None
    profile_key: str = ''


def transform(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or abs(np.linalg.det(matrix[:3, :3]) - 1) > 1e-6):
        raise ValueError('invalid rigid transform')
    return matrix


def observed_pose(data, now_s, after_stamp_ns=0, *, object_name):
    """Accept one fresh canonical model pose, with an exposure shared by its scene."""
    key = data.get('exposure_key', {})
    stamp = data.get('stamp_ns', 0)
    if (data.get('schema') != 'hack_perception.model_poses/v1' or data.get('status') != 'ok'
            or data.get('frame_id') != 'pelvis' or type(stamp) is not int
            or stamp <= after_stamp_ns or not 0 <= now_s - stamp / 1e9 <= 1.75
            or key.get('stamp_ns') != stamp or not key.get('session_id')
            or type(key.get('epoch')) is not int):
        raise ValueError('invalid or stale model pose response')
    matches = [obj for obj in data.get('objects', []) if obj.get('name') == object_name]
    if len(matches) != 1 or matches[0].get('pose_status') != 'ok' or matches[0].get('exposure_key') != key:
        raise ValueError('canonical pose unavailable or ambiguous: ' + object_name)
    return transform(matches[0]['T_pelvis_object'])


def targets(object_pose, profile):
    world = transform(object_pose)
    result = {}
    for phase in ('pregrasp', 'approach', 'tcp'):
        tcp = world @ transform(profile['T_reference_' + phase][profile['hand']])
        w, x, y, z = _quat_wxyz(tcp[:3, :3])
        result[phase] = pose(*tcp[:3, 3], Quaternion(x=float(x), y=float(y), z=float(z), w=float(w)))
    return result


def candidate(data, now_s, obj, container, db, after_stamp_ns, attempt=1, profile_key=None):
    world = observed_pose(data, now_s, after_stamp_ns, object_name=obj.name)
    if container is None:
        raise ValueError('recorded_profile_requires_observed_destination')
    side = choose_arm(obj, container)
    keys = db.get('by_name', {}).get(obj.name, {}).get('recorded_profiles', [])
    choices = [PROFILES[key] for key in keys if PROFILES[key]['hand'] == side and PROFILES[key]['object_name'] == obj.name]
    if profile_key:
        choices = [p for p in choices if p['key'] == profile_key]
    if not choices:
        raise ValueError('no_recorded_profile_for_object_and_arm: ' + obj.name + '/' + side)
    profile = choices[min(max(attempt - 1, 0), len(choices) - 1)]
    row = next(o for o in data['objects'] if o.get('name') == obj.name)
    center, size = np.asarray(row['center_pelvis_m']), np.asarray(row['size_model_m'])
    if (center.shape != (3,) or size.shape != (3,) or not np.isfinite(center).all()
            or not np.isfinite(size).all() or np.min(size) <= 0):
        raise ValueError('recorded_object_bounds_unavailable')
    tcp = world @ transform(profile['T_reference_tcp'][side])
    poses = targets(world, profile)
    lift = deepcopy(poses['tcp'])
    # Main verifies a measured rise >= 5 cm; the 47 mm diagnostic lift is not a sorting lift.
    lift.position.z += profile['lift_m']
    center_tcp = tcp[:3, :3].T @ (center - tcp[:3, 3])
    corners = np.asarray(list(product((-.5, .5), repeat=3))) * size
    corners_tcp = corners @ world[:3, :3].T @ tcp[:3, :3]
    key = profile['key']
    result = RecordedGrasp(side, poses['pregrasp'], poses['tcp'], lift, key + '_close',
                           profile['effort_limit'], 1., key + '_preshape', True,
                           tuple(center_tcp), tuple(map(tuple, corners_tcp)), dict(data['exposure_key']), key)
    place(result, container, db)  # reject a known footprint mismatch before touching the object
    return result


def refreshed(previous, current):
    """A pregrasp does not license descent after a reset or a large pose change."""
    if previous.profile_key != current.profile_key or previous.arm != current.arm:
        raise ValueError('recorded_profile_changed_after_pregrasp')
    if any(previous.exposure[k] != current.exposure[k] for k in ('session_id', 'epoch')):
        raise ValueError('recorded_pose_episode_changed')
    if current.exposure['stamp_ns'] <= previous.exposure['stamp_ns']:
        raise ValueError('recorded_pose_not_refreshed')
    a, b = previous.grasp, current.grasp
    shift = np.linalg.norm([getattr(a.position, k) - getattr(b.position, k) for k in 'xyz'])
    relative = np.asarray(matrix_from_quat(a.orientation)).T @ matrix_from_quat(b.orientation)
    angle = np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))
    if shift > .02 or angle > .15:
        raise ValueError('recorded_pose_moved_after_pregrasp')
    return current


def place(grasp, container, db):
    """Put the whole fitted object above the opening while retaining the grasp orientation.

    A rigid object-to-hand offset is an assumption, not proof that contact survived the carry.
    Main's contact monitor and post-release RGB-D verifier still decide the outcome.
    """
    p = db.get('place', {})
    q = grasp.grasp.orientation
    tcp_rotation = np.asarray(matrix_from_quat(q))
    tray_rotation = np.asarray(matrix_from_quat(container.pose.orientation))
    origin = np.array([getattr(container.pose.position, k) for k in 'xyz'])
    size = np.array([getattr(container.size, k) for k in 'xy'])
    if (not np.isfinite(origin).all() or not np.isfinite(size).all() or np.min(size) <= 0
            or not np.isfinite(tray_rotation).all()
            or not np.allclose(tray_rotation.T @ tray_rotation, np.eye(3), atol=1e-6)
            or abs(np.linalg.det(tray_rotation) - 1) > 1e-6):
        raise ValueError('invalid_destination_geometry')
    corners = np.asarray(grasp.corners_in_tcp) @ tcp_rotation.T @ tray_rotation
    margin = max(.015, p.get('recorded_rim_margin', .015))
    room = size / 2 - np.max(np.abs(corners[:, :2]), axis=0) - margin
    if np.min(room) < 0:
        raise ValueError('recorded_object_does_not_fit_destination')
    toward_robot = tray_rotation.T @ -origin
    length = np.linalg.norm(toward_robot[:2])
    xy = np.zeros(2) if length < 1e-6 else toward_robot[:2] / length * p.get('inset', .06)
    xy = np.clip(xy, -room, room)

    def at_height(height):
        center = origin + tray_rotation @ [*xy, height - np.min(corners[:, 2])]
        return tcp_from_local(center, grasp.center_in_tcp, q)

    approach = at_height(p.get('approach_height', .20))
    return PlacePlan(grasp.arm, approach, at_height(p.get('release_height', .10)), deepcopy(approach), 0.)
