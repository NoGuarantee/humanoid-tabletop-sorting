"""Recorded grasp geometry for the normal sorting mission, from canonical RGB-D poses."""
from copy import deepcopy
from dataclasses import dataclass
from itertools import product

import numpy as np
from geometry_msgs.msg import Quaternion

from hack_motion.recorded_grasp import CLOSE, PRESHAPE, RHINO
from hack_motion.robot import _quat_wxyz
from .grasp_planner import GraspCandidate, PlacePlan, choose_arm, matrix_from_quat, pose, tcp_from_local


def uses_recorded(db, name):
    return db.get('by_name', {}).get(name, {}).get('recorded_profile') == 'rhino_e71'


def evaluation_db(db, enabled):
    """Opt in to a full mission trial; default skip stays until bench qualification."""
    result = deepcopy(db)
    if enabled:
        result.setdefault('by_name', {}).setdefault(RHINO['object_name'], {}).update(
            skip=False, recorded_profile='rhino_e71')
    return result


@dataclass
class RecordedGrasp(GraspCandidate):
    center_in_tcp: tuple = ()
    corners_in_tcp: tuple = ()  # offsets from the object's centre, expressed in TCP axes
    exposure: dict = None


def transform(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or abs(np.linalg.det(matrix[:3, :3]) - 1) > 1e-6):
        raise ValueError('invalid rigid transform')
    return matrix


def observed_pose(data, now_s, after_stamp_ns=0):
    """Accept one fresh canonical model pose, with an exposure shared by its scene."""
    key = data.get('exposure_key', {})
    stamp = data.get('stamp_ns', 0)
    if (data.get('schema') != 'hack_perception.model_poses/v1' or data.get('status') != 'ok'
            or data.get('frame_id') != 'pelvis' or type(stamp) is not int
            or stamp <= after_stamp_ns or not 0 <= now_s - stamp / 1e9 <= 1.75
            or key.get('stamp_ns') != stamp or not key.get('session_id')
            or type(key.get('epoch')) is not int):
        raise ValueError('invalid or stale model pose response')
    matches = [obj for obj in data.get('objects', []) if obj.get('name') == RHINO['object_name']]
    if len(matches) != 1 or matches[0].get('pose_status') != 'ok' or matches[0].get('exposure_key') != key:
        raise ValueError('Rhino canonical pose unavailable or ambiguous')
    return transform(matches[0]['T_pelvis_object'])


def targets(object_pose):
    world = transform(object_pose)
    result = {}
    for phase in ('pregrasp', 'approach', 'tcp'):
        tcp = world @ transform(RHINO['T_reference_' + phase]['right'])
        w, x, y, z = _quat_wxyz(tcp[:3, :3])
        result[phase] = pose(*tcp[:3, 3], Quaternion(x=float(x), y=float(y), z=float(z), w=float(w)))
    return result


def candidate(data, now_s, obj, container, db, after_stamp_ns):
    world = observed_pose(data, now_s, after_stamp_ns)
    if container is None or choose_arm(obj, container) != RHINO['hand']:
        raise ValueError('recorded_profile_requires_right_arm_and_observed_destination')
    row = next(o for o in data['objects'] if o.get('name') == RHINO['object_name'])
    center, size = np.asarray(row['center_pelvis_m']), np.asarray(row['size_model_m'])
    if (center.shape != (3,) or size.shape != (3,) or not np.isfinite(center).all()
            or not np.isfinite(size).all() or np.min(size) <= 0):
        raise ValueError('recorded_object_bounds_unavailable')
    tcp = world @ transform(RHINO['T_reference_tcp']['right'])
    poses = targets(world)
    lift = deepcopy(poses['tcp'])
    # Main verifies a measured rise >= 5 cm; the 47 mm diagnostic lift is not a sorting lift.
    lift.position.z += .12
    center_tcp = tcp[:3, :3].T @ (center - tcp[:3, 3])
    corners = np.asarray(list(product((-.5, .5), repeat=3))) * size
    corners_tcp = corners @ world[:3, :3].T @ tcp[:3, :3]
    result = RecordedGrasp('right', poses['pregrasp'], poses['tcp'], lift, CLOSE,
                           RHINO['effort_limit'], 1., PRESHAPE, True,
                           tuple(center_tcp), tuple(map(tuple, corners_tcp)), dict(data['exposure_key']))
    place(result, container, db)  # reject a known footprint mismatch before touching the object
    return result


def refreshed(previous, current):
    """A pregrasp does not license descent after a reset or a large pose change."""
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
