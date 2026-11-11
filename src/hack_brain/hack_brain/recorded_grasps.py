"""Recorded grasp geometry for the normal sorting mission, from canonical RGB-D poses."""
from copy import deepcopy
from dataclasses import dataclass
from itertools import product

import numpy as np
from geometry_msgs.msg import Quaternion

from hack_motion.recorded_grasp import PROFILES
from hack_motion.robot import _quat_wxyz
from hack_motion.recorded_geometry import axis_angle
from hack_motion.recorded_carry import _minimum_rotating_support
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


def recorded_arms(db):
    return tuple((name, tuple(sorted({PROFILES[key]['hand'] for key in entry['recorded_profiles']})))
                 for name, entry in db.get('by_name', {}).items() if entry and entry.get('recorded_profiles'))


@dataclass
class RecordedGrasp(GraspCandidate):
    center_in_tcp: tuple = ()
    corners_in_tcp: tuple = ()  # offsets from the object's centre, expressed in TCP axes
    exposure: dict = None
    profile_key: str = ''
    observed_center: tuple = ()
    contact_object_id: str = ''
    orientation_mode: str = ''


def selected_object(scene, target):
    """Retain the exact cluster chosen by the FSM, including when names repeat."""
    matches = [o for o in scene.objects if (o.name or o.track_id) == target.id
               and (not target.track_id or o.track_id == target.track_id)]
    if len(matches) != 1:
        raise ValueError('selected_sensor_object_unavailable_or_ambiguous')
    return matches[0]


def _tcp_drift(previous_pose, current_matrix):
    shift = np.linalg.norm(np.array([getattr(previous_pose.position, k) for k in 'xyz']) - current_matrix[:3, 3])
    relative = np.asarray(matrix_from_quat(previous_pose.orientation)).T @ current_matrix[:3, :3]
    angle = np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1))
    return float(shift), float(angle)


def associate_model_pose(data, reference_point, reference_stamp_ns, *, object_name, previous=None):
    """Bind a canonical hypothesis to an independently selected measured surface.

    A visible surface centre need not be the complete model centre. It must lie
    inside exactly one fitted model's oriented bounds, with 2 cm allowance for
    depth/fit error. No score/nearest tie-break; unrelated regions remain in the
    response. DESCEND also applies the existing 2 cm / .15 rad pregrasp drift
    gate before counting compatible poses, rather than after picking one.
    """
    point = np.asarray(reference_point, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all() or data.get('stamp_ns', 0) <= reference_stamp_ns:
        raise ValueError('invalid_recorded_pose_reference')
    matches, evidence, aligned = [], [], {}
    for index, row in enumerate(data.get('objects', [])):
        if row.get('name') != object_name:
            continue
        if row.get('exposure_key') != data.get('exposure_key'):
            raise ValueError('mixed_recorded_pose_exposure')
        if row.get('pose_status') != 'ok':
            continue
        world = transform(row['T_pelvis_object'])
        center, size = np.asarray(row['center_pelvis_m']), np.asarray(row['size_model_m'])
        if center.shape != (3,) or size.shape != (3,) or not np.isfinite([center, size]).all() or np.min(size) <= 0:
            raise ValueError('invalid_recorded_pose_bounds')
        gap = np.maximum(np.abs((point-center) @ world[:3, :3]) - size/2, 0.)
        compatible = bool(np.max(gap) <= .02)
        detail = dict(index=index, outside_m=gap.tolist())
        if previous is not None:
            profile = PROFILES[previous.profile_key]
            # PR9 relative_observation_error compares declared cyclic aliases
            # around model Z. Keep the fresh translation/tilt, but express the
            # pose in the same observable basis as the accepted pregrasp.
            # This is one model instance; distinct instances stay ambiguous.
            mode = row.get('orientation_mode', '')
            order = {'c2': 2, 'c4': 4}.get(mode, 1) if mode == previous.orientation_mode else 1
            variants = []
            for k in range(order):
                equivalent = world @ axis_angle([0., 0., 1.], 2*np.pi*k/order)
                shift, angle = _tcp_drift(previous.grasp, equivalent @ transform(profile['T_reference_tcp'][previous.arm]))
                variants.append((shift, angle, k, equivalent))
            accepted = [v for v in variants if v[0] <= .02 and v[1] <= .15]
            compatible = compatible and len(accepted) == 1
            shift, angle, k, equivalent = accepted[0] if len(accepted) == 1 else min(variants, key=lambda v: v[1])
            detail.update(tcp_shift_m=shift, tcp_angle_rad=angle, symmetry_order=order, symmetry_index=k)
            if compatible:
                aligned[index] = equivalent
        detail['compatible'] = compatible
        evidence.append(detail)
        if compatible:
            matches.append(index)
    if len(matches) != 1:
        raise ValueError('recorded_pose_association_ambiguous' if matches else 'recorded_pose_association_unmatched')
    result = deepcopy(data)
    selected = matches[0]
    for index, row in enumerate(result['objects']):
        if row.get('name') == object_name and index != selected:
            row['unassociated_name'], row['name'] = row['name'], None
        if index in aligned:
            row['T_pelvis_object_observed'] = row['T_pelvis_object']
            row['T_pelvis_object'] = aligned[index].tolist()
    result['target_association'] = dict(reference_point=point.tolist(), reference_stamp_ns=reference_stamp_ns,
                                       selected_index=selected, candidates=evidence)
    return result


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


def candidate(data, now_s, obj, container, db, after_stamp_ns, attempt=1, profile_key=None, disabled_arm=''):
    world = observed_pose(data, now_s, after_stamp_ns, object_name=obj.name)
    if container is None:
        raise ValueError('recorded_profile_requires_observed_destination')
    side = choose_arm(obj, container)
    keys = db.get('by_name', {}).get(obj.name, {}).get('recorded_profiles', [])
    choices = [PROFILES[key] for key in keys
               if PROFILES[key]['hand'] != disabled_arm and PROFILES[key]['object_name'] == obj.name]
    if profile_key:
        choices = [p for p in choices if p['key'] == profile_key]
    else:
        # PR9 E71 physically carried right -> tray_left. Prefer the task's side
        # when demonstrated, but do not invent or require its mirrored grasp.
        preferred = [p for p in choices if p['hand'] == side]
        choices = preferred or choices
    if not choices:
        raise ValueError('no_recorded_profile_for_object_and_arm: ' + obj.name + '/' + side)
    profile = choices[min(max(attempt - 1, 0), len(choices) - 1)]
    side = profile['hand']
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
    result.observed_center = tuple(center)
    result.contact_object_id = obj.track_id
    result.orientation_mode = row.get('orientation_mode', '')
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
    matrix = np.eye(4)
    matrix[:3, :3] = matrix_from_quat(current.grasp.orientation)
    matrix[:3, 3] = [getattr(current.grasp.position, k) for k in 'xyz']
    shift, angle = _tcp_drift(previous.grasp, matrix)
    if shift > .02 or angle > .15:
        raise ValueError('recorded_pose_moved_after_pregrasp')
    return current


def place(grasp, container, db):
    """Put the whole fitted object above the opening, applying the recorded placement yaw.

    A rigid object-to-hand offset is an assumption, not proof that contact survived the carry.
    Main's contact monitor and post-release RGB-D verifier still decide the outcome.
    """
    p = db.get('place', {})
    initial_rotation = np.asarray(matrix_from_quat(grasp.grasp.orientation))
    settings = PROFILES[grasp.profile_key].get('placement', {})
    tray_rotation = np.asarray(matrix_from_quat(container.pose.orientation))
    origin = np.array([getattr(container.pose.position, k) for k in 'xyz'])
    size = np.array([getattr(container.size, k) for k in 'xy'])
    if (not np.isfinite(origin).all() or not np.isfinite(size).all() or np.min(size) <= 0
            or not np.isfinite(tray_rotation).all()
            or not np.allclose(tray_rotation.T @ tray_rotation, np.eye(3), atol=1e-6)
            or abs(np.linalg.det(tray_rotation) - 1) > 1e-6):
        raise ValueError('invalid_destination_geometry')
    # E71 uses a deliberate common yaw about the measured opening normal.
    # This changes neither the learned grasp nor the object-to-hand transform.
    angle = settings.get('yaw_preference_rad', 0.)
    tcp_rotation = axis_angle(tray_rotation[:, 2], angle)[:3, :3] @ initial_rotation
    # The native arm allowed wrist angles that the official ROS bridge clamps.
    # A configured placement pitch retains the recorded grasp and changes the
    # carried object's attitude; all bounds and swept support below use it.
    tcp_rotation = axis_angle([0., 1., 0.], settings.get('pitch_preference_rad', 0.))[:3, :3] @ tcp_rotation
    w, x, y, z = _quat_wxyz(tcp_rotation)
    q = Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))
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

    release_height = settings.get('release_clearance_m', p.get('release_height', .10))
    approach_height = (settings['carry_clearance_m'] if 'carry_clearance_m' in settings else
                       release_height + settings['approach_clearance_m'] if 'approach_clearance_m' in settings else
                       p.get('approach_height', .20))
    release, approach = at_height(release_height), at_height(approach_height)
    points = np.asarray(grasp.corners_in_tcp) + grasp.center_in_tcp
    initial_floor = np.min(points @ initial_rotation[2])
    final_floor = np.min(points @ tcp_rotation[2])
    swept_floor = _minimum_rotating_support(points, initial_rotation, tcp_rotation, np.array([0., 0., 1.]))
    lift_z = grasp.lift.position.z
    required_floor = max(lift_z + initial_floor, approach.position.z + final_floor)
    approach.position.z = max(lift_z, approach.position.z, required_floor - swept_floor)
    return PlacePlan(grasp.arm, approach, release, deepcopy(approach), 0.)
