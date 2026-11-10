"""Execute the E71 right-hand recipe through main's MoveTcp and Hand actions."""
from copy import deepcopy

import numpy as np
from geometry_msgs.msg import Quaternion

from hack_motion.recorded_grasp import CLOSE, PRESHAPE, RHINO
from hack_motion.robot import _quat_wxyz
from .grasp_planner import pose


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


def pick_recorded(clients, observe, read_tcp, wait, emit):
    """One attempt. observe reacquires after preparation and pregrasp.

    read_tcp returns the measured right TCP Pose. wait uses bounded sim time.
    emit records completed phases. Failures retain the commanded hand position.
    Completion describes motor execution; the caller assesses physical hold.
    """
    side = RHINO['hand']

    def move(stage, target, mode):
        if not clients.move_tcp(side, target, mode, pos_tol=.005):
            raise RuntimeError(stage + '_motion_failed')
        emit(stage, {})

    def hand(stage, preset):
        result = clients.hand(side, preset, RHINO['effort_limit'])
        if result is None or not result.success:
            raise RuntimeError(stage + '_hand_failed')
        emit(stage, dict(position=list(result.position), effort=list(result.effort),
                         object_detected=bool(result.object_detected)))

    observe()  # Populate the normal motion collision world before preparation.
    move('ready', pose(0., 0., 0., Quaternion(w=1.)), 'ready')
    initial = observe()
    hand('open', 'open')
    hand('preshape', PRESHAPE)
    move('pregrasp', targets(initial)['pregrasp'], 'direct')
    current = observe()
    move('approach', targets(current)['approach'], 'linear')
    hand('close', CLOSE)
    lift = deepcopy(read_tcp())
    lift.position.z += RHINO['test_lift_m']
    move('lift', lift, 'linear')
    wait(RHINO['hold_s'])
    emit('hold_interval_completed', dict(duration_s=RHINO['hold_s']))
    return dict(commands_completed=True, physical_hold_confirmed=None)
