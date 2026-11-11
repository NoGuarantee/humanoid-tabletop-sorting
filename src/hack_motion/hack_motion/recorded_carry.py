"""Swept support from PR9 e499c89; checked carry through main MoveTcp."""
import math
import copy
import numpy as np
from .recorded_geometry import interpolate, rotation_error, as_transform, quaternion_matrix, matrix_quaternion

CARRY_ROTATION_STEP = .04


def orientation_arc_deviation(quaternions, start, finish, tolerance):
    """Largest angular departure from the requested short SLERP arc (wxyz).

    A previous segment may finish within its tolerance, behind the nominal
    waypoint. Comparing the next path to its final orientation alone counts
    that permitted tracking error plus the deliberate turn as unwanted motion.
    Limit the requested turn separately, then check every supplied orientation
    against the measured-start -> goal arc. This never licenses a large turn.
    """
    rows = np.asarray([start, finish, *quaternions], dtype=float)
    norms = np.linalg.norm(rows, axis=1)
    if rows.ndim != 2 or rows.shape[1] != 4 or not np.isfinite(rows).all() or np.min(norms) < 1e-12:
        raise ValueError('invalid_carry_orientation')
    rows = rows / norms[:, None]
    a, b = rows[:2]
    if a @ b < 0:
        b = -b
    half_turn = math.acos(float(np.clip(a @ b, -1., 1.)))
    if 2*half_turn > CARRY_ROTATION_STEP + tolerance + .001:
        raise ValueError('recorded_carry_rotation_step_exceeded')
    got = rows[2:]
    closest = np.maximum(np.abs(got @ a), np.abs(got @ b))
    if half_turn > 1e-8:
        tangent = (b - math.cos(half_turn)*a) / math.sin(half_turn)
        x, y = got @ a, got @ tangent
        # |dot(q, SLERP(t))| has its interior maximum at this angle modulo pi.
        critical = np.arctan2(y, x) % math.pi
        inside = critical <= half_turn
        closest[inside] = np.hypot(x[inside], y[inside])
    return float(np.max(2*np.arccos(np.clip(closest, 0., 1.)), initial=0.))


def pose_matrix(pose):
    matrix = np.eye(4)
    matrix[:3, 3] = [getattr(pose.position, k) for k in 'xyz']
    matrix[:3, :3] = quaternion_matrix([getattr(pose.orientation, k) for k in 'wxyz'])
    return as_transform(matrix)


def with_matrix(prototype, matrix):
    matrix = as_transform(matrix)
    result = copy.deepcopy(prototype)
    for k, value in zip('xyz', matrix[:3, 3]):
        setattr(result.position, k, float(value))
    for k, value in zip('wxyz', matrix_quaternion(matrix)):
        setattr(result.orientation, k, float(value))
    return result


def carry_waypoints(start, target, rotation_step=CARRY_ROTATION_STEP, distance_step=.08, raise_offset=(0., 0.)):
    """PR9 raise -> bounded traverse/turn; retain the start orientation on the raise.

    Target height includes the swept-object reserve calculated by the brain.
    Each waypoint is still planned and checked by the existing motion owner.
    """
    start, target = as_transform(start), as_transform(target)
    offset = np.asarray(raise_offset, dtype=float)
    if offset.shape != (2,) or not np.isfinite(offset).all() or np.linalg.norm(offset) > .080001:
        raise ValueError('invalid_recorded_carry_inward_offset')
    if np.linalg.norm(offset) > 1e-8 and (offset @ start[:2, 3] >= 0 or target[2, 3] <= start[2, 3]+.001):
        raise ValueError('recorded_carry_offset_requires_inward_raise')
    raised = start.copy()
    raised[2, 3] = max(start[2, 3], target[2, 3])
    raised[:2, 3] += offset
    across = target.copy()
    across[2, 3] = raised[2, 3]
    count = max(1, math.ceil(np.linalg.norm(across[:3, 3] - raised[:3, 3]) / distance_step),
                math.ceil(rotation_error(raised, across) / rotation_step))
    if count > 128:
        raise ValueError('recorded carry waypoint budget exceeded')
    return ([raised] if raised[2, 3] > start[2, 3] + .001 else []) + [
        interpolate(raised, across, i/count) for i in range(1, count+1)] + (
            [target] if across[2, 3] > target[2, 3] + .001 else [])

def _minimum_rotating_support(points, start, finish, up):
    """Exact minimum vertical vertex support on the shortest SLERP arc."""
    from .recorded_geometry import matrix_quaternion
    # Match geometry.interpolate's global-quaternion sign choice exactly.
    # At pi, converting start.T @ finish independently can pick the other arc.
    a, b = matrix_quaternion(start), matrix_quaternion(finish)
    if a @ b < 0:
        b = -b
    q = np.r_[a @ b, a[0]*b[1:]-b[0]*a[1:]-np.cross(a[1:], b[1:])]
    sine = float(np.linalg.norm(q[1:]))
    local_up = start.T @ up
    if sine < 1e-12:
        return float(np.min(points @ local_up))
    axis = q[1:]/sine
    angle = 2*math.atan2(sine, float(q[0]))
    axial = (points @ axis)[:, None]*axis
    constant = axial @ local_up
    cosine = (points-axial) @ local_up
    sinusoid = np.cross(axis, points) @ local_up
    minimum = np.minimum(points @ local_up, points @ (finish.T @ up))
    critical = (np.arctan2(sinusoid, cosine)+math.pi) % (2*math.pi)
    interior = critical <= angle
    minimum[interior] = np.minimum(minimum[interior],
        constant[interior]-np.hypot(cosine[interior], sinusoid[interior]))
    return float(minimum.min())
