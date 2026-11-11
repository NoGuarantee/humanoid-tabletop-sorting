# Rigid-transform helpers reused from PR9 e499c89 (hack_grasp.geometry).
"""Small, ROS-independent rigid-transform helpers; metres and radians."""
import math

import numpy as np


def as_transform(value, name="transform"):
    out = np.asarray(value, dtype=float)
    if out.shape != (4, 4) or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(out[3], [0., 0., 0., 1.], atol=1e-7):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = out[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5)
            or not np.isclose(np.linalg.det(rotation), 1., atol=2e-5)):
        raise ValueError(f"{name} must contain a proper rotation")
    return out.copy()


transform = as_transform


def unit(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("axis must be a finite three-vector")
    length = float(np.linalg.norm(value))
    if length < 1e-12:
        raise ValueError("axis must not be zero")
    return value / length


def inverse(value):
    value = as_transform(value)
    out = np.eye(4)
    out[:3, :3] = value[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ value[:3, 3]
    return out


def translation(value):
    out = np.eye(4)
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("translation must be a finite three-vector")
    out[:3, 3] = value
    return out


def rotation_error(left, right):
    """Geodesic angle between rotations or homogeneous transforms."""
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape == (4, 4):
        a = as_transform(a)[:3, :3]
    if b.shape == (4, 4):
        b = as_transform(b)[:3, :3]
    if a.shape != (3, 3) or b.shape != (3, 3) or not np.isfinite([a, b]).all():
        raise ValueError("rotations must be finite 3x3 matrices")
    return math.acos(float(np.clip((np.trace(a.T @ b) - 1.) / 2., -1., 1.)))


def quaternion_matrix(quaternion):
    """Quaternion wxyz -> rotation, normalizing finite nonzero input."""
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("quaternion must be a finite nonzero wxyz vector")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def pose_matrix(position_wxyz):
    value = np.asarray(position_wxyz, dtype=float)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("pose must be xyz followed by wxyz")
    out = translation(value[:3])
    out[:3, :3] = quaternion_matrix(value[3:])
    return out


def matrix_quaternion(rotation):
    """Proper rotation -> unit quaternion wxyz, stable near pi."""
    r = np.asarray(rotation, dtype=float)
    if r.shape == (4, 4):
        r = as_transform(r)[:3, :3]
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("rotation must be finite 3x3")
    candidates = np.array([
        1+r[0, 0]+r[1, 1]+r[2, 2], 1+r[0, 0]-r[1, 1]-r[2, 2],
        1-r[0, 0]+r[1, 1]-r[2, 2], 1-r[0, 0]-r[1, 1]+r[2, 2]])
    i = int(candidates.argmax())
    q = np.zeros(4)
    q[i] = .5 * math.sqrt(max(float(candidates[i]), 0.))
    divisor = 4 * q[i]
    if i == 0:
        q[1:] = [r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]]
        q[1:] /= divisor
    elif i == 1:
        q[0], q[2], q[3] = (r[2, 1]-r[1, 2])/divisor, (r[0, 1]+r[1, 0])/divisor, (r[0, 2]+r[2, 0])/divisor
    elif i == 2:
        q[0], q[1], q[3] = (r[0, 2]-r[2, 0])/divisor, (r[0, 1]+r[1, 0])/divisor, (r[1, 2]+r[2, 1])/divisor
    else:
        q[0], q[1], q[2] = (r[1, 0]-r[0, 1])/divisor, (r[0, 2]+r[2, 0])/divisor, (r[1, 2]+r[2, 1])/divisor
    return q / np.linalg.norm(q)


def interpolate(start, finish, fraction):
    start, finish = as_transform(start), as_transform(finish)
    fraction = float(fraction)
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("interpolation fraction must be in [0,1]")
    a, b = matrix_quaternion(start), matrix_quaternion(finish)
    dot = float(a @ b)
    if dot < 0:
        b, dot = -b, -dot
    dot = min(1., max(-1., dot))
    if dot > .9995:
        q = a + fraction * (b-a)
    else:
        angle = math.acos(dot)
        q = (math.sin((1-fraction)*angle)*a + math.sin(fraction*angle)*b) / math.sin(angle)
    out = translation((1-fraction)*start[:3, 3] + fraction*finish[:3, 3])
    out[:3, :3] = quaternion_matrix(q)
    return out


def transform_points(matrix, points):
    matrix = as_transform(matrix)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite Nx3")
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def axis_angle(axis, angle):
    axis = unit(axis)
    angle = float(angle)
    if not math.isfinite(angle):
        raise ValueError("angle must be finite")
    x, y, z = axis
    k = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    out = np.eye(4)
    out[:3, :3] += math.sin(angle)*k + (1-math.cos(angle))*(k @ k)
    return out
