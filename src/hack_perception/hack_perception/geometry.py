"""Depth and frame helpers shared by the detectors."""
import numpy as np


def depth_to_m(msg):
    """sensor_msgs/Image 16UC1 (mm) -> float32 HxW metres, 0 where there is no data (closer 0.15 m / farther 6 m)."""
    d = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(msg.height, msg.width)
    return d.astype(np.float32) * 1e-3


def rgb_to_np(msg):
    """sensor_msgs/Image rgb8 -> uint8 HxWx3."""
    return np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.width, 3)


def intrinsics(info):
    """sensor_msgs/CameraInfo -> fx, fy, cx, cy."""
    k = info.k
    return k[0], k[4], k[2], k[5]


def deproject(u, v, z, info):
    """Pixel (u, v) arrays with depth z (metres) -> Nx3 points in the optical camera frame (z forward)."""
    fx, fy, cx, cy = intrinsics(info)
    u, v, z = np.asarray(u, float), np.asarray(v, float), np.asarray(z, float)
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)


def transform_matrix(tf):
    """geometry_msgs/TransformStamped -> 4x4 homogeneous matrix."""
    t, q = tf.transform.translation, tf.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = [t.x, t.y, t.z]
    return m


def apply(m, pts):
    """Apply a 4x4 transform to Nx3 points."""
    pts = np.asarray(pts, float)
    return pts @ m[:3, :3].T + m[:3, 3]
