"""Backprojection, rigid registration and table support from measured depth.

Transforms use column-vector convention: p_pelvis = R @ p_object + t. ICP
scores observed-to-model distances because the back side is unobserved. It is
a local optimizer, not a confidence estimate or a proof of object identity.
"""

import math
import numpy as np


def depth_mask_to_pelvis(depth_mm, mask, K, T_pelvis_camera):
    """Backproject an original-pixel mask using optical-axis Z and exact TF.

    Returns valid pixels in row-major order in metres. The caller is responsible
    for matching camera/depth timestamps and ensuring RGB-depth registration.
    """
    depth, mask = np.asarray(depth_mm), np.asarray(mask)
    k, transform = np.asarray(K, float), np.asarray(T_pelvis_camera, float)
    if depth.ndim != 2 or depth.dtype != np.uint16 or mask.dtype != np.bool_ or depth.shape != mask.shape:
        raise ValueError('expected matching 16UC1 millimetres and binary mask')
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError('expected finite pinhole calibration')
    if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError('expected finite rigid optical camera to pelvis transform')
    _rotation(transform[:3, :3])
    y, x = np.where(mask & (depth >= 150) & (depth <= 6000))
    z = depth[y, x].astype(float)/1000
    rays = np.column_stack((x, y, np.ones(len(x)))) @ np.linalg.inv(k).T
    optical = rays * (z / rays[:, 2])[:, None]
    return optical @ transform[:3, :3].T + transform[:3, 3]


def _points(value):
    p = np.asarray(value, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError('expected at least three finite 3D points')
    return p


def _rotation(value):
    r = np.asarray(value, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or abs(np.linalg.det(r)-1) > 1e-5:
        raise ValueError('expected proper rotation matrix')
    return r


def rigid_fit(source, target):
    """Least squares proper rigid transform for corresponding points."""
    a, b = _points(source), _points(target)
    if a.shape != b.shape:
        raise ValueError('correspondence sizes differ')
    ac, bc = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a-ac).T @ (b-bc))
    d = np.eye(3)
    d[-1, -1] = np.linalg.det(vt.T @ u.T)
    r = vt.T @ d @ u.T
    return r, bc - r @ ac


def estimate_table_support(depth, k, transform):
    """Fit support from depth points in the pelvis workspace."""
    y, x = np.indices(depth.shape)
    y, x = y[::4, ::4], x[::4, ::4]
    d = depth[::4, ::4].astype(float) / 1000
    rays = np.stack((x, y, np.ones_like(x)), axis=-1) @ np.linalg.inv(k).T
    points = (rays * (d / rays[..., 2])[..., None]) @ transform[:3, :3].T + transform[:3, 3]
    valid = (d >= .15) & (d <= 6)
    valid &= ((points >= (.05, -1.2, -.6)) & (points <= (1.4, 1.2, .6))).all(-1)
    return fit_support(points[valid])


def fit_support(points):
    distance_m = .004
    min_up = math.cos(math.radians(15.))
    if len(points) < 32:
        return None, "insufficient_observed_support_samples"
    if len(points) > 4096:
        points = points[np.linspace(0, len(points)-1, 4096, dtype=int)]
    # Fixed internal algorithm seed is unrelated to any task/scene seed.
    rng = np.random.default_rng(0)
    planes = []
    for _ in range(128):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b-a, c-a)
        norm = np.linalg.norm(n)
        if norm < 1e-8:
            continue
        n /= norm
        if n[2] < 0:
            n = -n
        if n[2] < min_up:
            continue
        d = -float(n@a)
        if any(float(n@old[1]) > .9995 and abs(d-old[2]) < 2*distance_m for old in planes):
            continue
        inliers = np.abs(points@n+d) <= distance_m
        planes.append((int(inliers.sum()), n, d, inliers))
    if not planes:
        return None, "horizontal_support_plane_unobserved"
    planes.sort(key=lambda p:p[0], reverse=True)
    best = planes[0]
    if best[0] < max(32, .20*len(points)):
        return None, "support_plane_insufficient_consensus"
    if any(p[0] >= best[0]*.85 and abs(p[2]-best[2]) > 4*distance_m
           for p in planes[1:]):
        return None, "multiple_similarly_supported_horizontal_planes"
    observed = points[best[3]]
    centre = np.mean(observed, axis=0)
    _, singular, axes = np.linalg.svd(observed-centre, full_matrices=False)
    if singular[1] < 1e-7:
        return None, "support_patch_collinear"
    n = axes[-1]
    if n[2] < 0:
        n = -n
    if n[2] < min_up:
        return None, "refined_support_plane_not_horizontal"
    d = -float(n@centre)
    return (n, d), None
