"""Test projected points from an RGB-D template against measured free space."""

import numpy as np
from scipy.ndimage import minimum_filter


def depth_view(sensor):
    depth = sensor["depth"].astype(np.float32)
    valid = (depth >= 150) & (depth <= 6000)
    depth[~valid] = np.inf
    # Use the nearest measured neighbor at boundaries so that a nearer surface
    # can explain occlusion without penalizing a shifted or sparse projection.
    minimum = minimum_filter(depth, size=3, mode="constant", cval=np.inf)
    return sensor["K"], sensor["T"], valid, minimum


def free_space_fraction(points, pose, view):
    k, camera, valid, depth = view
    local = (points @ pose[:3, :3].T + pose[:3, 3] - camera[:3, 3]) @ camera[:3, :3]
    local = local[local[:, 2] > .15]
    z = local[:, 2]
    pixels = np.floor((local @ k.T)[:, :2] / z[:, None]).astype(np.intp)
    u, v = pixels.T
    height, width = depth.shape
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[inside], v[inside], z[inside]
    measured = valid[v, u]
    u, v, z = u[measured], v[measured], z[measured]
    index = v * width + u
    ordered = np.lexsort((z, index))
    index = index[ordered]
    front = ordered[np.r_[True, index[1:] != index[:-1]]] if len(ordered) else ordered
    if len(front) < 30:
        return None
    nearer = depth[v[front], u[front]] - 1000 * z[front] > 8.
    return float(np.mean(nearer))


def rerank(points, pose, loss, residual, options, views):
    """Change a fit only if another candidate resolves a strong depth contradiction."""
    baseline = [(view, free_space_fraction(points, pose, view)) for view in views]
    baseline = [(view, score) for view, score in baseline if score is not None]
    if not baseline:
        return pose, residual
    worst = max(score for _, score in baseline)
    if worst < .15:
        return pose, residual
    best_cost = loss + .010 * worst
    best_pose, best_residual = pose, residual
    for option_loss, option_residual, candidate in options:
        if option_residual > min(.012, residual + .003):
            continue
        fractions = [free_space_fraction(points, candidate, view) for view, _ in baseline]
        if any(fraction is None for fraction in fractions):
            continue
        candidate_worst = max(fractions)
        cost = option_loss + .010 * candidate_worst
        if candidate_worst <= worst - .10 and cost < best_cost:
            best_cost, best_pose, best_residual = cost, candidate, option_residual
    return best_pose, best_residual
