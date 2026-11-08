"""Sensor-only trimmed ICP initialized by a predicted model pose."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .geometry import rigid_fit


def residual(tree, cloud, pose, trim):
    distance, index = tree.query((cloud - pose[:3, 3]) @ pose[:3, :3], workers=1)
    count = max(3, int(len(distance) * trim))
    keep = np.argsort(distance)[:count]
    return float(np.sqrt(np.mean(distance[keep] ** 2))), index, keep


def refine(model, cloud, initial, *, iterations, trim, max_translation, max_rotation_deg):
    model, cloud, initial = (np.asarray(array, float) for array in (model, cloud, initial))
    if model.shape[1:] != (3,) or cloud.shape[1:] != (3,) or len(model) < 3 or len(cloud) < 30:
        raise ValueError("insufficient point data")
    if initial.shape != (4, 4) or not np.isfinite(initial).all() or not np.isfinite(cloud).all():
        raise ValueError("invalid initial pose/cloud")
    tree = cKDTree(model)
    baseline, _, _ = residual(tree, cloud, initial, trim)
    best = baseline
    current = initial.copy()
    for _ in range(iterations):
        _, index, keep = residual(tree, cloud, current, trim)
        rotation, translation = rigid_fit(model[index[keep]], cloud[keep])
        candidate = np.eye(4)
        candidate[:3, :3], candidate[:3, 3] = rotation, translation
        angle_deg = Rotation.from_matrix(initial[:3, :3].T @ rotation).magnitude() * 180 / np.pi
        distance = np.linalg.norm(translation - initial[:3, 3])
        if distance > max_translation or angle_deg > max_rotation_deg:
            break
        cost, _, _ = residual(tree, cloud, candidate, trim)
        if cost > best + 1e-10:
            break
        difference = np.linalg.norm(candidate - current)
        current, best = candidate, cost
        if difference < 1e-6:
            break
    return current if best < baseline - 1e-6 else initial.copy()
