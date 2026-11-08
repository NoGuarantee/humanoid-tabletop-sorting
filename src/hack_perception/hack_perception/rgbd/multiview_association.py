"""Pair simultaneous YOLO instances by measured point sets and color."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .geometry import depth_mask_to_pelvis

REJECTED_COST = 1e3


def color_signature(rgb, mask, depth):
    valid = mask & (depth >= 150) & (depth <= 6000)
    color = rgb[valid].astype(float) / 255
    if len(color) < 30:
        return None
    maximum, minimum = color.max(axis=1), color.min(axis=1)
    saturation = (maximum - minimum) / np.maximum(maximum, 1e-6)
    good = (maximum > .10) & (saturation > .18)
    if good.sum() < 20:
        return None
    color = color[good]
    return np.mean(color / np.maximum(color.sum(axis=1, keepdims=True), 1e-5), axis=0)


def descriptors(rgb, depth, k, transform, raster, objects):
    result = []
    for obj in objects:
        mask = raster == obj["instance_id"]
        points = depth_mask_to_pelvis(depth, mask, k, transform)
        if len(points) < 30:
            continue
        if len(points) > 700:
            points = points[np.linspace(0, len(points) - 1, 700, dtype=int)]
        result.append(dict(id=obj["instance_id"], category=obj["class_id"], points=points,
                           center=np.median(points, axis=0), color=color_signature(rgb, mask, depth)))
    return result


def association(head, torso):
    """Global one-to-one assignment, with local ambiguity and distance gates."""
    if not head or not torso:
        return []
    cost = np.full((len(head), len(torso)), REJECTED_COST)
    for i, a in enumerate(head):
        for j, b in enumerate(torso):
            if a["category"] != b["category"]:
                continue
            center = float(np.linalg.norm(a["center"] - b["center"]))
            if center > .10:
                continue
            # Partial views overlap only in some points. Compare lower quantiles.
            da = cKDTree(b["points"]).query(a["points"], workers=1)[0]
            db = cKDTree(a["points"]).query(b["points"], workers=1)[0]
            near = float((np.quantile(da, .25) + np.quantile(db, .25)) / 2)
            color = None if a["color"] is None or b["color"] is None else float(np.linalg.norm(a["color"] - b["color"]))
            value = .5 * center + .5 * near + (.015 * color if color is not None else .003)
            if value <= .06:
                cost[i, j] = value
    rows, columns = linear_sum_assignment(cost)
    accepted = []
    for i, j in zip(rows, columns):
        value = cost[i, j]
        if value >= REJECTED_COST:
            continue
        row, column = np.delete(cost[i], j), np.delete(cost[:, j], i)
        row_alt = float(row.min()) if len(row) else REJECTED_COST
        column_alt = float(column.min()) if len(column) else REJECTED_COST
        if row_alt - value < .004 or column_alt - value < .004:
            continue
        accepted.append((head[i]["id"], torso[j]["id"]))
    return accepted
