"""Recover sorting-tray openings from YOLO masks and registered RGB-D."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from ..classify import DEFAULT_TRAY
from .geometry import depth_mask_to_pelvis, estimate_table_support


TRAY_CLASS = 7
# NSSR tray geometry and task colour labels. The centre and yaw are fitted per exposure.
TRAY_SIZE_M = (.22, .32, .07)
TRAY_COLORS = ("red", "green", "blue")


@dataclass(frozen=True)
class Tray:
    id: str
    color: str
    center: np.ndarray
    yaw: float
    size: tuple[float, float, float] = TRAY_SIZE_M


def _fit_partial_rim(projected, lo, hi, half):
    """Fit a clipped rectangle with two opposite sides and an adjacent side."""
    span = hi - lo
    candidates = []
    for full in (0, 1):
        clipped = 1 - full
        complete = np.abs(span[:, full] - 2 * half[full]) < .014
        partial = (span[:, clipped] < 2 * half[clipped] - .016) & (span[:, clipped] > .065)
        for angle in np.flatnonzero(complete & partial):
            for side in (-1, 1):
                center = (lo[angle] + hi[angle]) / 2
                edge = lo[angle, clipped] if side == -1 else hi[angle, clipped]
                # Depth samples span the inner and outer faces of the 8 mm wall.
                center[clipped] = edge - side * half[clipped]
                local = projected[angle] - center
                distance = np.abs(np.max(np.abs(local) - half, axis=1))
                error = np.quantile(distance, .90)
                if error > .009:
                    continue
                # Require all three walls beyond their intersection. A corner
                # or a lone stripe can fit several openings with different centres.
                in_full = np.abs(local[:, full]) < half[full] - .035
                in_clip = np.abs(local[:, clipped]) < half[clipped] - .020
                on_clipped_side = np.abs(local[:, clipped] - side * half[clipped]) < .009
                on_full_sides = [np.abs(local[:, full] - sign * half[full]) < .009 for sign in (-1, 1)]
                if np.count_nonzero(in_full & on_clipped_side) < 15:
                    continue
                if any(np.count_nonzero(in_clip & wall) < 15 for wall in on_full_sides):
                    continue
                candidates.append((error + abs(span[angle, full] - 2 * half[full]), angle, center))
    if not candidates:
        return None
    score, angle, center = min(candidates, key=lambda candidate: candidate[0])
    return angle, center, score


def _opening(sensor, mask, plane, tray_of_color):
    rgb = sensor["rgb"][mask].astype(float)
    if len(rgb) < 30:
        return None
    # Prefer the dominant plastic hue and allow pale highlights on the blue tray.
    channels = np.argsort(rgb, axis=1)
    brightest = np.take_along_axis(rgb, channels[:, 2:], axis=1)[:, 0]
    next_brightest = np.take_along_axis(rgb, channels[:, 1:2], axis=1)[:, 0]
    darkest = rgb.min(axis=1)
    chromatic = ((brightest >= 25) & (brightest - darkest >= .035 * brightest)
                 & (brightest - next_brightest >= .01 * brightest))
    votes = np.bincount(channels[chromatic, 2], minlength=3)
    color = votes.argmax()
    if votes[color] < 30 or votes[color] < .75 * votes.sum():
        return None
    label = TRAY_COLORS[color]
    tray_id = tray_of_color.get(label)
    if tray_id is None:
        return None
    plastic = mask.copy()
    plastic[mask] = chromatic & (channels[:, 2] == color)
    points = depth_mask_to_pelvis(sensor["depth"], plastic, sensor["K"], sensor["T"])
    normal, offset = plane
    height = points @ normal + offset
    # A wall top lies one tray height above the measured support. The lower
    # wall strip also contributes to the footprint when the rim is thin in RGB.
    rim = (height > .045) & (height < .085)
    if np.count_nonzero(rim) < 30:
        return None
    tangent = Rotation.align_vectors([normal], [[0., 0., 1.]])[0].as_matrix()[:, :2]
    xy = points[rim] @ tangent

    # Search orientations modulo pi. Score by measured short/long spans, with
    # trimmed extrema to remove stray segmentation pixels near wall edges.
    angles = np.deg2rad(np.arange(180))
    c, s = np.cos(angles), np.sin(angles)
    rotations = np.stack((c, -s, s, c), axis=1).reshape(-1, 2, 2)
    sampled = xy[np.linspace(0, len(xy) - 1, min(1024, len(xy)), dtype=int)]
    projected = sampled @ rotations
    lo, hi = np.quantile(projected, (.01, .99), axis=1)
    expected = np.asarray(TRAY_SIZE_M[:2])
    centers = (lo + hi) * .5
    # Check that measured wall samples are near the rectangle perimeter.
    # A rotated box can match the spans of an obliquely clipped mask.
    extent_error = np.abs(hi - lo - expected)
    boundary = np.max(np.abs(projected - centers[:, None, :]) - expected / 2, axis=2)
    wall_error = np.quantile(np.abs(boundary), .90, axis=1)
    supported = (extent_error <= .016).all(axis=1) & (wall_error < .012)
    if supported.any():
        scores = wall_error + np.linalg.norm(extent_error, axis=1)
        best = np.argmin(np.where(supported, scores, np.inf))
        center_xy, score = centers[best], scores[best]
    else:
        partial = _fit_partial_rim(projected, lo, hi, expected / 2)
        if partial is None:
            return None
        best, center_xy, score = partial
        score += .016  # Prefer a complete rim when another view has one.
    axes = tangent @ rotations[best]
    # The tallest coloured samples are direct measurements of the rim, using
    # the table normal to separate them from the floor of the tray.
    rim_height = np.quantile(height[rim], .90)
    if abs(rim_height - TRAY_SIZE_M[2]) > .012:
        return None
    center = axes @ center_xy + normal * (rim_height - offset)
    yaw = float((np.arctan2(axes[1, 0], axes[0, 0]) + np.pi / 2) % np.pi - np.pi / 2)
    return Tray(tray_id, label, center, yaw), float(score)


def detect_trays(sensors, segmentation, plane, tray_of_color):
    """Return one geometrically supported opening per observed NSSR tray."""
    if plane is None:
        return []
    best = {}
    for sensor, segment in zip(sensors, segmentation):
        if sensor is None:
            continue
        raster, proposals, _ = segment
        for proposal in proposals:
            if proposal["class_id"] != TRAY_CLASS:
                continue
            fitted = _opening(sensor, raster == proposal["instance_id"], plane, tray_of_color)
            if fitted is None:
                continue
            tray, error = fitted
            score = (error, -proposal["confidence"])
            if tray.id not in best or score < best[tray.id][0]:
                best[tray.id] = (score, tray)
    return [best[key][1] for key in sorted(best)]


def estimate_scene_geometry(sensors, segmentation, tray_of_color=None):
    """Find table support and tray openings in the exposure's pelvis frame."""
    for sensor, segment in zip(sensors, segmentation):
        if sensor is None:
            continue
        raster, _, _ = segment
        plane, _ = estimate_table_support(
            sensor["depth"], sensor["K"], sensor["T"], exclude=raster != 0, nssr_sample_centers=True)
        if plane is not None:
            mapping = DEFAULT_TRAY if tray_of_color is None else tray_of_color
            return plane, detect_trays(sensors, segmentation, plane, mapping)
    return None, []
