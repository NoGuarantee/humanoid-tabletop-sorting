"""Fast tray openings from measured rim height and connected plastic colour.

No learned masks or remembered layout: candidates must pass the same measured
rectangle/three-wall support as model-segmented openings.
"""
import numpy as np

from . import table_clusters as tc
from .rgbd.scene_geometry import _opening


def color_openings(sensor, cloud, plane, tray_of_color):
    if plane is None:
        return []
    rgb = sensor['rgb'].astype(float)
    high, low = rgb.max(axis=2), rgb.min(axis=2)
    channel = rgb.argmax(axis=2)
    height = plane.height(cloud)
    support = ((height > .003) & (height < .085) & (high >= 25)
               & (high-low >= .035*high)
               & (cloud[..., 0] > tc.TABLE_X[0]) & (cloud[..., 0] < tc.TABLE_X[1])
               & (np.abs(cloud[..., 1]) < tc.TABLE_Y[1]))
    normal = plane.normal
    measured_plane = normal, -float(normal @ plane.point)
    found = {}
    for color in range(3):
        if ('red', 'green', 'blue')[color] not in tray_of_color:
            continue
        observed = support & (channel == color)
        # Floor plastic connects separate visible rim strips. Rasterisation
        # can leave one-pixel gaps at corners. Dilate only
        # for connectivity; the fit still receives original measured pixels.
        padded = np.pad(observed, 1)
        h, w = observed.shape
        joined = np.logical_or.reduce([padded[y:y+h, x:x+w] for y in range(3) for x in range(3)])
        pixels = np.argwhere(joined)
        if len(pixels) < 30:
            continue
        labels, _ = tc.connected_cells(pixels)
        counts = np.bincount(labels)
        for ident in np.flatnonzero(counts >= 30):
            members = pixels[labels == ident]
            mask = np.zeros(cloud.shape[:2], dtype=bool)
            mask[members[:, 0], members[:, 1]] = True
            mask &= observed
            fitted = _opening(sensor, mask, measured_plane, tray_of_color)
            if fitted is not None:
                tray, error = fitted
                if tray.id not in found or error < found[tray.id][1]:
                    found[tray.id] = tray, error
    return [found[key][0] for key in sorted(found)]
