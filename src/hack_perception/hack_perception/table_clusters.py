"""Table plane and geometric object clusters from one organised point cloud in `pelvis`. Pure numpy, no ROS.

Geometry only: no names or categories. Pipeline: RANSAC table plane -> points above it within the table area -> connected
components on a 2D grid in the plane -> trays and the robot's arms dropped by simple rules -> a pose for each cluster.

The pelvis is not level in the simulator: captured frames show the table rising along +x by about 0.07 m/m. That fit is
an observation, not a pose to hardcode. The plane below is fitted from the frame; PRIOR_Z and TABLE_X are search bounds,
which the rules allow. `table_z` is that plane at TABLE_REF_XY. Heights are measured along the plane normal; the table
under an object in pelvis is `top_z - size.z`.
"""
import math
from dataclasses import dataclass, field

import numpy as np

TABLE_REF_XY = (0.38, 0.0)  # where table_z is evaluated: centre of the spawn area x 0.26..0.50, y -0.52..0.52
PRIOR_Z = (-0.30, -0.05)  # plane search window in pelvis: keeps out the floor (-0.75) and what stands on the table
TABLE_X = (0.21, 1.00)  # table area in pelvis, a little wider than the top; the robot does not walk
TABLE_Y = (-1.25, 1.25)
MAX_TILT = math.radians(15.0)  # a hypothesis further from pelvis z is a wall or a box side
RANSAC_ITERS = 100  # the table is at least half of the window on every frame
RANSAC_SAMPLES = 2000
RANSAC_TOL = 0.004
REFINE_TOL = 0.003
MIN_TABLE_POINTS = 2000

MIN_HEIGHT = 0.005  # above the plane; the table itself stays within 2 mm
CELL = 0.005  # grid step, 8-connected: 1 pixel is 2-3 mm on the table; the spawn gap between objects is 2 cm
MIN_POINTS = 40  # fewer is a speck
BORDER_PX = 2  # a cluster this close to the frame edge is cut by it

TOP_PERCENTILE = 99.5  # highest point / footprint ends, robust to a stray pixel
EXTENT_PERCENTILE = 0.5

# What is not an object. Trays are 0.22 x 0.32 x 0.07; the largest object seen is 0.13 long, the tallest 0.16.
TRAY_LONG = 0.25  # longer than this is a tray, or a tray with something touching it
TRAY_HEIGHT = (0.05, 0.09)
TRAY_FLOOR_H = 0.014  # the tray floor stands 8-10 mm above the table
TRAY_FLOOR_FRAC = 0.5  # share of cells at floor height: tells a tray cut by the frame from a bowl (at most 0.18)
TRAY_WALL = 0.012  # a piece of tray wall between two objects (seen from the torso) is this thin and has the tray's height
MAX_HEIGHT = 0.28  # taller is an arm: nothing in the pools stands this high
MAX_LOWEST = 0.04  # a whole object stands on the table, and a hand floats above it. Not for cut ones: the frame may hide
#                    the lower part of a box, so a hand cut by the frame needs ARM_RADIUS
MIN_CUT_POINTS = 150  # a sliver at the frame edge has no usable pose
ARM_RADIUS = 0.07  # points this close to an arm segment (from tf) belong to the robot
NEAR_EDGE_X = 0.225  # the table starts about here in pelvis: a cluster reaching past it comes from the robot


@dataclass
class Plane:
    normal: np.ndarray  # unit, z > 0
    point: np.ndarray
    n_inliers: int = 0
    rms: float = 0.0

    def height(self, pts):
        return (pts - self.point) @ self.normal

    def z_at(self, x, y):
        n, p = self.normal, self.point
        return p[2] - (n[0] * (x - p[0]) + n[1] * (y - p[1])) / n[2]

    def basis(self):
        """Unit vectors in the plane: e1 is pelvis x projected on it, e2 = n x e1 is close to pelvis y."""
        e1 = np.array([1.0, 0.0, 0.0]) - self.normal[0] * self.normal
        e1 /= np.linalg.norm(e1)
        return e1, np.cross(self.normal, e1)


@dataclass
class Cluster:
    center: np.ndarray  # pelvis, centre of the visible footprint at mid height
    yaw: float  # main axis, [-pi/2, pi/2]
    size: np.ndarray  # along the main axis / across / height above the plane
    top_z: float  # pelvis z of the top, taken above the footprint centre
    partially_visible: bool
    n_points: int
    track_id: str = ""
    rejected: str = ""  # empty for an object, otherwise why it is not one
    floor_frac: float = 0.0
    lowest: float = 0.0  # lowest point above the plane
    min_x: float = 0.0
    pixels: np.ndarray = field(default=None, repr=False)  # indices into the flattened cloud


@dataclass
class TableScene:
    plane: Plane
    table_z: float
    objects: list
    rejected: list


def fit_table_plane(pts, rng=None):
    """RANSAC on Nx3 finite points, refined by least squares. Returns Plane or None."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    q = pts[(z > PRIOR_Z[0]) & (z < PRIOR_Z[1]) & (x > TABLE_X[0]) & (x < TABLE_X[1])
            & (y > TABLE_Y[0]) & (y < TABLE_Y[1])]
    if len(q) < MIN_TABLE_POINTS:
        return None
    rng = np.random.default_rng(0) if rng is None else rng  # fixed seed: same frame, same answer
    sample = q[rng.choice(len(q), min(RANSAC_SAMPLES, len(q)), replace=False)]
    tri = sample[rng.integers(0, len(sample), (RANSAC_ITERS, 3))]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(normals, axis=1)
    ok = norm > 1e-9
    normals, tri = normals[ok] / norm[ok, None], tri[ok]
    keep = np.abs(normals[:, 2]) > math.cos(MAX_TILT)
    normals, tri = normals[keep], tri[keep]
    if len(normals) == 0:
        return None
    dist = np.abs(sample @ normals.T - np.sum(normals * tri[:, 0], axis=1))  # samples x hypotheses
    best = int(np.argmax(np.sum(dist < RANSAC_TOL, axis=0)))
    normal, point = normals[best], tri[best, 0]
    for tol in (RANSAC_TOL, REFINE_TOL):
        inliers = q[np.abs((q - point) @ normal) < tol]
        if len(inliers) < MIN_TABLE_POINTS:
            return None
        point = inliers.mean(axis=0)
        d = inliers - point
        normal = np.linalg.eigh(d.T @ d)[1][:, 0]
    if normal[2] < 0:
        normal = -normal
    rms = float(np.sqrt(np.mean(((inliers - point) @ normal) ** 2)))
    return Plane(normal, point, len(inliers), rms)


def connected_cells(cells):
    """8-connected components of grid cells (Nx2, sorted by row, then column). Returns labels and their number.

    Works on runs of neighbouring cells in a row, so Python only loops over a few hundred runs, not over every cell.
    """
    rows, cols = cells[:, 0], cells[:, 1]
    new_run = np.r_[True, (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1] + 1)]
    starts = np.flatnonzero(new_run)
    run_of_cell = np.cumsum(new_run) - 1
    run_row = rows[starts].tolist()
    run_lo = cols[starts].tolist()
    run_hi = cols[np.r_[starts[1:], len(cols)] - 1].tolist()
    parent = list(range(len(starts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    previous = 0  # first run that can still touch the current row
    for i in range(len(starts)):
        while run_row[previous] < run_row[i] - 1:
            previous += 1
        for j in range(previous, i):
            if run_row[j] == run_row[i]:
                break
            if run_hi[j] < run_lo[i] - 1:
                continue
            if run_lo[j] > run_hi[i] + 1:
                break
            parent[find(j)] = find(i)
    roots = np.array([find(i) for i in range(len(starts))])
    unique_roots, labels = np.unique(roots, return_inverse=True)
    return labels[run_of_cell], len(unique_roots)


def _pca_yaw(uv):
    """Main axis of 2D points as an angle in [-pi/2, pi/2]."""
    d = uv - uv.mean(axis=0)
    cov = d.T @ d
    w, v = np.linalg.eigh(cov)
    axis = v[:, 1]
    yaw = math.atan2(axis[1], axis[0])
    if yaw > math.pi / 2:
        yaw -= math.pi
    elif yaw < -math.pi / 2:
        yaw += math.pi
    return yaw


def _describe(uvh, cell_uv, plane, partially, pixels):
    """Pose of one cluster. uvh: points in plane coordinates, cell_uv: centres of its occupied cells.

    The axis comes from occupied cells, not points: a side facing the camera holds far more points than the top.
    """
    yaw = _pca_yaw(cell_uv)
    c, s = math.cos(yaw), math.sin(yaw)
    along = uvh[:, 0] * c + uvh[:, 1] * s
    across = -uvh[:, 0] * s + uvh[:, 1] * c
    lo_a, hi_a = np.percentile(along, [EXTENT_PERCENTILE, 100 - EXTENT_PERCENTILE])
    lo_b, hi_b = np.percentile(across, [EXTENT_PERCENTILE, 100 - EXTENT_PERCENTILE])
    mid_a, mid_b = (lo_a + hi_a) / 2, (lo_b + hi_b) / 2
    u, v = mid_a * c - mid_b * s, mid_a * s + mid_b * c
    height = float(np.percentile(uvh[:, 2], TOP_PERCENTILE))
    e1, e2 = plane.basis()
    base = plane.point + u * e1 + v * e2  # on the table under the footprint centre
    return Cluster(center=base + 0.5 * height * plane.normal, yaw=yaw,
                   size=np.array([hi_a - lo_a, hi_b - lo_b, height]),
                   top_z=float((base + height * plane.normal)[2]), partially_visible=partially,
                   n_points=len(uvh), pixels=pixels)


def near_segments(pts, segments, radius):
    """Mask of the Nx3 points closer than radius to any of the segments, each a pair of 3-vectors."""
    near = np.zeros(len(pts), dtype=bool)
    for a, b in segments:
        a, b = np.asarray(a, float), np.asarray(b, float)
        ab = b - a
        t = np.clip((pts - a) @ ab / max(float(ab @ ab), 1e-9), 0.0, 1.0)
        near |= np.sum((pts - a - t[:, None] * ab) ** 2, axis=1) < radius * radius
    return near


def clusters_above(cloud, plane, robot_segments=()):
    """All clusters above the plane in the table area, before filtering. cloud: HxWx3, NaN where no depth."""
    h, w = cloud.shape[:2]
    pts = cloud.reshape(-1, 3)
    with np.errstate(invalid="ignore"):
        height = plane.height(pts)
        mask = (height > MIN_HEIGHT) & (pts[:, 0] > TABLE_X[0]) & (pts[:, 0] < TABLE_X[1]) \
            & (pts[:, 1] > TABLE_Y[0]) & (pts[:, 1] < TABLE_Y[1])
    idx = np.flatnonzero(mask)
    if len(robot_segments):
        idx = idx[~near_segments(pts[idx], robot_segments, ARM_RADIUS)]
    if len(idx) == 0:
        return []
    e1, e2 = plane.basis()
    rel = pts[idx] - plane.point
    uvh = np.stack([rel @ e1, rel @ e2, height[idx]], axis=1)
    grid = np.floor(uvh[:, :2] / CELL).astype(np.int64)
    keys = (grid[:, 0] << 32) + grid[:, 1]  # one integer per cell, sorted by row and column: np.unique(axis=0) is far slower
    first, inverse = np.unique(keys, return_index=True, return_inverse=True)[1:]
    cells = grid[first]
    cell_h = np.zeros(len(cells))
    np.maximum.at(cell_h, inverse, uvh[:, 2])
    cell_label, n = connected_cells(cells)
    point_label = cell_label[inverse]
    rows, cols = np.unravel_index(idx, (h, w))
    out = []
    order = np.argsort(point_label, kind="stable")
    starts = np.searchsorted(point_label[order], np.arange(n + 1))
    for i in range(n):
        members = order[starts[i]:starts[i + 1]]
        if len(members) < MIN_POINTS:
            continue
        r, c = rows[members], cols[members]
        partially = bool(r.min() < BORDER_PX or c.min() < BORDER_PX or r.max() >= h - BORDER_PX
                         or c.max() >= w - BORDER_PX)
        on = cell_label == i
        cluster = _describe(uvh[members], (cells[on] + 0.5) * CELL, plane, partially, idx[members])
        cluster.floor_frac = float(np.mean(cell_h[on] < TRAY_FLOOR_H))
        cluster.lowest = float(np.percentile(uvh[members, 2], 100 - TOP_PERCENTILE))
        cluster.min_x = float(np.min(pts[idx[members], 0]))
        out.append(cluster)
    return out


def rejection(c):
    """Why a cluster is not an object, or an empty string."""
    long_side, height = max(c.size[0], c.size[1]), c.size[2]
    if height > MAX_HEIGHT or c.min_x < NEAR_EDGE_X:
        return "robot"
    if long_side > TRAY_LONG:
        return "tray"
    if TRAY_HEIGHT[0] < height < TRAY_HEIGHT[1]:
        if c.floor_frac > TRAY_FLOOR_FRAC or (not c.partially_visible and min(c.size[0], c.size[1]) < TRAY_WALL):
            return "tray"
    if c.partially_visible and c.n_points < MIN_CUT_POINTS:
        return "sliver"
    if c.lowest > MAX_LOWEST and not c.partially_visible:
        return "floating"
    return ""


def detect(cloud, robot_segments=()):
    """TableScene from an organised HxWx3 cloud in pelvis (NaN where there is no depth), or None without a table.

    robot_segments: pairs of points in pelvis along the robot's forearms and hands, their surroundings are ignored.
    """
    pts = cloud.reshape(-1, 3)
    plane = fit_table_plane(pts[np.isfinite(pts[:, 2])])
    if plane is None:
        return None
    objects, rejected = [], []
    for c in clusters_above(cloud, plane, robot_segments):
        c.rejected = rejection(c)
        (rejected if c.rejected else objects).append(c)
    objects.sort(key=lambda c: (c.center[1], c.center[0]))
    return TableScene(plane, float(plane.z_at(*TABLE_REF_XY)), objects, rejected)


class Tracker:
    """Gives an object the same track_id between calls while it stays where it was.

    Whole objects match by centre. One cut by the frame has its centre shifted, so it matches anything whose footprint
    holds it. A track keeps the pose from the last whole view, and stays after the object leaves the frame.
    """
    RADIUS = 0.03  # a whole object moved further than this is a new object
    TOP_TOL = 0.03

    def __init__(self):
        self.tracks = []  # dicts: id, xy, top_z, long, partial
        self.next_id = 0

    def _distance(self, c, t):
        """Distance between a cluster and a track, or None if they cannot be one object."""
        d = float(np.hypot(c.center[0] - t["xy"][0], c.center[1] - t["xy"][1]))
        if not c.partially_visible and not t["partial"]:
            return d if d < self.RADIUS and abs(c.top_z - t["top_z"]) < self.TOP_TOL else None
        reach = 0.5 * max(max(c.size[0], c.size[1]), t["long"]) + 0.01
        return d if d < reach else None  # the top of a cut object may be out of the frame: no check on it

    def update(self, clusters):
        """Set track_id on each cluster."""
        pairs = []
        for i, c in enumerate(clusters):
            for j, t in enumerate(self.tracks):
                d = self._distance(c, t)
                if d is not None:
                    pairs.append((d, i, j))
        used_c, used_t = set(), set()
        for d, i, j in sorted(pairs):
            if i not in used_c and j not in used_t:
                used_c.add(i)
                used_t.add(j)
                self._assign(clusters[i], self.tracks[j])
        for i, c in enumerate(clusters):
            if i not in used_c:
                t = {"id": f"obj_{self.next_id}", "partial": True, "xy": None, "top_z": 0.0, "long": 0.0}
                self.next_id += 1
                self.tracks.append(t)
                self._assign(c, t)

    @staticmethod
    def _assign(c, t):
        c.track_id = t["id"]
        if t["partial"] or not c.partially_visible:  # a cut view never overwrites a whole one
            t.update(xy=(float(c.center[0]), float(c.center[1])), top_z=c.top_z,
                     long=float(max(c.size[0], c.size[1])), partial=c.partially_visible)
