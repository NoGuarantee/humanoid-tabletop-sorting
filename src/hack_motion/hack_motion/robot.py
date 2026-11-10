"""Robot constants: motor ids, joint limits, groups. Source: hackathon/docs/INTERFACES.md of the simulator repo."""
import math
from .recorded_grasp import HAND_TARGETS

MAX_SPEED = 0.8  # rad/s, hard limit of every arm/head/waist motor in the sim and on the real robot

# joint name -> (motor id, command group, lower limit, upper limit)
JOINTS = {
    "head_roll_joint": (1, "head", -0.45, 0.45),
    "head_pitch_joint": (2, "head", -0.44, 0.44),  # plus = down
    "head_yaw_joint": (3, "head", -1.57, 1.57),  # plus = left
    "body_yaw_joint": (31, "waist", -2.97, 3.14),  # plus = left
    "shoulder_pitch_l_joint": (11, "arm", -2.97, 2.97),
    "shoulder_roll_l_joint": (12, "arm", -0.26, 2.62),
    "shoulder_yaw_l_joint": (13, "arm", -2.97, 2.97),
    "elbow_pitch_l_joint": (14, "arm", -2.62, 0.26),
    "elbow_yaw_l_joint": (15, "arm", -2.97, 2.97),
    "wrist_pitch_l_joint": (16, "arm", -0.79, 1.05),
    "wrist_roll_l_joint": (17, "arm", -1.66, 1.31),
    "shoulder_pitch_r_joint": (21, "arm", -2.97, 2.97),
    "shoulder_roll_r_joint": (22, "arm", -2.62, 0.26),
    "shoulder_yaw_r_joint": (23, "arm", -2.97, 2.97),
    "elbow_pitch_r_joint": (24, "arm", -2.62, 0.26),
    "elbow_yaw_r_joint": (25, "arm", -2.97, 2.97),
    "wrist_pitch_r_joint": (26, "arm", -0.79, 1.05),
    "wrist_roll_r_joint": (27, "arm", -1.31, 1.66),
}

ARM_JOINTS = {
    side: [f"{j}_{side[0]}_joint" for j in
           ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_yaw", "wrist_pitch", "wrist_roll")]
    for side in ("left", "right")
}

TCP_LINK = {"left": "left_tcp_link", "right": "right_tcp_link"}

# The collision world is what perception measured: the table plane and the trays it saw (perception/scene). Measured
# scene poses must not ship in the solution (docs/ANALYSIS.md, section 8 item 9); search bounds, sizes, which trays
# exist (/sim/task) and the robot's own geometry may. The poses measured in the simulator for the probes in tools/ are in
# tools/scene_measured.py.
#
# `table` everywhere below is the plane (a, b, c, d) from SceneObjects.table_plane: a x + b y + c z + d = 0 in pelvis,
# (a, b, c) unit with c > 0. In the sim it is z = -0.179 + 0.070 x, 4 degrees: the top is not level in pelvis.
# The table cuboid covers TABLE_X x TABLE_Y on that plane, perception's search bounds for the table (table_clusters):
# the cameras see the top from x 0.24 to 0.98. The real top is 3 cm thick; 8 cm here so that a coarse path does not
# step through it. It cannot go down to the floor: the locked knees stand under the table at x 0.20, so a solid block
# puts every pose in collision (0/32 IK seeds).
TABLE_X = (0.21, 1.0)
TABLE_Y = (-1.25, 1.25)
TABLE_THICKNESS = 0.08

# The sorting trays (hackathon/docs/TASKS.md); /sim/task says which of them the task has. 0.22 x 0.32, 0.071 from the top
# to the rim. A tray is (id, x, y, z, yaw, size_x, size_y, size_z) from perception: the middle of the opening in pelvis,
# yaw of the 0.22 side in the table plane. Solid blocks: hands stay above the rim, and an 8 mm wall is thinner than a
# step of a coarse path.
TRAY_IDS = ("tray_left", "tray_right", "tray_center")
TRAY_SIZE = (0.22, 0.32, 0.071)


def table_normal(table):
    """Unit normal of the table plane, up. Level when there is no table."""
    if not table:
        return (0.0, 0.0, 1.0)
    return tuple(float(v) for v in table[:3])


def table_z(table, x, y=0.0):
    """z of the table top in pelvis at (x, y)."""
    a, b, c, d = table
    return -(d + a * x + b * y) / c


def table_axes(table, yaw=0.0):
    """Unit axes in pelvis: along `yaw`, across it, and the table normal.

    `yaw` is measured in the table plane from pelvis +x projected onto it, as perception reports it.
    """
    nx, ny, nz = table_normal(table)
    ex = (1.0 - nx * nx, -nx * ny, -nx * nz)  # pelvis +x minus its normal part
    norm = math.sqrt(sum(v * v for v in ex))
    e1 = tuple(v / norm for v in ex)
    e2 = (ny * e1[2] - nz * e1[1], nz * e1[0] - nx * e1[2], nx * e1[1] - ny * e1[0])  # normal x e1
    yc, ys = math.cos(yaw), math.sin(yaw)
    long = tuple(yc * u + ys * v for u, v in zip(e1, e2))
    across = tuple(-ys * u + yc * v for u, v in zip(e1, e2))
    return long, across, (nx, ny, nz)


def _box(center, dims, axes):
    """A cuRobo cuboid: {dims, pose [x, y, z, w, x, y, z]} with these axes as its x, y, z."""
    long, across, normal = axes
    quat = _quat_wxyz((
        (long[0], across[0], normal[0]),
        (long[1], across[1], normal[1]),
        (long[2], across[2], normal[2]),
    ))
    return {"dims": [float(v) for v in dims], "pose": [float(v) for v in center] + list(quat)}


def table_cuboid(table):
    """The table slab under the plane, over the search bounds."""
    axes = table_axes(table)
    x, y = sum(TABLE_X) / 2, sum(TABLE_Y) / 2
    top = (x, y, table_z(table, x, y))
    normal = axes[2]
    center = tuple(t - n * TABLE_THICKNESS / 2 for t, n in zip(top, normal))
    length = (TABLE_X[1] - TABLE_X[0]) / axes[0][0]  # along the plane, over the x bounds
    return _box(center, (length, TABLE_Y[1] - TABLE_Y[0], TABLE_THICKNESS), axes)


def tray_cuboid(tray, table):
    """The solid block of one tray: from the opening down to the table along its normal."""
    _, x, y, z, yaw, sx, sy, sz = tray
    axes = table_axes(table, yaw)
    normal = axes[2]
    center = (x - normal[0] * sz / 2, y - normal[1] * sz / 2, z - normal[2] * sz / 2)
    return _box(center, (sx, sy, sz), axes)


# Neighbours on the table. Without them the arm plans through whatever is standing there and pushes it; a pushed
# object that leaves the table is -10. 1 cm on every side, on top of cuRobo's 1 cm activation distance: the drives
# still close the last few millimetres. Wider than that and two objects with the 2 cm spawn gap become one block,
# and the fingers cannot reach the one being grasped.
OBJECT_MARGIN = 0.01
EXCLUDE_PAD = 0.02  # m past the measured half-size. The grasp aims at the centre; a cluster cut by the frame is off by this
FINGERTIP = 0.21  # m from the TCP to the fingertips along -z. The same number as grasp_planner and perception
VIA_CLEAR = 0.04  # m, fingertips above a neighbour the straight plan could not pass
HAND_PAD = 0.12  # m, the open hand past an object's bounding circle when deciding what the approach must clear
CUBOID_CACHE = 16  # table + the three trays + the objects below, allocated once so a later update stays in place
MAX_OBJECTS = CUBOID_CACHE - 1 - len(TRAY_IDS)


def fingertip(position, xyzw):
    """(x, y, z) of the fingertips. `xyzw` is the ROS orientation of the TCP."""
    x, y, z, w = xyzw
    zx = 2 * (x * z + w * y)
    zy = 2 * (y * z - w * x)
    zz = 1 - 2 * (x * x + y * y)
    return (position[0] - FINGERTIP * zx, position[1] - FINGERTIP * zy, position[2] - FINGERTIP * zz)


def _quat_wxyz(rows):
    """(w, x, y, z) of a rotation matrix given by its three rows."""
    (r00, r01, r02), (r10, r11, r12), (r20, r21, r22) = rows
    trace = r00 + r11 + r22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return 0.25 * scale, (r21 - r12) / scale, (r02 - r20) / scale, (r10 - r01) / scale
    if r00 > r11 and r00 > r22:
        scale = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        return (r21 - r12) / scale, 0.25 * scale, (r01 + r10) / scale, (r02 + r20) / scale
    if r11 > r22:
        scale = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        return (r02 - r20) / scale, (r01 + r10) / scale, 0.25 * scale, (r12 + r21) / scale
    scale = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
    return (r10 - r01) / scale, (r02 + r20) / scale, (r12 + r21) / scale, 0.25 * scale


def footprint_gap(obj, point, table=None):
    """Metres from `point` to this object's measured footprint in the table plane, negative over it."""
    _, ox, oy, oz, yaw, sx, sy, _, _ = obj
    long, across, _ = table_axes(table, yaw)
    dx, dy, dz = point[0] - ox, point[1] - oy, point[2] - oz
    along = dx * long[0] + dy * long[1] + dz * long[2]
    cross = dx * across[0] + dy * across[1] + dz * across[2]
    out_along, out_cross = abs(along) - sx / 2, abs(cross) - sy / 2
    if out_along > 0.0 and out_cross > 0.0:
        return math.hypot(out_along, out_cross)
    return max(out_along, out_cross)


def _in_footprint(obj, point, table=None):
    """True when `point` is over this object's measured footprint, EXCLUDE_PAD past the half-size."""
    _, ox, oy, oz, yaw, sx, sy, _, _ = obj
    long, across, _ = table_axes(table, yaw)
    dx, dy, dz = point[0] - ox, point[1] - oy, point[2] - oz
    along = dx * long[0] + dy * long[1] + dz * long[2]
    cross = dx * across[0] + dy * across[1] + dz * across[2]
    return abs(along) <= sx / 2 + EXCLUDE_PAD and abs(cross) <= sy / 2 + EXCLUDE_PAD


def _cuboid(obj, names, table=None):
    """(name, {dims, pose}) of one object. `names` are cuboids already in the scene, so a repeated id gets a suffix."""
    ident, ox, oy, oz, yaw, sx, sy, sz, _ = obj
    name = ident or "object"
    suffix = 2
    while name in names:
        name = f"{ident or 'object'}_{suffix}"
        suffix += 1
    dims = [sx + 2 * OBJECT_MARGIN, sy + 2 * OBJECT_MARGIN, max(sz, 0.005) + 2 * OBJECT_MARGIN]
    return name, _box((ox, oy, oz), dims, table_axes(table, yaw))


# Holds the cuboid cache in place before perception has found the table (setup.sh warms the planner with no scene).
NO_TABLE = {"dims": [0.01, 0.01, 0.01], "pose": [0.0, 0.0, -10.0, 1.0, 0.0, 0.0, 0.0]}


def world_scene(table=None, trays=(), objects=(), near=None):
    """Table, the trays perception has seen, and the detected objects around the one being reached.

    `table` is the plane (a, b, c, d), None before perception found it. `trays` are tray tuples (see TRAY_SIZE),
    `objects` are `(id, x, y, z, yaw, size_x, size_y, size_z, top_z)` in pelvis, metres and radians.
    `near` is the fingertips of the goal. The object under them is left out: the fingers have to go into it,
    and a cuboid there makes every grasp unreachable. Returns `(scene, [ids left out])`.
    """
    cuboids = {"table": table_cuboid(table)} if table else {"no_table": dict(NO_TABLE)}
    for tray in trays:
        cuboids[tray[0]] = tray_cuboid(tray, table)
    scene = {"cuboid": cuboids}
    left_out = []
    kept = []
    for obj in objects:
        if obj[5] <= 0.0 or obj[6] <= 0.0:
            continue
        if near is not None and _in_footprint(obj, near, table):
            left_out.append(obj[0])
            continue
        kept.append(obj)
    if near is not None and len(kept) > MAX_OBJECTS:
        kept.sort(key=lambda obj: (obj[1] - near[0]) ** 2 + (obj[2] - near[1]) ** 2)
    for obj in kept[:MAX_OBJECTS]:
        name, cub = _cuboid(obj, scene["cuboid"], table)
        scene["cuboid"][name] = cub
    return scene, left_out


def _xy_segment_distance(point, a, b):
    """Metres from `point` to the segment ab, both in the table plane."""
    ax, ay = a
    bx, by = b
    px, py = point
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def on_the_way(obj, start_xyz, goal_xyz, pad=HAND_PAD):
    """True when this object's bounding circle comes within `pad` of the straight line from start to goal."""
    dist = _xy_segment_distance((obj[1], obj[2]), start_xyz[:2], goal_xyz[:2])
    return dist <= 0.5 * math.hypot(obj[5], obj[6]) + pad


def clearance_z(top_zs, start_z, goal_z, fingertip=FINGERTIP, margin=VIA_CLEAR):
    """TCP z that puts the fingertips `margin` above these neighbour tops, or None.

    None when there is nothing to clear, or both the start and the goal are already that high:
    a waypoint there would be the pose that just failed. A top-down hand carries its fingertips
    `fingertip` m below the TCP.
    """
    if not top_zs:
        return None
    need = max(top_zs) + fingertip + margin
    if need <= max(start_z, goal_z) + 0.01:
        return None
    return need


def sphere_cuboid_overlap(center, radius, dims, pose):
    """Metres the sphere penetrates the cuboid. Negative when the sphere is clear of it.

    `pose` is (x, y, z, w, x, y, z), the same layout as a cuRobo cuboid.
    """
    px, py, pz = pose[:3]
    w, x, y, z = pose[3:7]
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    dx, dy, dz = center[0] - px, center[1] - py, center[2] - pz
    local = (r00 * dx + r10 * dy + r20 * dz,
             r01 * dx + r11 * dy + r21 * dz,
             r02 * dx + r12 * dy + r22 * dz)
    half = (dims[0] / 2, dims[1] / 2, dims[2] / 2)
    outside = tuple(max(0.0, abs(v) - h) for v, h in zip(local, half))
    gap = math.sqrt(outside[0] ** 2 + outside[1] ** 2 + outside[2] ** 2) - radius
    if all(abs(v) <= h for v, h in zip(local, half)):
        return min(h - abs(v) for v, h in zip(local, half)) + radius
    return -gap


def spheres_hit(spheres, cuboids):
    """True when any sphere (x, y, z, r) penetrates a cuboid. r <= 0 is an unused sphere slot."""
    for x, y, z, r in spheres:
        if r <= 0.0:
            continue
        for cub in cuboids.values():
            if sphere_cuboid_overlap((x, y, z), r, cub["dims"], cub["pose"]) > 0.0:
                return True
    return False


# Joint moves that bring a hanging arm up in front of the chest, the start of every cuRobo plan. cuRobo cannot plan
# from the hang, where the forearm is inside the thigh, and the hanging hand is below table height: any elbow fold from
# the hang swings the fingers forward 0.4 m under the pelvis, into the table edge. Out to the side first, fold the elbow
# there, then bring the upper arm back in. On this path no arm sphere in front of x 0.20 goes below z -0.09 in
# pelvis (tools/reach_pose_probe.py); it ends with the TCP near (0.37, +-0.30, 0.09). Right arm values; roll flips
# sign on the left. Each step lists the joints that differ from home.
RAISE_STEPS = (
    {"shoulder_roll": -1.2},
    {"shoulder_roll": -1.2, "elbow_pitch": -2.0},
    {"shoulder_roll": -0.2, "elbow_pitch": -2.0},
)


# At shoulder_roll = +-pi/2 the shoulder_yaw axis lies on the shoulder_pitch axis. Around that pose shoulder_pitch stops
# short of its command by up to 60 Nm over gravity, on both sides of the singularity, with no force at the wrist:
# 13 Nm at 1.3 rad, 30 Nm at 1.5, 47 Nm at pi/2 (tools/shoulder_roll_sweep.py). The planner stays below it.
PLAN_ROLL_LIMIT = 1.3  # rad


# The MoveTcp `ready` pose: out to the side with the elbow folded (RAISE_STEPS[1]), off the table and, unlike the
# end of the raise, out of the head camera's view of the table.
READY_STEP = 1


def ready_pose(arm):
    """Arm joints of the MoveTcp `ready` mode, in ARM_JOINTS[arm] order."""
    return raise_steps(arm)[READY_STEP]


def plan_limits(name):
    """(lower, upper) of a joint for planning: JOINTS narrowed by the shoulder singularity."""
    _, _, lo, hi = JOINTS[name]
    if name.startswith("shoulder_roll_"):
        lo, hi = max(lo, -PLAN_ROLL_LIMIT), min(hi, PLAN_ROLL_LIMIT)
    return lo, hi


def raise_steps(arm):
    """[[7 positions in ARM_JOINTS[arm] order], ...] for the moves above."""
    side = arm[0]
    sign = -1.0 if side == "l" else 1.0
    out = []
    for step in RAISE_STEPS:
        q = {f"{j}_{side}_joint": (sign * v if j == "shoulder_roll" else v) for j, v in step.items()}
        out.append([q.get(name, 0.0) for name in ARM_JOINTS[arm]])
    return out

# Hand presets: Inspire actuators 1 little, 2 ring, 3 middle, 4 index, 5 thumb bend, 6 thumb rotation; 1.0 open, 0.0 closed.
# Starting points only, tune them on real objects (owner: grasping).
HAND_PRESETS = {
    **HAND_TARGETS,
    "open": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "power": [0.2, 0.2, 0.2, 0.2, 0.5, 0.2],  # example from INTERFACES.md
    # Index + thumb. A finger presses only as hard as its target is past the contact, whatever the force limit:
    # at 0.3/0.4 the thumb stopped at its target without touching and the cartridge slipped out on the lift.
    # Deep targets: 5 of 7 into the tray against 1 of 3 (pool 4, 2026-11-09). Keep the index target at 0.1 or
    # more: the right index stops at 0.08 in free air, and a target under that reads as an object.
    "pinch": [1.0, 1.0, 1.0, 0.1, 0.0, 0.1],
    # Flat pinch: thumb already across (actuator 6) and left open as the anvil. Bending it on a 3 cm
    # dish drove the pad into the table (the reading went to 1.34, past open). Only the index closes.
    # It still slips on the lift (bowl, soap dish, rhino; 2026-11-09): neither force 0.7/1.0 nor an index target
    # of 0.0 kept it, so the grip itself has to change, not its strength.
    "pinch_ready": [1.0, 1.0, 1.0, 1.0, 1.0, 0.1],
    "pinch_flat": [1.0, 1.0, 1.0, 0.55, 1.0, 0.1],
    "tube50": [0.35, 0.35, 0.35, 0.35, 0.5, 0.1],  # TODO tune on a 29 mm tube
    "tube15": [1.0, 1.0, 0.25, 0.25, 0.4, 0.1],  # TODO tune on a 17 mm tube
}


def clip(name, value):
    _, _, lo, hi = JOINTS[name]
    return min(max(value, lo), hi)
