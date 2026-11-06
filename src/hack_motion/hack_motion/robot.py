"""Robot constants: motor ids, joint limits, groups. Source: hackathon/docs/INTERFACES.md of the simulator repo."""
import math

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

# Temporary collision world only. Measured table and tray poses must not ship in the solution: the table, trays, racks
# and the cabinet are found by the cameras, like the objects (docs/ANALYSIS.md, section 8 item 9). Search bounds, sizes
# from the docs, which trays exist (/sim/task) and the robot's own geometry may stay. Replace TABLE_PLANE / TRAYS with
# the plane and containers from perceive.
# The table in pelvis, as a cuRobo scene. There is one scene and the robot does not walk (hackathon/docs/TASKS.md): near
# edge at x +0.22, 0.76 deep, 2.44 wide (hackathon/config/scene.yaml, and the simulator log prints the same). The top is
# not level in pelvis: z = -0.1792 + 0.0700 x, 4 degrees. Fitted to depth, and the open fingers touch it there at x 0.28,
# 0.38 and 0.50 (tools/table_touch.py, tools/motion_trace.py gaps), where a level top at -0.18 is 2.4-4.0 cm off.
# The real top is 3 cm thick; 8 cm here so that a coarse path does not step through it. It cannot go down to the
# floor: the locked knees stand under the table at x 0.20, so a solid block puts every pose in collision (0/32 IK seeds).
TABLE_SCENE = {"cuboid": {"table": {"dims": [0.76, 2.44, 0.08],
                                   "pose": [0.6028, 0.0, -0.1771, 0.99939, 0.0, -0.03494, 0.0]}}}

# The table top in pelvis is z = a + b x (fit on depth frames, pose from /tf, rms 0.3 mm): 4 deg, cause not established.
TABLE_PLANE = (-0.1792, 0.0700)
_TILT = math.atan(TABLE_PLANE[1])
_ABOVE_PELVIS = -TABLE_PLANE[0] * math.cos(_TILT)  # m, pelvis origin over the table top along the table normal

# The trays on the table (hackathon/config/scene.yaml: x, y from the pelvis along the table, 0.22 x 0.32 x 0.07, 1 mm over
# the top). The place does not depend on the layout seed, the set depends on the task (/sim/task): sort2 has no tray_center,
# and objects may lie where it would stand. Depth frames of layout 1 agree within 3 mm: rims x 0.237..0.456, y 0.530..0.850
# on the left, x 0.238..0.457, y -0.848..-0.529 on the right, x 0.426..0.643, y -0.158..0.160 in the centre, 71 mm above
# the plane. Solid blocks: hands stay above the rim, and an 8 mm wall is thinner than a step of a coarse path.
TRAYS = {"tray_left": (0.34, 0.69), "tray_right": (0.34, -0.69), "tray_center": (0.53, 0.0)}
TRAY_SIZE = (0.22, 0.32, 0.071)  # to the rim, from the table top


def table_z(x):
    """z of the table top in pelvis at this x."""
    return TABLE_PLANE[0] + TABLE_PLANE[1] * x


def on_table(along, y, height):
    """Pelvis (x, y, z) of a point `along` m out on the table axis from the pelvis, `height` m above the top."""
    c, s = math.cos(_TILT), math.sin(_TILT)
    z_level = height - _ABOVE_PELVIS
    return along * c - z_level * s, y, along * s + z_level * c


def tray_rim(name):
    """Pelvis (x, y, z) of the middle of the opening of a tray."""
    return on_table(*TRAYS[name], TRAY_SIZE[2])


def table_scene(trays=()):
    """TABLE_SCENE with these trays on the table, tilted with it."""
    scene = {"cuboid": dict(TABLE_SCENE["cuboid"])}
    for name in trays:
        x, y, z = on_table(*TRAYS[name], TRAY_SIZE[2] / 2)
        scene["cuboid"][name] = {"dims": list(TRAY_SIZE),
                                 "pose": [x, y, z, math.cos(_TILT / 2), 0.0, -math.sin(_TILT / 2), 0.0]}
    return scene


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
MAX_OBJECTS = CUBOID_CACHE - 1 - len(TRAYS)


def fingertip(position, xyzw):
    """(x, y, z) of the fingertips. `xyzw` is the ROS orientation of the TCP."""
    x, y, z, w = xyzw
    zx = 2 * (x * z + w * y)
    zy = 2 * (y * z - w * x)
    zz = 1 - 2 * (x * x + y * y)
    return (position[0] - FINGERTIP * zx, position[1] - FINGERTIP * zy, position[2] - FINGERTIP * zz)


def _object_axes(yaw):
    """Unit axes of an object in pelvis: long, across, table normal. `yaw` is in the table plane, as perception reports it."""
    c, s = math.cos(_TILT), math.sin(_TILT)
    yc, ys = math.cos(yaw), math.sin(yaw)
    e1 = (c, 0.0, s)  # pelvis +x projected onto the table, perception's e1 for this tilt
    long = (yc * e1[0], ys, yc * e1[2])
    across = (-ys * e1[0], yc, -ys * e1[2])
    return long, across, (-s, 0.0, c)


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


def _in_footprint(obj, point):
    """True when `point` is over this object's measured footprint, EXCLUDE_PAD past the half-size."""
    _, ox, oy, oz, yaw, sx, sy, _, _ = obj
    long, across, _ = _object_axes(yaw)
    dx, dy, dz = point[0] - ox, point[1] - oy, point[2] - oz
    along = dx * long[0] + dy * long[1] + dz * long[2]
    cross = dx * across[0] + dy * across[1] + dz * across[2]
    return abs(along) <= sx / 2 + EXCLUDE_PAD and abs(cross) <= sy / 2 + EXCLUDE_PAD


def _cuboid(obj, names):
    """(name, {dims, pose}) of one object. `names` are cuboids already in the scene, so a repeated id gets a suffix."""
    ident, ox, oy, oz, yaw, sx, sy, sz, _ = obj
    long, across, normal = _object_axes(yaw)
    quat = _quat_wxyz((
        (long[0], across[0], normal[0]),
        (long[1], across[1], normal[1]),
        (long[2], across[2], normal[2]),
    ))
    name = ident or "object"
    suffix = 2
    while name in names:
        name = f"{ident or 'object'}_{suffix}"
        suffix += 1
    dims = [sx + 2 * OBJECT_MARGIN, sy + 2 * OBJECT_MARGIN, max(sz, 0.005) + 2 * OBJECT_MARGIN]
    return name, {"dims": dims, "pose": [ox, oy, oz, *quat]}


def world_scene(trays=(), objects=(), near=None):
    """Table, the trays of the task, and the detected objects around the one being reached.

    `objects` are `(id, x, y, z, yaw, size_x, size_y, size_z, top_z)` in pelvis, metres and radians.
    `near` is the fingertips of the goal. The object under them is left out: the fingers have to go into it,
    and a cuboid there makes every grasp unreachable. Returns `(scene, [ids left out])`.
    """
    scene = table_scene(trays)
    left_out = []
    kept = []
    for obj in objects:
        if obj[5] <= 0.0 or obj[6] <= 0.0:
            continue
        if near is not None and _in_footprint(obj, near):
            left_out.append(obj[0])
            continue
        kept.append(obj)
    if near is not None and len(kept) > MAX_OBJECTS:
        kept.sort(key=lambda obj: (obj[1] - near[0]) ** 2 + (obj[2] - near[1]) ** 2)
    for obj in kept[:MAX_OBJECTS]:
        name, cub = _cuboid(obj, scene["cuboid"])
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
    "open": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "power": [0.2, 0.2, 0.2, 0.2, 0.5, 0.2],  # example from INTERFACES.md
    "pinch": [1.0, 1.0, 1.0, 0.3, 0.4, 0.1],  # TODO tune: index + thumb
    "tube50": [0.35, 0.35, 0.35, 0.35, 0.5, 0.1],  # TODO tune on a 29 mm tube
    "tube15": [1.0, 1.0, 0.25, 0.25, 0.4, 0.1],  # TODO tune on a 17 mm tube
}


def clip(name, value):
    _, _, lo, hi = JOINTS[name]
    return min(max(value, lo), hi)
