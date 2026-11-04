"""Robot constants: motor ids, joint limits, groups. Source: hackathon/docs/INTERFACES.md of the simulator repo."""

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

# The table in pelvis, as a cuRobo scene. There is one scene and the robot does not walk (hackathon/docs/TASKS.md): top
# at z -0.18, near edge at x +0.22, 0.76 deep, 2.44 wide (hackathon/config/scene.yaml, and the simulator log prints the
# same). The real top is 3 cm thick; 8 cm here so that a coarse path does not step through it. It cannot go down to the
# floor: the locked knees stand under the table at x 0.20, so a solid block puts every pose in collision (0/32 IK seeds).
# At this thickness the leg spheres clear it by about 3 cm.
TABLE_SCENE = {"cuboid": {"table": {"dims": [0.76, 2.44, 0.08], "pose": [0.60, 0.0, -0.22, 1.0, 0.0, 0.0, 0.0]}}}

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
