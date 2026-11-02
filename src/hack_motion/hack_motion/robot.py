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
