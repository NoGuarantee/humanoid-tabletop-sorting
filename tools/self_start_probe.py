#!/usr/bin/env python3
"""Which sphere pairs collide at a given joint state, per self-collision buffer. Runs where cuRobo is installed.

A start state inside the buffer fails every plan from it, so `no collision-free joint path` repeats until the arm moves.
"""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src/hack_motion"))

from hack_motion.curobo_robot import load_robot_cfg  # noqa: E402

ORDER = ["body_yaw_joint"] + [f"{j}_{s}_joint" for s in "lr" for j in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_yaw", "wrist_pitch", "wrist_roll")]
# Starts of the failed plans in move_tcp_check (body_yaw, left arm, right arm)
STARTS = {
    "high": [0.265, -0.466, 0.387, -0.693, -1.859, 0.343, -0.065, 0.299,
             -1.953, -1.158, 2.309, -1.689, -0.529, 0.61, 1.035],
    "low": [-0.532, 0.776, 0.613, 0.35, -2.356, 0.29, -0.233, 0.732,
            -1.574, 0.003, 1.984, -1.474, -1.914, 1.03, -0.257],
    "high5": [-0.319, 0.32, 0.457, 0.119, -2.286, 0.246, -0.034, 0.382,
              -2.263, 0.119, 2.573, -1.816, -2.36, 0.52, 0.097],
}
BUFFERS = (0.0, 0.002, 0.005, 0.01)


def pairs(kin_cfg, by_name):
    import torch
    from curobo._src.cost.cost_self_collision import SelfCollisionCost
    from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
    from curobo._src.robot.kinematics.kinematics import Kinematics

    kin = Kinematics(kin_cfg)
    js = kin.default_joint_state.clone()
    position = js.position.view(-1).clone()
    for i, name in enumerate(js.joint_names):
        if name in by_name:
            position[i] = by_name[name]
    js.position = position.view(1, -1)
    cost = SelfCollisionCost(SelfCollisionCostCfg(weight=1.0, store_pair_distance=True,
                                                  self_collision_kin_config=kin_cfg.self_collision_config))
    cost.setup_batch_tensors(1, 1)
    cost.forward(kin.compute_kinematics(js).robot_spheres)
    dist = cost._pair_distance.view(-1)
    hit = dist > 0
    if not torch.any(hit):
        return []
    sph = kin_cfg.self_collision_config.collision_pairs[hit].to(torch.int64)
    link_idx = kin_cfg.kinematics_config.link_sphere_idx_map.to(torch.int64)[sph].cpu().tolist()
    names = {v: k for k, v in kin_cfg.kinematics_config.link_name_to_idx_map.items()}
    worst = {}
    for (a, b), d in zip(link_idx, dist[hit].cpu().tolist()):
        key = tuple(sorted((names[a], names[b])))
        worst[key] = max(worst.get(key, 0.0), d)
    return sorted(((a, b, d) for (a, b), d in worst.items()), key=lambda x: -x[2])


def main():
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg

    robot = ROOT / "src/hack_motion/robot"
    urdf = next(robot.glob("*.urdf"))
    cfg = load_robot_cfg(str(urdf), str(robot / "tienkung_curobo.yml"))
    for label, q in STARTS.items():
        by_name = dict(zip(ORDER, q))
        for buffer in BUFFERS:
            kin = copy.deepcopy(cfg["kinematics"])
            kin["self_collision_buffer"] = {k: buffer for k in kin["self_collision_buffer"]}
            found = pairs(KinematicsCfg.from_data_dict(kin), by_name)
            text = ", ".join(f"{a}-{b} {d * 1e3:.1f} mm" for a, b, d in found[:6]) or "free"
            print(f"{label} buffer {buffer * 1e3:4.1f} mm: {len(found)} pairs: {text}", flush=True)


if __name__ == "__main__":
    main()
