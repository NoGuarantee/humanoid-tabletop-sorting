#!/usr/bin/env python3
"""Probe arm poses to raise the hand from the hanging ready pose by joints, before the first cuRobo plan.

cuRobo has no table yet (it comes from perception), so the joint move out of the hang must keep the arm out of
the space in front of the pelvis below the top of the side trays. For each candidate the straight joint path from ready is
sampled; reported: the lowest arm sphere bottom in front of FRONT_X, self-collisions on the way, and the final TCP.
FRONT_X and TRAY_TOP are the sim scene (scene.yaml), used only to judge the candidates here.

  bash tools/remote.sh dev -- 'python3 tools/reach_pose_probe.py'

torch and cuRobo come from the dev image (remote.sh dev-image), not from curobo_install.sh.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src" / "hack_motion"))

from fit_tienkung_spheres import READY_POSE, colliding_pairs, load_runtime_cfg  # noqa: E402

FRONT_X = 0.20  # table near edge 0.22 in front of the pelvis, minus 2 cm
TRAY_TOP = -0.11  # table top -0.18 plus the tray height 0.07: the side trays are right under the swung-out hand
CANDIDATES = {  # right arm, a chain of joint moves from ready; the left arm is mirrored (roll and yaw flip)
    "side_fold": [{"shoulder_roll": -0.9}, {"shoulder_roll": -0.9, "elbow_pitch": -1.6},
                  {"shoulder_roll": -0.2, "elbow_pitch": -1.6}],
    "side_up": [{"shoulder_roll": -0.9}, {"shoulder_roll": -0.9, "elbow_pitch": -2.0},
                {"shoulder_roll": -0.2, "elbow_pitch": -2.0}],
    "side_fwd": [{"shoulder_roll": -0.9}, {"shoulder_roll": -0.9, "elbow_pitch": -1.6},
                 {"shoulder_roll": -0.3, "elbow_pitch": -1.6, "shoulder_pitch": -0.4}],
    "back_fold": [{"shoulder_pitch": 0.6}, {"shoulder_pitch": 0.6, "elbow_pitch": -2.2},
                  {"shoulder_pitch": -0.2, "elbow_pitch": -2.0}],
    "side_only": [{"shoulder_roll": -0.9}, {"shoulder_roll": -1.2, "elbow_pitch": -1.0}],
    "wide_fold": [{"shoulder_roll": -1.2}, {"shoulder_roll": -1.2, "elbow_pitch": -1.6},
                  {"shoulder_roll": -0.2, "elbow_pitch": -1.6}],
    "wide_up": [{"shoulder_roll": -1.2}, {"shoulder_roll": -1.2, "elbow_pitch": -2.0},
                {"shoulder_roll": -0.2, "elbow_pitch": -2.0}],
    "wider_up": [{"shoulder_roll": -1.4}, {"shoulder_roll": -1.4, "elbow_pitch": -2.0},
                 {"shoulder_roll": -0.2, "elbow_pitch": -2.0}],
}
MIRROR = {"shoulder_roll", "shoulder_yaw", "elbow_yaw", "wrist_roll"}


def main():
    from curobo._src.robot.kinematics.kinematics import Kinematics
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
    from curobo.types import JointState

    kin_cfg = KinematicsCfg.from_data_dict(load_runtime_cfg()["kinematics"])
    kin = Kinematics(kin_cfg)
    names = list(kin.joint_names)
    idx = {v: k for k, v in kin_cfg.kinematics_config.link_name_to_idx_map.items()}
    sphere_link = [idx[i] for i in kin_cfg.kinematics_config.link_sphere_idx_map.tolist()]
    tcp = {"r": "right_tcp_link", "l": "left_tcp_link"}
    ready = torch.tensor([READY_POSE.get(j, 0.0) for j in names])
    for label, chain in CANDIDATES.items():
        arm = [i for i, link in enumerate(sphere_link)
               if link.endswith(("_r_link",)) and not link.startswith(("hip", "knee")) or link == "R_base_link"]
        worst, who, hits = 1.0, "", set()
        start = ready
        for step, joints in enumerate(chain):
            goal = ready.clone()
            for side in "rl":
                for j, v in joints.items():
                    sign = -1.0 if side == "l" and j in MIRROR else 1.0
                    goal[names.index(f"{j}_{side}_joint")] = sign * v
            for t in torch.linspace(0, 1, 21):
                q = (start + t * (goal - start)).view(1, -1)
                st = kin.compute_kinematics(JointState.from_position(q.cuda(), joint_names=names))
                sph = st.robot_spheres.view(-1, 4)[arm].cpu()
                front = (sph[:, 0] + sph[:, 3] > FRONT_X) & (sph[:, 3] > 0)  # r <= 0: unused sphere slot
                if front.any():
                    bottom = torch.where(front, sph[:, 2] - sph[:, 3], torch.full_like(sph[:, 2], 9.0))
                    k = int(bottom.argmin())
                    if float(bottom[k]) < worst:
                        worst, who = float(bottom[k]), f"{sphere_link[arm[k]]}@s{step}t{float(t):.2f}"
                for a_, b_, _ in colliding_pairs(kin_cfg, q[0]):
                    hits.add((a_, b_))
            start = goal
        st = kin.compute_kinematics(JointState.from_position(goal.view(1, -1).cuda(), joint_names=names))
        i = st.tool_frames.index(tcp["r"])
        p = st.tool_poses.position[0, 0, i].cpu().tolist()
        verdict = "CLEAR" if worst > TRAY_TOP + 0.05 and not hits else "no"
        print(f"{label:11s} {chain[-1]}  lowest in front {worst:+.3f} {who}  tcp ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})  "
              f"self {sorted(hits) or 'none'}  {verdict}", flush=True)


if __name__ == "__main__":
    main()
