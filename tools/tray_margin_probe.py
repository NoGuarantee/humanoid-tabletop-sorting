#!/usr/bin/env python3
"""Reach of a top-down hand over a side tray, and a path down to the release height.

The place pose is the tray opening with the fingertips at grasp_db place heights (approach 0.20 m,
release 0.10 m) and the TCP 0.21 m higher. left_tcp_link has no self-collision margin; this checks
that the left tray is then reachable and that a wrist folded to its stops still meets the forearm.

  bash tools/remote.sh dev --timeout 900 -- 'python3 tools/tray_margin_probe.py'
"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "hack_motion"))

from hack_motion import robot  # noqa: E402
from hack_motion.curobo_planner import CuroboPlanner, Unreachable  # noqa: E402
from hack_motion.robot import ARM_JOINTS, raise_steps, tray_rim  # noqa: E402

ROBOT = ROOT / "src" / "hack_motion" / "robot"
CFG = ROBOT / "tienkung_curobo.yml"
# Wrist angles where the raw elbow_yaw and tcp spheres overlap most (local sweep of the two joints).
WORST_WRIST = {"pitch": -0.7853981633974483, "roll": 1.3089969389957472}


def target_at(x, y, tip_z):
    return NS(position=NS(x=x, y=y, z=tip_z + 0.21),
              orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))


def start_q(planner):
    q = {n: planner.ready.get(n, 0.0) for n in planner.joint_names}
    for arm in ARM_JOINTS:
        q.update(zip(ARM_JOINTS[arm], raise_steps(arm)[-1]))
    return q


def pairs(planner, sol):
    import torch
    from curobo.types import JointState

    q = torch.tensor([sol], device="cuda", dtype=torch.float32)
    js = JointState.from_position(q, joint_names=planner.joint_names)
    kin = planner._self_kin
    spheres = kin.compute_kinematics(js).robot_spheres
    cost = planner._self_cost
    cost.setup_batch_tensors(1, 1)
    cost.forward(spheres)
    dist = cost._pair_distance.view(-1)
    hit = dist > 0
    if not bool(hit.any().item()):
        return []
    cfg = kin.config
    sph = cfg.self_collision_config.collision_pairs[hit].to(torch.int64)
    link_idx = cfg.kinematics_config.link_sphere_idx_map.to(torch.int64)[sph].cpu().tolist()
    names = {v: k for k, v in cfg.kinematics_config.link_name_to_idx_map.items()}
    worst = {}
    for (a, b), d in zip(link_idx, dist[hit].cpu().tolist()):
        key = tuple(sorted((names[a], names[b])))
        worst[key] = max(worst.get(key, 0.0), d)
    return sorted(worst.items(), key=lambda kv: -kv[1])


def seeds(planner, arm, target, q):
    import torch
    from curobo.types import JointState

    goal, state = planner._goal(arm, target, q)
    result = planner.mp.ik_solver.solve_pose(
        goal, current_state=state, return_seeds=planner.mp.ik_solver.config.num_seeds)
    link = robot.TCP_LINK[arm]
    i = goal.tool_frames.index(link)
    want = goal.position[0, 0, i, 0]
    sols = result.solution.detach().reshape(-1, result.solution.shape[-1])
    feas = result.feasible
    out = []
    for s in range(sols.shape[0]):
        ok = bool(feas) if isinstance(feas, bool) else bool(feas.detach().reshape(-1)[s].item())
        sol = sols[s].detach()
        js = JointState.from_position(sol.view(1, -1), joint_names=planner.joint_names)
        got = planner.mp.compute_kinematics(js).tool_poses.position[0, 0, i]
        err = torch.linalg.norm(got - want).item()
        out.append((ok, err, sol.cpu().tolist()))
    return out


def show_goal(planner, arm, target, q, label):
    found = seeds(planner, arm, target, q)
    close = [s for s in found if s[1] < 0.015]
    ok = [s for s in close if s[0]]
    print(f"  {label}: {len(ok)} feasible of {len(close)} seeds within 1.5 cm "
          f"({len(found)} seeds)", flush=True)
    close.sort(key=lambda s: (0 if s[0] else 1, s[1]))
    for ok_s, err, sol in close[:3]:
        hit = pairs(planner, sol)
        note = "clear" if not hit else ", ".join(f"{a}-{b} {d * 1e3:.1f} mm" for (a, b), d in hit[:4])
        by = dict(zip(planner.joint_names, sol))
        side = arm[0]
        print(f"    {'ok' if ok_s else 'blocked'} {err * 100:.2f} cm  "
              f"wrist_pitch {by[f'wrist_pitch_{side}_joint']:+.3f}  "
              f"wrist_roll {by[f'wrist_roll_{side}_joint']:+.3f}  "
              f"body {by['body_yaw_joint']:+.3f}  {note}", flush=True)
    if not ok:
        return None
    return min(ok, key=lambda s: s[1])[2]


def worst_wrist(planner, arm):
    """Self-collision of the folded wrist at the joint limits, from the raised arm."""
    q = start_q(planner)
    side = arm[0]
    # Pitch limits match on both arms. Roll limits are mirrored, and so is the sphere layout.
    q[f"wrist_pitch_{side}_joint"] = WORST_WRIST["pitch"]
    q[f"wrist_roll_{side}_joint"] = WORST_WRIST["roll"] if side == "l" else -WORST_WRIST["roll"]
    sol = [q.get(n, 0.0) for n in planner.joint_names]
    hit = pairs(planner, sol)
    note = "clear" if not hit else ", ".join(f"{a}-{b} {d * 1e3:.1f} mm" for (a, b), d in hit[:4])
    print(f"  folded {arm} wrist at the limits: {note}", flush=True)


def plan_to(planner, arm, target, q, mode, label=None):
    label = label or mode
    try:
        _, waypoints, _ = planner.plan(arm, target, q, mode)
    except Unreachable as exc:
        print(f"  {label} unreachable: {exc}", flush=True)
        return None
    print(f"  {label} path, {len(waypoints)} waypoints", flush=True)
    return waypoints[-1]


def main():
    planner = CuroboPlanner(ROBOT / "tiangong2pro_with_hands.urdf", CFG)
    planner.trays = tuple(robot.TRAYS)
    print(planner.load(None), flush=True)
    q = start_q(planner)
    for arm, tray in (("left", "tray_left"), ("right", "tray_right")):
        x, y, z = tray_rim(tray)
        print(f"{arm} {tray} opening ({x:.3f},{y:.3f},{z:.3f})", flush=True)
        for tip in (0.10, 0.16, 0.20):
            show_goal(planner, arm, target_at(x, y, z + tip), q, f"+{tip:.2f} m")
        worst_wrist(planner, arm)
        approach = target_at(x, y, z + 0.20)
        release = target_at(x, y, z + 0.10)
        sol = show_goal(planner, arm, approach, q, "approach again")
        if sol is not None:
            import torch
            at = dict(zip(planner.joint_names, sol))
            q_seed = torch.tensor([sol], device="cuda", dtype=torch.float32)
            gap = float(planner._self_overlap(q_seed)[0].item())
            print(f"  approach seed vs start clearance: {gap * 1e3:.2f} mm", flush=True)
            planner._goal(arm, release, at)
            print(f"  start stepped {planner.start_moved:.4f} rad", flush=True)
            plan_to(planner, arm, release, at, "linear", "linear from the clear seed")
        plan_to(planner, arm, approach, q, "direct")


if __name__ == "__main__":
    main()
