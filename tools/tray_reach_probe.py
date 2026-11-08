#!/usr/bin/env python3
"""IK error of a top-down hand over a side tray, from the raised-arm pose.

The place aim is the tray opening (scene_measured.tray_rim). Heights are fingertip heights above that opening;
the TCP is 0.21 m higher, as grasp_planner.tcp_at does for top_down. The other hand stays put, as MoveTcp
plans. A feasible seed is preferred; otherwise the closest seed is shown with the joints on a stop.

  bash tools/remote.sh dev -- 'source /opt/ros/humble/setup.bash && python3 tools/tray_reach_probe.py'
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "hack_motion"))

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria  # noqa: E402
from hack_motion import robot  # noqa: E402
from hack_motion.curobo_planner import CuroboPlanner  # noqa: E402
from hack_motion.robot import ARM_JOINTS, plan_limits, raise_steps  # noqa: E402
from scene_measured import tray_rim, use  # noqa: E402

ROBOT = ROOT / "src" / "hack_motion" / "robot"


def target_at(x, y, tip_z, yaw):
    """TCP whose fingertips are at (x, y, tip_z): top_down keeps +z up, tips 0.21 m below the palm."""
    return NS(position=NS(x=x, y=y, z=tip_z + 0.21),
              orientation=NS(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)))


def stops(names, sol):
    """Joints within 0.02 rad of a planning limit."""
    out = []
    for name, value in zip(names, sol):
        if name not in robot.JOINTS:
            continue
        lo, hi = plan_limits(name)
        if value - lo < 0.02:
            out.append(f"{name} {value:+.3f}=lo")
        elif hi - value < 0.02:
            out.append(f"{name} {value:+.3f}=hi")
    return ", ".join(out) or "none"


def best(planner, arm, target, q):
    import torch
    from curobo.types import JointState

    goal, state = planner._goal(arm, target, q)
    result = planner.mp.ik_solver.solve_pose(
        goal, current_state=state, return_seeds=planner.mp.ik_solver.config.num_seeds)
    link = robot.TCP_LINK[arm]
    other = robot.TCP_LINK["right" if arm == "left" else "left"]
    i = goal.tool_frames.index(link)
    j = goal.tool_frames.index(other)
    want = goal.position[0, 0, i, 0]
    want_q = goal.quaternion[0, 0, i, 0]
    hold = goal.position[0, 0, j, 0]
    sols = result.solution.detach().reshape(-1, result.solution.shape[-1])
    feas = result.feasible
    found = None
    for s in range(sols.shape[0]):
        ok = bool(feas) if isinstance(feas, bool) else bool(feas.detach().reshape(-1)[s].item())
        js = JointState.from_position(sols[s].view(1, -1), joint_names=planner.joint_names)
        got = planner.mp.compute_kinematics(js).tool_poses
        err = torch.linalg.norm(got.position[0, 0, i] - want).item()
        hold_err = torch.linalg.norm(got.position[0, 0, j] - hold).item()
        dot = torch.abs(torch.dot(got.quaternion[0, 0, i], want_q)).clamp(max=1.0).item()
        ang = 2.0 * torch.acos(torch.tensor(dot)).item()
        # A feasible seed wins over a closer one that is in collision or past a limit.
        rank = (0 if ok else 1, err)
        if found is None or rank < found[0]:
            found = (rank, err, ang, ok, hold_err, got.position[0, 0, i].detach().cpu().tolist(),
                     sols[s].detach().cpu().tolist())
    return want.detach().cpu().tolist(), found


def self_note(planner, sol):
    """Pairs whose spheres overlap, including the self-collision buffer."""
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
    n = int(hit.sum().item())
    if n == 0:
        return "self clear"
    cfg = kin.config
    sph = cfg.self_collision_config.collision_pairs[hit].to(torch.int64)
    link_idx = cfg.kinematics_config.link_sphere_idx_map.to(torch.int64)[sph].cpu().tolist()
    names = {v: k for k, v in cfg.kinematics_config.link_name_to_idx_map.items()}
    worst = {}
    for (a, b), d in zip(link_idx, dist[hit].cpu().tolist()):
        key = tuple(sorted((names[a], names[b])))
        worst[key] = max(worst.get(key, 0.0), d)
    pairs = ", ".join(f"{a}-{b} {d * 100:.1f} cm" for (a, b), d in sorted(worst.items(), key=lambda kv: -kv[1]))
    return f"self {n} pairs: {pairs}"


def show(planner, arm, target, q, label):
    want, got = best(planner, arm, target, q)
    if got is None:
        print(f"{label}  no seed  want ({want[0]:.3f},{want[1]:.3f},{want[2]:.3f})", flush=True)
        return
    _, err, ang, ok, hold_err, pos, sol = got
    by = dict(zip(planner.joint_names, sol))
    side = arm[0]
    extra = "" if ok else "  " + self_note(planner, sol)
    print(f"{label}  {'ok' if ok else 'blocked'}  {err * 100:.1f} cm  {math.degrees(ang):.1f} deg  "
          f"other {hold_err * 100:.1f} cm  "
          f"d ({pos[0] - want[0]:+.3f},{pos[1] - want[1]:+.3f},{pos[2] - want[2]:+.3f})  "
          f"body {by['body_yaw_joint']:+.3f}  roll {by[f'shoulder_roll_{side}_joint']:+.3f}  "
          f"pitch {by[f'shoulder_pitch_{side}_joint']:+.3f}  "
          f"stops {stops(planner.joint_names, sol)}{extra}", flush=True)


def main():
    planner = CuroboPlanner(ROBOT / "tiangong2pro_with_hands.urdf", ROBOT / "tienkung_curobo.yml")
    use(planner)
    print(planner.load(None), flush=True)
    q = {n: planner.ready.get(n, 0.0) for n in planner.joint_names}
    for arm in ARM_JOINTS:
        q.update(zip(ARM_JOINTS[arm], raise_steps(arm)[-1]))
    rims = {name: tray_rim(name) for name in ("tray_left", "tray_right")}
    for name, (x, y, z) in rims.items():
        print(f"{name} opening ({x:.3f},{y:.3f},{z:.3f})", flush=True)
    x, y, z = rims["tray_left"]
    print("left tray, yaw 0, fingertip height above the opening", flush=True)
    for tip in (0.10, 0.12, 0.14, 0.16, 0.18, 0.20):
        show(planner, "left", target_at(x, y, z + tip, 0.0), q, f"  +{tip:.2f} m")
    print("left tray, +0.20 m, yaw 0, across the opening", flush=True)
    for yy in (0.55, 0.60, 0.65, 0.69):
        show(planner, "left", target_at(x, yy, z + 0.20, 0.0), q, f"  y {yy:.2f}")
    planner.mp.ik_solver.update_tool_pose_criteria({"right_tcp_link": ToolPoseCriteria.disabled()})
    print("other hand free, left tray centre, yaw 0", flush=True)
    for tip in (0.10, 0.16, 0.18, 0.20):
        show(planner, "left", target_at(x, y, z + tip, 0.0), q, f"  +{tip:.2f} m")
    del planner
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    for limit in (1.45, 1.55):
        robot.PLAN_ROLL_LIMIT = limit
        planner = CuroboPlanner(ROBOT / "tiangong2pro_with_hands.urdf", ROBOT / "tienkung_curobo.yml")
        use(planner)
        print(f"roll limit {limit:.2f}: {planner.load(None)}", flush=True)
        q = {n: planner.ready.get(n, 0.0) for n in planner.joint_names}
        for arm in ARM_JOINTS:
            q.update(zip(ARM_JOINTS[arm], raise_steps(arm)[-1]))
        for tip in (0.16, 0.20):
            show(planner, "left", target_at(x, y, z + tip, 0.0), q, f"  +{tip:.2f} m")
        del planner
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
