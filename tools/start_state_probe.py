#!/usr/bin/env python3
"""Why a move_tcp start state cannot be planned from. cuRobo only, no ROS and no bench.

Takes the 15 planner joints from an `Unreachable ... from [...]` message (body yaw, left arm, right arm), prints the
self-collision margin and the worst overlap with each cuboid of the sort3 table scene, then plans `direct` to a goal.

  bash tools/remote.sh dev --timeout 600 -- 'python3 tools/start_state_probe.py \
    --start -0.422 -0.518 0.353 -0.989 -1.541 0.265 0.004 0.13 -2.02 0.053 2.95 -0.904 -2.535 1.03 0.48 \
    --goal right 0.311 -0.323 0.232 0.29'
"""
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "hack_motion"))
sys.path.insert(0, str(ROOT / "tools"))

from hack_motion.curobo_planner import CuroboPlanner, Unreachable  # noqa: E402
from hack_motion.robot import ARM_JOINTS, TCP_LINK, TRAYS, ready_pose, table_scene  # noqa: E402
from linear_tray_probe import overlaps, raised, self_margin, spheres, target  # noqa: E402

ROBOT = ROOT / "src" / "hack_motion" / "robot"


def show(p, scene, q, name):
    sph, names = spheres(p, q)
    hit = overlaps(sph, names, scene)[:3]
    text = " | ".join(f"{d * 100:+.1f} cm {cub} {link} at {at}" for d, cub, link, at in hit)
    print(f"{name}: self {self_margin(p, q) * 100:+.2f} cm; {text}", flush=True)


def arm_joints(p, q, arm):
    side = "_r_" if arm == "right" else "_l_"
    return " ".join(f"{q[n]:+.2f}" for n in p.joint_names if side in n)


def chain(p, scene, steps):
    """Plan the steps one after another from the raised arms, as one pick_one run does."""
    q = raised(p)
    for mode, arm, x, y, z, yaw in steps:
        try:
            names, way, _ = p.plan(arm, None if mode == "ready" else target(x, y, z, yaw), q, mode)
        except Unreachable as e:
            print(f"{mode} ({x:.3f},{y:.3f},{z:.3f}) yaw {yaw:.2f} FAIL {e}", flush=True)
            return 1
        q = dict(q)
        q.update(zip(names, way[-1]))
        print(f"{mode:6s} ({x:.3f},{y:.3f},{z:.3f}) yaw {yaw:+.2f}: body {q['body_yaw_joint']:+.2f} "
              f"{arm} [{arm_joints(p, q, arm)}] moved {p.start_moved:.3f}", flush=True)
    show(p, scene, q, "end")
    return 0


# pick_one runs 5 and 6 on 2026-11-07: cartridge at yaw 0.6 into tray_right, then park and the next pre-grasp.
# With --ready the arm goes back to the raised pose before each park, and the release (no plan offline: the real
# arm lags its target, so the replayed carry ends elsewhere) is left out.
READY = ("ready", "right", 0, 0, 0, 0)
RUNS_5_6 = (
    ("direct", "right", 0.40, -0.35, 0.30, 0.0),  # park
    ("direct", "right", 0.371, -0.365, 0.235, 0.6),  # pre
    ("linear", "right", 0.371, -0.365, 0.115, 0.6),  # grasp
    ("linear", "right", 0.371, -0.365, 0.265, 0.6),  # lift
    ("linear", "right", 0.371, -0.365, 0.326, 0.6),  # carry up
    ("linear", "right", 0.347, -0.690, 0.326, 0.6),  # carry over
    ("linear", "right", 0.347, -0.690, 0.226, 0.6),  # release
    ("linear", "right", 0.347, -0.690, 0.326, 0.6),  # retreat
    ("direct", "right", 0.40, -0.35, 0.30, 0.0),  # park of run 6
    ("direct", "right", 0.311, -0.323, 0.232, 0.29),  # pre of run 6
)


def why(p, q, name):
    """Each constraint and hybrid cost of the graph planner's feasibility check (the one behind "Start or End state
    in collision") for one joint dict: name and the largest value over the horizon."""
    import torch

    gp = p.mp.graph_planner
    x = torch.tensor([[q[n] for n in p.joint_names]], device="cuda", dtype=torch.float32)
    buf = gp._max_act_buffer
    buf[:] = x.unsqueeze(1)
    cc = gp.feasibility_rollout.compute_metrics_from_action(buf).costs_and_constraints
    feasible = bool(cc.get_feasible(include_all_hybrid=False, sum_horizon=True)[0].item())
    parts = []
    for group in (cc.constraints, cc.hybrid_costs_constraints):
        if group is None:
            continue
        for n, v in zip(group.names, group.values):
            if v is not None:
                parts.append(f"{n} {float(v[0].max().item()):.4g}")
    print(f"{name}: feasible {feasible}; " + ", ".join(parts), flush=True)


def tcp(p, q, arm):
    import torch
    from curobo.types import JointState

    t = torch.tensor([[q[n] for n in p.joint_names]], device="cuda", dtype=torch.float32)
    poses = p.mp.compute_kinematics(JointState.from_position(t, joint_names=p.joint_names)).tool_poses
    i = poses.as_goal().tool_frames.index(TCP_LINK[arm])
    return [round(v, 3) for v in poses.position[0, 0, i].tolist()]


def sweep(p, scene):
    """From both arms at ready_pose: pre-grasp (direct) and grasp (linear 0.12 down) over the spawn area, then the
    carry to the tray and back to ready. Objects spawn at x 0.26..0.50, |y| <= 0.52 (pools.yaml area_x, area_y)."""
    start = raised(p)
    for arm in ARM_JOINTS:
        start.update(zip(ARM_JOINTS[arm], ready_pose(arm)))
    show(p, scene, start, "ready both")
    for arm in ARM_JOINTS:
        print(f"{arm} ready tcp {tcp(p, start, arm)}", flush=True)
    ok = total = 0
    for arm, sign in (("right", -1.0), ("left", 1.0)):
        for x in (0.28, 0.38, 0.48):
            for y in (0.05, 0.25, 0.45):
                for yaw in (-0.6, 0.0, 0.6):
                    total += 1
                    q = dict(start)
                    steps = (("direct", 0.23), ("linear", 0.11))
                    try:
                        for mode, z in steps:
                            names, way, _ = p.plan(arm, target(x, sign * y, z, yaw), q, mode)
                            q.update(zip(names, way[-1]))
                        ok += 1
                        res = "ok"
                    except Unreachable as e:
                        res = f"FAIL {mode}: {str(e)[:90]}"
                    print(f"{arm} ({x:.2f},{sign * y:+.2f}) yaw {yaw:+.1f}: {res}", flush=True)
                    if res == "ok" and x == 0.38 and yaw == 0.0:
                        try:
                            for mode, z in (("linear", 0.33),):
                                names, way, _ = p.plan(arm, target(x, sign * y, z, yaw), q, mode)
                                q.update(zip(names, way[-1]))
                            names, way, _ = p.plan(arm, target(0.347, sign * 0.69, 0.33, yaw), q, "linear")
                            q.update(zip(names, way[-1]))
                            names, way, _ = p.plan(arm, None, q, "ready")
                            print(f"  carry and ready from it: ok, {len(way)} waypoints", flush=True)
                        except Unreachable as e:
                            print(f"  carry and ready FAIL {str(e)[:120]}", flush=True)
    print(f"pre-grasp and grasp from ready: {ok}/{total}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, nargs=15)
    ap.add_argument("--goal", nargs=5, metavar=("ARM", "X", "Y", "Z", "YAW"))
    ap.add_argument("--chain", action="store_true", help="replay pick_one runs 5 and 6 from the raised arms")
    ap.add_argument("--ready", choices=("left", "right"), help="plan this arm back to ready_pose before --goal")
    ap.add_argument("--sweep", action="store_true", help="pre-grasps over the spawn area from both arms at ready_pose")
    args = ap.parse_args()
    p = CuroboPlanner(str(ROBOT / "tiangong2pro_with_hands.urdf"), str(ROBOT / "tienkung_curobo.yml"))
    p.trays = tuple(sorted(TRAYS))
    print(p.load(None), flush=True)
    scene = table_scene(p.trays)
    if args.sweep:
        return sweep(p, scene)
    if args.chain:
        steps = RUNS_5_6
        if args.ready:
            steps = (READY, *RUNS_5_6[:6], READY, *RUNS_5_6[8:])
        return chain(p, scene, steps)
    if args.start is None:
        sys.exit("--start or --chain")
    q = raised(p)
    if len(p.joint_names) != 15:
        sys.exit(f"planner has {len(p.joint_names)} joints: {p.joint_names}")
    q.update(zip(p.joint_names, args.start))
    for name, value in zip(p.joint_names, args.start):
        print(f"  {name} {value:+.3f}", flush=True)
    show(p, scene, q, "start")
    why(p, q, "start")
    if args.ready:
        goal = dict(q)
        goal.update(zip(ARM_JOINTS[args.ready], ready_pose(args.ready)))
        goal["body_yaw_joint"] = p.ready.get("body_yaw_joint", 0.0)
        show(p, scene, goal, "ready goal")
        why(p, goal, "ready goal")
        try:
            names, way, _ = p.plan(args.ready, None, q, "ready")
        except Unreachable as e:
            print(f"ready FAIL {e}", flush=True)
            return 1
        q = dict(q)
        q.update(zip(names, way[-1]))
        print(f"ready OK {len(way)} waypoints, start moved {p.start_moved:.3f} rad", flush=True)
        show(p, scene, q, "ready")
    if not args.goal:
        return 0
    arm, x, y, z, yaw = args.goal[0], *map(float, args.goal[1:])
    try:
        names, way, _ = p.plan(arm, target(x, y, z, yaw), q, "direct")
    except Unreachable as e:
        print(f"direct FAIL {e}", flush=True)
        return 1
    end = dict(q)
    end.update(zip(names, way[-1]))
    print(f"direct OK {len(way)} waypoints, start moved {p.start_moved:.3f} rad", flush=True)
    show(p, scene, end, "end")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
