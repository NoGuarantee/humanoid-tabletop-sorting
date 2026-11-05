#!/usr/bin/env python3
"""Why does the goal after the head tilt sometimes have no IK seed? cuRobo only, no ROS, no bench.

Plays the goals of move_tcp_check through the planner with perfect tracking, then solves the tilted goal: head level
and tilted, other hand held and free, several tries, and lists the sphere pairs the seeds collide on.

  bash tools/remote.sh dev -- 'python3 tools/tilt_goal_probe.py [--low] [--tries 8]'
"""
import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "hack_motion"))
sys.path.insert(0, str(ROOT / "tools"))
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria  # noqa: E402
from hack_motion.curobo_planner import CuroboPlanner, TCP_LINK  # noqa: E402
from hack_motion.robot import ARM_JOINTS, raise_steps  # noqa: E402

ROBOT = ROOT / "src" / "hack_motion" / "robot"
GOALS = [("right", 0.40, -0.35, 0.30, 0.0), ("right", 0.30, -0.15, 0.25, 0.8), ("right", 0.45, -0.30, 0.25, 1.57),
         ("left", 0.40, 0.35, 0.30, 0.0), ("left", 0.30, 0.15, 0.25, -0.8)]
LOW_GOALS = [("right", 0.40, -0.35, 0.10, 0.0), ("right", 0.30, -0.15, 0.10, 0.8), ("right", 0.45, -0.30, 0.08, 1.57),
             ("left", 0.40, 0.35, 0.10, 0.0), ("left", 0.30, 0.15, 0.07, -0.8)]
TILTED = ("right", 0.40, -0.20, 0.25, 0.0)
OTHER = {"left": "right", "right": "left"}
HEIGHTS = [("right", 0.28, -0.33, 0.0), ("right", 0.38, -0.33, 0.0), ("right", 0.50, -0.33, 0.0), ("left", 0.28, 0.33, 0.0),
           ("left", 0.38, 0.33, 0.0), ("left", 0.50, 0.33, 0.0), ("right", 0.45, -0.30, 1.57), ("left", 0.30, 0.15, -0.8)]


def target(x, y, z, yaw):
    return NS(position=NS(x=x, y=y, z=z), orientation=NS(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)))


def hits(p, sols):
    """Counter of sphere pairs (by link) with positive overlap over the rows of sols [N x dof]; rows in self-collision."""
    import torch
    from curobo.types import JointState

    n = sols.shape[0]
    state = JointState.from_position(sols, joint_names=p.joint_names)
    p._self_cost.setup_batch_tensors(n, 1)
    p._self_cost.forward(p._self_kin.compute_kinematics(state).robot_spheres)
    dist = p._self_cost._pair_distance.view(n, -1)
    cfg = p._self_kin.config.self_collision_config
    pairs = cfg.collision_pairs.to(torch.int64)
    kin = p._self_kin.config.kinematics_config
    index = kin.link_sphere_idx_map.tolist()
    links = {i: name for name, i in kin.link_name_to_idx_map.items()}
    out = Counter()
    for row, col in (dist > 0).nonzero().tolist():
        a, b = pairs[col].tolist()
        out[tuple(sorted((links[index[a]], links[index[b]])))] += 1
    return out, int((dist > 0).any(dim=1).sum().item())


def solve(p, q, goal, tries, free):
    p.mp.ik_solver.update_tool_pose_criteria({TCP_LINK[OTHER[goal[0]]]: ToolPoseCriteria.disabled() if free
                                              else ToolPoseCriteria()})
    arm, x, y, z, yaw = goal
    counts = []
    pairs = Counter()
    collide = 0
    for _ in range(tries):
        g, state = p._goal(arm, target(x, y, z, yaw), q)
        r = p.mp.ik_solver.solve_pose(g, current_state=state, return_seeds=p.mp.ik_solver.config.num_seeds)
        counts.append(len(p._ranked(r, g, arm, state, 0.01, 0.1)))
        sols = r.solution.detach().reshape(-1, r.solution.shape[-1])
        got, n = hits(p, sols)
        pairs += got
        collide += n
    p.mp.ik_solver.update_tool_pose_criteria({TCP_LINK[OTHER[goal[0]]]: ToolPoseCriteria()})
    print(f"    {'free' if free else 'held'}: solutions per try {counts}; "
          f"seeds in self-collision {collide}/{tries * sols.shape[0]}; top pairs {pairs.most_common(6)}", flush=True)


def heights(p, q):
    """The lowest TCP the planner accepts over the tilted table: IK solutions per height, top_down."""
    for arm, x, y, yaw in HEIGHTS:
        row = []
        for z in [0.13 - 0.005 * i for i in range(21)]:
            g, state = p._goal(arm, target(x, y, z, yaw), q)
            r = p.mp.ik_solver.solve_pose(g, current_state=state, return_seeds=p.mp.ik_solver.config.num_seeds)
            row.append(f"{z:.3f}:{len(p._ranked(r, g, arm, state, 0.01, 0.1))}")
        print(f"{arm} ({x}, {y}) yaw {yaw}: table {-0.1792 + 0.07 * x:.4f}  " + " ".join(row), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--low", action="store_true")
    parser.add_argument("--tries", type=int, default=8)
    parser.add_argument("--heights", action="store_true", help="the lowest TCP the planner accepts, then stop")
    args = parser.parse_args()
    p = CuroboPlanner(ROBOT / "tiangong2pro_with_hands.urdf", ROBOT / "tienkung_curobo.yml")
    print(p.load(None), flush=True)
    q = {n: p.ready.get(n, 0.0) for n in p.joint_names}
    for arm in ARM_JOINTS:
        q.update(zip(ARM_JOINTS[arm], raise_steps(arm)[-1]))
    for goal in (LOW_GOALS if args.low else GOALS):
        arm, x, y, z, yaw = goal
        names, way, _ = p.plan(arm, target(x, y, z, yaw), q, "direct")
        q.update(zip(names, way[-1]))
        print(f"{goal}: {len(way)} waypoints, moved {p.start_moved:.3f}", flush=True)
    print("q " + " ".join(f"{n.replace('_joint', '')} {v:+.3f}" for n, v in q.items()), flush=True)
    if args.heights:
        heights(p, q)
        return
    for head in (0.0, 0.2, 0.4):
        print(p.load({"head_pitch_joint": head}), flush=True)
        import torch
        now = torch.tensor([[q[n] for n in p.joint_names]], device="cuda", dtype=torch.float32)
        print(f"  head {head}: start in self-collision pairs {hits(p, now)[0].most_common(6)}", flush=True)
        solve(p, q, TILTED, args.tries, False)
        solve(p, q, TILTED, args.tries, True)


if __name__ == "__main__":
    main()
