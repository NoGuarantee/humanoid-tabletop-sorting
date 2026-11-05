#!/usr/bin/env python3
"""Straight lift and a drop into each tray, cuRobo only, no ROS and no bench.

The lift of move_tcp_check (+5 cm from the first goal, and from the low goal) is planned with the
sort3 trays and without them. Then each opening is approached and the hand is lowered to the place
release. `start moved` is how far the planner had to step off the self-collision buffer: above
about 0.08 rad that step throws a held object.

  bash tools/remote.sh dev --timeout 900 -- 'python3 tools/linear_tray_probe.py'
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "hack_motion"))

from hack_motion.curobo_planner import CuroboPlanner, Unreachable  # noqa: E402
from hack_motion.robot import ARM_JOINTS, TRAYS, raise_steps, table_scene, tray_rim  # noqa: E402

ROBOT = ROOT / "src" / "hack_motion" / "robot"
GOAL = ("right", 0.40, -0.35, 0.30, 0.0)
LOW = ("right", 0.40, -0.35, 0.10, 0.0)


def target(x, y, z, yaw):
    return NS(position=NS(x=x, y=y, z=z), orientation=NS(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)))


def raised(p):
    q = {n: p.ready.get(n, 0.0) for n in p.joint_names}
    for arm in ARM_JOINTS:
        q.update(zip(ARM_JOINTS[arm], raise_steps(arm)[-1]))
    return q


def spheres(p, q_by_name):
    """Robot spheres [S, 4] and the link name of each, for one joint dict."""
    import torch
    from curobo.types import JointState

    q = torch.tensor([[q_by_name[n] for n in p.joint_names]], device="cuda", dtype=torch.float32)
    state = JointState.from_position(q, joint_names=p.joint_names)
    sph = p._self_kin.compute_kinematics(state).robot_spheres.detach().reshape(-1, 4)
    kin = p._self_kin.config.kinematics_config
    index = kin.link_sphere_idx_map.tolist()
    links = {i: name for name, i in kin.link_name_to_idx_map.items()}
    names = [links[index[i]] if i < len(index) else "?" for i in range(sph.shape[0])]
    return sph, names


def overlaps(sph, names, scene):
    """Worst sphere-in-cuboid overlap (m) per cuboid. Positive means the sphere is inside."""
    import torch

    out = []
    for name, cub in scene["cuboid"].items():
        dims = torch.tensor(cub["dims"], device=sph.device, dtype=sph.dtype)
        pose = cub["pose"]
        w, x, y, z = pose[3], pose[4], pose[5], pose[6]
        rot = torch.tensor([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ], device=sph.device, dtype=sph.dtype)
        center = torch.tensor(pose[:3], device=sph.device, dtype=sph.dtype)
        local = (sph[:, :3] - center) @ rot
        half = dims / 2
        # distance outside the box along each axis; negative is inside
        outside = (local.abs() - half).clamp(min=0)
        gap = torch.linalg.norm(outside, dim=1) - sph[:, 3]
        inside = (local.abs() <= half).all(dim=1)
        depth = torch.where(inside, (half - local.abs()).min(dim=1).values + sph[:, 3], -gap)
        i = int(depth.argmax().item())
        out.append((float(depth[i].item()), name, names[i], [round(v, 3) for v in sph[i, :3].tolist()]))
    return sorted(out, reverse=True)


def reach(p, q, goal):
    arm, x, y, z, yaw = goal
    names, way, _ = p.plan(arm, target(x, y, z, yaw), q, "direct")
    arrived = dict(q)
    arrived.update(zip(names, way[-1]))
    print(f"  direct {goal}: {len(way)} waypoints, start moved {p.start_moved:.3f} rad", flush=True)
    return arrived


def lift(p, q, goal, dz):
    arm, x, y, z, yaw = goal
    try:
        names, way, _ = p.plan(arm, target(x, y, z + dz, yaw), q, "linear")
    except Unreachable as e:
        print(f"  linear +{dz:.2f} m FAIL {e}", flush=True)
        return False
    print(f"  linear +{dz:.2f} m OK {len(way)} waypoints, start moved {p.start_moved:.3f} rad", flush=True)
    return True


def pose_collision(p, q, goal, scene):
    """IK seeds of this TCP pose: how close in joint space, and the worst cuboid overlap."""
    arm, x, y, z, yaw = goal
    g, state = p._goal(arm, target(x, y, z, yaw), q)
    result = p.mp.ik_solver.solve_pose(g, current_state=state, return_seeds=p.mp.ik_solver.config.num_seeds)
    sols = p._ranked(result, g, arm, state, 0.01, 0.1)
    print(f"  IK of the lift goal: {len(sols)} within 1 cm", flush=True)
    current = state.position.detach().reshape(-1)
    for sol in sols[:3]:
        dist = float((sol - current).norm().item())
        arrived = dict(zip(p.joint_names, sol.tolist()))
        sph, names = spheres(p, arrived)
        hit = overlaps(sph, names, scene)[0]
        print(f"    {dist:.2f} rad from here, worst overlap {hit[0] * 100:.1f} cm "
              f"{hit[1]} by {hit[2]} at {hit[3]}", flush=True)


def self_margin(p, q_by_name):
    """Metres of self-collision overlap. Positive means the spheres already intersect."""
    import torch
    from curobo.types import JointState

    q = torch.tensor([[q_by_name[n] for n in p.joint_names]], device="cuda", dtype=torch.float32)
    state = JointState.from_position(q, joint_names=p.joint_names)
    p._self_cost.setup_batch_tensors(1, 1)
    p._self_cost.forward(p._self_kin.compute_kinematics(state).robot_spheres)
    return float(p._self_cost._pair_distance.max().item())


def descend(p, q, name, arm):
    """Direct to the pose above the opening, then a straight drop to the release height."""
    rx, ry, rz = tray_rim(name)
    for yaw in (0.0, 1.57, -1.57):
        approach = (arm, rx, ry, rz + 0.41, yaw)  # palm: fingertips 20 cm above the rim
        release = (arm, rx, ry, rz + 0.31, yaw)  # palm: fingertips 10 cm above the rim
        try:
            here = reach(p, q, approach)
        except Unreachable as e:
            print(f"  yaw {yaw:+.2f} approach FAIL {e}", flush=True)
            continue
        try:
            p.plan(arm, target(rx, ry, rz + 0.31, yaw), here, "linear")
            print(f"  yaw {yaw:+.2f} release linear OK, start moved {p.start_moved:.3f}", flush=True)
        except Unreachable as e:
            print(f"  yaw {yaw:+.2f} release linear FAIL {e}", flush=True)


def main():
    sets = [(), tuple(sorted(TRAYS))]
    for trays in sets:
        p = CuroboPlanner(str(ROBOT / "tiangong2pro_with_hands.urdf"), str(ROBOT / "tienkung_curobo.yml"))
        p.trays = trays
        print(p.load(None), flush=True)
        scene = table_scene(trays)
        q = raised(p)
        print(f"trays {list(trays)}", flush=True)
        for goal in (GOAL, LOW):
            try:
                here = reach(p, q, goal)
            except Unreachable as e:
                print(f"  direct {goal} FAIL {e}", flush=True)
                continue
            sph, names = spheres(p, here)
            hit = overlaps(sph, names, scene)[:4]
            print("  at the goal:", " | ".join(
                f"{d * 100:.1f} cm {n} {link}" for d, n, link, _ in hit), flush=True)
            print(f"  self overlap {self_margin(p, here) * 100:.2f} cm", flush=True)
            lift(p, here, goal, 0.05)
            arm, x, y, z, yaw = goal
            pose_collision(p, here, (arm, x, y, z + 0.05, yaw), scene)
        if trays:
            for name, arm in (("tray_right", "right"), ("tray_center", "right"), ("tray_left", "left")):
                print(f"descend {arm} into {name}", flush=True)
                descend(p, q, name, arm)


if __name__ == "__main__":
    main()
