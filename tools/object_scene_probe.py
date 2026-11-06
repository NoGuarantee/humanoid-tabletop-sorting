#!/usr/bin/env python3
"""Cuboids of neighbouring objects, and one plan that has to go around one.

Without --plan this only checks the geometry, no GPU: the box is tilted with the table, a grasp aimed at an
object leaves that object out, and a neighbour stays in.

  python3 tools/object_scene_probe.py
  bash tools/remote.sh dev --timeout 900 -- 'python3 tools/object_scene_probe.py --plan'

--plan needs cuRobo. It plans a move with an empty world, puts a box on a sphere of that path, and plans
again: the new path has to stay outside the box. Then the same box under the fingertips is left out and the
move still plans.
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "hack_motion"))

from hack_motion.robot import (  # noqa: E402
    MAX_OBJECTS, OBJECT_MARGIN, TRAYS, _object_axes, fingertip, table_z, world_scene)

ROBOT = ROOT / "src" / "hack_motion" / "robot"


def obj(ident, x, y, z, yaw, sx, sy, sz):
    return (ident, x, y, z, yaw, sx, sy, sz, z + sz / 2)


def check_geometry():
    tilt = math.atan(0.07)
    want = (math.cos(tilt / 2), 0.0, -math.sin(tilt / 2), 0.0)
    scene, left = world_scene((), [obj("flat", 0.40, -0.30, 0.0, 0.0, 0.10, 0.04, 0.06)])
    pose = scene["cuboid"]["flat"]["pose"]
    dot = abs(sum(a * b for a, b in zip(pose[3:], want)))
    if dot < 1 - 1e-6:
        raise SystemExit(f"yaw 0 should be the table tilt, quaternion dot {dot}")
    if left:
        raise SystemExit(f"nothing to leave out, got {left}")
    dims = scene["cuboid"]["flat"]["dims"]
    if any(abs(a - b) > 1e-9 for a, b in zip(dims, (0.10 + 2 * OBJECT_MARGIN, 0.04 + 2 * OBJECT_MARGIN,
                                                    0.06 + 2 * OBJECT_MARGIN))):
        raise SystemExit(f"margin not applied: {dims}")

    long, _, _ = _object_axes(math.pi / 2)
    turned, _ = world_scene((), [obj("long", 0.40, 0.0, 0.0, math.pi / 2, 0.12, 0.03, 0.05)])
    got = _column(turned["cuboid"]["long"]["pose"])
    if abs(abs(sum(a * b for a, b in zip(got, long))) - 1) > 1e-6:
        raise SystemExit(f"yaw pi/2 long axis {got}, want {long}")

    here = obj("here", 0.42, -0.20, -0.05, 0.4, 0.08, 0.05, 0.07)
    there = obj("there", 0.42, 0.25, -0.05, 0.0, 0.08, 0.05, 0.07)
    goal = fingertip((0.42, -0.20, -0.05 + 0.21), (0.0, 0.0, 0.0, 1.0))
    scene, left = world_scene(tuple(sorted(TRAYS)), [here, there], goal)
    if left != ["here"] or "there" not in scene["cuboid"] or "here" in scene["cuboid"]:
        raise SystemExit(f"exclusion: left out {left}, cuboids {list(scene['cuboid'])}")
    if not all(name in scene["cuboid"] for name in ("table",) + tuple(sorted(TRAYS))):
        raise SystemExit(f"table or a tray missing: {list(scene['cuboid'])}")

    many = [obj(f"o{i}", 0.3, -0.4 + 0.05 * i, 0.0, 0.0, 0.02, 0.02, 0.04) for i in range(MAX_OBJECTS + 3)]
    scene, left = world_scene((), many, (0.3, -2.0, 0.0))
    far = f"o{MAX_OBJECTS + 2}"
    if left or len(scene["cuboid"]) != 1 + MAX_OBJECTS or "o0" not in scene["cuboid"] or far in scene["cuboid"]:
        raise SystemExit(f"kept {sorted(scene['cuboid'])}, left out {left}")
    print("geometry ok", flush=True)


def _column(pose):
    """First column of the rotation: the cuboid's long axis."""
    w, x, y, z = pose[3:]
    return (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))


def target(x, y, z, yaw=0.0):
    return NS(position=NS(x=x, y=y, z=z),
              orientation=NS(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)))


def spheres_at(planner, q_by_name):
    import torch
    from curobo.types import JointState

    q = torch.tensor([[q_by_name[n] for n in planner.joint_names]], device="cuda", dtype=torch.float32)
    state = JointState.from_position(q, joint_names=planner.joint_names)
    sph = planner.mp.compute_kinematics(state).robot_spheres.detach().reshape(-1, 4)
    return sph


def worst_overlap(sph, cub):
    """Metres a sphere centre sits inside the cuboid, minus its radius. Positive is a hit."""
    import torch

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
    outside = (local.abs() - half).clamp(min=0)
    gap = torch.linalg.norm(outside, dim=1) - sph[:, 3]
    inside = (local.abs() <= half).all(dim=1)
    depth = torch.where(inside, (half - local.abs()).min(dim=1).values + sph[:, 3], -gap)
    return float(depth.max().item())


def raised(planner):
    from hack_motion.robot import ARM_JOINTS, raise_steps

    q = {n: planner.ready.get(n, 0.0) for n in planner.joint_names}
    for arm in ARM_JOINTS:
        q.update(zip(ARM_JOINTS[arm], raise_steps(arm)[-1]))
    return q


def joints_of(base, names, waypoint):
    q = dict(base)
    q.update(zip(names, waypoint))
    return q


def plan(planner, q, goal):
    from hack_motion.curobo_planner import Unreachable

    arm, x, y, z = goal
    try:
        names, way, _ = planner.plan(arm, target(x, y, z), q, "direct")
    except Unreachable as e:
        print(f"  FAIL {e}", flush=True)
        return None, None
    print(f"  {planner.scene_note}, {len(way)} waypoints", flush=True)
    return names, way


def path_overlap(planner, q, names, way, cub):
    step = max(1, len(way) // 16)
    worst = -math.inf
    for waypoint in way[::step] + [way[-1]]:
        hit = worst_overlap(spheres_at(planner, joints_of(q, names, waypoint)), cub)
        worst = max(worst, hit)
    return worst


def blocking_box(planner, q, names, way):
    """A cuboid around a mid-path sphere, clear of the start and the goal. None if the path stays in the body."""
    start = spheres_at(planner, joints_of(q, names, way[0]))
    goal = spheres_at(planner, joints_of(q, names, way[-1]))
    ends = torch_cat(start, goal)
    mid = way[len(way) // 2]
    sph = spheres_at(planner, joints_of(q, names, mid))
    for i in range(sph.shape[0]):
        centre = sph[i, :3]
        radius = float(sph[i, 3].item())
        if float(centre[0]) < 0.30 or float(centre[2]) < table_z(float(centre[0])) + 0.05:
            continue
        dist = (ends[:, :3] - centre).norm(dim=1) - ends[:, 3]
        if float(dist.min().item()) < radius + 0.06:
            continue
        side = 2 * (radius + 0.015)
        x, y, z = (float(v) for v in centre.tolist())
        return obj("block", x, y, z, 0.0, side, side, side)
    return None


def torch_cat(a, b):
    import torch
    return torch.cat([a, b], dim=0)


def check_plan():
    from hack_motion.curobo_planner import CuroboPlanner

    planner = CuroboPlanner(str(ROBOT / "tiangong2pro_with_hands.urdf"), str(ROBOT / "tienkung_curobo.yml"))
    planner.trays = tuple(sorted(TRAYS))
    print(planner.load(None), flush=True)
    q = raised(planner)
    goal = ("right", 0.48, -0.32, 0.22)
    print("empty world", flush=True)
    names, way = plan(planner, q, goal)
    if way is None:
        raise SystemExit("the goal is unreachable with nothing on the table")
    box = blocking_box(planner, q, names, way)
    if box is None:
        raise SystemExit("no mid-path sphere clear of the start and the goal")
    print(f"box at {[round(v, 3) for v in box[1:4]]} size {box[5]:.3f}", flush=True)
    bare = world_scene((), [box])[0]["cuboid"]["block"]
    before = path_overlap(planner, q, names, way, bare)
    print(f"  empty path inside that box by {before * 100:.1f} cm", flush=True)
    if before <= 0.0:
        raise SystemExit("the box does not meet the empty path")

    planner.objects = (box,)
    print("with the box", flush=True)
    names, way = plan(planner, q, goal)
    if way is None:
        raise SystemExit("no way around the box")
    if "left out" in planner.scene_note:
        raise SystemExit(f"the box is not the goal and was left out: {planner.scene_note}")
    after = path_overlap(planner, q, names, way, planner._scene["cuboid"]["block"])
    print(f"  path overlap {after * 100:.1f} cm", flush=True)
    if after > 0.005:
        raise SystemExit("the path still goes through the box")

    grasp = obj("grasp", goal[1], goal[2], goal[3] - 0.21, 0.0, 0.08, 0.05, 0.08)
    neighbour = obj("neighbour", goal[1], goal[2] + 0.18, grasp[3], 0.0, 0.06, 0.06, 0.06)
    planner.objects = (grasp, neighbour)
    print("grasp on the object, neighbour beside it", flush=True)
    names, way = plan(planner, q, goal)
    if way is None:
        raise SystemExit("leaving the grasped object out still has no plan")
    if planner.scene_note.find("grasp") < 0 or "neighbour" not in planner._scene["cuboid"]:
        raise SystemExit(f"scene {planner.scene_note}, cuboids {list(planner._scene['cuboid'])}")
    print("plan ok", flush=True)


def main():
    check_geometry()
    if "--plan" in sys.argv:
        check_plan()


if __name__ == "__main__":
    main()
