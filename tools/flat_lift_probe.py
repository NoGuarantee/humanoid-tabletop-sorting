#!/usr/bin/env python3
"""Why the leaned soap-dish lift is in collision. cuRobo only, no ROS.

The pose is the one pick_one reached: right TCP (0.355, -0.266, 0.049), roll -45°.
Prints the worst cuboid overlap at the grasp, at the first axis step of the lift,
and at the lift, then tries the straight lift.

  bash tools/remote.sh dev --timeout 900 -- 'python3 tools/flat_lift_probe.py'
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "hack_motion"))
sys.path.insert(0, str(ROOT / "tools"))

from hack_motion.curobo_planner import CuroboPlanner, Unreachable  # noqa: E402
from hack_motion.robot import ARM_JOINTS, TRAYS, table_scene  # noqa: E402
from linear_tray_probe import overlaps, raised, self_margin, spheres  # noqa: E402

ROBOT = ROOT / "src" / "hack_motion" / "robot"
# quat_from_rpy(-45°, 0, 0), the first candidate of the last pick_one
ROLL = NS(x=-math.sin(math.radians(45) / 2), y=0.0, z=0.0, w=math.cos(math.radians(45) / 2))
POSES = (
    ("pre", 0.355, -0.266, 0.149),
    ("grasp", 0.355, -0.266, 0.049),
    ("axis", 0.355, -0.341, 0.124),  # first TCP-axis step of the world-vertical lift
    ("lift", 0.355, -0.266, 0.199),
)


def pose(x, y, z):
    return NS(position=NS(x=x, y=y, z=z), orientation=ROLL)


def show(p, scene, q, name):
    sph, names = spheres(p, q)
    hit = overlaps(sph, names, scene)[:3]
    text = " | ".join(f"{d * 100:+.1f} cm {cub} {link}" for d, cub, link, _ in hit)
    print(f"  {name}: self {self_margin(p, q) * 100:+.2f} cm; {text}", flush=True)


def ik_to(p, q, name, x, y, z):
    g, state = p._goal("right", pose(x, y, z), q)
    result = p.mp.ik_solver.solve_pose(g, current_state=state, return_seeds=p.mp.ik_solver.config.num_seeds)
    sols = p._ranked(result, g, "right", state, 0.01, 0.1)
    print(f"  IK {name}: {len(sols)} within 1 cm", flush=True)
    if not sols:
        return None
    arrived = dict(q)
    arrived.update(zip(p.joint_names, sols[0].tolist()))
    return arrived


def main():
    p = CuroboPlanner(str(ROBOT / "tiangong2pro_with_hands.urdf"), str(ROBOT / "tienkung_curobo.yml"))
    p.trays = tuple(sorted(TRAYS))
    print(p.load(None), flush=True)
    scene = table_scene(p.trays)
    q = raised(p)
    here = q
    for name, x, y, z in POSES:
        arrived = ik_to(p, here, name, x, y, z)
        if arrived is None:
            continue
        show(p, scene, arrived, name)
        if name == "pre":
            here = arrived
    print("linear pre -> grasp, then up", flush=True)
    try:
        names, way, _ = p.plan("right", pose(0.355, -0.266, 0.049), here, "linear")
    except Unreachable as e:
        print(f"  linear descent FAIL {e}", flush=True)
        return
    grasp = dict(here)
    grasp.update(zip(names, way[-1]))
    print(f"  linear descent OK {len(way)} waypoints, start moved {p.start_moved:.3f}", flush=True)
    show(p, scene, grasp, "descent end")
    here = grasp
    z = 0.049
    for dz in (0.15, 0.13):
        z += dz
        try:
            names, way, _ = p.plan("right", pose(0.355, -0.266, z), here, "linear")
        except Unreachable as e:
            print(f"  linear to {z:.3f} FAIL {e}", flush=True)
            return 1
        print(f"  linear to {z:.3f} OK {len(way)} waypoints", flush=True)
        here = dict(here)
        here.update(zip(names, way[-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
