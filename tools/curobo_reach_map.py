#!/usr/bin/env python3
"""Map of the fingers-down TCP poses IK reaches over the table, per hand yaw, from the raised arm pose.

"held": the other hand keeps its pose, as MoveTcp plans. "free": its goal is disabled. x, y span the object spawn
area of the simulator (objects.yaml), used only to judge the result here.

  bash tools/remote.sh dev -- 'python3 tools/curobo_reach_map.py'

torch and cuRobo come from the dev image (remote.sh dev-image), not from curobo_install.sh.

--tray maps the tray area instead (place heights over the trays, scene.yaml), with the three trays in the world.
--arm left mirrors y. --roll-limit R replaces robot.PLAN_ROLL_LIMIT.
"""
import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "hack_motion"))
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria  # noqa: E402
from hack_motion.curobo_planner import CuroboPlanner  # noqa: E402
from hack_motion import robot  # noqa: E402
from hack_motion.robot import ARM_JOINTS, raise_steps  # noqa: E402

ROBOT = ROOT / "src" / "hack_motion" / "robot"
XS = (0.26, 0.32, 0.38, 0.44, 0.50)
YS = (0.0, -0.10, -0.20, -0.30, -0.40, -0.52)
ZS = (0.20, 0.10, 0.05)
YAWS = (0.0, math.pi / 4, math.pi / 2, -math.pi / 4, -math.pi / 2, math.pi)

parser = argparse.ArgumentParser()
parser.add_argument("--tray", action="store_true")
parser.add_argument("--arm", default="right", choices=("left", "right"))
parser.add_argument("--roll-limit", type=float, default=robot.PLAN_ROLL_LIMIT)
args = parser.parse_args()
robot.PLAN_ROLL_LIMIT = args.roll_limit
SIGN = -1 if args.arm == "left" else 1  # YS are for the right arm
if args.tray:
    XS = (0.28, 0.35, 0.42, 0.535)
    YS = (0.0, -0.58, -0.62, -0.66, -0.69, -0.74)
    ZS = (0.32, 0.26, 0.22, 0.20, 0.18, 0.16)
    YAWS = (0.0, math.pi / 2)


def down(yaw):  # rpy (0, 0, yaw), xyzw
    return NS(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))


p = CuroboPlanner(ROBOT / "tiangong2pro_with_hands.urdf", ROBOT / "tienkung_curobo.yml")
if args.tray:
    p.trays = tuple(robot.TRAYS)
p.load(None)
q = {n: p.ready.get(n, 0.0) for n in p.joint_names}
for arm in ARM_JOINTS:
    q.update(zip(ARM_JOINTS[arm], raise_steps(arm)[-1]))
for mode in ("held", "free"):
    if mode == "free":
        other = "right_tcp_link" if args.arm == "left" else "left_tcp_link"
        p.mp.ik_solver.update_tool_pose_criteria({other: ToolPoseCriteria.disabled()})
    for yaw in YAWS:
        print(f"{mode} yaw {yaw:+.2f}  x: " + " ".join(f"{x:.2f}" for x in XS), flush=True)
        for z in ZS:
            for y in YS:
                row = []
                for x in XS:
                    goal, state = p._goal(args.arm, NS(position=NS(x=x, y=y * SIGN, z=z), orientation=down(yaw)), q)
                    r = p.mp.ik_solver.solve_pose(goal, current_state=state, return_seeds=4)
                    row.append(" ok " if p._ranked(r, goal, args.arm, state, 0.01, 0.1) else "  . ")
                print(f"  z {z:+.2f} y {y:+.2f} " + " ".join(row), flush=True)
