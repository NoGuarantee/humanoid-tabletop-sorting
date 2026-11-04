#!/usr/bin/env python3
"""Map of the fingers-down TCP poses IK reaches over the table, per hand yaw, from the raised arm pose.

"held": the other hand keeps its pose, as MoveTcp plans. "free": its goal is disabled. x, y span the object spawn
area of the simulator (objects.yaml), used only to judge the result here.

  bash tools/remote.sh dev -- 'bash tools/curobo_install.sh && python3 tools/curobo_reach_map.py'
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "hack_motion"))
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria  # noqa: E402
from hack_motion.curobo_planner import CuroboPlanner  # noqa: E402
from hack_motion.robot import ARM_JOINTS, raise_steps  # noqa: E402

ROBOT = ROOT / "src" / "hack_motion" / "robot"
XS = (0.26, 0.32, 0.38, 0.44, 0.50)
YS = (0.0, -0.10, -0.20, -0.30, -0.40, -0.52)
ZS = (0.20, 0.10, 0.05)
YAWS = (0.0, math.pi / 4, math.pi / 2, -math.pi / 4, -math.pi / 2, math.pi)


def down(yaw):  # rpy (0, 0, yaw), xyzw
    return NS(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))


p = CuroboPlanner(ROBOT / "tiangong2pro_with_hands.urdf", ROBOT / "tienkung_curobo.yml")
p.load(None)
q = {n: p.ready.get(n, 0.0) for n in p.joint_names}
for arm in ARM_JOINTS:
    q.update(zip(ARM_JOINTS[arm], raise_steps(arm)[-1]))
for mode in ("held", "free"):
    if mode == "free":
        p.mp.ik_solver.update_tool_pose_criteria({"left_tcp_link": ToolPoseCriteria.disabled()})
    for yaw in YAWS:
        print(f"{mode} yaw {yaw:+.2f}  x: " + " ".join(f"{x:.2f}" for x in XS), flush=True)
        for z in ZS:
            for y in YS:
                row = []
                for x in XS:
                    goal, state = p._goal("right", NS(position=NS(x=x, y=y, z=z), orientation=down(yaw)), q)
                    r = p.mp.ik_solver.solve_pose(goal, current_state=state, return_seeds=4)
                    row.append(" ok " if p._pick(r, goal, "right", state, 0.01, 0.1) is not None else "  . ")
                print(f"  z {z:+.2f} y {y:+.2f} " + " ".join(row), flush=True)
