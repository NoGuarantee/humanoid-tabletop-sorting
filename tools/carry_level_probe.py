#!/usr/bin/env python3
"""Why a carry finds no level path: the cup carry of sort3 seed 13051 (2026-11-10), cuRobo only, no ROS.

Start joints and goal are from the motion log of that run: the right hand holds the cup over its grasp and goes
to tray_right. Plans the straight leg, the level path with the grasp yaw, with a free yaw, and the plain direct,
each with the measured sort3 trays and without them, and prints what cuRobo says.

  NSSR_WHO=jobs-probe bash tools/remote.sh dev --timeout 900 -- 'python3 tools/carry_level_probe.py'
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "hack_motion"))

from hack_motion.curobo_planner import CuroboPlanner, Unreachable  # noqa: E402
from scene_measured import use  # noqa: E402

ROBOT = ROOT / "src" / "hack_motion" / "robot"
START = [0.714, -1.052, -0.136, 0.165, -1.057, -0.891, -0.77, -0.851, -2.093, -0.02, 2.345, -1.231, 1.779, -0.733,
         -0.921]
GOAL = NS(position=NS(x=0.320, y=-0.635, z=0.326), orientation=NS(x=0.0, y=0.0, z=-0.442, w=0.897))


def attempt(label, fn):
    try:
        names, way, _ = fn()
        print(f"  {label}: {len(way)} waypoints; {p.scene_note}", flush=True)
    except Unreachable as e:
        print(f"  {label}: unreachable: {e}", flush=True)
    except Exception as e:  # a probe: show everything
        print(f"  {label}: {type(e).__name__}: {e}", flush=True)


for trays in (None, ()):
    p = CuroboPlanner(str(ROBOT / "tiangong2pro_with_hands.urdf"), str(ROBOT / "tienkung_curobo.yml"))
    use(p, trays)
    print(p.load(None), flush=True)
    q = dict(zip(p.joint_names, START))
    print(f"trays {'sort3' if trays is None else 'none'}", flush=True)
    p.sync_world(GOAL)
    for free in (False, True):
        attempt(f"level free_yaw={free}", lambda: p._plan_level("right", GOAL, q, free)[0])
    p.carrying = False
    attempt("linear", lambda: p.plan("right", GOAL, q, "linear"))
    attempt("direct", lambda: p.plan("right", GOAL, q, "direct"))
    p.carrying = True
    attempt("carry direct", lambda: p.plan("right", GOAL, q, "direct"))

    # Where the free-yaw carry ends, and the lower into the tray from there with the yaw it chose.
    names, way, _ = p.plan("right", GOAL, q, "direct")
    end = dict(q)
    end.update(zip(names, way[-1]))
    goal, state = p._goal("right", GOAL, end)
    i = goal.tool_frames.index("right_tcp_link")
    w, x, y, z = p.mp.compute_kinematics(state).tool_poses.quaternion[0, 0, i].tolist()
    from hack_motion.robot import top_down, yaw  # noqa: E402
    heading = yaw((x, y, z, w))
    print(f"  carry ends at yaw {math.degrees(heading):.0f} deg (grasp {math.degrees(yaw((0, 0, -0.442, 0.897))):.0f})")
    p.carrying = False
    ox, oy, oz, ow = top_down(heading)
    low = NS(position=NS(x=0.320, y=-0.635, z=0.226), orientation=NS(x=ox, y=oy, z=oz, w=ow))
    attempt("lower 10 cm, kept yaw", lambda: p.plan("right", low, end, "linear"))
    attempt("lower 10 cm, grasp yaw", lambda: p.plan("right", NS(position=low.position, orientation=GOAL.orientation),
                                                         end, "linear"))
