#!/usr/bin/env python3
"""Time CuroboPlanner.plan without the simulator: direct, linear, direct again, with one CUDA graph mode.

  bash tools/remote.sh dev -- 'bash tools/curobo_install.sh && python3 tools/curobo_bench.py reset [head_pitch]'

Mode: reset (graphs, re-captured on shape change) or off. Each plan starts at the last waypoint of the previous one.
"""
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "hack_motion"))

from hack_motion.curobo_planner import CuroboPlanner, Unreachable  # noqa: E402

ROBOT = ROOT / "src" / "hack_motion" / "robot"
DOWN = NS(x=1.0, y=0.0, z=0.0, w=0.0)  # rpy (pi, 0, 0): palm down, as grasp_planner.top_down(0)
STEPS = [
    ("right", 0.40, -0.35, 0.30, "direct"),
    ("right", 0.40, -0.35, 0.35, "linear"),
    ("right", 0.35, -0.40, 0.22, "direct"),
    ("right", 0.45, -0.30, 0.38, "direct"),
    ("right", 0.45, -0.30, 0.43, "linear"),
    ("left", 0.40, 0.35, 0.30, "direct"),
    ("right", 0.40, -0.35, 0.30, "direct"),
]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "reset"
    head = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    planner = CuroboPlanner(ROBOT / "tiangong2pro_with_hands.urdf", ROBOT / "tienkung_curobo.yml")
    t0 = time.perf_counter()
    print(planner.load(None, graphs=mode), flush=True)
    print("warm_cache", planner.warm_cache(), f"{time.perf_counter() - t0:.1f} s", flush=True)
    q = {name: planner.ready.get(name, 0.0) for name in planner.joint_names}
    for rnd in range(2):
        if rnd == 1:  # brain tilts the head to look at the table: the planner is rebuilt with the new lock
            t0 = time.perf_counter()
            print(planner.load({"head_pitch_joint": head}, graphs=mode).split(" joints")[0],
                  f"{time.perf_counter() - t0:.1f} s", flush=True)
        for arm, x, y, z, how in STEPS:
            target = NS(position=NS(x=x, y=y, z=z), orientation=DOWN)
            t0 = time.perf_counter()
            try:
                names, waypoints, dt = planner.plan(arm, target, q, how)
                q.update(zip(names, waypoints[-1]))
                out = f"ok {len(waypoints)} pts dt {dt}"
            except Unreachable as e:
                out = f"unreachable: {e}"
            except Exception as e:
                out = f"ERROR {type(e).__name__}: {e}"
            print(f"{mode} r{rnd} {arm} ({x:.2f},{y:.2f},{z:.2f}) {how:6s} "
                  f"{(time.perf_counter() - t0) * 1e3:7.0f} ms  {out}", flush=True)


if __name__ == "__main__":
    main()
