#!/usr/bin/env python3
"""Compile cuRobo / Warp kernels for TienKung into the on-disk cache.

setup.sh runs this in the fresh solution container, before colcon. The cache (~/.cache/warp and the NVRTC
cache) is then there for run.sh in the same container. A second process still builds CUDA graphs; this only
removes the cold compile, which was 25 s of the first IK (docs/ANALYSIS.md, 7.3).
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "hack_motion"))  # the Python package, not the ROS package root

from hack_motion.curobo_planner import CuroboPlanner  # noqa: E402

ROBOT = ROOT / "src" / "hack_motion" / "robot"


def main():
    t0 = time.perf_counter()
    planner = CuroboPlanner(ROBOT / "tiangong2pro_with_hands.urdf", ROBOT / "tienkung_curobo.yml")
    print("[curobo_warmup]", planner.load(None), flush=True)
    stats = planner.warm_cache()
    stats["total_s"] = round(time.perf_counter() - t0, 1)
    print("[curobo_warmup]", stats, flush=True)
    print("CUROBO_WARMUP " + json.dumps(stats), flush=True)
    if not stats["ik_ok"]:
        sys.exit("warmup IK did not reach a sampled configuration")


if __name__ == "__main__":
    main()
