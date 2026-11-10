#!/bin/bash
# Starts the whole solution. The evaluator runs: cd /solution && source install/setup.bash && bash run.sh
# ROS_DOMAIN_ID and the DDS profile are already set in the container: do not change them here.
cd "$(dirname "$0")" || exit 1
if [ -f weights/.rgbd-ready ] && [ -f weights/rgbd-b33b2debe3c1/bundle.json ]; then
    export NSSR_RGBD_MODEL_DIR="${NSSR_RGBD_MODEL_DIR:-$PWD/weights/rgbd-b33b2debe3c1}"
    # Full ROS lift/carry/release qualification is tracked in ROADMAP.
    # Enable by default only after full-placement qualification on all three seeds.
    export NSSR_RECORDED_GRASPS="${NSSR_RECORDED_GRASPS:-false}"
    export NSSR_RECORDED_OBJECTS="${NSSR_RECORDED_OBJECTS:-philips}"
    export NSSR_RGBD_ON_DEMAND="${NSSR_RGBD_ON_DEMAND:-true}"
else
    # Failed/incomplete setup must not make the mission retry unavailable poses.
    export NSSR_RECORDED_GRASPS=false
fi
exec ros2 launch hack_bringup solution.launch.py "$@"
