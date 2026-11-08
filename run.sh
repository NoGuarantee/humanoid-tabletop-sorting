#!/bin/bash
# Starts the whole solution. The evaluator runs: cd /solution && source install/setup.bash && bash run.sh
# ROS_DOMAIN_ID and the DDS profile are already set in the container: do not change them here.
cd "$(dirname "$0")" || exit 1
if [ -z "${NSSR_RGBD_MODEL_DIR:-}" ] && [ -f weights/.rgbd-ready ] &&
   [ -f weights/rgbd-b33b2debe3c1/bundle.json ]; then
    export NSSR_RGBD_MODEL_DIR="$PWD/weights/rgbd-b33b2debe3c1"
fi
exec ros2 launch hack_bringup solution.launch.py "$@"
