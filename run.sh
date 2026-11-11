#!/bin/bash
# Starts the whole solution. The evaluator runs: cd /solution && source install/setup.bash && bash run.sh
# ROS_DOMAIN_ID and the DDS profile are already set in the container: do not change them here.
cd "$(dirname "$0")" || exit 1
# Explicit launch arguments take precedence over the environment, including
# when deciding whether to start the optional RGB-D worker.
requested="${NSSR_RECORDED_GRASPS:-false}"
for arg in "$@"; do
    case "$arg" in recorded_grasps:=*) requested="${arg#recorded_grasps:=}";; esac
done
if [ -f weights/.rgbd-ready ] && [ -f weights/rgbd-b33b2debe3c1/bundle.json ]; then
    # Full ROS lift/carry/release qualification is tracked in ROADMAP.
    # Enable by default only after full-placement qualification on all three seeds.
    export NSSR_RECORDED_GRASPS="$requested"
    case "$requested" in true|True|TRUE)
        export NSSR_RGBD_MODEL_DIR="${NSSR_RGBD_MODEL_DIR:-$PWD/weights/rgbd-b33b2debe3c1}"
        export NSSR_RECORDED_OBJECTS="${NSSR_RECORDED_OBJECTS:-philips}"
        export NSSR_RGBD_ON_DEMAND="${NSSR_RGBD_ON_DEMAND:-true}"
        ;;
    esac
else
    # Failed/incomplete setup must not make the mission retry unavailable poses.
    export NSSR_RECORDED_GRASPS=false
    set -- "$@" recorded_grasps:=false
fi
exec ros2 launch hack_bringup solution.launch.py "$@"
