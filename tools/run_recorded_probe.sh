#!/usr/bin/env bash
# Run inside the team's ROS development container on a freshly owned simulator.
set -euo pipefail
cd "$(dirname "$0")/.."
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to hack_bringup
source install/setup.bash
model_dir=${NSSR_RGBD_MODEL_DIR:-$PWD/weights/rgbd-b33b2debe3c1}
test -f "$model_dir/bundle.json" || { echo 'Run tools/stage_perception_rgbd.sh first'; exit 1; }
nodes=$(ros2 node list --no-daemon)
if [[ "$nodes" == *'/hack_motion'* || "$nodes" == *'/hack_brain'* || "$nodes" == *'/hack_perception'* ]]; then
  echo 'Solution nodes already own this ROS domain'
  exit 1
fi
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill -INT -- "-$pid" 2>/dev/null || true; done
  sleep 2
  for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  sleep 2
  for pid in "${pids[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done
  wait || true
}
trap cleanup EXIT
setsid ros2 run hack_motion motion_node --ros-args -p use_sim_time:=true &
pids+=("$!")
setsid ros2 run hack_perception perception_node --ros-args -p use_sim_time:=true \
  -p rgbd_model_dir:="$model_dir" -p rgbd_enable_model_poses:=true \
  -p rgbd_on_demand:=true -p rgbd_device:=cuda -p rgbd_pool:=4 \
  -p rgbd_model_names:=Schleich_African_Black_Rhino &
pids+=("$!")
timeout --signal=INT --kill-after=10s 600 python3 tools/pick_recorded.py "$@"
