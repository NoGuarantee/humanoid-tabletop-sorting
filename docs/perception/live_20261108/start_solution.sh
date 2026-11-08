#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
cd /solution
colcon build --packages-up-to hack_perception hack_motion --parallel-workers 2
source install/setup.bash
export PYTHONPATH=/runtime:$PYTHONPATH PYTHONDONTWRITEBYTECODE=1
export YOLO_OFFLINE=true YOLO_AUTOINSTALL=false
PYTHONPATH="${NSSR_PERCEPTION_SOURCE:-/solution/src/hack_perception}:$PYTHONPATH" python3 -m hack_perception.perception_node --ros-args -p use_sim_time:=true -p rgbd_model_dir:=/models -p rgbd_enable_model_poses:=true -p rgbd_on_demand:=true -p rgbd_device:=cpu -p rgbd_model_names:=Schleich_African_Black_Rhino,Nescafe_Memento_Latte_Caramel_8_08_oz_23_g_packets_64_oz_184_g > /evidence/perception.log 2>&1 & perception=$!
python3 -m hack_motion.motion_node --ros-args -p use_sim_time:=true -p planner:=ik > /evidence/motion.log 2>&1 & motion=$!
trap 'kill -INT "$perception" "$motion" 2>/dev/null || true; wait || true' EXIT
sleep 4
timeout 480 python3 /probe/live_probe.py > /evidence/probe.log 2>&1
