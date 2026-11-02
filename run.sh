#!/bin/bash
# Starts the whole solution. The evaluator runs: cd /solution && source install/setup.bash && bash run.sh
# ROS_DOMAIN_ID and the DDS profile are already set in the container: do not change them here.
exec ros2 launch hack_bringup solution.launch.py
