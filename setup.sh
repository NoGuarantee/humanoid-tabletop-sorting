#!/bin/bash
# Installs cuRobo and compiles its kernels. The evaluator runs this after requirements.txt and before colcon
# (hackathon/tools/solution.sh). cuRobo v0.8 is a PEP 621 package: the image's pip 22 and setuptools 59.6 cannot
# build it, and colcon needs that setuptools left alone, so the wheel is built the way tools/curobo_install.sh does.
# This script runs once and is outside the sim-time limit, while the start of run.sh counts on every layout
# (organisers' answer, docs/ANALYSIS.md, section 8). So the kernels are compiled here, not in the motion node.
#
# Never fails: solution.sh runs this under set -e, and a non-zero exit would skip run.sh, so /sim/task/done would
# never be published. Without cuRobo the nodes still start and MoveTcp answers with the error.
cd "$(dirname "$0")"
rm -f weights/.rgbd-ready
if bash tools/curobo_install.sh; then
    python3 tools/curobo_warmup.py || echo "[setup] cuRobo warmup failed, the first MoveTcp compiles the kernels"
else
    echo "[setup] cuRobo install FAILED, MoveTcp will not work"
fi
# Presence of the pre-staged archive explicitly opts in to the RGB-D scene worker.
if [ -f weights/geometry_only_ros_bundle.tar.gz ]; then
    if bash tools/stage_perception_rgbd.sh &&
       /usr/bin/python3 -m pip install 'numpy<2' -r tools/perception_rgbd_requirements.txt &&
       YOLO_OFFLINE=true YOLO_AUTOINSTALL=false /usr/bin/python3 -c           'import cv2, numpy, scipy, PIL, torch, torchvision, ultralytics; assert numpy.__version__.split(".")[0] == "1"'; then
        touch weights/.rgbd-ready
    else
        echo "[setup] RGB-D setup failed, using baseline perception"
    fi
fi
exit 0
