#!/bin/bash
# Installs cuRobo and compiles its kernels. The evaluator runs this after requirements.txt and before colcon
# (hackathon/tools/solution.sh). cuRobo v0.8 is a PEP 621 package: the image's pip 22 and setuptools 59.6 cannot
# build it, and colcon needs that setuptools left alone, so the wheel is built the way tools/curobo_install.sh does.
# This script runs once and is outside the sim-time limit, while the start of run.sh counts on every layout
# (organisers' answer, docs/ANALYSIS.md, section 8). So the kernels are compiled here, not in the motion node.
#
# Never fails: solution.sh runs this under set -e, and a non-zero exit would skip run.sh, so /sim/task/done would
# never be published. Without cuRobo the nodes still start and MoveTcp answers with the error.
cd "$(dirname "$0")" || exit 0
setup_started=$SECONDS
if bash tools/curobo_install.sh; then
    python3 tools/curobo_warmup.py || echo "[setup] cuRobo warmup failed, the first MoveTcp compiles the kernels"
else
    echo "[setup] cuRobo install FAILED, MoveTcp will not work"
fi
# Keep a disabled recorded mode cheap, including during evaluator installation.
# Use the same default and explicit opt-out as run.sh.
rgbd_started=$SECONDS
case "${NSSR_RECORDED_GRASPS:-true}" in
true|True|TRUE)
rm -f weights/.rgbd-ready
# The bundled 12 MB archive works in a clean checkout without GitHub credentials.
if bash tools/stage_perception_rgbd.sh &&
       /usr/bin/python3 -m pip install 'numpy<2' -r tools/perception_rgbd_requirements.txt &&
       PYTHONPATH="$PWD/src/hack_perception${PYTHONPATH:+:$PYTHONPATH}" YOLO_OFFLINE=true YOLO_AUTOINSTALL=false /usr/bin/python3 -c \
           'import numpy; assert numpy.__version__.split(".")[0] == "1"; from hack_perception.rgbd.pose_backend import PoseBackend; PoseBackend("weights/rgbd-b33b2debe3c1", device="cpu")'; then
    touch weights/.rgbd-ready
else
    echo "[setup] RGB-D setup failed, using baseline perception"
fi
;;
*) echo "[setup] RGB-D disabled, using baseline perception" ;;
esac
echo "[setup] RGB-D elapsed $((SECONDS-rgbd_started))s; total $((SECONDS-setup_started))s"
exit 0
