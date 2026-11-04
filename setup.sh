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
bash tools/curobo_install.sh || { echo "[setup] cuRobo install FAILED, MoveTcp will not work"; exit 0; }
python3 tools/curobo_warmup.py || echo "[setup] cuRobo warmup failed, the first MoveTcp compiles the kernels"
exit 0
