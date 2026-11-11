#!/bin/bash
# Installs torch and cuRobo into the solution image the way the evaluator installs requirements.txt
# (/usr/bin/python3 -m pip), and times it. Candidate pins for requirements.txt; see tools/curobo_check.py.
# cuRobo v0.8 compiles its kernels at runtime with NVRTC (cuda-core), so the image needs no nvcc.
set -eu
TORCH=${TORCH:-torch==2.8.0}                  # the PyPI build is CUDA 12.8, has sm_120 (Blackwell)
CUROBO=${CUROBO:-"nvidia-curobo[cu12] @ git+https://github.com/NVlabs/curobo.git@v0.8.0"}
# The evaluator is Linux x86-64. Warp's PyPI 1.19 wheel uses CUDA 13.4 and
# disables CUDA on R570 (our 4090). Keep the same version with its official
# CUDA 12 build, matching torch/cuRobo without requiring a host driver update.
WARP=${WARP:-"https://github.com/NVIDIA/warp/releases/download/v1.19.0/warp_lang-1.19.0+cu12-py3-none-manylinux_2_28_x86_64.whl"}
which git >/dev/null || { echo "[curobo_install] no git in the image"; exit 1; }
PIP="/usr/bin/python3 -m pip install -q --no-warn-script-location"
t0=$SECONDS
$PIP "numpy<2" "$TORCH"
echo "[curobo_install] torch took $((SECONDS - t0)) s"
# The image's Debian pip 22.0 leaks the system packaging 21.3 into isolated builds (the latest setuptools_scm needs
# packaging>=26.2), and the system setuptools 59.6 cannot read PEP 621 metadata. colcon relies on that setuptools, so
# do not upgrade it: build the cuRobo wheel with fresh build tools from a temp dir, then install the wheel normally.
# cuRobo is pure Python (kernels ship as .cu sources), so the wheel is py3-none-any.
t1=$SECONDS
TMP=$(mktemp -d)
$PIP --target "$TMP/bt" "setuptools>=61,<75" "setuptools_scm<8" "packaging>=23" wheel
PYTHONPATH="$TMP/bt" SETUPTOOLS_SCM_PRETEND_VERSION=0.8.0 /usr/bin/python3 -m pip wheel -q --no-deps \
  --no-build-isolation -w "$TMP/whl" "$CUROBO"
whl=$(ls "$TMP"/whl/nvidia_curobo-*.whl)
echo "[curobo_install] built $(basename "$whl") ($(du -h "$whl" | cut -f1))"
# Checked on the image (2026-11-03). warp 1.20 drops Python 3.10, the image has 3.10.
$PIP "numpy<2" "$WARP" "cuda-core==1.2.2" "$whl[cu12]"
rm -rf "$TMP"
echo "[curobo_install] curobo took $((SECONDS - t1)) s, total $((SECONDS - t0)) s"
/usr/bin/python3 -m pip list 2>/dev/null | grep -i -E '^(torch|nvidia-curobo|warp-lang|cuda-core|numpy|nvidia-cuda-nvrtc)' || true
du -sh /usr/local/lib/python3.10/dist-packages 2>/dev/null || true
