#!/usr/bin/env bash
set -eo pipefail
cd /nssr_sim
export HACK_TASK=sort3 HACK_POOL=4 HACK_RENDER_INTERVAL=3
unset HACK_PLUGIN HACK_OBJECTS HACK_LAYOUT HACK_CAMS HACK_SEED
export PYTHONPATH=/nssr_sim/source:/nssr_sim${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HACK_KIT_ARGS="--/plugins/carb.tasking.plugin/threadCount=8 --/renderer/multiGpu/enabled=false --/rtx/raytracing/fractionalCutoutOpacity=true --/rtx/pathtracing/fractionalCutoutOpacity=true --/rtx/translucency/maxRefractionBounces=4 --/app/runLoops/main/rateLimitEnabled=false --/app/renderer/resolution/width=2560 --/app/renderer/resolution/height=1440"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PXR_WORK_THREAD_LIMIT=8
exec timeout --signal=INT --kill-after=20 1000 bash scripts/start_sim.sh --perf_stats --step_hz 100 --headless --device cpu
