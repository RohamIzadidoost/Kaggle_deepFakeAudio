#!/bin/bash
# Matched comparison for the advisor's decisive table. Everything below runs
# with float32 audio, so the tail-adaptation row and the frozen multi-view
# teacher are measured under identical input precision. Without this the
# teacher would get a free 4.0% relative EER head start from the fp16 buffer.
set -u
cd /home/general/Desktop/deepfake_project
source env/bin/activate
export CACHE_ON_CPU=1 PUBA_FP32=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=========== [$(date '+%F %T')] fp32 source + tail adaptation, native ITW ==========="
PUBA_CORPUS=itw PUBA_SKEW=0.3718 PUBA_ADAPT_N=8000 PUBA_Q=0.3 \
  PUBA_CKPTS=ssl_aasist_wavefake PUBA_ARMS=source,ours_fixed python protocol_a_public.py
echo "=========== [$(date '+%F %T')] DONE ==========="
