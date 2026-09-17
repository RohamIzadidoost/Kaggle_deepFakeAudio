#!/bin/bash
# Experiment E: the same prevalence/budget grid on a second corpus and
# checkpoint. Pre-registered in PREREGISTERED_excess_dose_response.md.
set -u
cd /home/general/Desktop/deepfake_project
source env/bin/activate
export CACHE_ON_CPU=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for k in 0.50 0.10 0.01 0.90; do
  echo "=========== [$(date '+%F %T')] DF SKEW $k ==========="
  PUBA_CORPUS=df2021 PUBA_SKEW=$k PUBA_ADAPT_N=8000 PUBA_Q=0.3 \
    PUBA_CKPTS=deepfense_w2v2_aasist_s2 PUBA_ARMS=source python protocol_a_public.py; sleep 8
  for qq in 0.02 0.15 0.30; do
    PUBA_CORPUS=df2021 PUBA_SKEW=$k PUBA_Q=$qq PUBA_ADAPT_N=8000 \
      PUBA_CKPTS=deepfense_w2v2_aasist_s2 PUBA_ARMS=ours_fixed python protocol_a_public.py; sleep 8
  done
done
echo "=========== [$(date '+%F %T')] DF PREVALENCE GRID DONE ==========="
