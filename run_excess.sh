#!/bin/bash
# Pre-registered in PREREGISTERED_excess_dose_response.md before launch.
# Experiment B runs first because it is the falsification test.
# Every cell uses an 8,000-clip adaptation pool, so the optimization budget is
# matched across the whole grid; native prevalence is expressed as PUBA_SKEW
# equal to the native prior so it joins the same family (and so the audit
# reconstructs its adaptation draw with the same 8,000 budget).
set -u
cd /home/general/Desktop/deepfake_project
source env/bin/activate
export CACHE_ON_CPU=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run () {  # run <ckpt> <skew> <arm> <q>
  PUBA_CORPUS=itw PUBA_SKEW=$2 PUBA_Q=$4 PUBA_ADAPT_N=8000 \
    PUBA_CKPTS=$1 PUBA_ARMS=$3 python protocol_a_public.py
  sleep 8
}

echo "=========== [$(date '+%F %T')] EXPERIMENT B: large q at native prevalence ==========="
run ssl_aasist_wavefake 0.3718 source 0.3
for qq in 0.30 0.40 0.45 0.50; do run ssl_aasist_wavefake 0.3718 ours_fixed $qq; done

echo "=========== [$(date '+%F %T')] EXPERIMENT A: dose-response through the gap ==========="
for qq in 0.06 0.11 0.15 0.19 0.23; do run ssl_aasist_wavefake 0.01 ours_fixed $qq; done
for qq in 0.08 0.13 0.17 0.21 0.25; do run ssl_aasist_wavefake 0.97 ours_fixed $qq; done

echo "=========== [$(date '+%F %T')] EXPERIMENT C: second checkpoint ==========="
for k in 0.01 0.97; do
  run deepfense_w2v2_aasist_s42 $k source 0.3
  for qq in 0.02 0.15 0.30; do run deepfense_w2v2_aasist_s42 $k ours_fixed $qq; done
done
echo "=========== [$(date '+%F %T')] EXCESS QUEUE DONE ==========="
