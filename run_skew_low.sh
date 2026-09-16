#!/bin/bash
# Low-spoof-prevalence extension of run_skew_sweep.sh (ICASSP reviewer item 4).
# Predictions pre-registered in PREREGISTERED_low_prevalence_predictions.md
# before launch. Descending order so a partial run still leaves a curve that
# joins the existing native-prevalence point (0.372).
set -u
cd /home/general/Desktop/deepfake_project
source env/bin/activate
export CACHE_ON_CPU=1
for k in 0.10 0.05 0.02 0.01; do
  echo "=========== [$(date '+%F %T')] SKEW $k ==========="
  PUBA_CORPUS=itw PUBA_SKEW=$k PUBA_ADAPT_N=8000 PUBA_CKPTS=ssl_aasist_wavefake \
    PUBA_ARMS=source python protocol_a_public.py
  for qq in 0.02 0.10 0.30; do
    PUBA_CORPUS=itw PUBA_SKEW=$k PUBA_Q=$qq PUBA_ADAPT_N=8000 \
      PUBA_CKPTS=ssl_aasist_wavefake PUBA_ARMS=ours_fixed python protocol_a_public.py
  done
  PUBA_CORPUS=itw PUBA_SKEW=$k PUBA_ADAPT_N=8000 PUBA_CKPTS=ssl_aasist_wavefake \
    PUBA_ARMS=ours_bbse python protocol_a_public.py
done
echo "=========== [$(date '+%F %T')] LOW-PREVALENCE SWEEP DONE ==========="
