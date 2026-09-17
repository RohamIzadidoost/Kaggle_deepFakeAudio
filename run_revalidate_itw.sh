#!/bin/bash
# Stage A of the re-validation: the In-the-Wild prevalence grid, rescored and
# re-adapted under per-utterance waveform normalisation. This is the paper's
# centrepiece, and it is the part most exposed to the batched-normalisation bug,
# because changing a pool's prevalence changes its batch composition and so its
# scores independently of any adaptation.
#
# Deliberately no `set -e`: one failed cell must not abandon the queue. The
# runner's resume guard now distinguishes normalisation mode, so re-running this
# script after an interruption costs only the cells that are missing.
set -u
cd /home/general/Desktop/deepfake_project
source env/bin/activate
export CACHE_ON_CPU=1 PUBA_PERSAMPLE_NORM=1 PUBA_FP32=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for k in 0.01 0.02 0.05 0.10 0.3718 0.90 0.95 0.97; do
  echo "=========== [$(date '+%F %T')] PSN SKEW $k ==========="
  PUBA_CORPUS=itw PUBA_SKEW=$k PUBA_ADAPT_N=8000 PUBA_CKPTS=ssl_aasist_wavefake \
    PUBA_ARMS=source python protocol_a_public.py; sleep 8
  for qq in 0.02 0.10 0.30; do
    PUBA_CORPUS=itw PUBA_SKEW=$k PUBA_Q=$qq PUBA_ADAPT_N=8000 \
      PUBA_CKPTS=ssl_aasist_wavefake PUBA_ARMS=ours_fixed python protocol_a_public.py; sleep 8
  done
  PUBA_CORPUS=itw PUBA_SKEW=$k PUBA_ADAPT_N=8000 PUBA_CKPTS=ssl_aasist_wavefake \
    PUBA_ARMS=ours_bbse python protocol_a_public.py; sleep 8
done
echo "=========== [$(date '+%F %T')] STAGE A DONE ==========="
