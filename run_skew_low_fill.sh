#!/bin/bash
# Gap-fill pass for run_skew_low.sh. Re-runs only what is missing: the runner's
# own resume guard skips every (ckpt, arm, setting, seed, eval_sub, q, skew)
# already in results_protocol_a_public_itw.csv, and cached source scores are
# reused, so completed cells cost seconds.
#
# Why this pass exists: one cell (q=0.02 at pi=0.10) died with CUDA OOM while
# other desktop processes held ~680 MB of the 10 GB card and PyTorch had 1.75 GB
# reserved-but-unallocated. expandable_segments removes that fragmentation
# headroom problem. Batch size is deliberately NOT lowered: it would change the
# number of optimization steps and break comparability with the cells that
# already ran. The sleep lets each process fully release the device before the
# next one starts.
set -u
cd /home/general/Desktop/deepfake_project
source env/bin/activate
export CACHE_ON_CPU=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for k in 0.10 0.05 0.02 0.01; do
  echo "=========== [$(date '+%F %T')] FILL SKEW $k ==========="
  PUBA_CORPUS=itw PUBA_SKEW=$k PUBA_ADAPT_N=8000 PUBA_CKPTS=ssl_aasist_wavefake \
    PUBA_ARMS=source python protocol_a_public.py; sleep 10
  for qq in 0.02 0.10 0.30; do
    PUBA_CORPUS=itw PUBA_SKEW=$k PUBA_Q=$qq PUBA_ADAPT_N=8000 \
      PUBA_CKPTS=ssl_aasist_wavefake PUBA_ARMS=ours_fixed python protocol_a_public.py; sleep 10
  done
  PUBA_CORPUS=itw PUBA_SKEW=$k PUBA_ADAPT_N=8000 PUBA_CKPTS=ssl_aasist_wavefake \
    PUBA_ARMS=ours_bbse python protocol_a_public.py; sleep 10
done
echo "=========== [$(date '+%F %T')] FILL DONE ==========="
