#!/bin/bash
set -u
cd /home/general/Desktop/deepfake_project
source env/bin/activate
export CACHE_ON_CPU=1
for k in 0.90 0.95 0.97; do
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
echo "=========== [$(date '+%F %T')] SKEW SWEEP DONE ==========="
