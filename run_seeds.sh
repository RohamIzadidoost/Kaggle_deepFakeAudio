#!/bin/bash
# Seed replication on the cells the claims rest on. PUBA_SEED varies both the
# 8,000-clip adaptation draw and the augmentation noise, so these are full
# repetitions, not re-evaluations. Each invocation reports the source arm too,
# so every cached score array has a matching history row for the audit.
set -u
cd /home/general/Desktop/deepfake_project
source env/bin/activate
export CACHE_ON_CPU=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run () {  # run <corpus> <ckpt> <skew> <q> <seed>
  PUBA_CORPUS=$1 PUBA_CKPTS=$2 PUBA_SKEW=$3 PUBA_Q=$4 PUBA_SEED=$5 \
    PUBA_ADAPT_N=8000 PUBA_ARMS=source,ours_fixed python protocol_a_public.py
  sleep 8
}
for sd in 1 2; do
  echo "=========== [$(date '+%F %T')] SEED $sd ==========="
  for qq in 0.06 0.15 0.30; do run itw ssl_aasist_wavefake 0.01 $qq $sd; done
  for qq in 0.08 0.17 0.30; do run itw ssl_aasist_wavefake 0.97 $qq $sd; done
  run itw ssl_aasist_wavefake 0.3718 0.50 $sd
  run df2021 deepfense_w2v2_aasist_s2 0.50 0.02 $sd
done
echo "=========== [$(date '+%F %T')] SEED REPLICATION DONE ==========="
