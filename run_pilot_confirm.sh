#!/bin/bash
# Confirm the frozen multi-view teacher where the first pilot could only
# suggest it: the full In-the-Wild pool rather than a 10k sample, and a second
# corpus with a second checkpoint. Crop-only views, since additive noise was
# shown to cost 84% relative EER.
set -u
cd /home/general/Desktop/deepfake_project
source env/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=========== [$(date '+%F %T')] full In-the-Wild, WF, 8 crop views ==========="
PILOT_N=40000 PILOT_K=8 PILOT_NOISE=0 PILOT_CORPUS=itw \
  PILOT_CKPT=ssl_aasist_wavefake PILOT_OUT=pilot_views_itw_full.npz python pilot_views.py
sleep 10
echo "=========== [$(date '+%F %T')] ASVspoof2021-DF, DF-2, 8 crop views ==========="
PILOT_N=30000 PILOT_K=8 PILOT_NOISE=0 PILOT_CORPUS=df2021 \
  PILOT_CKPT=deepfense_w2v2_aasist_s2 PILOT_OUT=pilot_views_df_full.npz python pilot_views.py
echo "=========== [$(date '+%F %T')] CONFIRMATION DONE ==========="
