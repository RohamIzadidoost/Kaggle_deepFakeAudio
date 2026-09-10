#!/bin/bash
# Sequential GPU job queue for the ICASSP strengthening pass.
# One GPU, so everything is serial. Each stage is resumable: every runner has a
# resume guard keyed on its own result rows, so re-running this script after an
# interruption picks up where it stopped instead of redoing finished cells.
#
#   nohup bash run_icassp_jobs.sh > jobs_icassp.log 2>&1 &
set -u
cd /home/general/Desktop/deepfake_project
source env/bin/activate
export CACHE_ON_CPU=1          # 10 GB card: waveform cache stays in host RAM

stage () {
  echo "=================================================================="
  echo "[$(date '+%F %T')] STAGE $1"
  echo "=================================================================="
}

# --- 0. Smoke the two new adaptive_pipeline arms before spending 15 GPU-h ----
stage "0a smoke: TTA_BASELINES"
ADAPTIVE_SMOKE=1 TTA_BASELINES=1 BASELINE_EPOCHS=1 python adaptive_pipeline.py || exit 1
stage "0b smoke: STOP_TRACE"
ADAPTIVE_SMOKE=1 STOP_TRACE=1 STOP_EPOCHS=2 python adaptive_pipeline.py || exit 1

# --- 1. Official ASVspoof2021-DF, remaining seeds, ALL arms -----------------
# Re-planned once seed 2 came back: the DF arms are NOT the predicted null they
# looked like. On a checkpoint at 4.51% EER, naive symmetric-q TTA drives
# accuracy 98.87 -> 62.96 and EER 4.51 -> 5.84, i.e. it CREATES a 31-point
# calibration deficit on the field's standard benchmark, precisely the way the
# hidden balanced-pool assumption predicts. That is a headline, not a null, and
# it needs more than one seed. (Resume-safe: rows already recorded are skipped,
# so this is also the safety net if the first launch died.)
stage "1 DF official, seeds 42/240, all arms"
PUBA_CKPTS=deepfense_w2v2_aasist_s42,deepfense_w2v2_aasist_s240 \
  PUBA_ARMS=source,ours_fixed,ours_bbse python protocol_a_public.py

# --- 2. Source confusion matrices, for the prior sweep (CPU-cheap) -----------
stage "2 cache source confusion matrices"
python cache_source_M.py

# --- 3. Full In-the-Wild benchmark on published checkpoints ------------------
# All 31,779 clips (the paper's existing third-party arm caps every pool at
# 6,000). True P(fake) = 0.372 -- skewed the OPPOSITE way from DF's 0.972, so
# the prior correction is tested in both directions.
stage "3 In-the-Wild (full) on published checkpoints"
PUBA_CORPUS=itw \
  PUBA_CKPTS=ssl_aasist_wavefake,deepfense_w2v2_aasist_s2,deepfense_w2v2_aasist_s42,deepfense_w2v2_aasist_s240 \
  PUBA_ARMS=source,ours_fixed,ours_bbse python protocol_a_public.py

# --- 4. Modern TTA baselines on the leave-one-corpus-out grid ----------------
stage "4 SHOT / ETA / EATA / SAR vs ours vs Tent"
ADAPTIVE_SMOKE=0 TTA_BASELINES=1 python adaptive_pipeline.py

# --- 5. Instrumented budget traces, for the label-free stopping rule ---------
stage "5 stop-trace, E=32, seeds 0-1"
ADAPTIVE_SMOKE=0 STOP_TRACE=1 STOP_EPOCHS=32 OUR_SEEDS=0,1 python adaptive_pipeline.py

# --- 6. Third seed for the stopping rule, if the budget allows ---------------
stage "6 stop-trace, seed 2"
ADAPTIVE_SMOKE=0 STOP_TRACE=1 STOP_EPOCHS=32 OUR_SEEDS=2 python adaptive_pipeline.py

stage "ALL STAGES DONE"
