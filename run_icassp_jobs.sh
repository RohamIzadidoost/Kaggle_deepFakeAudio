#!/bin/bash
# Sequential GPU job queue for the ICASSP strengthening pass. One GPU, so
# everything is serial. Every runner has its own resume guard keyed on its
# result rows, so re-running this script after an interruption resumes rather
# than redoing finished cells.
#
#   nohup bash run_icassp_jobs.sh > jobs_icassp.log 2>&1 &
#
# Ordered by (reviewer value x needs-to-exist-by-morning), not by cost.
set -u
cd /home/general/Desktop/deepfake_project
source env/bin/activate
export CACHE_ON_CPU=1          # 10 GB card: waveform cache stays in host RAM

stage () {
  echo "=================================================================="
  echo "[$(date '+%F %T')] STAGE $*"
  echo "=================================================================="
}

# --- 0. Source confusion matrices (unblocks the CPU-only prior sweep) --------
stage "0 cache source confusion matrices  (~5 min)"
python cache_source_M.py

# --- 1. Modern TTA baselines AT DEPLOYMENT PREVALENCE ------------------------
# Tent / SHOT / ETA / SAR were all developed and evaluated on class-balanced
# benchmarks. This runs them unmodified on the official ASVspoof2021-DF eval
# pool (97.2% spoof) against a checkpoint at 4.51% EER. The naive symmetric-q
# arm already fell over there (98.87 -> 62.96 accuracy, EER 4.51 -> 5.84) while
# the prior-corrected arm improved it (4.51 -> 4.02). If the published
# baselines fall over too, the balanced-pool assumption is a property of the
# field's evaluation practice, not a quirk of our recipe.
stage "1 DF official: tent / shot / eta / sar on a SOTA checkpoint  (~4 h)"
PUBA_CKPTS=deepfense_w2v2_aasist_s2 PUBA_ARMS=tent,shot,eta,sar \
  python protocol_a_public.py

# --- 2. The headline, replicated across independently trained checkpoints ----
stage "2 DF official, seeds 42/240, all arms  (~5.2 h)"
PUBA_CKPTS=deepfense_w2v2_aasist_s42,deepfense_w2v2_aasist_s240 \
  PUBA_ARMS=source,ours_fixed,ours_bbse python protocol_a_public.py

# --- 3. Full In-the-Wild, the other direction of skew ------------------------
# 31,779 clips, true P(fake) = 0.372 -- skewed the OPPOSITE way from DF's 0.972.
# A prior correction that only ever helps on spoof-heavy pools would be a
# special case; this is the test that it is not.
stage "3 In-the-Wild (all 31,779) on published checkpoints  (~3.7 h)"
PUBA_CORPUS=itw \
  PUBA_CKPTS=ssl_aasist_wavefake,deepfense_w2v2_aasist_s2,deepfense_w2v2_aasist_s42,deepfense_w2v2_aasist_s240 \
  PUBA_ARMS=source,ours_fixed,ours_bbse python protocol_a_public.py

# --- 4. Modern TTA baselines on the BALANCED leave-one-corpus-out grid -------
# The counterpart to stage 1. Together they separate "this method is worse"
# from "this method assumes balance".
stage "4 SHOT / ETA / EATA / SAR vs ours vs Tent, leave-one-corpus-out  (~4.2 h)"
ADAPTIVE_SMOKE=1 TTA_BASELINES=1 BASELINE_EPOCHS=1 python adaptive_pipeline.py || exit 1
ADAPTIVE_SMOKE=0 TTA_BASELINES=1 python adaptive_pipeline.py

# --- 5. Seed variance for the DF headline -----------------------------------
# Adaptation is stochastic (pool draw, batch order, augmentation), so the
# headline needs an error bar. Scoring a FIXED 50k-trial subset instead of all
# 400k cuts each cell from ~59 to ~32 min; the subset is identical across every
# seed and arm so the comparison stays paired, and the official pooled number
# still comes from the full-eval rows above. 3 checkpoints x 2 extra seeds
# gives 9 paired cells in total.
stage "5 DF seed variance, 50k fixed subset, seeds 1-2  (~6 h)"
for sd in 1 2; do
  PUBA_SEED=$sd PUBA_EVAL_SUB=50000 \
    PUBA_CKPTS=deepfense_w2v2_aasist_s2,deepfense_w2v2_aasist_s42,deepfense_w2v2_aasist_s240 \
    PUBA_ARMS=source,ours_fixed,ours_bbse python protocol_a_public.py
done

# --- 6. Instrumented budget traces ------------------------------------------
stage "6 stop-trace, E=32, seeds 0-1  (~10.3 h)"
ADAPTIVE_SMOKE=1 STOP_TRACE=1 STOP_EPOCHS=2 python adaptive_pipeline.py || exit 1
ADAPTIVE_SMOKE=0 STOP_TRACE=1 STOP_EPOCHS=32 OUR_SEEDS=0,1 python adaptive_pipeline.py

# --- 7. Third seed for the stopping rule ------------------------------------
stage "7 stop-trace, seed 2  (~5.2 h)"
ADAPTIVE_SMOKE=0 STOP_TRACE=1 STOP_EPOCHS=32 OUR_SEEDS=2 python adaptive_pipeline.py

# --- 8. Baselines on In-the-Wild, at the third prior -------------------------
stage "8 In-the-Wild: tent / shot / eta / sar  (~1.7 h)"
PUBA_CORPUS=itw PUBA_CKPTS=ssl_aasist_wavefake PUBA_ARMS=tent,shot,eta,sar \
  python protocol_a_public.py

stage "ALL STAGES DONE"
