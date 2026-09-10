#!/bin/bash
# Follow-up GPU queue. Run ONLY after run_icassp_jobs.sh has finished (one GPU).
#   nohup bash run_icassp_jobs2.sh > jobs_icassp2.log 2>&1 &
set -u
cd /home/general/Desktop/deepfake_project
source env/bin/activate
export CACHE_ON_CPU=1

stage () { echo "=============== [$(date '+%F %T')] STAGE $* ==============="; }

# --- 8b. The prior-free alternative a reader will propose first -------------
# A fixed CONFIDENCE threshold instead of a quantile needs no prior at all. It
# trades one assumption for another, and the trade is the paper's own thesis:
# a quantile rule is calibration-free but prior-dependent; a confidence rule is
# prior-free but calibration-dependent -- and calibration is exactly what our
# measurements say does not transfer across corpora. Running it settles which
# assumption costs more on a real deployment pool.
stage "8b DF + In-the-Wild: fixed-confidence pseudo-labels  (~1.6 h)"
PUBA_CKPTS=deepfense_w2v2_aasist_s2 PUBA_ARMS=ours_conf python protocol_a_public.py
PUBA_CORPUS=itw PUBA_CKPTS=ssl_aasist_wavefake PUBA_ARMS=ours_conf \
  python protocol_a_public.py

# --- 9. Which half of the objective produces the DF gain? --------------------
# Same prior-corrected pseudo-label budget, consistency term removed. The
# fixed-q arm already shows the budget is what prevents the damage; this asks
# whether the budget alone is also what produces the 4.51 -> 4.02 improvement.
stage "9 DF: prior-corrected self-training, no consistency  (~1 h)"
PUBA_CKPTS=deepfense_w2v2_aasist_s2 PUBA_ARMS=ours_bbse_nocons \
  python protocol_a_public.py

# --- 10. Two more seeds for the leave-one-corpus-out baseline grid -----------
# ckpt_ext holds all ten seeds; the resume guard is keyed on
# (seed, target, method, setting), so this only adds folds.
stage "10 leave-one-corpus-out baselines, seeds 3-4  (~2.8 h)"
ADAPTIVE_SMOKE=0 TTA_BASELINES=1 OUR_SEEDS=3,4 python adaptive_pipeline.py

# --- 11. In-the-Wild consistency ablation, for the second prior --------------
stage "11 In-the-Wild: prior-corrected self-training, no consistency  (~0.6 h)"
PUBA_CORPUS=itw PUBA_CKPTS=ssl_aasist_wavefake PUBA_ARMS=ours_bbse_nocons \
  python protocol_a_public.py

stage "FOLLOW-UP QUEUE DONE"
