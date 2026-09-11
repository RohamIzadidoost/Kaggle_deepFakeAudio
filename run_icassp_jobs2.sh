#!/bin/bash
# Follow-up GPU queue. Run ONLY after run_icassp_jobs.sh has finished (one GPU).
#   nohup bash run_icassp_jobs2.sh > jobs_icassp2.log 2>&1 &
set -u
cd /home/general/Desktop/deepfake_project
source env/bin/activate
export CACHE_ON_CPU=1

stage () { echo "=============== [$(date '+%F %T')] STAGE $* ==============="; }

# --- 8a. The repo's own correctness gate ------------------------------------
# verify_reduction.py asserts that adapt_adaptive() with both switches off is
# bitwise identical to the published adapt(). If it fails, every
# adaptive-vs-fixed comparison in the paper is confounded by a refactor. The
# ICASSP pass added 242 lines to adaptive_pipeline.py; a diff against the last
# validated commit shows every change is additive (the only deletions are a
# stale comment and one `if` that became `elif`), so adapt/adapt_adaptive/
# set_tta_params/score/tent are untouched -- but the gate is cheap and the repo
# treats it as load-bearing, so run it rather than reason about it.
stage "8a verify_reduction (correctness gate)"
python verify_reduction.py || echo "!! VERIFY_REDUCTION FAILED -- adaptive results are confounded"

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

# --- 8c. What did the prior ESTIMATOR cost? ---------------------------------
# On In-the-Wild, BBSE returned pi_hat=0.600 against a true 0.372 (+0.228),
# because its q vector is the checkpoint's prediction rate at tau=0.5 and that
# checkpoint arrives with a +16.7-point calibration deficit -- so the deficit
# passes straight into the prior. The GMM, which reads histogram shape rather
# than a thresholded count, was off by 0.050. These arms price the difference:
# same machinery, same budget rule, three different sources of the prior
# (BBSE already run, GMM, and the true labels as an upper bound).
stage "8c In-the-Wild: GMM prior and oracle prior  (~1.2 h)"
PUBA_CORPUS=itw PUBA_CKPTS=ssl_aasist_wavefake PUBA_ARMS=ours_gmm,ours_oracle \
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
