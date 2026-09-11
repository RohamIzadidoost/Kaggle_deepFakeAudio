#!/bin/bash
# Merged remaining queue, re-ordered by value after the purity bound changed the
# priorities. Replaces the tail of run_icassp_jobs.sh (stages 6-8) and all of
# run_icassp_jobs2.sh. Every runner resumes from its own result rows, so nothing
# already finished is redone.
#
#   nohup bash run_icassp_jobs3.sh > jobs_icassp3.log 2>&1 &
set -u
cd /home/general/Desktop/deepfake_project
source env/bin/activate
export CACHE_ON_CPU=1
stage () { echo "=============== [$(date '+%F %T')] STAGE $* ==============="; }

# --- A. Does a small symmetric budget obsolete the prior correction? ---------
# The counting bound says the confident-quantile rule can be sound only while
# q <= min(pi, 1-pi). q is ours to choose; pi is not identifiable. If q=0.02
# matches prior-corrected TTA on DF, the simpler rule wins outright.
stage "A DF: tail budget sweep q in {0.02, 0.05, 0.10}  (~3 h)"
for qq in 0.02 0.05 0.10; do
  PUBA_Q=$qq PUBA_CKPTS=deepfense_w2v2_aasist_s2 PUBA_ARMS=ours_fixed \
    python protocol_a_public.py
done

# --- B. What did the prior ESTIMATOR cost, vs a perfect one? -----------------
stage "B In-the-Wild: GMM prior and oracle prior  (~1.2 h)"
PUBA_CORPUS=itw PUBA_CKPTS=ssl_aasist_wavefake,deepfense_w2v2_aasist_s42 \
  PUBA_ARMS=ours_gmm,ours_oracle python protocol_a_public.py

# --- C. The prior-free alternative a reader proposes first -------------------
stage "C DF + ITW: fixed-confidence pseudo-labels  (~1.6 h)"
PUBA_CKPTS=deepfense_w2v2_aasist_s2 PUBA_ARMS=ours_conf python protocol_a_public.py
PUBA_CORPUS=itw PUBA_CKPTS=ssl_aasist_wavefake PUBA_ARMS=ours_conf \
  python protocol_a_public.py

# --- D. Out-of-sample validation of the label-free guards -------------------
stage "D stop-trace, E=32, seeds 0-1  (~10.3 h)"
ADAPTIVE_SMOKE=1 STOP_TRACE=1 STOP_EPOCHS=2 python adaptive_pipeline.py || \
  echo "!! stop-trace smoke failed -- skipping the real run"
ADAPTIVE_SMOKE=0 STOP_TRACE=1 STOP_EPOCHS=32 OUR_SEEDS=0,1 python adaptive_pipeline.py

# --- E. Which half of the objective produces the DF gain? -------------------
stage "E DF: prior-corrected self-training, no consistency  (~1 h)"
PUBA_CKPTS=deepfense_w2v2_aasist_s2 PUBA_ARMS=ours_bbse_nocons python protocol_a_public.py

# --- F. The repo's own correctness gate --------------------------------------
stage "F verify_reduction"
python verify_reduction.py || echo "!! VERIFY_REDUCTION FAILED -- adaptive results confounded"

# --- G. Remaining breadth, in descending value -------------------------------
stage "G In-the-Wild: tent / shot / eta / sar  (~1.7 h)"
PUBA_CORPUS=itw PUBA_CKPTS=ssl_aasist_wavefake PUBA_ARMS=tent,shot,eta,sar \
  python protocol_a_public.py
stage "H leave-one-corpus-out baselines, seeds 3-4  (~2.8 h)"
ADAPTIVE_SMOKE=0 TTA_BASELINES=1 OUR_SEEDS=3,4 python adaptive_pipeline.py
stage "I stop-trace, seed 2  (~5.2 h)"
ADAPTIVE_SMOKE=0 STOP_TRACE=1 STOP_EPOCHS=32 OUR_SEEDS=2 python adaptive_pipeline.py

stage "ALL DONE"
