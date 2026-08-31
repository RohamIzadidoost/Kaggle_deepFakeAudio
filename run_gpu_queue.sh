#!/usr/bin/env bash
# Overnight GPU queue for the ~24h window ending 2026-08-27.
#
# Ordered by what it buys the ICASSP paper, most valuable first, so that if the
# GPU is lost early the work already done is the work most worth having. Every
# stage is resumable: run_public_ckpt_multi.sh skips (seed, target, method,
# family) rows already present in the CSV, so re-running this script continues
# rather than duplicating.
#
# Nothing here trains a source model. Every stage is adaptation-only (~4.7 GB
# measured), which is why it fits a 10 GB card at all.
#
#   ./run_gpu_queue.sh                # everything, in order
#   STAGES="2 3" ./run_gpu_queue.sh   # just those stages
set -u
cd "$(dirname "$0")"
source env/bin/activate

STAGES="${STAGES:-1 2 3 4 5}"
LOG=queue_log.txt
say() { echo "[$(date +%H:%M:%S)] === $* ===" | tee -a "$LOG"; }

has_stage() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

# --- Stage 1: ASVspoof2021-DF as a FIFTH corpus ------------------------------
# The highest-value new information available. The precondition claim predicts
# adaptation should fail where source ranking is weak, and 2021-DF is where our
# own method regresses (26.33 -> 42.46 EER, Protocol A). If third-party models
# regress there too, the precondition survives a risky prediction it could have
# failed -- that is worth more than another seed of a target we already have.
# Balanced 3000/class, same rule as every other target, so the skew is NOT the
# variable here; ranking quality is.
if has_stage 1; then
  say "stage 1: ASVspoof2021-DF, four checkpoints, seed 0"
  for c in ssl_aasist_wavefake deepfense_w2v2_aasist_s42 \
           deepfense_w2v2_aasist_s2 deepfense_w2v2_aasist_s240; do
    python public_ckpt_tta.py --mode signcheck --ckpt "$c" \
      --manifest manifest_tgt_asvspoof2021df_seed0.csv --target asvspoof2021df \
      --limit 200 --log run_log_public_ckpt_multi.txt 2>&1 | tail -3 | tee -a "$LOG"
    python public_ckpt_tta.py --mode ours --ckpt "$c" \
      --manifest manifest_tgt_asvspoof2021df_seed0.csv --target asvspoof2021df \
      --seed 0 --batch 16 --out results_public_ckpt_multi.csv \
      --log run_log_public_ckpt_multi.txt 2>&1 | tail -4 | tee -a "$LOG"
  done
fi

# --- Stage 2: Tent + self-training-only across every cell --------------------
# Tent is the load-bearing contrast (PROJECT_LOG.md S5): naive entropy
# minimisation collapses where our structure does not. That claim currently
# rests on our own model plus two public checkpoints on ONE corpus. Running it
# across the whole grid turns "Tent collapsed on our data" into "Tent collapses
# across four independent models and five corpora, ours does not".
# st_only gives the component decomposition on the same cells -- and the
# recorded finding is that the ablation *inverts* between checkpoints, so more
# cells is the only way to tell which pattern is typical.
if has_stage 2; then
  say "stage 2: tent + st_only across all cells, seed 0"
  MODES="tent st_only" SEEDS=0 ./run_public_ckpt_multi.sh 2>&1 | tee -a "$LOG"
  for c in ssl_aasist_wavefake deepfense_w2v2_aasist_s42 \
           deepfense_w2v2_aasist_s2 deepfense_w2v2_aasist_s240; do
    for m in tent st_only; do
      python public_ckpt_tta.py --mode "$m" --ckpt "$c" \
        --manifest manifest_tgt_asvspoof2021df_seed0.csv --target asvspoof2021df \
        --seed 0 --batch 16 --out results_public_ckpt_multi.csv \
        --log run_log_public_ckpt_multi.txt 2>&1 | tail -3 | tee -a "$LOG"
    done
  done
fi

# --- Stage 3: seeds 1-2 of the `ours` grid -----------------------------------
# Each cell becomes a mean over three pools rather than one draw. Note the pool
# itself is reseeded (the fake class is resampled), so this is not merely
# adaptation noise -- it is sampling variability of the target pool, which is
# the thing a single-seed point cannot show.
if has_stage 3; then
  say "stage 3: seeds 1-2, ours, all cells"
  SEEDS="1 2" MODES=ours ./run_public_ckpt_multi.sh 2>&1 | tee -a "$LOG"
fi

# --- Stage 4: inductive check on the new cells -------------------------------
# Adapt on half the pool, evaluate on the disjoint half. The paper reports this
# alongside every transductive number as the anti-memorisation control; right
# now the public arm has it for In-the-Wild only.
if has_stage 4; then
  say "stage 4: inductive check, seed 0"
  for c in ssl_aasist_wavefake deepfense_w2v2_aasist_s42 \
           deepfense_w2v2_aasist_s2 deepfense_w2v2_aasist_s240; do
    for t in arabic dataset2 in_the_wild asvspoof2021df; do
      python public_ckpt_tta.py --mode ours --ckpt "$c" \
        --manifest "manifest_tgt_${t}_seed0.csv" --target "$t" --seed 0 \
        --batch 16 --inductive --out results_public_ckpt_multi.csv \
        --log run_log_public_ckpt_multi.txt 2>&1 | tail -3 | tee -a "$LOG"
    done
  done
  python public_ckpt_tta.py --mode ours --ckpt ssl_aasist_wavefake \
    --manifest manifest_tgt_asvspoof2019_seed0.csv --target asvspoof2019 --seed 0 \
    --batch 16 --inductive --out results_public_ckpt_multi.csv \
    --log run_log_public_ckpt_multi.txt 2>&1 | tail -3 | tee -a "$LOG"
fi

# --- Stage 5: Phase-1 detector re-run on the repaired decoder ----------------
# Not part of the ICASSP work: this belongs to the companion dataset study, and
# ran here only because the card was free. It closes a question the decoder
# repair opened -- the Phase-1 detector's published 4.81% EER was trained
# 2026-07-22, and the librosa fallback went hollow 2026-08-17; the timestamps say
# the training pre-dates the breakage, but that is inference from a directory
# mtime. A 106k-param CNN, minutes on this card. Results live on the
# dataset-study branch, not here.
if has_stage 5; then
  say "stage 5: retrain AttentiveSpecCNN on the repaired decoder"
  python train.py --epochs 15 --out attentive_spec_cnn_postfix.pt 2>&1 \
    | tail -25 | tee -a "$LOG"
fi

say "queue complete"
