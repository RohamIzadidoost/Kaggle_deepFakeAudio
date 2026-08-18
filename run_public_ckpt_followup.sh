#!/usr/bin/env bash
# Follow-up runs, queued behind the main grid.
#
# The main grid runs the ablation (st_only), the anti-memorization check
# (inductive) and the Tent contrast on ash56 ONLY. That leaves the ablation
# asymmetric: DeepFense is where the AUC gain was ~10x larger, so whether the
# same self-training/consistency split holds there is exactly the question the
# ash56 ablation raises and cannot answer.
#
# This waits for the main launcher to exit, then fills those three cells for
# DeepFense s42 -- if there is still GPU window left.
#
#   MAIN_PID=<launcher pid> ./run_public_ckpt_followup.sh

set -u
cd "$(dirname "$0")"

MAIN_PID="${MAIN_PID:-0}"
# 21 h window opened ~17:05 on 2026-08-17; stop starting new runs after this.
DEADLINE="${DEADLINE:-$(date -d '2026-08-18 13:30' +%s)}"
BATCH="${BATCH:-16}"
LOG=run_log_public_ckpt.txt

say() { echo "[$(date +%H:%M:%S)] [followup] $*" | tee -a "$LOG"; }

if [ "$MAIN_PID" != "0" ]; then
  say "waiting for main grid (PID $MAIN_PID) to finish..."
  while kill -0 "$MAIN_PID" 2>/dev/null; do sleep 120; done
  say "main grid finished"
fi

source env/bin/activate

run() {
  local desc="$1"; shift
  local now; now=$(date +%s)
  # each of these is ~35-70 min; don't start one we cannot finish
  if [ "$now" -ge "$DEADLINE" ]; then
    say "past deadline, skipping: $desc"; return 0
  fi
  say ">>> $desc"
  if timeout 10800 python public_ckpt_tta.py "$@" --batch "$BATCH"; then
    say "<<< done: $desc"
  else
    say "<<< FAILED: $desc -- continuing"
  fi
}

CK=deepfense_w2v2_aasist_s42
ASH=ssl_aasist_wavefake

# Ordered by value. The grid measured `st_only` and `ours` but never
# `cons_only`, so the 2x2 behind the paper's central mechanistic claim -- that
# self-training recalibrates while consistency adapts the representation, the
# latter carrying ~89% of the EER gain -- has a missing cell. Whether
# consistency ALONE suffices decides between "the structure matters" and "just
# use the consistency term", which is a materially different paper.

# 1. the missing ablation cell, on the checkpoint the 89% figure came from
run "$ASH cons_only seed0" --mode cons_only --ckpt "$ASH" --seed 0

# 2. does the same decomposition hold where the AUC gain was ~10x larger?
run "$CK st_only seed0"    --mode st_only   --ckpt "$CK"  --seed 0
run "$CK cons_only seed0"  --mode cons_only --ckpt "$CK"  --seed 0

# 3. anti-memorization on the second family, so the inductive check is not
#    a property of one checkpoint
run "$CK ours INDUCTIVE seed0" --mode ours --ckpt "$CK" --seed 0 --inductive

# 4. does Tent's AUC-degradation signature reproduce, or was it one model?
run "$CK tent seed0" --mode tent --ckpt "$CK" --seed 0

say "FOLLOWUP DONE"
