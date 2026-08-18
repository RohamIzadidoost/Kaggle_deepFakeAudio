#!/usr/bin/env bash
# Run our published TTA on public checkpoints, once the GPU is actually free.
#
# The box is shared: another project's sweep (openlabel.sweep.train) holds ~6.7 GB
# of the 3080. This waits for that PID to exit AND for memory to actually come
# back before touching the GPU, so we never OOM it or ourselves.
#
# Experiments are ordered by value, each writes to results_public_ckpt.csv
# incrementally, and one failure does not stop the rest -- an interrupted run
# still leaves usable rows (same discipline as the overnight pipeline).
#
#   ./run_public_ckpt.sh            # wait for GPU, then run the grid
#   WAIT_PID=0 ./run_public_ckpt.sh # skip the wait (GPU already free)

set -u
cd "$(dirname "$0")"

WAIT_PID="${WAIT_PID:-50253}"     # the other project's training sweep
MAX_WAIT_SEC="${MAX_WAIT_SEC:-46800}"   # 13 h ceiling
NEED_FREE_MB="${NEED_FREE_MB:-8000}"
BATCH="${BATCH:-16}"
LOG=run_log_public_ckpt.txt

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------- wait for GPU
if [ "$WAIT_PID" != "0" ]; then
  say "waiting for PID $WAIT_PID (other project's sweep) to finish..."
  waited=0
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 120
    waited=$((waited + 120))
    if [ "$waited" -ge "$MAX_WAIT_SEC" ]; then
      say "hit ${MAX_WAIT_SEC}s ceiling and PID $WAIT_PID is still alive; not preempting it. Exiting."
      exit 1
    fi
    [ $((waited % 1800)) -eq 0 ] && say "  still waiting (${waited}s elapsed)"
  done
  say "PID $WAIT_PID has exited after ${waited}s"
fi

# memory can lag process exit
for _ in $(seq 1 60); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "${FREE:-0}" -ge "$NEED_FREE_MB" ] && break
  say "  only ${FREE}MB free, need ${NEED_FREE_MB}MB; waiting..."
  sleep 60
done
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "${FREE:-0}" -lt "$NEED_FREE_MB" ]; then
  say "GPU still busy (${FREE}MB free). Exiting rather than fighting for memory."
  exit 1
fi
say "GPU free: ${FREE}MB. Starting."

source env/bin/activate

run() {  # run <description> <args...>
  local desc="$1"; shift
  local rc err
  err=$(mktemp)
  # GPU memory headroom is the one thing that could not be tested while the
  # other project's sweep held the card, so fall back on batch size rather than
  # losing the run. The OOM traceback goes to stderr, not to the python log, so
  # capture it here.
  for b in "$BATCH" 8 4; do
    say ">>> $desc (batch $b)"
    # plain redirect, not process substitution: we must be able to grep $err
    # immediately after, with no race against a background tee flushing
    timeout 10800 python public_ckpt_tta.py "$@" --batch "$b" 2>"$err"
    rc=$?
    cat "$err" >&2
    if [ $rc -eq 0 ]; then say "<<< done: $desc"; rm -f "$err"; return 0; fi
    if [ "$b" != "4" ] && grep -qi "out of memory\|CUDA out of memory" "$err" 2>/dev/null; then
      say "<<< OOM at batch $b, retrying smaller"
      continue
    fi
    say "<<< FAILED (rc=$rc): $desc -- continuing"
    rm -f "$err"
    return 1
  done
  rm -f "$err"
}

# ------------------------------------------------------------ 0. gates on GPU
# Polarity must be re-confirmed on the device we actually score on.
#
# These are BLOCKING. HANDOFF_PUBLIC_CKPT.md: "Do not proceed past a failed gate
# -- a silently broken port makes every adapted number meaningless." Since this
# runs unattended, a non-blocking gate would quietly produce a full grid of
# garbage. `run` returns non-zero on failure, and here that aborts everything.
for CK in ssl_aasist_wavefake deepfense_w2v2_aasist_s42; do
  if ! run "GATE signcheck $CK" --mode signcheck --ckpt "$CK" --limit 400; then
    say "GATE FAILED for $CK -- aborting the entire run rather than producing"
    say "numbers from a model whose score polarity is unverified."
    exit 1
  fi
done
say "all polarity gates passed on GPU"

# --------------------------------------------------- 1. headline: ash56 on ITW
run "ash56 source (full ITW)"        --mode source --ckpt ssl_aasist_wavefake --seed 0
run "ash56 ours transductive seed0"  --mode ours   --ckpt ssl_aasist_wavefake --seed 0

# -------------------------------- 2. second training corpus (ASVspoof2019) ---
run "deepfense s42 ours seed0" --mode ours --ckpt deepfense_w2v2_aasist_s42 --seed 0

# ------------------------------------------- 3. anti-memorization + decomposition
run "ash56 ours INDUCTIVE seed0" --mode ours    --ckpt ssl_aasist_wavefake --seed 0 --inductive
run "ash56 st_only seed0"        --mode st_only --ckpt ssl_aasist_wavefake --seed 0

# Tent is the load-bearing contrast: PROJECT_LOG S5 records that naive entropy
# minimisation collapses to ~chance here. If it collapses on a third-party
# checkpoint too while ours improves, the structure -- not our source model --
# is what is doing the work.
run "ash56 tent seed0" --mode tent --ckpt ssl_aasist_wavefake --seed 0

# ------------------------------------------------------- 4. seed variance -----
for S in 1 2; do
  run "ash56 ours seed$S" --mode ours --ckpt ssl_aasist_wavefake --seed "$S"
done
for CK in deepfense_w2v2_aasist_s2 deepfense_w2v2_aasist_s240; do
  if [ ! -s "public_ckpt/$CK/best_model.pth" ]; then
    say "skip $CK (not downloaded)"; continue
  fi
  run "$CK ours seed0" --mode ours --ckpt "$CK" --seed 0
done

say "ALL DONE"
