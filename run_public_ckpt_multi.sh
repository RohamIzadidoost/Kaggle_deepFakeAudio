#!/usr/bin/env bash
# Extend the public-checkpoint TTA arm from In-the-Wild to every EER target.
#
# Why: the ICASSP paper's AUC-vs-gain relation rests on four corpora scored with
# ONE model (ours), so the correlation is ecological -- within each target the
# sign reverses. Each (public checkpoint, corpus) cell here is an independent
# point on that plot, produced by a model we did not train, at no training cost.
#
# Pools come from make_target_manifests.py, which reproduces extended_pipeline.py's
# target pools exactly (size-gated against results_ext.csv), so every row is
# directly comparable to our own source model's row for the same corpus.
#
# DeepFense x asvspoof2019 is deliberately absent: those checkpoints were fitted
# on ASVspoof2019 LA train, which is where our asvspoof2019 pool is drawn from.
# public_ckpt_tta.py refuses that pair; it is memorisation, not transfer.
#
#   ./run_public_ckpt_multi.sh            # seed 0, all pairs
#   SEEDS="0 1 2" ./run_public_ckpt_multi.sh
set -u

SEEDS="${SEEDS:-0}"
BATCH="${BATCH:-16}"
OUT="${OUT:-results_public_ckpt_multi.csv}"
LOG="${LOG:-run_log_public_ckpt_multi.txt}"
MODES="${MODES:-ours}"

WAVEFAKE_TARGETS="asvspoof2019 arabic dataset2 in_the_wild"
DEEPFENSE_TARGETS="arabic dataset2 in_the_wild"

run_one() {
  local ckpt="$1" target="$2" seed="$3" mode="$4"
  # already recorded? (resume without duplicating rows)
  if [ -f "$OUT" ] && awk -F, -v s="$seed" -v t="$target" -v f="$ckpt" -v m="$mode" \
      'NR>1 && $1==s && $2==t && $3==m && $5==f {found=1} END{exit !found}' "$OUT"; then
    echo "[skip] $ckpt / $target / seed $seed / $mode already in $OUT"
    return 0
  fi
  echo "[run ] $ckpt / $target / seed $seed / $mode"
  for b in "$BATCH" 8 4; do
    if python public_ckpt_tta.py --mode "$mode" --ckpt "$ckpt" \
         --manifest "manifest_tgt_${target}_seed${seed}.csv" \
         --target "$target" --seed "$seed" --batch "$b" \
         --out "$OUT" --log "$LOG"; then
      return 0
    fi
    echo "[warn] $ckpt / $target failed at batch $b; retrying smaller" | tee -a "$LOG"
  done
  echo "[FAIL] $ckpt / $target / seed $seed / $mode -- continuing" | tee -a "$LOG"
  return 0   # one failure must not kill the grid
}

t_start=$(date +%s)
for seed in $SEEDS; do
  for mode in $MODES; do
    for target in $WAVEFAKE_TARGETS; do
      run_one ssl_aasist_wavefake "$target" "$seed" "$mode"
    done
    for s in 42 2 240; do
      for target in $DEEPFENSE_TARGETS; do
        run_one "deepfense_w2v2_aasist_s${s}" "$target" "$seed" "$mode"
      done
    done
  done
done
echo "grid done in $(( ($(date +%s) - t_start) / 60 )) min -> $OUT" | tee -a "$LOG"
