#!/usr/bin/env bash
# Add the three ASVspoof2021 tracks as targets for OUR model, at ten seeds.
#
# Motivation is the paper's own Limitations: "Four targets, ten seeds ... Seeds
# do not buy corpora." Source training now costs ~6.6 min, so corpora are
# affordable in a way they were not when the grid ran on a cloud allocation.
#
# Protocol note. These three are NEVER in a source pool (SOURCE_CORPORA in
# train_source_local.py pins it to the original four), so one source model per
# seed -- trained on all four originals plus MLAAD -- serves all three new
# targets. That is a clean leave-one-corpus-out: train on four corpora, test on
# an unseen fifth. It also means the original four targets' source models are
# untouched, which the pool-hash gate confirms.
#
# What must be said when reporting these: all three share ASVspoof2019 lineage,
# and ASVspoof2019 IS in the source pool. They are a CONDITION-SHIFT arm (codec,
# and for PA replay) rather than genuinely new corpora. PA is a different
# detection task altogether -- a real voice replayed, not synthesised -- and is
# reported separately from the synthetic-speech targets, never pooled with them.
#
#   SEED=3 ./run_new_targets.sh
set -u
cd "$(dirname "$0")"
source env/bin/activate

SEED="${SEED:?set SEED}"
NEW="asvspoof2021la asvspoof2021df asvspoof2021pa"
BASE="ckpt_ext/source_asvspoof2021la_seed${SEED}.pt"

export CACHE_ON_CPU=1 ADAPTIVE_SMOKE=0 INCLUDE_MLAAD=1 NEW_TARGETS=1

if [ ! -f "$BASE" ]; then
  echo "[new-targets] training shared source model for seed $SEED"
  python train_source_local.py --seeds "$SEED" --targets asvspoof2021la || exit 1
fi

# One model serves all three: the source pool does not depend on which new
# target is held out, because none of them is ever in it. Hard-linked rather
# than copied -- three 194 MB duplicates per seed would be 5.8 GB of identical
# bytes across ten seeds.
for t in $NEW; do
  dst="ckpt_ext/source_${t}_seed${SEED}.pt"
  [ -f "$dst" ] || ln "$BASE" "$dst"
done

OUR_SEEDS="$SEED" OUR_TARGETS="$(echo $NEW | tr ' ' ',')" OUR_E_SWEEP="4" \
  python adaptive_pipeline.py
