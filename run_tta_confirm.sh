#!/bin/bash
# Confirm the winning config (self-training + consistency, NO anchor) is:
#   (1) stable across 3 seeds, and
#   (2) generalises to a SECOND unseen OOD corpus (ASVspoof2021-DF, which was not
#       in the source model's training pool), not just In-the-Wild.
set -u
cd /home/general/Desktop/deepfake_project
PY=env/bin/python

run () {  # $1=target_csv  $2=tag
  for s in 0 1 2; do
    $PY tta.py --mode realmanifold --no-anchor --seed $s \
        --adapt_csv "$1" --tag "$2" 2>&1 | grep -aE "^\["
  done
}

echo "### st+consistency on In-the-Wild (3 seeds) ###"
run protocols/protoB_itw_test.csv st_cons_itw

echo "### st+consistency on ASVspoof2021-DF (3 seeds, 2nd OOD target) ###"
run protocols/protoA_test.csv st_cons_df

echo "=== TTA CONFIRM DONE ==="
grep -E "st_cons_" tta_results.txt
