#!/bin/bash
# Validate the Real-Manifold TTA signal: multi-seed stability, component ablation,
# and an inductive (adapt-on-A, eval-on-disjoint-B) check to rule out transductive
# overfitting. All on held-out In-the-Wild, unsupervised.
set -u
cd /home/general/Desktop/deepfake_project
PY=env/bin/python
: > tta_results.txt

echo "### multi-seed (full method) ###"
for s in 0 1 2; do
  $PY tta.py --mode realmanifold --seed $s --tag full 2>&1 | grep -aE "^\["
done

echo "### ablation (seed 0) ###"
$PY tta.py --mode realmanifold --no-anchor --no-consistency --seed 0 --tag st_only      2>&1 | grep -aE "^\["
$PY tta.py --mode realmanifold --no-consistency            --seed 0 --tag st_anchor     2>&1 | grep -aE "^\["
$PY tta.py --mode realmanifold --no-anchor                 --seed 0 --tag st_cons       2>&1 | grep -aE "^\["

echo "### inductive (adapt on ITW-A, eval on disjoint ITW-B) ###"
$PY tta.py --mode realmanifold --seed 0 --adapt_csv protocols/protoB_itw_A.csv \
      --eval_csv protocols/protoB_itw_B.csv --tag inductive 2>&1 | grep -aE "^\["

echo "=== TTA VALIDATION DONE ==="
echo "---- collected ----"; cat tta_results.txt
