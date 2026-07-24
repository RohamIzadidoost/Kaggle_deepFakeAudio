#!/bin/bash
# DA-RAD ablation on Protocol B: train on the multi-source pool, test on held-out
# In-the-Wild. Each row adds one DA-RAD component. Single-seed, reduced epochs —
# the first direct test of whether DA-RAD closes the cross-corpus gap.
set -u
cd /home/general/Desktop/deepfake_project
PY=env/bin/python
COMMON="--train_csv protocols/protoB_train.csv --val_csv protocols/protoB_itw_test.csv \
        --encoder xlsr --epochs 5 --batch_size 16 --lr 1e-3 --workers 8 --seed 42"

run () {  # $1=name  $2..=extra flags
  name=$1; shift
  echo ">>> config: $name  ($*)"
  $PY trainer.py $COMMON "$@" --out protocols/abl_${name}.pt > protocols/abl_${name}.log 2>&1
  eer=$(grep -aoE "Best val EER: [0-9.]+%" protocols/abl_${name}.log | tail -1)
  echo "$name | ${eer:-FAILED}" >> ablation_results.txt
  echo ">>> $name done: ${eer:-FAILED}"
}

echo "DA-RAD ablation (test = held-out In-the-Wild EER, lower=better)" > ablation_results.txt
run xlsr      --no-rawboost --lambda_grl 0   --lambda_anchor 0
run rawboost  --rawboost    --lambda_grl 0   --lambda_anchor 0
run grl       --rawboost    --lambda_grl 0.5 --lambda_anchor 0
run darad     --rawboost    --lambda_grl 0.5 --lambda_anchor 0.3
echo "ABLATION DONE" >> ablation_results.txt
cat ablation_results.txt
