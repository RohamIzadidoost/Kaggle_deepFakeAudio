#!/bin/bash
# GO/NO-GO: does DA-RAD help once the encoder can ADAPT (top-4 XLS-R layers
# fine-tuned)? Tests on held-out In-the-Wild. Three configs:
#   xlsr    = fine-tuned baseline (no DG)          -> should be MUCH better than frozen (31%)
#   anchor  = baseline + Real-Anchored loss only   -> isolates the C2 novelty
#   darad   = full DA-RAD (RawBoost+GRL+Anchor)    -> the method
# If anchor/darad beat xlsr -> signal to scale up on better GPUs.
set -u
cd /home/general/Desktop/deepfake_project
PY=env/bin/python
COMMON="--train_csv protocols/protoB_train.csv --val_csv protocols/protoB_itw_test.csv \
        --encoder xlsr --n_trainable_layers 4 --epochs 4 --batch_size 16 --lr 1e-4 \
        --workers 8 --seed 42"

run () { name=$1; shift
  echo ">>> $name ($*)"
  $PY trainer.py $COMMON "$@" --out protocols/ft_${name}.pt > protocols/ft_${name}.log 2>&1
  eer=$(grep -aoE "Best val EER: [0-9.]+%" protocols/ft_${name}.log | tail -1)
  echo "$name | ${eer:-FAILED}" >> gonogo_results.txt
  echo ">>> $name done: ${eer:-FAILED}"
}

echo "GO/NO-GO fine-tuned (top-4 layers) — test=held-out In-the-Wild, lower EER better" > gonogo_results.txt
run xlsr    --no-rawboost --lambda_grl 0   --lambda_anchor 0
run anchor  --no-rawboost --lambda_grl 0   --lambda_anchor 0.3
run darad   --rawboost    --lambda_grl 0.5 --lambda_anchor 0.3
echo "GONOGO DONE" >> gonogo_results.txt
cat gonogo_results.txt
