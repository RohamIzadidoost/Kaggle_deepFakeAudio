---
name: project-goal
description: "The deepfake-audio project's assignment, datasets, and the paper it must beat"
metadata: 
  node_type: memory
  type: project
  originSessionId: c7d6f656-ee0b-4dbe-b6da-6a4cf7966d06
  modified: 2026-07-22T05:41:50.133Z
---

University assignment (professor-assigned). Task: read the IEEE Access paper
"Unmasking the Fake: Machine Learning Approach for Deepfake Voice Detection"
(Gujjar et al., 2024, DOI 10.1109/ACCESS.2024.3521026), then merge + preprocess +
clean three datasets, then build a deep-learning model with a "reasonable,
explainable, novel" architecture that improves on the paper's accuracy.

The paper's headline is 99.93% accuracy (GNB on MFCC+GNB+NMF "transfer" features)
but on a tiny, different Kaggle set (DEEP-VOICE), with internally inconsistent
metrics (99.93% acc alongside 0.50 real-recall) and leakage (SMOTE-before-split,
1s segments split across train/test). It is NOT directly comparable — see
[[paper-critique]]. Real target metric is EER on ASVspoof2021 DF, not accuracy.

The three datasets (Kaggle → local `data/` folder):
- dataset_1 = ASVspoof 2021 (kaggle mohammedabdeldayem/avsspoof-2021) — mostly fake
- dataset_2 = Adarsh audio-deepfake-detection (kaggle adarshsingh0903/...) — real_samples/ + TTS generator folders
- dataset_3 = MLAAD v5 (kaggle trapka/mlaad...) — all fake, multilingual

Merged manifest: 614,764 rows, 19,251 real / 595,513 fake (~97% fake — heavy
imbalance is a core design constraint). Pipeline (build_manifest.py →
split_manifest.py → deepfake_dataset.py) works; see [[pipeline-status]].
