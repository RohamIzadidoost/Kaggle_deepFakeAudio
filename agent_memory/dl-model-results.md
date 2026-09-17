---
name: dl-model-results
description: AttentiveSpecCNN deep-learning model — architecture and test results vs baseline
metadata: 
  node_type: memory
  type: project
  originSessionId: c7d6f656-ee0b-4dbe-b6da-6a4cf7966d06
  modified: 2026-07-22T06:44:25.614Z
---

model.py = AttentiveSpecCNN: compact 2D CNN over log-mel (1,80,401) + a learnable
TEMPORAL ATTENTION pooling layer (the novelty vs the paper's static MFCC→GNB→NMF
feature engineering; attention weights = per-frame saliency for explainability).
106K params. train.py trains it on the speaker-disjoint splits, evaluates with EER.

Trained 15 epochs on RTX 3080 (~58s/epoch AFTER the audio-decoding fix; see
[[pipeline-status]]). Test set (3,803 files, speaker-disjoint), VALID audio:
  EER 4.81%, balanced acc 93.84%, F1(fake) 93.49%, AUC 0.992, accuracy 93.90%.

vs best classical baseline (RF on MFCC): EER 11.33%, bal-acc 87.51%.
=> EER cut 11.33% -> 4.81% (~58% relative), balanced acc +6.3 pts. Combined table
in results.csv. Checkpoint: attentive_spec_cnn.pt.

(Pre-fix contaminated run gave 8.29% EER off a silence shortcut; the fixed run is
BOTH more trustworthy AND better, since real audio teaches genuine artifacts.)

explain.py DONE: Grad-CAM (time-freq) + temporal-attention figures per clip in
explanations/. All three deliverable parts (reasonable/novel/explainable) covered.
