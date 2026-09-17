---
name: baseline-results
description: Paper-reproduction (MFCC+GNB+NMF) baseline numbers — the target the DL model must beat
metadata: 
  node_type: memory
  type: project
  originSessionId: c7d6f656-ee0b-4dbe-b6da-6a4cf7966d06
  modified: 2026-07-22T06:44:13.800Z
---

paper_baseline.py reproduces the paper's MFCC-GNB XtractNet on OUR speaker-disjoint
test set (3,803 files). Results (baseline_results.csv), EER lower=better.
NOTE: numbers below are AFTER the audio-decoding fix (see [[pipeline-status]]);
pre-fix numbers were contaminated by zero-filled silence.

Best baseline = RF on raw MFCC: EER 11.33%, F1(fake) 87.86%, bal-acc 87.51%,
AUC 0.96. KNC on MFCC next (EER 13.15%).

Key findings for the report:
- Honest evaluation gives ~13% EER / ~85% balanced acc, NOT the paper's 99.93%
  accuracy — confirms the paper's number came from a leaky/tiny setup.
- The paper's headline GNB model is the WORST here (EER 24-29%).
- The paper's core "novelty", the GNB+NMF transfer step, does NOT help under
  leakage-free eval — it slightly WORSENS the best models (RF 13.33→14.54,
  KNC 13.12→15.19 EER). Good talking point: their feature engineering doesn't
  generalize.

=> The DL model needs to beat ~13% EER / ~85% balanced accuracy. See [[project-goal]].
