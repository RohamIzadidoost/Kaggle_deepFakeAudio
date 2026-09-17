---
name: cross-dataset-results
description: Leave-one-source-out generalization test — model does NOT transfer across datasets
metadata: 
  node_type: memory
  type: project
  originSessionId: c7d6f656-ee0b-4dbe-b6da-6a4cf7966d06
  modified: 2026-07-22T07:23:44.230Z
---

cross_dataset.py trains a fresh AttentiveSpecCNN on two sources, tests on the
held-out third (10 epochs). Results (cross_dataset_results.csv):
- Hold out ASVspoof: EER 44.9%, AUC 0.56 (~chance)
- Hold out dataset_2: EER 57.5%, AUC 0.42 (BELOW chance — inverted!)
- Hold out MLAAD: 48.4% fake-recall on unseen generators (~chance)

KEY FINDING (great for the report, shows integrity): the strong in-domain
result (4.81% EER, model trained on all 3 sources — see [[dl-model-results]]) does
NOT reflect universal deepfake-artifact detection. When a source is fully unseen,
performance collapses to near/below chance. So the detector leans heavily on
dataset-specific cues (recording/domain signatures), the SAME failure mode the
paper had. In-domain vs cross-domain gap quantifies this reliance.

Caveats: single-run, fixed-0.5-threshold probes; AUC is the robust signal.
ASVspoof-holdout fold is also real-starved (only ~2.3k LibriSpeech real to train
on). MLAAD holdout has no real audio so only fake-recall is reportable.

Natural next step if pursuing: domain-generalization (augmentation, more diverse
real speech, source-adversarial training) to close the gap.
