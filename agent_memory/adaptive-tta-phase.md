---
name: adaptive-tta-phase
description: Adaptive q/E work — three mechanisms tested and killed pre-GPU; lambda stays fixed; awaiting the 35GB GPU run
metadata: 
  node_type: memory
  type: project
  originSessionId: a7e7d7d9-af89-4792-b7a8-336e8fb579a1
  modified: 2026-08-10T03:58:58.296Z
---

Phase 4 (started 2026-08-07): making the TTA knobs adapt from unlabeled target
audio. User originally asked for **q and lambda** adaptive; the delivered method
adapts **q and E, with lambda fixed at 0.3** — three proposed mechanisms were
tested on CPU and killed before any GPU time:

1. **lambda gated by pseudo-label churn under augmentation** — confounded. Moves
   lambda 0.017 across AUC .60–.99 but 0.165 across augmentation magnitude; it
   reads `augment()`'s own constants, not the corpus.
2. **`select_q_max`** — returns the grid minimum q=0.05 for every corpus on real
   audio (13/128 clips vs 78 at q=0.3). Same confound.
3. **E stopping on epoch label churn** — churn measured *exactly* 0.0000 always
   (quantile labels only move if the ranking reorders). Stopped at E_MIN every
   time and measured worse than fixed E (10.16 vs 9.38 EER). Replaced with
   `score_shift` (mean per-clip |delta score|), which decays properly.

Kept as *recorded negative results* in `test_adaptive_tta.py` — do not
resurrect without new evidence.

**Separate finding worth publishing:** `tail_gap` predicts Δ₂ at r=+0.74,
p=0.006. The paper (`main.tex:939-956`) calls this a failure since a reliability
proxy needs r<0, but the sign IS the result — high separation = saturated scores
= consistency freezes an overconfident boundary, a *calibration* property, which
is where `main.tex:952-956` predicted the answer would be. Source AUC, which
needs labels, predicts worse (r=−0.397). It doesn't cash out though: only 0.24
EER pooled separates always-lambda=0.3 from a per-point oracle, and LOTO lands
inside that and flips with grid resolution. Hence lambda fixed.

Effect sizes that set priorities: q on skewed pools ~16 EER (Protocol A), E=4→8
worth 1.2–1.6, lambda ≤0.5 and fragile.

**Not yet run:** the real grid and the Protocol A arm, both awaiting the 35 GB
GPU the user requested for Mon 2026-08-10. See [[sota-phase-status]] and
`PROJECT_LOG.md` §11.
