---
name: operating-point-law
description: Confident-tail TTA only moves the operating point; contamination fraction Delta predicts where it lands (51/51 runs)
metadata:
  type: project
---

Established 2026-09-16 over 51 adaptation runs (4 released checkpoints, 2
corpora, spoof prevalence 0.01-0.972, budgets q 0.02-0.50):

- Adaptation shifts the **achievable accuracy ceiling** (best-threshold accuracy,
  a pure ranking quantity) by a median of 0.11 points, and the **operating
  point** (gap between that ceiling and accuracy at 0.5) by 16.55 — 155x more,
  with 0/51 exceptions. AUC moves by the same 0.11 median.
- The controlling quantity is the contamination fraction
  `Delta = sum_c (b_c - pi_c)^+` implied by the counting bound, NOT pseudo-label
  purity. Post-adaptation threshold gap rises monotonically with Delta in 10 of
  11 checkpoint/prevalence strata (Spearman +1.000).
- Operational rule, exact on 51/51 runs: adaptation raises accuracy iff its
  post-adaptation threshold gap ends below the source's.
- Purity and q both fail to order outcomes: q=0.50 (the largest possible
  symmetric budget) at pi=0.37 gains +7.5 BA while q=0.15 at pi=0.01 gains +2.6;
  purity 0.107 gains +9.4 where purity 0.320 loses 0.4.

**Why:** this replaced the earlier "accuracy collapses from 98.9% to 63%"
headline, which is a threshold artifact — in balanced accuracy that same
"collapse" is an improvement on DF-42. See [[ba-vs-accuracy-artifact]].

**How to apply:** report EER/AUC/BA and the trivial baseline whenever accuracy
appears; frame confident-tail TTA as operating-point placement, not detector
improvement. Reproduce with `audit_icassp.py` (writes contamination.csv).
