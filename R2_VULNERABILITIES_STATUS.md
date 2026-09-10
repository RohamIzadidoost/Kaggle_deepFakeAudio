# R2 reviewer vulnerabilities — status (2026-09-09)

A reviewer summary flagged two downgrade risks. Both are now addressed in
`main_icassp.tex` (commit history on branch `icassp-breadth`). Full numbers:
`FINDINGS_AUTONOMOUS_RUN.md` Addendum 6.

## 1. The "one-line rule" equivalence — ANSWERED (carve-out, not full defeat)

**Claim under attack:** a label-free median-threshold shift is statistically
indistinguishable from the full gradient TTA over 133 cells, so why the gradient
method?

**What we found** (`r2_vuln1_subpop.py`, stage U — per-generator ROC-AUC before/
after `adapt()`, 5 seeds):

| target | src AUC | per-sub-population AUC source → adapted | p | cells up |
|---|---|---|---|---|
| ASVspoof2019 (A01–A06) | 0.99 | 0.9907 → 0.9941 (+1.1 EER) | **1.9e-9** | **30/30** |
| dataset2 (8 TTS) | 0.71 | 0.8112 → 0.8127 | 0.65 (ns) | 22/40 |

A threshold move **cannot change any ROC-AUC**. Where the source ranking is
strong, the gradient method raises within-attack-family AUC on *every* attack
(largest on the two hardest, A04/A06). Where source AUC is 0.71 there is no
separability to sharpen and the effect is null.

**Verdict:** the median-rule equivalence holds *exactly where the method reduces
to re-calibration* — weak source ranking, or third-party checkpoints whose
ranking adaptation degrades. Given a usable source ranking, adaptation does
something threshold-free. That is the paper's answer; it does not claim the
gradient step is always necessary.

## 2. The class-balance precondition — MOSTLY REMOVED (threshold), bounded (self-training)

**Claim under attack:** accuracy collapses (87.3 → 60.8) when the target pool is
90% fake; deepfakes in the wild are a minority class.

**What we found** (`r2_vuln2_prevalence_cpu.py`, CPU — real ITW scores +
synthetic AUC ensembles, 50–95% fake):

* A parallel opus session had already killed the score-distribution strategies
  (2-component GMM, Otsu, k-means, FreeMatch, consistency-filter, prototype
  anchoring): every one buys skew-awareness by spending calibration-invariance.
* **BBSE was not tested and it works.** `p̂ = M⁻¹q`, M = source-validation
  confusion matrix, q = target prediction histogram. It estimates target
  prevalence to within **1.4%** at any skew (GMM: 28% error at 90% fake), and
  re-pointing the decision threshold at the `p̂` quantile matches the **labelled**
  accuracy-optimal threshold to within **0.4 points at every skew**, ties the
  median rule at 50/50, and is robust to moderate calibration drift (decisive
  monotone-logit-shift test). It works because it reads *source error structure*
  and the *target prediction rate*, not the target score distribution — which is
  exactly what miscalibration corrupts.
* **What BBSE does not fix:** the confident-tail pseudo-labels. Symmetric q=0.3
  real-tail purity goes 0.85 → 0.13 as skew goes 0.5 → 0.95; BBSE-asymmetric
  tails roughly double it (→ 0.52) but it stays noisy under severe imbalance.
* **Caveat:** under extreme skew the raw-accuracy-optimal threshold is not the
  balanced-accuracy-optimal one. BBSE recovers *raw* accuracy at the true prior;
  if the operator wants a balanced operating point under 95%-skew, no threshold
  rule (BBSE or oracle) gets above ~71% balanced accuracy.

**Verdict:** the balance requirement is largely removable for the *threshold*
via label-free prior estimation; the *gradient self-training* degrades
gracefully but stays bounded under severe skew. The paper states it this way.

## Files added

| file | what |
|---|---|
| `r2_vuln1_subpop.py` / `.csv` / `.out` | V1 per-generator separability (GPU, stage U) |
| `r2_vuln2_prevalence_cpu.py` / `.csv` / `.out` | V2 BBSE/GMM prevalence + drift (CPU) |
| `analyze_stage_r2.py`, `analyze_stage_t.py` | R2 / T analysis helpers |

## Update 2026-09-10 — dynamic q, done (stages V/V2/Y)

The "GPU confirmation" above is done and it is a **full result, not just a
confirmation**. `adapt_dynq()` (adaptive_pipeline.py): published `adapt()` with
the confident-tail budget split by a BBSE prevalence estimate. Controlled skew
sweep, ten seeds:

| target | skew | fixed q=0.3 | **dynq** | p (dynq vs fixed) |
|---|---|---|---|---|
| ASVspoof2019 | 0.90 | 5.34 | **3.52** | 0.004 |
| In-the-Wild  | 0.90 | 13.30 | **11.61** | 0.006 |
| ASVspoof2019 | 0.70 | 4.08 | 3.06 | 0.016 |
| arabic | 0.90 | 21.27 | 20.27 | 0.064 |
| dataset2 (AUC 0.71) | any | ~35 | ~35 | ns |

Fixed q=0.3 degrades EER under skew; dynq reverses it on the targets with a
usable source ranking (monotone in skew — the worse the skew, the more it
helps). Flat on the low-AUC target, same precondition as the whole paper.
`r2_vuln1_subpop` was V1; dynq is the V2 self-training fix. Both folded into
`main_icassp.tex`. Protocol A (97%-spoof): partial — a guarded variant is safe
but does not reliably beat source (`FINDINGS` Addendum 7).

## Not done / possible next

* Protocol A at a stronger / better-calibrated single-corpus source model
  (the 97%-skew case only failed where the source model was itself broken).
* V1 on more targets with generator labels (only asvspoof2019 + dataset2 have
  usable sub-populations locally; in_the_wild / arabic are single-generator).
* `main.tex` (11-page full version) still has the pre-breadth claims.
