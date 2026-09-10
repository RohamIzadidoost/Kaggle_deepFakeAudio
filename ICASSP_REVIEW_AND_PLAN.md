# ICASSP review of `main_icassp.tex`, and the 3-GPU-day plan to fix it

Written 2026-09-10, autonomously. Section 1 is me reviewing the paper as an
ICASSP reviewer would. Section 2 is the plan. Section 3 tracks execution.

---

## 1. Review

**Recommendation as submitted: 3/5 — Weak Reject (borderline).**
Not because the work is weak; because the *paper* argues itself out of its own
contribution. The evidence base is unusually thorough and unusually honest, and
that honesty is currently deployed as self-refutation rather than as scope.

### Summary of submission
A fine-tuned XLS-R detector's ROC-AUC transfers cross-corpus while its decision
threshold does not. The authors exploit this with an unsupervised TTA method
(confident-quantile pseudo-label self-training + channel consistency, on
LayerNorm affine + head only) and evaluate it on a 4-target 10-seed
leave-one-corpus-out protocol, 2 lineage-independent corpora, and 9 third-party
checkpoints over 369 cells.

### Strengths
- S1. The ranking-vs-calibration diagnosis is genuinely useful and, as far as I
  know, not stated this cleanly for audio anti-spoofing.
- S2. The third-party arm (9 checkpoints, 5 architectures, 10 corpora, 369
  cells) is far beyond what a 4-page submission normally carries, and the
  architecture-independence evidence (an XLS-R-designed rule re-calibrating a
  mel-spectrogram AST unchanged) is compelling.
- S3. The mechanism test (proxy A-distance *rises*) is the right experiment and
  the answer is counter-intuitive and well-argued.
- S4. Statistics are handled properly: per-target Wilcoxon, Holm correction,
  bootstrap, explicit seed counts, admitted nulls.

### Weaknesses (ordered by how much they cost the score)

**W1 — The paper defeats itself with the median-threshold control.**
"A label-free median-threshold rule captures most of the accuracy gain" and
"*beats* it over 63 third-party cells" appear in the abstract and conclusion.
A reviewer reads that as: the contribution is a one-line heuristic and the
gradient machinery is unnecessary. The carve-out (per-attack AUC) is real but
is presented as a rescue clause, not as the claim.
*This is the single largest cost, and it is largely a framing error:* EER and
AUC are threshold-invariant. A median-threshold move **cannot change EER or
AUC by construction**. Every headline number in Table I is EER/AUC. So the
median rule is not a competitor to the main table at all — it is a competitor
only on `acc@0.5`, a metric the paper did not need to report. The paper
volunteered a strawman and then let it win.

**W2 — The backbone is not competitive, so "improves EER" has no absolute anchor.**
The only protocol with published baselines (ASVspoof2021-DF) puts the source
model at 26.33% EER against published 8.54% (Wav2Vec2-AASIST) and 1.95%
(Wav2DF-TSL). The paper says this plainly, which is honest, but it converts
every gain into "relative improvement on a weak model." At ICASSP that is a
downgrade to *significant-but-unanchored*. Worse, the diagnosis under test
("mis-calibration is the cross-corpus failure mode") is precisely the kind of
claim that a reviewer will suspect is an artefact of an under-trained source
model. **We must show the effect on a competitive detector.**
Looking at the code: `protocol_a.py` trains the top-4 XLS-R layers *and* the
head at a single `LR = 2e-4`. Published SSL anti-spoofing recipes use
~1e-6 for the SSL block. The run log shows the source loss bottoming at 1e-4
(memorisation). This is very likely a recipe bug, not a ceiling.

**W3 — Missing every modern TTA baseline.**
Tent (ICLR 2021) is the only TTA comparison point. Tent's instability is *known*
and the literature's answer to it — SAR (reliable + sharpness-aware entropy),
EATA (sample-efficient anti-forgetting), SHOT (information-maximisation +
pseudo-labels) — is exactly the family this paper's method belongs to. A
reviewer will write: "the authors show Tent collapses, then propose a stability
fix, without comparing to the published stability fixes." That objection alone
can sink the paper, and it is cheap for us to close.

**W4 — The headline hyper-parameter is admittedly mis-specified and unselectable.**
"E is not a hyperparameter but a mis-specified budget" and the strongest results
(Arabic +3.65, AST reversals) come from E=32, chosen post hoc. There is no
label-free rule for picking it. Reviewer: "your best numbers require an oracle
you say you don't have."

**W5 — Method novelty is thin as stated.**
Confident-quantile pseudo-labels + consistency regularisation + norm-layer-only
adaptation is a recombination of FixMatch/SHOT/Tent components. The genuinely
new piece — a *prior-corrected* pseudo-label budget that removes the
class-balance precondition — is buried in the Limitations section as a patch.
That is backwards. Class skew is the single most-cited reason TTA is not
deployed in anti-spoofing (real pools are >>50% bonafide, benchmark pools are
97% spoof); a method that is provably skew-robust label-free is a contribution.

**W6 — The submission is eight studies wearing a four-page coat.**
Six contributions in the intro, six clauses in the abstract, eight bolded
sub-results in Section 4. No single sentence a reviewer can repeat back.

**W7 — minor:** RawNet2Lite at AUC 0.40–0.65 reads as a broken baseline rather
than a finding about SSL. Stating "we did not run adaptation on it" invites
"then why is it in the paper?"

### What would move this to Strong Accept
1. Show the effect survives on a *competitive* detector, with an absolute
   number next to published ones.
2. Beat the modern TTA baselines that were designed to fix exactly the failure
   mode this paper starts from.
3. Make the threshold-free (EER/AUC) claim the headline, and demote the
   median-threshold rule from rival to corollary.
4. Turn the admitted E defect into a label-free selection rule.
5. Lead with the skew-robust prior-corrected method as *the method*.

---

## 2. Plan (3 GPU days, RTX 3080 10 GB)

Budget: ~72 wall-hours, ~55 h of GPU compute, 4 jobs, each independently
useful so a failure in one does not sink the pass.

| Job | Kills | Est. GPU-h |
|---|---|---|
| **A. Competitive Protocol-A source model + TTA on it** | W2 | 16 |
| **B. Modern TTA baselines: SAR / EATA / SHOT** | W3, W5 | 14 |
| **C. Label-free step budget + stopping rule** | W4 | 8 |
| **D. Threshold-free breadth + skew-robust headline** | W1, W5 | 10 |
| E. Rewrite (no GPU) | W1, W6, W7 | 0 |

Details and gates are recorded per job in Section 3 as they run.

## 3. Execution log

(appended as jobs complete)

### 2026-09-10 17:30–18:00 — build-out (no GPU contention)

New code, all smoke-tested before any GPU hour was spent on it:

* `protocol_a_public.py` — the method on **published** checkpoints over a
  **whole** benchmark (no 6k subsample): `PUBA_CORPUS=df2021` is the official
  ASVspoof2021-DF eval partition (400,435 trials on disk, 97.2% spoof),
  `PUBA_CORPUS=itw` is all 31,779 In-the-Wild clips. Arms: source /
  fixed-q TTA / BBSE-prior-corrected TTA, plus a label-free median-threshold
  control computed for free on every row.
  Decode and GPU work are overlapped by one prefetch thread — measured 533
  clips/s decode vs 204 clips/s GPU, so serial execution capped throughput at
  147; the overlap gets 196 and saves ~4 GPU-h over the nine passes.
* `tta_baselines.py` — SHOT, ETA/EATA, SAR, against a callback protocol so the
  same implementations run on our detector and on third-party checkpoints.
* `adaptive_pipeline.py` — two new env-gated grid arms, `TTA_BASELINES=1`
  (the above vs ours vs Tent, same folds and seeds) and `STOP_TRACE=1`
  (one long instrumented budget per fold, per-epoch label-free monitors).
* `analyze_stop_rule.py` — fits and leave-one-target-out validates the
  stopping rules from the trace.

**Two ported constants were silently vacuous at C=2 and had to be fixed.**
Both would have shipped a baseline that is a no-op and reads as a negative
result — the exact failure this repo's own notes warn about:

1. *EATA's redundancy filter.* `|cos(p, ema)| < 0.05` is calibrated for
   C=1000, where two confident vectors are near-orthogonal. On the binary
   simplex the minimum attainable cosine is ~0.12, so **nothing** ever passes
   and ETA becomes inert after the first batch. Fixed by running the
   reliability filter alone — which is EATA's own published ablation — rather
   than inventing a new threshold.
2. *SAR's model-recovery trigger.* Published `e0 = 0.2` sits far below
   ImageNet's `E0 = 0.4·ln 1000 = 2.76`. At C=2, `E0 = 0.277`, so the constant
   0.2 lands *inside* the reliable band, the guard fires on every batch and SAR
   is frozen at its initial weights (verified: 32 resets in 32 steps, zero net
   movement). Fixed by keeping SAR's relative position,
   `e0 = (0.2/ln 1000)·ln C = 0.020`. Verified: 0 resets, real movement.

Both now log a warning when the reliability filter admits <1% of samples, so a
future inert baseline announces itself.

### Framing decision recorded before the results land (Job E, no GPU)

The median-threshold "equivalence" (W1) is not a finding that needs new
experiments to defeat. It needs the paper to state what it already knows:

* **A threshold cannot change EER or AUC.** Both are computed from the score
  ROC; neither takes a threshold as input. So the median rule's row in
  Table~I would be *bit-identical to source-only*. Every headline number in
  this paper is EER/AUC. The median rule therefore cannot explain any of them
  and never could.
* The 94%-recovery result is about `acc@0.5` alone — a metric this paper did
  not have to report. Read correctly it is not a rival method, it is the
  **confirmation of the mechanism**: if the gain at a fixed threshold is
  recoverable by moving the threshold, then the gain at a fixed threshold *was*
  a threshold problem. That is the paper's own thesis, measured.
* **And the median rule was only ever evaluated where its assumption holds.**
  Checked: all 63 cells in `threshold_control_multi.csv` have `pos_rate`
  between 0.489 and 0.539 — the extended pipeline balances every target pool, so
  the one-line rule was tested exclusively at the 50/50 prevalence at which the
  median *is* the optimal threshold by construction. On the official
  ASVspoof2021-DF eval at its real 97.2% spoof rate the same rule gives 55.8%
  accuracy against the checkpoint's shipped 98.6% (smoke run, 2,000 trials).

So: median-rule demoted from rival to corollary; BBSE prior estimation promoted
from limitation-section patch to the component that makes the calibration story
survive contact with a deployment-realistic prior.

### Job A1, first cell — official ASVspoof2021-DF eval, published checkpoint

`deepfense_w2v2_aasist_s2` (XLS-R 300M + AASIST, trained on ASVspoof2019-LA
train), scored over the official DF eval partition, 400,435 trials on disk,
true spoof rate **97.22%**:

| | value |
|---|---|
| EER | **4.51%** |
| AUC | 0.9923 |
| accuracy @ shipped tau=0.5 | **98.87%** |
| accuracy @ **median rule** | **59.06%** |
| calibration deficit (100-EER) - acc | **-3.38** |

Three things this settles.

1. **An absolute anchor at last.** 4.51% EER on the official protocol against
   published Wav2Vec2-AASIST 8.54% and challenge-era DF top-1 ~15.6%. Every
   claim we now make about adaptation is made on a detector that is
   state-of-the-art on the benchmark, not on our own 26.33% model. W2 is
   answered by substitution rather than by retraining.
2. **The one-line median rule is destroyed at deployment prevalence.** 59.06%
   vs the shipped 98.87%. The rule was only ever competitive because every pool
   it was tested on had been balanced to ~50/50 by our own pipeline. W1 is
   answered with data, on the field's standard benchmark.
3. **This cell has no calibration deficit to repair** (deficit -3.38: the
   shipped threshold is already *better* than the ranking's break-even). Our
   own scope map predicts adaptation is inert here, so the TTA arms on this
   corpus are a falsification test of the scope map, not a bid for a better
   number. Reporting a predicted null on a SOTA checkpoint is worth more than
   another gain on a weak one.

### Job A1 continued — the DF arms are a headline, not a null

Same checkpoint, same official 400,435-trial eval, adapting on an unlabelled
20,000-clip subsample of it:

| arm | EER | AUC | acc@0.5 | deficit |
|---|---|---|---|---|
| source | **4.510** | .9923 | **98.87** | -3.38 |
| naive fixed-$q{=}0.3$ TTA (published config) | 5.836 | .9846 | **62.96** | **+31.20** |
| median-threshold rule (label-free, no gradient) | 4.510 (unchanged, by construction) | — | **59.06** | — |

The published configuration, applied unmodified to a state-of-the-art
checkpoint on the field's standard benchmark, **damages it**: accuracy
98.87 -> 62.96, EER 4.51 -> 5.84, and it manufactures a 31-point calibration
deficit where there was none. The mechanism is exactly the one the prior
argument predicts — symmetric $q{=}0.3$ labels the bottom 30% of a 97.2%-spoof
pool "confidently real", so self-training is taught that a great deal of spoof
is bona fide.

And the prior estimate that fixes it lands: **BBSE $\hat\pi_{fake}$ = 0.9715 vs
true 0.9708**, on a third-party checkpoint we did not train, at 97% skew, from
$M=[[.997,.003],[0,1]]$ measured on the checkpoint's own training corpus. The
budget becomes (0.017, 0.583) instead of (0.3, 0.3).

This reframes the whole paper's contribution and is worth spending the two
remaining seeds on: the class-balance assumption buried inside every
confident-quantile TTA method is not a footnote, it is a *failure mode on the
standard benchmark*, and a label-free prior estimate removes it.

### Job A1 complete for seed 2 — the headline

Official ASVspoof2021-DF eval, 400,435 trials on disk, 97.22% spoof,
`deepfense_w2v2_aasist_s2` (XLS-R 300M + AASIST, ASVspoof2019-LA train).
Everything below is label-free: no target labels, no retraining, 16,706
adapted parameters.

| arm | EER | AUC | acc@0.5 | deficit |
|---|---|---|---|---|
| source | 4.510 | .9923 | 98.87 | -3.38 |
| naive fixed-$q{=}0.3$ (the published config) | 5.836 | .9846 | 62.96 | +31.20 |
| **prior-corrected (BBSE) budget** | **4.020** | **.9931** | **98.95** | -2.97 |
| — same model, disjoint/inductive half | 4.034 | .9931 | 98.95 | — |

* **4.51 -> 4.02% EER, an 11% relative reduction**, on a checkpoint that was
  already better than the published Wav2Vec2-AASIST Protocol-A baseline
  (8.54%). Both EER *and* AUC improve, so no threshold rule can account for it.
* The naive configuration goes the other way and **damages** the same
  checkpoint (EER +1.33, accuracy -35.9). The entire difference between the two
  arms is one label-free number: $\hat\pi_{fake}=0.9715$ against a true 0.9708.
* The inductive check reproduces it (4.034 vs 4.020), so it is not transduction.

### The prior sweep, from cached scores (CPU, no GPU)

`analyze_prevalence_rules.py` resamples the same score distribution to a range
of target priors. Raw accuracy:

| target prior | shipped $\tau{=}0.5$ | median rule | BBSE threshold | oracle |
|---|---|---|---|---|
| 0.50 | 88.27 | **95.55** | 88.34 | 95.70 |
| 0.70 | 92.73 | 81.91 | 92.75 | 95.72 |
| 0.90 | 97.30 | 62.01 | 97.30 | 97.34 |
| 0.9722 (true) | 98.92 | 58.96 | 98.94 | 99.09 |

The median rule is the *best* rule at 50/50 and the worst by 40 points at the
benchmark's actual prior. That is not a coincidence, it is its assumption. The
paper's "a one-line rule recovers 94% of the gain" was measured on pools our
own pipeline had balanced to 0.489-0.539.

**Two things to state honestly about BBSE here.** (i) Its prior estimate carries
a positive bias that is largest at balanced priors ($\hat\pi$ 0.61 at a true
0.50) and shrinks toward the truth at high skew (0.9738 at 0.9722): BBSE assumes
*label* shift, and DF's codec conditions are also a covariate shift, so the
in-domain $M$ under-states target error. (ii) On a checkpoint whose shipped
threshold is already near-optimal, the BBSE *threshold* buys essentially nothing
over $\tau{=}0.5$ (98.94 vs 98.92). The value of the prior estimate here is in
the **pseudo-label budget**, not the threshold — which is exactly where the
4.51 -> 4.02 came from, and is the claim the paper should make.

### Self-audit of claims (no GPU)

Checked every number in the v2 manuscript against a source in this repo. All
carried over from v1 verify. Two corrections and one open item:

* **Corrected (was over-stated).** I first wrote that EATA's redundancy filter
  is "unreachable" at $C{=}2$. Measured: the minimum attainable cosine over the
  binary simplex is $0.71$ at an EMA of $(.5,.5)$, $0.39$ at $(.7,.3)$, $0.11$
  at $(.9,.1)$ — all above the published $0.05$ — but $0.031$ at $(.97,.03)$
  and below. So it is not unreachable; it is degenerate in two different ways
  depending on where the running average sits, and on a 97%-spoof pool the
  second regime (admits only minority-predicted samples) is the live one.
  Paper and code comment now say exactly that.
* **OPEN, must be fixed before submission.** The published Protocol-A EER
  figures used as the scale row of Table~I (Wav2DF-TSL 1.95, SSL+SLS 2.87,
  Wav2Vec2-AASIST 8.54, challenge DF top-1 15.64) are carried over from
  `protocol_a.py`'s docstring and are **not currently cited**. Each needs a
  bibliography entry and a verification against the source paper. Do not submit
  with an uncited comparison table.
* The DF eval is 400,435 of 611,829 official trials (part03 absent locally).
  Stated in the setup section with the uniformity check; keep it stated.
