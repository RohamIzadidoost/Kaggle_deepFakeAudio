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

### A claim of ours that this pass weakened, and a better one that replaces it

`prior_estimators.py` compares four label-free prior estimators on the cached
official-DF scores of `deepfense_w2v2_aasist_s2`, resampled to a range of
priors. Mean absolute error over all priors:

| estimator | needs | mean abs. error |
|---|---|---|
| 2-component GMM | nothing | **0.028** |
| SLD/EM~(Saerens et al.) | only the *source prior* (a number on the model card) | 0.040 |
| BBSE | labelled source-domain data | 0.041 |
| mean predicted probability | nothing (assumes calibration) | 0.042 |

**This does not reproduce the repo's earlier "BBSE 1.4% vs GMM 28%".** That
comparison was measured on In-the-Wild scores plus synthetic ensembles and on
the RawBoost Protocol-A model, whose score histogram is degenerate. On a
checkpoint with AUC $.992$ the histogram is cleanly bimodal and the GMM has an
easy job. Both results are real; they are about different score distributions.

Two consequences, both improvements:

1. **The paper should not claim BBSE is the best estimator.** It should claim
   what is actually true and more useful: *any* prior estimate fixes the budget,
   because at the real DF prior all four agree to within $0.02$ and give
   effectively the same tail split. BBSE is the default because it is the one
   that does not read the target score histogram, and so is the one that
   survives a degenerate one — which is a statement about robustness, not
   accuracy.
2. **The "needs labelled source data" limitation is largely removable.** SLD/EM
   needs only the source class balance, a number model cards state, and matches
   BBSE here. That un-excludes the three undocumented-provenance checkpoints.

All four estimators share the same bias pattern (over-estimating $P(\text{fake})$
by ~0.11 at a true 0.50, converging to the truth at high skew). A bias common to
four unrelated estimators is a property of the *shift*, not of any estimator:
DF's codec conditions raise the detector's real-class error above its
source-domain rate, and every one of these methods assumes that rate carries
over.

### Correction found while queueing: all ten seeds are available locally

`adaptive_pipeline.py` carried a comment saying only seeds 0-2 have local source
checkpoints. `ckpt_ext/` in fact holds **ten** seeds for each of the four EER
targets (90 files, 9.5 GB). The default stays at 0-2 because the six-arm
baseline grid costs ~21 min per fold, but the resume guard is keyed on
(seed, target, method, setting), so `OUR_SEEDS=3,4` after the queue's stage 4
simply adds those folds and lifts the baseline comparison to five seeds per
target without redoing anything. Queued as follow-up work.

### Stage 1, first arm: Tent's DF number is a trap, and catching it is a result

Tent on the official DF eval, same checkpoint and pool as the headline:

| arm | EER | AUC | distinct scores | largest tied value |
|---|---|---|---|---|
| source | 4.510 | .9923 | 2,762 | 9.8% |
| Tent | **4.062** | **.9765** | **1,247** | **93.0%** (98.07% sit at exactly 1.0) |
| ours, prior-corrected | **4.020** | **.9931** | 1,764 | 26.6% |

Read the EER column alone and Tent *improves* the checkpoint. It does not. It
saturates $98.07\%$ of $400{,}435$ scores to the numerical ceiling, leaving
$1{,}247$ distinct values in the entire pool, so its "EER" is a statement about
how the ROC breaks ties rather than about ranking. AUC, which tie-breaking
cannot rescue, falls $.9923\!\to\!.9765$.

Two things follow. First, our arm beats Tent on both metrics, and decisively on
the one that is well-defined. Second --- and this is the more useful point for
the paper --- **an entropy-minimisation collapse can present as an EER
improvement**, so a TTA study that reports EER without a resolution check can
record a collapse as a win. `score_resolution.py` now computes distinct-score
counts and largest-tie fraction for every cached arm and flags any arm whose
scores are more than half a single tied value.

This also sharpens the manuscript's existing Tent narrative: on balanced pools
Tent is unpredictably catastrophic; on the deployment-prevalence pool it is
quietly destructive in a way the field's headline metric does not show.

### Stage 1, second arm: SHOT damages it too — and its loss says why

SHOT on the same checkpoint and official DF pool: EER $4.51\!\to\!6.38$,
AUC $.9923\!\to\!.9725$, accuracy $98.87\!\to\!77.64$, deficit
$-3.38\!\to\!+15.99$. No score saturation (6,973 distinct values, largest tie
$2.5\%$), so unlike Tent this is a genuine ranking and calibration loss, not a
tie-breaking artefact.

The mechanism is not incidental, it is **written into the objective**. SHOT
minimises $\mathbb{E}[H(p)] - H(\mathbb{E}[p])$; the second term is maximised
when the *mean prediction over the pool is uniform*. On a two-class problem that
is an explicit hard-coded prior of $0.5$. Running it on a $97.2\%$-spoof pool
asks it to push $47$ points of probability mass into the bona fide class.

That makes the paper's claim sharper than "these methods happen to assume
balance". Two of the four modern TTA baselines assume it in different places:
Tent implicitly (confident predictions on a skewed pool are all one class, so
entropy minimisation saturates them), SHOT *explicitly*, in the diversity term
it was given precisely to stop Tent-style collapse. The fix for Tent's collapse
introduced the prior assumption in a more literal form.

Running tally on the official DF eval, all label-free, same checkpoint and pool:

| method | EER | AUC | acc@0.5 |
|---|---|---|---|
| source | 4.51 | .9923 | 98.87 |
| Tent | 4.06$^\ddagger$ | .9765 | 98.52 |
| SHOT | 6.38 | .9725 | 77.64 |
| symmetric-$q$ TTA (published) | 5.84 | .9846 | 62.96 |
| **prior-corrected TTA (ours)** | **4.02** | **.9931** | **98.95** |

$^\ddagger$ tie-breaking artefact; 98.07% of scores saturate to 1.0.

### Stage 1, third arm: ETA, and why an entropy filter cannot save you here

ETA on the same checkpoint and pool: EER $4.51\!\to\!7.44$, AUC
$.9923\!\to\!.9288$ — the worst ranking damage of any arm — while accuracy at
$\tau{=}0.5$ is *unchanged* ($98.93$). Score resolution: $97.71\%$ of scores
saturate to exactly $1.0$, $1{,}235$ distinct values left, largest tie $96.65\%$.

The logged reliable-sample rate is the tell: **0.996, 0.997, 0.998, 0.998**
across the four epochs. ETA's reliability filter, whose whole job is to exclude
the high-entropy samples that destabilise entropy minimisation, excluded
essentially *nothing*. With the filter inert, ETA reduces to Tent — and lands
worse.

This is not an implementation problem, it is a structural one, and it is the
paper's own thesis pointed at a baseline: **an entropy-based reliability filter
cannot detect over-confidence, because over-confidence is low entropy.** It
separates "confident" from "uncertain", not "confident and right" from
"confident and wrong" — and mis-calibration under corpus shift manufactures
precisely the latter. Any TTA safeguard keyed on the model's own confidence
inherits the broken calibration it is supposed to protect against.

### The taxonomy, now complete enough to state

Four published approaches, one pool (official DF eval, 97.2% spoof), one
released SOTA checkpoint. Every one degrades it, each through its own
assumption, and every assumption is a version of the same one:

| method | EER | AUC | acc | how it assumes the target distribution |
|---|---|---|---|---|
| source | 4.51 | .9923 | 98.87 | — |
| Tent | 4.06$^\ddagger$ | .9765 | 98.52 | implicit: confident predictions on a skewed pool are all one class |
| ETA | 7.44$^\ddagger$ | **.9288** | 98.93 | its entropy filter is inert against over-confidence (admit rate 0.998) |
| SHOT | 6.38 | .9725 | **77.64** | explicit: the diversity term maximises $H(\mathbb{E}[p])$, a hard $\pi{=}\tfrac12$ |
| symmetric-$q$ TTA | 5.84 | .9846 | **62.96** | the quantile budget is symmetric regardless of the prior |
| **ours** | **4.02** | **.9931** | **98.95** | estimates the prior instead |

$^\ddagger$ EER computed over a pool that is $>93\%$ one tied score; read the AUC
column instead.

### Stage 1, fourth arm: SAR does not fail — it declines to adapt

I predicted SAR would land near ETA, since it layers sharpness-aware
minimisation on the same entropy filter that had just proved inert. It did not.

SAR: EER $4.510\!\to\!4.472$, AUC $.9923\!\to\!.9923$ (unchanged to four
decimals), accuracy $98.87\!\to\!98.88$. Score resolution is essentially the
source model's ($2{,}779$ distinct values vs $2{,}762$; largest tie $9.56\%$ vs
$9.82\%$). The logs say why: **308 model-recovery resets over 2,500 batches**,
each reverting to the pre-adaptation snapshot, with the reliability filter again
nearly inert (admit rate $0.988$).

So SAR is the one modern baseline that does not damage the checkpoint, and it
achieves that by **not moving it**. Its collapse guard does exactly what SAR
claims for it — prevention — and prevention is all that is delivered: a
$0.04$-point EER change with an identical AUC is not adaptation.

**This is partly my own doing and must be stated as such.** SAR's published
recovery trigger $e_0{=}0.2$ sits inside the $C{=}2$ reliable band and would
have reverted on essentially every batch; I re-derived it as
$e_0=(0.2/\ln 1000)\ln C = 0.020$ to keep its relative position. The conclusion
survives either constant — at $0.2$ SAR is inert by construction, at $0.020$ it
is inert by measurement — but the reader is entitled to know the threshold was
ours to choose.

### Stage 1 complete: the honest claim

Not "all four fail". The accurate statement, on the official DF eval at $97.2\%$
spoof against a released $4.51\%$-EER checkpoint:

| method | EER | AUC | acc | verdict |
|---|---|---|---|---|
| source | 4.51 | .9923 | 98.87 | — |
| Tent | 4.06$^\ddagger$ | .9765 | 98.52 | degrades ranking; EER flatters it |
| ETA | 7.44$^\ddagger$ | .9288 | 98.93 | worst ranking damage; filter inert |
| SHOT | 6.38 | .9725 | 77.64 | degrades both; explicit $\pi{=}\tfrac12$ |
| SAR | 4.47 | .9923 | 98.88 | **safe, but inert** (308 resets) |
| symmetric-$q$ TTA | 5.84 | .9846 | 62.96 | degrades both |
| **ours** | **4.02** | **.9931** | **98.95** | improves both |

**Three of five published/prior configurations degrade a state-of-the-art
detector at deployment prevalence; the fourth avoids damage only by reverting
its own updates. None of them improves it. Ours does.** That is a weaker
sentence than "all four fail" and a much more defensible one.

### Two framing corrections made before they became reviewer ammunition

1. **"Deployment prevalence" was the wrong phrase and a soft target.** A
   reviewer would object, correctly, that ASVspoof2021-DF's $97.2\%$ spoof rate
   is itself a benchmark artefact --- a real moderation pipeline sees mostly
   \emph{bona fide} audio, i.e. skew in the opposite direction. The claim is now
   stated the way it is actually true and is stronger for it: **the target prior
   is essentially never $\tfrac12$, in either direction.** DF is $0.972$,
   In-the-Wild is $0.372$, a deployed pipeline is lower still, and the balanced
   pools on which cross-corpus TTA is normally reported — this paper's own
   earlier ones included — are balanced because an evaluation pipeline made them
   so. Stage 3 (full In-the-Wild) is what turns that from a rhetorical point
   into a two-sided measurement.
2. **Baseline fairness now stated explicitly.** Every TTA arm adapts the same
   16,706 parameters with the same lr, batch and pass count; no baseline was
   tuned separately. The defence is not "we did our best" but a structural one:
   selecting TTA hyper-parameters per target requires target labels, which is
   precisely the resource the setting assumes absent. A matched budget is the
   only comparison the setting admits.

### The rank-agreement guard works, and it was free

Tested on CPU from score dumps already on disk, so contribution (iv) is
de-risked days before the stop-trace stage produces its out-of-sample version.
$\rho=\mathrm{Spearman}(s_{\text{adapted}}, s_{\text{source}})$ on the official DF
pool, against the labelled AUC change it is not allowed to see:

| arm | $\rho$ (label-free) | $\Delta$AUC | guard at $\rho\!\ge\!0.85$ |
|---|---|---|---|
| ETA | 0.312 | $-0.0635$ | abort |
| Tent | 0.438 | $-0.0158$ | abort |
| SHOT | 0.767 | $-0.0198$ | abort |
| TTA, symmetric $q$ (**ours, published**) | 0.816 | $-0.0077$ | **abort** |
| TTA, prior-corrected | 0.867 | $+0.0008$ | keep |
| SAR | 0.970 | $0.0000$ | keep |

Pearson $+0.84$ / Spearman $+0.89$ against $\Delta$AUC. As a keep/abort rule it
is **exact**: four harmful arms caught, none missed, no false alarms — and it
flags our own published configuration, which is the point. It is
method-agnostic; nothing in it knows which algorithm produced the scores.

Caveats recorded in the paper: $\rho$ conflates re-ordering (SHOT) with
resolution collapse through ties (Tent, ETA); and the threshold is fitted on
these six arms, with the out-of-sample validation still queued.

### Page budget

Adding all of this took the manuscript to 6 pages. Trimmed back to **4 content
pages + 1 references-only page**, the ICASSP format, by: folding the guard table
into Table I (same information, one table), dropping both figures — the score
distribution figure, whose message is now the deficit column, and the prior
sweep, whose key numbers are the median-rule row of Table I plus two sentences —
compressing the inherited breadth/mechanism material to a single paragraph, and
tightening related work, setup, limitations and conclusion. The cross-corpus
multi-baseline table was kept over the figures: a reviewer needs the baseline
comparison more than a plot. `fig_prior_rules.png` remains in the repo.

### s42 corrects the headline claim — and the correction is more on-thesis

Second checkpoint, `deepfense_w2v2_aasist_s42`, official DF eval, symmetric-$q$
arm:

| ckpt | source EER | fixed-$q$ EER | source acc | fixed-$q$ acc | deficit |
|---|---|---|---|---|---|
| s2  | 4.51 | **5.84** (worse) | 98.87 | 62.96 | $+31.20$ |
| s42 | 4.56 | **4.08** (better) | 98.78 | 61.80 | $+34.12$ |

So **the EER effect of the published configuration is checkpoint-dependent** —
it hurt s2 and helped s42 — while **the operating-point collapse replicates
almost exactly**: accuracy $62.96$ vs $61.80$, deficit $+31.2$ vs $+34.1$,
predicted fake rate $0.60$ vs $0.59$ on a pool that is $0.972$ fake.

The manuscript currently leads on "EER $4.51\!\to\!5.84$". That is a single-cell
claim and must be demoted. The replicated claim is the calibration one — which
is *better*, because operating-point failure is this paper's entire thesis. The
abstract and Table~I need rewriting around accuracy/deficit with the EER effect
reported honestly as varying.

### Two monitors, and they are complementary

Adding prediction-rate drift alongside rank agreement, both label-free:

| ckpt/arm | $\rho$ | prior gap | what it did | caught by |
|---|---|---|---|---|
| s2 ETA | 0.312 | 0.007 | AUC $-0.064$ | $\rho$ |
| s2 Tent | 0.438 | 0.013 | AUC $-0.016$ | $\rho$ |
| s2 SHOT | 0.767 | 0.225 | AUC $-0.020$, acc $-21$ | both |
| s2 fixed-$q$ | 0.816 | **0.372** | acc $-35.9$ | prior gap |
| s42 fixed-$q$ | 0.602 | **0.392** | acc $-37.0$ | both |
| s2 prior-corrected | 0.866 | 0.007 | AUC $+0.001$ | — (keep) |
| s2 SAR | 0.970 | 0.000 | inert | — (keep) |

Combined guard — keep iff $\rho\ge0.85$ \emph{and} gap $\le0.05$ — is exact over
two checkpoints and seven arms: five harmful arms caught, two clean ones kept.
The two monitors catch **different** failures: $\rho$ sees ranking collapse
(Tent, ETA destroy the ranking while leaving the operating point alone); the gap
sees operating-point collapse (symmetric-$q$ leaves the ranking nearly intact
and moves the predicted positive rate from $0.97$ to $0.59$). Neither alone is
sufficient — $\rho$ would have passed s2's fixed-$q$ arm at $0.816$.

**Honest note on the second monitor.** $\hat\pi$ is estimated by BBSE from the
frozen source model, and on these well-separated checkpoints $M$ is near
identity, so $\hat\pi\approx$ the source model's own predicted positive rate and
the monitor reduces to *prediction-rate drift from the source*. That is still
the right quantity and still label-free, but the paper must not dress it up as
requiring the shift estimator — the source row's gap of $0.000$ is true by
construction.

### Correction to the two-monitor claim above

I wrote that a $\rho$-only guard "would have passed" s2's symmetric-$q$ arm at
$\rho{=}0.816$. That is wrong: $0.816 < 0.85$, so $\rho$ aborts it too. Running
the combined guard properly:

* **$\rho$ alone at $\tau{=}0.85$ catches all five harmful arms.** It is
  sufficient on this set.
* **The prior gap alone misses ETA and Tent** (gaps $0.007$ and $0.013$) --- they
  destroy the ranking without moving the operating point.

So the second monitor is not *necessary* here, and the paper must not say it is.
What is true, and still worth reporting, is that the two are decisive on
different failures and each is weak on the other's:

| failure | $\rho$ margin below $0.85$ | gap margin above $0.05$ |
|---|---|---|
| ranking collapse (ETA, Tent) | $0.54$, $0.41$ — decisive | $-0.04$, $-0.04$ — blind |
| operating-point collapse (s2 fixed-$q$) | $0.034$ — **marginal** | $0.32$ — decisive |
| operating-point collapse (s42 fixed-$q$) | $0.25$ | $0.34$ — decisive |

$\rho$ catches the s2 calibration collapse by a $0.034$ margin on a threshold
fitted to these same seven points --- that is not a margin to ship a safety rule
on. The gap catches the same failure at $0.372$ against a $0.05$ threshold. The
honest recommendation is to run both because each has one failure mode it
detects with room to spare, not because either is individually incomplete.

Combined guard over 7 arms and 2 checkpoints: 5/5 harmful caught, 0 missed, 0
false alarms, 2 clean arms kept.

### The eighth arm settles the monitor question — against my previous correction

`s42/ours_bbse` landed: EER $4.560\!\to\!\mathbf{3.290}$ ($-27.8\%$ relative),
AUC $.9919\!\to\!.9958$, accuracy $98.78\!\to\!98.98$. The best result in the
study, and the $\rho\ge0.85$ guard **rejected it** ($\rho{=}0.828$) --- a false
alarm on our own headline.

So the correction I made an hour ago ("$\rho$ alone is sufficient") was itself an
artefact of a threshold fitted to seven points. Sweeping both thresholds over all
eight arms:

| guard | caught | missed | false alarms | kept |
|---|---|---|---|---|
| $\rho\ge0.85$ alone | 5/5 | 0 | **1** (our best arm) | 2 |
| gap $\le0.05$ alone | 3/5 | **2** (ETA, Tent) | 0 | 3 |
| $\rho\ge0.70$ alone | 3/5 | 2 | 0 | 3 |
| $\rho\ge0.80$ alone | 4/5 | 1 | 0 | 3 |
| **both, $\rho\ge0.70$--$0.80$ and gap $\le0.05$** | **5/5** | **0** | **0** | **3** |

**Both monitors are necessary after all**, and the combined guard is exact over a
$0.10$-wide plateau in $\rho$ rather than at a knife-edge. The single-monitor
variants each fail in their own way: $\rho$ tight enough to catch the calibration
collapses also rejects a beneficial arm; $\rho$ loose enough to keep the
beneficial arms misses the calibration collapses entirely; the gap is blind to
ranking collapse.

**What this episode actually shows, and the paper must say it.** Two thresholds
were fitted on eight points, and the first threshold I fitted was falsified by
the ninth cell to arrive. That is a small-sample warning, not a validated rule.
The paper should present the guard as a *diagnostic with a demonstrated
plateau*, report that the naive single-monitor version was falsified by our own
data, and rest the claim on the queued leave-one-target-out validation rather
than on these eight points.

### The headline replicates across independently trained checkpoints

| ckpt | source | symmetric-$q$ | **prior-corrected** | rel. gain |
|---|---|---|---|---|
| s2  | 4.510 | 5.836 | **4.020** | $-10.9\%$ |
| s42 | 4.560 | 4.080 | **3.290** | $-27.8\%$ |

AUC improves in both ($.9923\!\to\!.9931$, $.9919\!\to\!.9958$), and
prior-correction beats the symmetric budget on both ($-1.82$, $-0.79$ EER).
Disjoint/inductive rows track within $0.02$. s240 pending.

### Audit: the validated adaptation path was not touched

This pass added 242 lines to `adaptive_pipeline.py`. Diffing against the last
validated commit (655dd2d), the only *deletions* are a stale three-line comment
and one `if os.environ.get("DYNQ"):` that became an `elif` after the new
env-gated branches were inserted above it. `adapt()`, `adapt_adaptive()`,
`set_tta_params()`, `score()` and `tent()` are unchanged, so
`verify_reduction.py`'s invariant --- adaptive-with-switches-off is bitwise
identical to the published `adapt()` --- is structurally unaffected. It is run
anyway as the first stage of the follow-up queue rather than reasoned about,
since the repo treats it as load-bearing.

It was deliberately NOT run mid-queue: the card has ~4 GB free while the
official-DF stage is resident, and an OOM in the running job would cost more
than the check is worth.

### Symmetric-$q$ across all three checkpoints — the final form of the claim

| ckpt | $\Delta$EER | accuracy | deficit |
|---|---|---|---|
| s2  | $+1.33$ worse | $98.87\to62.96$ | $+31.20$ |
| s42 | $-0.48$ better | $98.78\to61.80$ | $+34.12$ |
| s240 | $+0.84$ worse | $99.11\to65.24$ | $+30.10$ |

Operating-point collapse replicates on all three (accuracy $62$--$65\%$, deficit
$+30$ to $+34$, on a $97.2\%$-spoof pool). The EER effect does not (worse on two,
better on one, mean $+0.56$). The paper claims the former and reports the latter
as varying.

### Stage 2 complete — the three-checkpoint replication, and what it forced

| ckpt | source | symmetric $q$ | prior-corrected |
|---|---|---|---|
| seed 2 | 4.51 / 98.87 | 5.84 / **62.96** | **4.02** / 98.95 |
| seed 42 | 4.56 / 98.78 | 4.08 / **61.80** | **3.29** / 98.98 |
| seed 240 | 3.82 / 99.11 | 4.66 / **65.24** | **3.80** / 98.74 |

(EER\,\% / accuracy\,\%, official DF eval, 400,435 trials.)

**Replicates on 3/3:** the operating-point collapse under a symmetric budget
(accuracy $62$--$65\%$, deficit $+30$ to $+34$), and prior-correction beating the
symmetric budget ($-1.82$, $-0.79$, $-0.86$ EER).

**Does not replicate:** the symmetric budget's effect on *ranking* — EER worse on
two checkpoints, better on one. And the size of prior-correction's gain over
source varies enormously: $1.27$, $0.49$, $0.02$ EER points, with AUC up on two
and flat on the third. Seed 240 is essentially a null.

So the manuscript now claims the unanimous controlled comparison
(prior-corrected vs symmetric budget, 3/3) and the replicated calibration
collapse, and reports the against-source magnitudes as varying. The earlier
"$11\%$ relative reduction" headline was a single cell and has been demoted
everywhere including the abstract.

### Manuscript rebuilt and refitted

Abstract, Table~I, the replication block, the guard section, limitations and
conclusion all rewritten around the three-checkpoint result. Table~I gained the
prediction-rate-gap column so both monitors are visible per arm. Refitting to
4 content pages + 1 references-only page took six trim passes; the last of it
came from float spacing (`\textfloatsep`, `\intextsep`, caption skips) rather
than from cutting further content. Discussion and Conclusion are now one
section. Seven `\PENDING` cells remain, all queued.

### Stage 3: the two-sided test, and a caveat to declare

`ssl_aasist_wavefake` on the complete In-the-Wild benchmark (31,779 clips,
prior $0.372$ — skewed the *opposite* way to DF's $0.972$):

| | EER | AUC | acc | deficit |
|---|---|---|---|---|
| source | 5.06 | .9854 | 78.24 | **+16.70** |
| symmetric $q{=}0.3$ | **4.55** | **.9882** | **95.38** | **+0.06** |

Two separable findings.

1. **The calibration-repair mechanism is confirmed quantitatively.** Where there
   *is* a deficit, adaptation closes it almost exactly: $16.70 \to 0.06$ points.
   On DF every deficit was negative and adaptation had nothing to repair. The
   scope map holds on both sides.
2. **The symmetric budget is harmless here.** At a prior of $0.372$ it is not
   badly mis-specified, and it improves EER, AUC and accuracy together. The same
   configuration that destroys three checkpoints at $0.972$ is fine at $0.372$,
   so the failure tracks *distance from $\tfrac12$* rather than the method's
   identity. That makes the paper's claim falsifiable rather than rhetorical,
   and it is the control the DF result needs.

**Caveat that must be declared in the paper.** BBSE needs a labelled sample of
the checkpoint's own training distribution. For the three DeepFense checkpoints
we have exactly that (ASVspoof2019-LA train, the documented training set). For
`ssl_aasist` we have *a* WaveFake mirror (`data/hf_wavefake`, 6k real / 6k fake
from a HuggingFace redistribution), not verifiably the same split the authors
trained on. The confusion matrix $M$ for that checkpoint is therefore measured
on an approximation of its source distribution, and the ITW BBSE row should say
so rather than imply the same provenance as the DF rows.

### BBSE fails on In-the-Wild, and the reason indicts the method section

`ssl_aasist_wavefake` on ITW, true $P(\text{fake}){=}0.3718$:

| estimator | reads | $\hat\pi$ | error |
|---|---|---|---|
| BBSE | source $M$ + target *prediction rate* | 0.5996 | **$+0.228$** |
| SLD/EM | source prior + target posteriors | 0.6141 | $+0.242$ |
| mean predicted probability | target posteriors | 0.5954 | $+0.224$ |
| **2-component GMM** | target score *histogram shape* | **0.4215** | **$+0.050$** |

Every estimator that reads the model's *thresholded prediction rate* is off by
$\approx0.23$; the one that reads the score histogram is off by $0.05$. The cause
is visible in the numbers: the model predicts $58.9\%$ fake on a pool that is
$37.2\%$ fake — that is the $+16.7$-point calibration deficit — and BBSE's
$q$ vector is exactly that prediction rate, taken at $\tau{=}0.5$ on a
mis-calibrated model. $M$ is near identity, so $\hat\pi \approx q$ and the
deficit passes straight through.

**This contradicts the method section as written.** I justified BBSE on the
grounds that it "reads source error structure and the target prediction rate
only — never the shape of the target score distribution, which is precisely what
mis-calibration corrupts". That is backwards. The prediction rate is a
*threshold-dependent* quantity, and thresholds are the one thing this paper
measures as not transferring. **BBSE's accuracy depends on the very calibration
whose failure the method exists to repair** — it is accurate on DF ($\hat\pi$
$0.9715$ vs a true $0.9708$) precisely because that checkpoint arrives
*well*-calibrated there (deficit $-3.4$), and inaccurate on ITW because that is
where the deficit is large. It is reliable exactly where it is not needed.

The GMM, which reads rank/shape rather than a thresholded count, is now better
on both corpora tested (DF: $0.028$ vs $0.041$ mean error; ITW: $0.050$ vs
$0.228$). The repo's original "BBSE $1.4\%$ vs GMM $28\%$" came from a
degenerate-histogram case (the RawBoost Protocol-A model) and does not
generalise.

**Consequences.** (i) The method section's justification for BBSE must be
rewritten, not softened. (ii) The default estimator should arguably be the GMM,
with BBSE as the fallback for degenerate histograms — the reverse of what the
manuscript says. (iii) The DF headline is unaffected: there $\hat\pi$ was right
to $0.0007$ and the correction is what preserved the checkpoint. (iv) The
practical damage on ITW is nil — with a prior wrong by $0.23$ the arm still
scored $4.542$ vs source $5.060$, i.e. the correction degrades gracefully — but
that is luck, not design.

Queued to quantify: ITW arms with the GMM prior and with the *oracle* prior
($0.3718$, budget $(0.377,0.223)$ against BBSE's $(0.240,0.360)$ — opposite
direction), to measure what the estimation error actually cost.

### Page fit resolved structurally

Word-level trimming kept the manuscript three lines over for several passes. The
fix was structural: the standalone Conclusion paragraph is gone and its one load-
bearing sentence moved into the introduction, where the same claim was already
being set up. Sections are now I Introduction, II Method, III Experimental
Setup, IV Results, V Limitations, with references alone on page 5. A paper this
dense does not need a conclusion that restates the abstract; the contributions
list does that work.

Headroom for the seven remaining `\PENDING` cells: most are table cells (neutral
on length), so the two prose PENDINGs (Sec. IV-C baselines, Sec. IV-D
out-of-sample guard validation) are the ones to watch on the next fit.

### The single strongest cell in the study: s42 on In-the-Wild

The same `deepfense_w2v2_aasist_s42` that scores $4.56\%$ EER on ASVspoof2021-DF
arrives on In-the-Wild with its ranking intact and its operating point
destroyed:

| | EER | AUC | acc@$0.5$ | deficit |
|---|---|---|---|---|
| source | 16.47 | .9171 | **39.08** | **$+44.45$** |
| median-threshold rule (no gradient) | 16.47 | .9171 | 78.10 | — |
| symmetric-$q$ TTA | **12.58** | **.9473** | **82.39** | $+5.02$ |
| — same model, disjoint/inductive half | **12.27** | **.9490** | 82.24 | $+5.50$ |

Accuracy $39.08\%$ against an attainable $83.53$: the detector is correct on
fewer clips than a majority-class predictor while still ranking at AUC $.917$.
This is "ranking transfers, thresholds don't" in its purest observed form, on a
released checkpoint, and it is the cleanest illustration the paper has.

**Why this cell answers the reviewer objection that sank v1.** Adaptation here
does three things at once: closes $39$ of the $44$ deficit points; beats the
label-free median rule on accuracy ($82.39$ vs $78.10$); and raises **AUC from
$.9171$ to $.9473$**. A threshold move cannot change AUC --- not by a little, but
by construction --- so on the cell with the largest calibration failure in the
study the gradient step did something no re-thresholding rule can do. And the
inductive row is slightly *better* than the transductive one ($12.27$/$.9490$
vs $12.58$/$.9473$), so the ranking gain is not memorisation of the adapt pool.

Note this is the \emph{symmetric-$q$} arm. On a pool at prior $0.372$ the
symmetric budget is close to right, so the prior correction is not what produces
this --- and should not be credited with it. What produces it is the confident-
tail self-training the paper already had. The prior correction's job is
elsewhere: keeping that same machinery from destroying a checkpoint at prior
$0.972$.

### The prior correction actively harms a cell, and no estimator we have can tell

`deepfense_s42` on In-the-Wild, continued:

| arm | EER | AUC | acc | deficit |
|---|---|---|---|---|
| source | 16.47 | .9171 | 39.08 | $+44.45$ |
| symmetric-$q$ | **12.58** | **.9473** | **82.39** | $+5.02$ |
| prior-corrected (BBSE) | 14.51 | .9329 | **47.99** | $+37.50$ |

BBSE returned $\hat\pi{=}0.9786$ against a true $0.3710$ --- an error of $+0.61$ ---
because the checkpoint predicts $98.1\%$ of the pool fake ($q{=}[0.019,0.981]$)
and $M$ is near identity, so the $44$-point calibration deficit passed straight
into the prior. The budget became $(0.013,0.587)$ on a pool that is $62.9\%$
real: exactly backwards. It cost $1.9$ EER points and $34$ accuracy points
against the symmetric budget on the one cell where the symmetric budget worked
best.

**My proposed label-free detector for this failure is falsified.** I expected
BBSE and a score mixture to disagree loudly when calibration breaks, giving a
free trust signal. They do not:

| ckpt (ITW) | deficit | BBSE err | GMM err | \|disagreement\| |
|---|---|---|---|---|
| `deepfense_s2` | $+2.05$ | $+0.074$ | $+0.001$ | 0.073 |
| `ssl_aasist` | $+16.70$ | $+0.228$ | $+0.050$ | 0.178 |
| `deepfense_s42` | $+44.45$ | $+0.607$ | $+0.539$ | **0.068** |

On the catastrophic cell both estimators fail *together* (errors $+0.61$ and
$+0.54$) and their disagreement is the *smallest* of the three. The GMM is not a
rescue either: when the model puts $98\%$ of clips at high scores, the mixture's
high-mean component simply has weight $0.91$. **Prior error grows with the
calibration deficit for every estimator we have, and we have no label-free
signal that says so.**

### What this does to the paper

The prior correction is not free insurance. Its value is conditional on an
estimate whose accuracy degrades exactly as the operating point degrades:

* DF, prior $0.972$, deficit $\approx-3$ (well-calibrated): $\hat\pi$ accurate to
  $0.0007$, correction **essential** --- prevents a $31$--$34$ point accuracy
  collapse on 3/3 checkpoints.
* ITW, prior $0.372$, deficits $+2$ / $+17$ / $+44$: correction mildly harmful /
  inert / **very harmful**, tracking the prior error.

The two regimes happen to be disjoint here --- where the correction was needed it
was accurate, where it was inaccurate it was not needed --- but that is an
accident of these corpora, not a property of the method. A deployment pool can
be both severely skewed *and* severely mis-calibrated, which is the case the
method cannot serve and cannot detect.

**So the headline should be the diagnosis, not the remedy.** The defensible
paper is: a failure mode nobody has measured (every confident-quantile TTA
recipe, and three of four modern TTA baselines, break a released SOTA detector
at the standard benchmark's own prior), the natural fix, and the precise
boundary of that fix. That is complete, fully supported by these runs, and does
not require the remedy to be universal. The queued GMM-prior and oracle-prior
arms now matter a great deal: the oracle arm prices what a *perfect* prior would
have bought on s42, separating "the correction is wrong" from "the estimate is
wrong".

### The limitation is an identifiability result, not an engineering gap

Two attempts at a label-free detector for the prior-estimation failure, both
falsified:

1. **BBSE-vs-mixture disagreement.** Falsified above: on the catastrophic cell
   both estimators fail together and disagree *least*.
2. **Where $\tau{=}0.5$ falls relative to the score mixture's crossover.** The
   crossover sits at $0.9935$--$0.9952$ on all seven cells regardless of a
   deficit spanning $-3.4$ to $+44.5$ ($r{=}{-}0.775$ driven entirely by
   variation in the fourth decimal). These detectors are confident enough that
   the mixture crossover is pinned near $1.0$ and carries no calibration
   information.

The reason neither works is structural, and stating it properly is better than
either detector would have been. The only quantity a label-free method observes
is the predicted positive rate,
$$\Pr[\hat y{=}1] \;=\; \pi\,\mathrm{TPR}_{\mathcal T} \;+\; (1-\pi)\,\mathrm{FPR}_{\mathcal T},$$
which is one equation in three unknowns. BBSE closes it by assuming
$\mathrm{TPR}_{\mathcal T},\mathrm{FPR}_{\mathcal T}$ equal their source values ---
the label-shift assumption. When the corpus also shifts covariates, that
assumption is exactly what fails, and the system is underdetermined. The data
show the collision directly:

| cell | predicted positive rate | true prior | verdict |
|---|---|---|---|
| DF / s2 | 0.974 | 0.972 | estimate correct, no deficit |
| ITW / s42 | 0.981 | 0.372 | estimate off by $0.61$, deficit $+44$ |

**The same observable, opposite truths.** An extreme prior and an extreme
calibration failure are indistinguishable from the model's outputs alone, so
*estimating the prior* and *detecting that the estimate is wrong* are the same
problem. That is why the error tracks the deficit at $r{=}0.998$: both are the
same unidentifiability, measured two ways.

This is the right form for the paper's limitation --- a statement about what is
recoverable, with an empirical law attached --- rather than "our estimator
sometimes fails". It also says precisely what would fix it: any side information
that pins down one of the three unknowns (a handful of target labels, a known
deployment prior, or a calibration set), none of which the strict label-free
setting allows.

### Stage 4: on balanced pools, SHOT is our peer — and that is the point

All seven arms on identical folds (4 leave-one-corpus-out targets x 3 seeds =
12 folds), mean EER:

| method | ASV19 | LS-TTS | ITW | Arabic | pooled | paired vs ours |
|---|---|---|---|---|---|---|
| source | 4.77 | 35.88 | **11.19** | 21.04 | 18.39 | $10/12$, $p{=}0.027$ |
| Tent | 22.16 | **34.07** | 23.89 | 48.89 | 32.25 | $9/12$, $p{=}0.012$ |
| ETA | 7.66 | 35.23 | 32.02 | 43.26 | 29.54 | $10/12$, $p{=}0.003$ |
| EATA | 14.76 | 38.55 | 25.31 | 46.75 | 31.34 | $9/12$, $p{=}0.016$ |
| SAR | 4.71 | 34.48 | 12.54 | 21.84 | 18.39 | $10/12$, $p{=}0.027$ |
| **SHOT** | **3.64** | 34.41 | 11.52 | 21.47 | **17.76** | **$7/12$, $p{=}0.47$** |
| ours | 3.76 | 34.64 | 11.27 | **20.11** | **17.44** | — |

**We do not beat SHOT on balanced pools**, and it is better than us on
ASVspoof2019. Reported plainly rather than buried: the paper now says so in the
section heading ("where the advantage disappears").

This is the thesis demonstrated on a competitor rather than on ourselves.
SHOT's diversity term $-H(\mathbb{E}[p])$ is a hard uniform prior; a uniform
prior costs nothing on a balanced pool and destroys a released checkpoint on a
skewed one. SHOT on DF: $6.38\%$ EER, $77.6\%$ accuracy. Ours: $4.02$, $98.95$.
The entire separation between the two methods lives in the regime the field does
not test.

Two more things worth keeping. Tent, ETA and EATA are unstable even on balanced
pools ($29$--$32\%$ pooled against a source-only $18.39$), so their DF failure is
not purely a prior effect. And SAR's pooled EER is identical to source-only to
two decimals --- inert on balanced pools exactly as it was on DF, which is now
two independent confirmations that its recovery scheme buys safety by declining
to adapt.

**Caveat on seed counts.** These are 3 seeds on `adaptive_pipeline` pools; the
manuscript's earlier 10-seed source-vs-ours numbers came from `extended_pipeline`
with different pool sampling (source ITW $11.19$ here vs $12.78$ there). The
two must not be mixed, and the paper now reports this run as its own internally
consistent comparison.

### Read-through caught three claims the experiments do not support

A full front-to-back read of the rendered PDF, after this much span-editing,
found three places where the text had drifted ahead of the evidence:

1. **The abstract oversold the remedy.** It was written before the In-the-Wild
   failure and claimed the correction "preserves accuracy and improves EER"
   without qualification. It now states the boundary as a result: the prior is
   not identifiable from a model's own outputs, the estimate degrades as the
   operating point does ($r{=}0.998$), and on a badly mis-calibrated checkpoint a
   wrong prior is worse than none. Contribution (iii) carries the same
   qualification.
2. **Method (d) claimed a per-step stopping rule we never ran.** The guard is
   used *post hoc*, as a keep/abort decision on a completed adaptation. The text
   now says exactly that and explicitly declines to claim the per-step version.
   It also now names both monitors, not just $\rho$.
3. **Setup claimed ten seeds per target.** True of the older `extended_pipeline`
   numbers, but Sec. IV-D is a three-seed `adaptive_pipeline` run and every
   ten-seed number has been removed from the paper. Setup now states the twelve
   folds actually used, and the dangling reference to the ten-seed work is gone.

The lesson for the remaining edits: verify the rendered text against the runs,
not the source against my memory of the runs.

### The result this pass was missing: an exact validity condition

Under a reliable ranking the bottom-$q$ pseudo-real bucket can contain at most
the pool's real clips, so its precision is
$$\mathrm{purity}(q)=\min(1,\ \pi_{\mathrm{real}}/q).$$
Checked against labels on all seven (checkpoint, corpus) cells: predicted vs
observed $r{=}0.9989$, mean absolute error $0.019$.

| $q$ | DF predicted | DF observed |
|---|---|---|
| 0.01 | 1.000 | 0.997 |
| 0.02 | 1.000 | 0.958 |
| 0.05 | 0.556 | 0.509 |
| 0.30 | 0.093 | **0.086** |

At the published $q{=}0.3$, **$92\%$ of DF's "confidently bona fide" pseudo-labels
are spoof**. The confident-quantile rule is therefore sound exactly while
$$q \le \min(\pi,\,1-\pi),$$
and that single inequality predicts every result in this study: In-the-Wild
satisfies it ($\min(.372,.628){=}.372>0.3$) and the recipe repairs checkpoints
there; DF violates it ($\min(.972,.028){=}.028\lll 0.3$) and the recipe destroys
them. It also explains SHOT --- a hard uniform prior is free when the condition
holds and fatal when it does not, which is exactly the balanced-vs-DF split we
measured.

**And it points past the remedy this pass was built around.** The
identifiability result says $\pi$ cannot be estimated; but $q$ is *ours to
choose* and $\pi$ is not. Shrinking $q$ below the most skewed prior anticipated
satisfies the condition without estimating anything: $q{=}0.02$ holds purity
$\ge0.956$ on all seven cells, spanning priors $0.372$ and $0.972$. If small-$q$
symmetric matches prior-corrected TTA on DF, it is the better recommendation ---
same benefit, no dependence on an unidentifiable quantity, one fewer moving
part. Queued as stage 8z (sweep $q\in\{0.02,0.05,0.10\}$ on DF).

Promoted into the abstract and contribution (ii); Secs. IV-A and IV-B now cite
the two equations instead of re-explaining the mechanism in prose, which paid
for the space.
