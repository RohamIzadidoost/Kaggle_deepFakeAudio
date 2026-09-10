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
