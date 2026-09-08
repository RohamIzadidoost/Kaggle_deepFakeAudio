# Findings — autonomous run, 2026-08-27 → 08-31

Scope: 9 third-party checkpoints (5 model families, one of them a spectrogram
transformer), 7 corpora, up to 10 seeds. Setup and gates in
`AUTONOMOUS_RUN_PLAN.md`; the prior smaller run is `PUBLIC_CKPT_MULTI_NOTES.md`.

**Bottom line: at the published configuration this is a calibration method, not
a ranking method.** Across 196 cells, 9 checkpoints and 6 corpora, adaptation
closes the calibration deficit at p = 5×10⁻²⁰ while its effect on ranking is
*not significant in either direction* (EER p=0.81, AUC p=0.22). The EER claim the
manuscript leads with does not survive at breadth.

The ranking gains are recoverable, but only by fixing a protocol bug: `E` counts
epochs rather than gradient steps, so the update budget silently scales with pool
size. At a corrected budget the ranking improvements appear, including a reversal
of the study's two largest degradations (§4). Final GPU cost: **72.5 h, 264 jobs,
0 failures.**

---

## 1. The claim that survives, overwhelmingly

Define per cell

    deficit = (100 − EER) − acc@0.5

the balanced accuracy attainable at the EER-optimal threshold minus what the
model actually delivers at its shipped 0.5 threshold — ranking it *has* but
cannot cash in. Computed before and after adaptation from each row's own
numbers, so the two share no term.

| | value |
|---|---|
| cells | **196** (9 checkpoints × 6 corpora × up to 10 seeds) |
| mean deficit **before** | 8.06 pts |
| mean deficit **after** | **0.39 pts** |
| reduced in | **152/196** cells |
| Wilcoxon | **p = 5.1 × 10⁻²⁰** |
| corr(before, after) | **+0.22** |

And the contrast that defines the paper — same 196 cells, same runs:

| measure | source → adapted | improved | Wilcoxon p |
|---|---|---|---|
| calibration deficit | 8.06 → **0.39** | 152/196 | **5.1 × 10⁻²⁰** |
| EER (ranking) | 28.91 → 28.65 | 97/196 | **0.81** — not significant |
| AUC (ranking) | 0.7577 → 0.7612 | 118/196 | 0.22 — not significant |

At the published E=4, across the full breadth of the study, the method
**reliably repairs the threshold and does not measurably improve ranking at
all**. That is the single most important sentence this run produced.

The last row carries the weight: the endpoint does not depend on the starting
point. Adaptation lands at the attainable operating point whether the model
arrived 34 points short or 1 point short.

Within model families:

| family | n | deficit before → after | improved | p |
|---|---|---|---|---|
| `ssl_aasist_wavefake` | 35 | 11.09 → −0.08 | **35/35** | <10⁻⁴ |
| `hf_ast_asv19` (AST) | 32 | 11.21 → −0.32 | **32/32** | <10⁻⁴ |
| `deepfense_w2v2_aasist` | 54 | 14.36 → 1.43 | 43/54 | <10⁻⁴ |
| `hf_w2v2_mothecreator` | 33 | 0.85 → 0.48 | 21/33 | 0.35 |
| `hf_w2v2_bisher` | 5 | 7.81 → 3.72 | 4/5 | n<6 |

`hf_ast_asv19` is the load-bearing row. It is an **Audio Spectrogram
Transformer** — it consumes a mel spectrogram, not a waveform. Our
parameter-selection rule (top-4 blocks' LayerNorms + head) was designed against
XLS-R and transfers to it unchanged, 32/32. That is evidence of
architecture-independence that no number of XLS-R+AASIST variants could provide.

`hf_w2v2_mothecreator` fails to move because it has nothing to fix: its deficit
starts at 0.85. See §5 — it is almost certainly ASVspoof-trained.

## 2. The claim that does NOT survive

At the published E=4, ten seeds per cell, paired Wilcoxon with Holm correction:

| family | target | src AUC | source → adapted | gain | p_holm |
|---|---|---|---|---|---|
| `deepfense_s42` | arabic | 0.73 | 32.73 → 28.98 | **+3.75** | 0.023 |
| `deepfense_s42` | in_the_wild | 0.93 | 15.11 → 12.64 | **+2.47** | 0.023 |
| `ssl_aasist` | dataset2 | 0.73 | 33.86 → 32.99 | +0.87 | 0.023 |
| `ssl_aasist` | in_the_wild | 0.99 | 4.81 → 4.61 | +0.20 | 0.023 |
| `deepfense_s42` | dataset2 | 0.59 | 43.03 → 42.54 | +0.49 | 0.109 |
| `hf_ast_asv19` | arabic | 0.74 | 32.05 → 31.94 | +0.11 | 0.234 |
| `mothecreator` | arabic | 0.51 | 48.88 → 48.86 | +0.03 | 0.680 |
| `mothecreator` | in_the_wild | 0.95 | 10.39 → 10.41 | −0.02 | 0.234 |
| `ssl_aasist` | arabic | 0.86 | 22.25 → 22.72 | **−0.47** | 0.023 |
| `mothecreator` | dataset2 | 0.53 | 48.98 → 49.54 | **−0.57** | 0.023 |
| `hf_ast_asv19` | dataset2 | 0.60 | 41.88 → 44.01 | **−2.14** | 0.023 |
| `hf_ast_asv19` | in_the_wild | 0.68 | 38.13 → 41.91 | **−3.79** | 0.023 |

**7/12 improved, 5/12 worsened; 8/12 significant after Holm — four of them
significant degradations.**

The manuscript currently claims the method "improves In-the-Wild EER on all
four" public checkpoints. That was true of the four checkpoints and the one
corpus originally tested. It is not true in general, and the counter-examples
are significant, not noise.

**These two results are consistent, not contradictory.** EER is threshold-free,
so it measures ranking; acc@0.5 measures where the threshold sits. Adaptation
reliably repairs the threshold and does *not* reliably improve ranking —
`hf_ast_asv19` is the clean illustration, repairing calibration in 32/32 cells
while its In-the-Wild ranking degrades by 3.79 EER.

## 3. The AUC precondition is not merely weak — it is unstable

| sample | n | r | p |
|---|---|---|---|
| earlier run (2 families, 5 corpora) | 43 | **−0.20** | 0.21 |
| this run (5 families, 6 corpora) | 159 | **+0.14** | 0.071 |
| final (9 checkpoints, 6 corpora) | 196 | **+0.10** | 0.183 |

The correlation **changes sign** when the set of models and corpora is
resampled. That instability is the finding: it is not a law with a weak
coefficient, it is an artefact of which models and corpora happen to be in the
sample. Retire the claim rather than restate it with a smaller number.

## 4. E is an epoch count, and that is a protocol bug — with a safety limit

On **our own model** (`ckpt_ext` seeds 0–2, four targets), EER gain over source
by adaptation epochs:

| target | src AUC | E=4 | E=8 | E=16 | E=32 | E=64 |
|---|---|---|---|---|---|---|
| arabic | 0.87 | +1.85 | +3.08 | +3.97 | +4.85 | **+6.43** |
| in_the_wild | 0.94 | +1.73 | +1.71 | +2.17 | +2.35 | +2.47 |
| asvspoof2019 | 0.99 | +1.08 | +1.18 | +1.39 | **+1.58** | +1.14 |
| dataset2 | 0.73 | −0.02 | −0.43 | −0.41 | −0.81 | **−1.35** |

Three targets improve with more epochs — Arabic dramatically (22.59 → 16.16
EER, still rising at E=64) — and dataset2, the weakest-ranking target,
degrades **monotonically**. ASVspoof2019 peaks at E=32 and falls at E=64,
consistent with exhausting headroom at 3.18% EER.

Because `E` counts epochs, the update budget is a function of pool size: 4
epochs over 31,779 clips is 5.3× the gradient steps of 4 epochs over 6,000. Every
pool in the main table is 4,447–6,000 clips. The defensible framing is a
protocol correction — **E should be a fixed number of gradient steps, not a
fixed number of epochs** — which is label-free, unlike tuning E.

### The mechanism I proposed for this was tested and FALSIFIED

The reading above — "more epochs amplify whatever the pseudo-labels say, so bad
ranking compounds" — predicts that raising E on a cell that degrades at E=4 makes
the degradation *larger*. Stage G tested exactly that on third-party checkpoints.
The prediction failed, and in the most useful direction:

| cell | src AUC | E=4 | E=16 | **E=32** |
|---|---|---|---|---|
| `hf_ast_asv19` / in_the_wild | 0.69 | **−3.97** | −0.60 | **+6.35** |
| `hf_ast_asv19` / dataset2 | 0.60 | **−2.11** | −2.11 | **+1.92** |
| `deepfense_s42` / in_the_wild | 0.93 | +2.37 | +6.85 | +7.72 |
| `ssl_aasist` / in_the_wild | 0.99 | +0.25 | +0.70 | +1.03 |
| `ssl_aasist` / dataset2 | 0.73 | +1.06 | +1.21 | +1.08 |
| `hf_w2v2_bisher` / dataset2 | 0.62 | +13.83 | +10.26 | +9.31 |
| `deepfense_s42` / dataset2 | 0.59 | +0.21 | −0.67 | −0.91 |

Both AST cells — the two largest significant degradations in the entire study —
**flip from significantly harmful to strongly beneficial**. E=32 beats E=4 in
5/7 cells and more than doubles the mean gain (+1.66 → +3.79 EER pts), though at
n=7 that is not significant (p=0.375).

The better reading is that **E=4 leaves the model mid-transit**: perturbed away
from its source solution but not re-converged, which can be worse than either
endpoint.

**All four §2 degradations were then tested at E=32. Three of four reverse:**

| cell | src AUC | gain E=4 (10 seeds) | gain E=32 |
|---|---|---|---|
| `hf_ast_asv19` / in_the_wild | 0.69 | −3.79 | **+6.35** |
| `hf_ast_asv19` / dataset2 | 0.60 | −2.14 | **+1.92** |
| `ssl_aasist` / arabic | 0.86 | −0.47 | **+0.86** |
| `hf_w2v2_mothecreator` / dataset2 | **0.53** | −0.57 | **−1.60** (worse) |

The one that does not recover is the one whose source model sits at **chance**
(AUC 0.527): with no ranking to exploit, more epochs amplify noise and the cell
degrades further. So the honest claim is *three of four* degradations are an
undertraining artefact of the published budget, with the exception marking a real
boundary — adaptation needs some usable ranking to converge toward.

That boundary is **not cleanly predicted by AUC alone**: `deepfense_s42`/dataset2
at AUC 0.585 degrades with E, while `hf_ast_asv19`/dataset2 at 0.595 recovers.
Same AUC, opposite behaviour, different models — the same instability as §3.

Two caveats it does not license. `deepfense_s42`/dataset2 does degrade
monotonically with E, and our own model's dataset2 degrades monotonically
through E=64, so "more is always better" is false. And source AUC does not
predict the E-slope either (0.60 recovers, 0.59 degrades, 0.62 declines while
staying strongly positive) — the same instability as §3.

## 5. Provenance matters, and one checkpoint proves it

`hf_w2v2_mothecreator` has an undocumented training set. Its measured pattern:

| corpus | source AUC |
|---|---|
| asvspoof2019 | 0.999 |
| asvspoof2021la | 0.999 |
| asvspoof2021pa (**replay**) | 0.998 |
| in_the_wild | 0.950 |
| dataset2 | 0.527 |
| arabic | 0.506 |

Near-perfect on every ASVspoof track — including replay, a different detection
task — and at chance on Arabic and dataset2. That is the signature of a model
trained on ASVspoof, so its ASVspoof cells are in-domain and cannot support a
cross-corpus claim. This is why `provenance` is tracked per checkpoint and why
undocumented checkpoints are kept in a separate arm.

## 6. What to do with the manuscript

1. **Lead with threshold repair.** It is the claim with 196 cells, five
   families, a different input modality, and p ≈ 10⁻²¹ behind it.
2. **Retire the AUC precondition** (§3), do not restate it weaker.
3. **Replace the third-party EER claim** with the honest version: significant in
   both directions, 7/12 up and 5/12 down (§2). The current "improves all four"
   sentence is true only of the original single-corpus arm and must be scoped to it.
4. **Report the E protocol bug** (§4) as a correction with its safety caveat.
5. **Keep the replay arm separate.** It is a different task and is reported as
   an extreme-OOD probe, never pooled.

## 7. Limits to state plainly

* Our own model's E curve has **3 seeds**, not 10 — only `ckpt_ext` seeds 0–2
  exist locally, and retraining source models is out of scope (and would not fit
  in 10 GB at the published batch size).
* The two 300M HF checkpoints run at **batch 8**, not 16, because the
  consistency term's second forward pass does not fit otherwise. Batch size
  changes gradient noise, so those rows are not silently comparable to batch-16
  rows; the batch is recorded in every row.
* Three of nine checkpoints have undocumented training data (§5).
* `asvspoof2021df` and `asvspoof2021la` share lineage with ASVspoof2019, so
  cells pairing them with ASVspoof2019-trained checkpoints are condition shift,
  not clean cross-corpus transfer.

---

# Addendum — the main method at ten seeds (2026-09-01)

Every result above concerns third-party checkpoints. This closes the gap that
made our own model the weakest-evidenced part of the study: only `ckpt_ext`
seeds 0-2 survived the cloud, so local work on the paper's own method was capped
at n=3, below the n=6 where a paired Wilcoxon can reach p<0.05 at all.

28 source checkpoints were trained here (~6.6 min each, 3.64 GiB peak; see
`FIDELITY_NOTE.md` for the fidelity gate and the MLAAD bug it caught). All 40
cells below are scored on one machine, so no cross-hardware confound remains.

## 1. The published configuration replicates locally

E=4, ten seeds, paired Wilcoxon with Holm correction:

| target | src AUC | source | adapted | gain | s.d. | p_holm |
|---|---|---|---|---|---|---|
| arabic | 0.868 | 21.04 | 19.63 | **+1.41** | 0.57 | **0.008** |
| asvspoof2019 | 0.990 | 4.77 | 3.63 | **+1.13** | 0.76 | **0.008** |
| in_the_wild | 0.954 | 11.19 | 10.07 | +1.12 | 1.41 | 0.074 |
| dataset2 | 0.701 | 35.88 | 36.04 | −0.16 | 0.83 | 0.695 |

The manuscript reports 5.03→3.47, 12.78→11.33, 22.50→20.99 and a dataset2 null
from the cloud run. Same pattern, same two targets surviving Holm, same nominal
In-the-Wild, same null. This is an independent replication on independently
trained checkpoints, not a re-reading of the same numbers.

## 2. The budget defect, now properly powered

E=32 against the published E=4, ten seeds each, same checkpoints and pools:

| target | src AUC | gain @E=4 | gain @E=32 | E=32 beats E=4 | p |
|---|---|---|---|---|---|
| arabic | 0.868 | +1.41 | **+3.65** | **10/10** | **0.0020** |
| dataset2 | 0.701 | −0.16 | −0.67 | 2/10 | 0.0645 |

On Arabic, correcting the update budget **more than doubles the gain, unanimously
across ten seeds**; against source directly, E=32 reaches p=0.0020. On dataset2 —
the target whose source ranking is weakest — it does not help and trends worse.

So the defect is real and costly where the method works, and correcting it is
*not* universally safe. Both halves have to be reported: "E should be a step
count" is right, but "raise E" is not a free win.

## 3. Caveats that belong in the paper

* **Mixed provenance.** Seeds 0-2 use cloud-trained checkpoints, seeds 3-9 were
  trained here. Cloud and local source EERs show no detectable systematic offset
  (Mann-Whitney p=0.38-1.00 per target), but at n=3 vs 7 that test has low power:
  it establishes no *detectable* offset, not none.
* **The same checkpoint scores differently on different hardware.** Seeds 0-2
  scored 0.00-1.24 EER apart between the H200 run in `results_ext.csv` and this
  3080 (ASVspoof2019 seed 2 matched exactly). Same weights, same pools, bf16
  autocast -- EER is a threshold-crossing statistic and moves where scores are
  dense. The manuscript already says single-checkpoint cross-corpus figures carry
  unreported error bars; this shows the same checkpoint does too.
* **E=32 was tested on two targets, not four**, chosen as the positive and
  negative cases. A full E sweep at ten seeds is ~66 GPU-hours.

---

# Addendum 2 — fairness, three more targets, and a label-free proxy (2026-09-02)

28.5 GPU-h, 31 jobs, 0 failures.

## 1. The E result is not a compute artefact (stage K)

We claim E=32 beats the published E=4, which hands our method 8x the updates.
Handed the same budget, the baselines do not benefit (ten seeds each):

| target | method | E=4 | E=32 | delta | better | p |
|---|---|---|---|---|---|---|
| arabic | tent | 44.90 | 49.88 | **+4.98** | 0/10 | 0.0020 |
| | st_only | 20.66 | 20.28 | −0.38 | 8/10 | 0.131 |
| | **ours** | 19.63 | **17.39** | **−2.24** | 10/10 | **0.0020** |
| dataset2 | tent | 40.56 | 42.81 | +2.25 | 3/10 | 0.232 |
| | st_only | 35.38 | 35.53 | +0.14 | 5/10 | 0.770 |
| | ours | 36.04 | 36.55 | +0.51 | 2/10 | 0.065 |

Tent gets significantly *worse* with more budget; st_only does not move. Only
the full method converts budget into gain. Note also that on dataset2 `ours`
degrades with E while `st_only` is flat -- where there is no headroom it is the
**consistency term** that goes bad, not the self-training.

## 2. The E curve, all four targets, ten seeds (stage M)

| target | src AUC | gain @E=4 | gain @E=32 | E=32 wins | p |
|---|---|---|---|---|---|
| arabic | 0.868 | +1.41 | +3.65 | 10/10 | 0.0020 |
| in_the_wild | 0.954 | +1.12 | +2.69 | 9/10 | 0.0039 |
| asvspoof2019 | 0.990 | +1.13 | +1.67 | 9/10 | 0.0098 |
| dataset2 | 0.701 | −0.16 | −0.67 | 2/10 | 0.065 |

Significant on three of four; the exception is the target with least
calibration headroom.

## 3. Three more targets, and what they do to the headline (stage L)

| target | src AUC | source | ours | gain | p | deficit before → after |
|---|---|---|---|---|---|---|
| asvspoof2021la | 0.846 | 27.68 | 32.05 | **−4.37** | 0.0020 | −0.82 → −4.09 |
| asvspoof2021df | 0.857 | 26.75 | 33.11 | **−6.36** | 0.0039 | −2.73 → −7.27 |
| asvspoof2021pa (replay) | 0.591 | 43.63 | 41.14 | **+2.49** | 0.0020 | 3.19 → 0.04 |

**Extending the target set erases the pooled EER benefit, while the calibration
result strengthens:**

| target set | n | EER gain | p | deficit | p |
|---|---|---|---|---|---|
| original 4 | 40 | +0.88 | 3.6e-05 | 8.63 → 1.53 | 1.3e-07 |
| **all 7** | 70 | **−0.68** | **0.80** | 4.88 → −0.74 | **9.7e-11** |
| synthetic only | 60 | −1.20 | 0.37 | 5.16 → −0.87 | 2.7e-09 |

This is the same split the 196 third-party cells produced (calibration
p=5e-20, EER p=0.81), now replicated on our own model. The reading is not that
the method is weaker than believed, but that **the EER gain was always a
consequence of calibration repair**: the original four targets arrive badly
miscalibrated (deficits 3.7-12.3) so repairing the threshold also moves EER;
the new three arrive already well-thresholded (−2.7 to +3.2), so there is
nothing to repair and self-training only drifts them off.

It does mean the paper's positive EER numbers are contingent on which targets
were available, not a general property.

**Independence caveat.** The three new targets are not independent corpora: all
share ASVspoof2019 lineage (which is in the source pool), LA and DF are closely
related to each other, and PA is a different detection task. "Seven targets"
overstates independence the same way "43 cells" did.

## 4. The precondition is diagnostic, not actionable (stage N)

Calibration deficit predicts the gain (r=+0.56, rho=+0.69, n=70) where source
AUC does not (r=−0.12, ns). But it is computed from EER and accuracy, so it
needs labels. Testing whether any statistic of the score distribution alone
tracks it:

| statistic | vs deficit | vs gain |
|---|---|---|
| `pred_rate` | **r=+0.72** (3e-12) | +0.36 (0.002) |
| `score_mean` | +0.69 (6e-11) | +0.30 (0.012) |
| `otsu_gap` | +0.31 (0.008) | +0.17 (ns) |
| `near_thresh` | −0.07 (ns) | −0.10 (ns) |

`otsu_gap` -- the one candidate needing neither labels nor a prevalence
assumption, and the one we expected to work -- largely failed.

**And `pred_rate` does not work as a decision rule.** The two harmful targets sit
at |pred_rate − 0.5| = 0.21-0.22, inside the range of the helpful ones
(0.14-0.34); no threshold separates them, and no tau beat adapting everything.
The r=+0.72 is misleading because the relation is **non-monotone**: deficit is
low at pred_rate ~0.29 and high at both extremes (0.16 and 0.70-0.81).

So the precondition explains results after the fact and cannot yet tell a
deployer whether to adapt. Report it as such.

---

# Addendum 3 — the method is a threshold rule with a prior assumption (2026-09-02/03)

Two experiments, one mechanism. Together they explain the EER null, the
calibration result, the dataset2 "null", the two new significant failures, and
the Protocol A collapse the paper reports without accounting for.

## 1. A one-line label-free baseline recovers 94% of the method

For each source checkpoint, balanced-pool accuracy at four thresholds (raw
accuracy, matching what `adaptive_pipeline.metrics` records), against the
adapted model at 0.5. Seventy cells, seven targets, ten seeds.

| | shipped 0.5 | **TTA @0.5** | otsu | **median** | oracle (labels) |
|---|---|---|---|---|---|
| pooled | 70.7 | **75.6** | 71.0 | **75.3** | 76.6 |

| vs TTA | delta | wins | p |
|---|---|---|---|
| otsu (label-free) | −4.62 | 22/70 | 3.5e-06 |
| median (label-free) | **−0.32** | 23/70 | **0.014** |
| oracle (uses labels) | +0.99 | 41/70 | 0.017 |

TTA beats thresholding at the median significantly but by **0.32 accuracy
points** -- it recovers 4.6 of TTA's 4.9-point gain, ~94%, with no training, no
gradients and none of the 16,706 adapted parameters. The labelled oracle is only
0.99 above TTA, so both sit near the ceiling of what any threshold can do. Per
target it is mixed: the median *beats* TTA by 3.9 on ASVspoof2019 and ties on
dataset2.

Method note: an earlier pass compared balanced accuracy against the recorded raw
accuracy and reported the two as statistically indistinguishable (p=0.088). That
comparison was invalid -- the pools are not exactly 50/50 -- and the corrected
run is what is reported here.

## 2. Skew the pool and both break, together

If the reason they nearly coincide is a shared balanced-pool assumption -- the
median predicts exactly 50% positive, `q=0.3` takes the top and bottom 30% as
pseudo-fake/real regardless of the true prior -- then skewing the pool should
break them. It does (four targets, five seeds):

| pool | EER gain | p | **accuracy gain** |
|---|---|---|---|
| balanced | +0.86 | 0.005 | **+7.6** |
| 70% fake | +0.70 | <0.001 | **−3.5** |
| 90% fake | −0.58 | 0.010 | **−26.5** |

Ranking is nearly flat across all three. Accuracy collapses: at 90% fake the
source scores 87.3% and adaptation drags it to 60.8%.

**So the method is a threshold rule that targets a balanced-prior operating
point.** Where the true prior is balanced that is a repair; where it is not, it
is damage. This is the same assumption the median baseline makes, which is why
S1 found them within 0.32 points.

It also accounts for the Protocol A result (26.33 -> 42.46 EER on a 97%-spoof
pool) that the manuscript reports as an unexplained limitation. Same mechanism,
now measured across a controlled skew sweep rather than observed once.

## 3. The actionable precondition, at last

S4 of Addendum 2 looked for a label-free proxy for the calibration deficit and
failed: otsu_gap did not track it, and pred_rate tracked it but could not
separate the harmful cells because the relation is non-monotone.

The skew sweep supplies what that search was for, and it is simpler: **this
method assumes the target pool is roughly class-balanced.** That is not
estimated from the audio at all -- deployment prevalence is something an operator
usually knows -- and it is checkable without a single label.

## 4. A silent failure worth recording

The first attempt at the skew experiment reported "queue complete", ten jobs,
0.6 min each, and wrote nothing. `run_grid`'s resume guard keys on
`done(seed, target, name, "transductive")` with the setting hardcoded, while the
skew change relabelled rows only at `record()` time -- so every skewed fold
looked already-done and was skipped. This is the resume-guard hazard the repo
already documents; it fails by succeeding quietly. The guard and the record now
share one computed setting label.

---

# Addendum 4 — two ASVspoof-independent corpora, and a correction (2026-09-03)

Addendum 2 reported that extending the target set from four to seven erased the
pooled EER benefit (+0.88 -> -0.68) and read it as "the paper's positive EER
numbers are contingent on which targets were to hand". It also flagged that the
three added targets were ASVspoof2021 relatives and so not independent corpora.

**That flag was the whole story.** WaveFake (vocoder artefacts over LJSpeech) and
a 2024 commercial-TTS corpus (ElevenLabs/Polly/Kokoro/Hume/Speechify) share no
ASVspoof lineage. On our own model, ten seeds each:

| target | src AUC | source | ours | gain | p | deficit |
|---|---|---|---|---|---|---|
| hf_wavefake | 0.938 | 13.52 | 11.05 | **+2.47** | 0.0020 | 15.59 -> 0.59 |
| hf_commercialtts | 0.935 | 14.17 | 11.24 | **+2.93** | 0.0020 | 5.77 -> 3.55 |

Both significantly positive, and **larger than any of the original four**
(mean +0.88).

| target set | n | EER gain | p |
|---|---|---|---|
| original 4 | 40 | +0.88 | <0.001 |
| + 3 ASVspoof2021 relatives | 70 | **-0.68** | 0.804 |
| **+ 2 independent corpora** | 60 | **+1.48** | **<0.001** |
| all 9 | 90 | +0.07 | 0.032 |

So adding corpora does not erase the benefit. Adding *ASVspoof relatives* does.

**The mechanism is the same one running through every result.** LA and DF arrive
with NEGATIVE deficit (-0.82, -2.73) because the source model has already seen
ASVspoof2019, their lineage: the threshold is roughly right, there is nothing to
repair, and adaptation only drifts it off. WaveFake arrives at deficit 15.59 and
is repaired to 0.59, which is where its +2.47 comes from.

Revised claim: the benefit is contingent not on which targets were available but
on **calibration headroom, which lineage-relatedness predicts**. That is both
more defensible and more useful than Addendum 2's reading, and it is consistent
with the skew result (Addendum 3): the method moves the threshold to a
balanced-prior operating point, which helps exactly when the shipped threshold
is wrong and hurts when it was already right.

## Third-party on the same two corpora: mixed

48 cells, 9 checkpoints, 3 seeds (the two in-domain pairs excluded by the runner).

| target | n | EER gain | p | deficit | p |
|---|---|---|---|---|---|
| hf_wavefake | 24 | **+5.12** | <0.0001 | 8.98 -> 4.14 | 0.32 |
| hf_commercialtts | 24 | **-2.66** | 0.034 | 3.37 -> 2.85 | 0.64 |
| pooled | 48 | +1.23 | 0.301 | 6.18 -> 3.49 | 0.24 |

Significant in both directions on different corpora, and -- unlike everywhere
else in this study -- the deficit closure is **not** significant on these two.
With n=24 per corpus and only three seeds this arm is underpowered relative to
the ten-seed cells; it should be reported as breadth, not as a test.

## Grand total, third-party arm

**257 cells, 9 checkpoints, 10 corpora:**

| measure | source -> adapted | p |
|---|---|---|
| calibration deficit | 7.77 -> **1.04** | **8.1e-20** |
| EER | +0.46 | 0.162 (ns) |

The split that has held from the first 43 cells to the last 257 is unchanged:
the threshold repair is overwhelming, the ranking effect is not significant.

## Corpus that did not arrive

`alexlicuriceanu/ro-dia-deepfake-audio` (Romanian, language diversity) failed
after five exponential-backoff retries against HTTP 429 on unauthenticated Hub
requests. Not load-bearing for any claim; worth retrying with an HF token.

---

# Addendum 5 — autonomous pass to strengthen the manuscript (2026-09-07)

Context: queue was fully drained (384/384, 0 failures). The manuscript at commit
`a60bfb4` had folded in the 196-cell breadth result but not Addenda 2-4, and
three claims it now leads with were thinner than the calibration-deficit result
beside them. This pass (a) rewrote the manuscript around Addenda 2-5 and (b)
queued three GPU stages, ordered so a lost GPU leaves the most valuable work
done.

## What went into the manuscript (no GPU)

* **Abstract + conclusion + contributions** rewritten around threshold-repair:
  the median-threshold control, the skew sweep, and the WaveFake/commercial-TTS
  positive result are now first-class, not addenda.
* **257-cell numbers** replace the 196-cell ones for every pooled third-party
  claim (deficit 7.77->1.04, p=8.1e-20, 187/257; EER +0.46, p=0.16 ns; AUC
  +0.005, p=0.04). Verified against `analyze_public_ckpt_multi.py`'s own dedup
  (mean over repeated runs per seed/target/family).
* **Within-family honesty at breadth.** The 196-era "35/35 and 32/32" becomes,
  deduped over 257: AST 39/39 (p=5e-8), WaveFake-XLS-R 39/40, deepfense_s42
  38/40 (p<1e-11) -- but a *no-op* where the shipped threshold is already right
  (stafford, mothecreator) and it *worsens* calibration on one w2v2 classifier
  (gustking, 2/12, deficit 8.87->13.53). Stated plainly.
* **Median-threshold control** (Addendum 3 S1) now in Results: 70 our-model
  cells, median 75.3 vs TTA 75.6 vs oracle 76.6; recovers 94% of the gain, TTA
  wins by 0.32 (p=0.014). Stage S extends this to the third-party grid.
* **Skew sweep** (Addendum 3 S2) replaces the anecdotal Protocol-A note in the
  class-balance limitation: 50/70/90% fake -> acc gain +7.6/-3.5/-26.5, ranking
  flat. Precondition stated as "operator knows deployment prevalence".
* **Matched-budget fairness** (stage K): E=32 hands the method 8x updates;
  Tent gets *worse* (+5 EER on Arabic), st_only flat -- so the E gain is the
  method's, not the compute's. One clause added to the E paragraph.
* **Page budget.** Additions pushed content onto p5 (violates ICASSP
  "refs-only"). Trimmed the Tent per-seed list, the bootstrap-CI sentence, the
  DANN/ASDG paragraph, the comparison-points list, and the conclusion back to
  4-page content + refs-only p5. No overfull boxes.

## What was queued (GPU), value-ordered

| stage | what | why | est |
|---|---|---|---|
| **S** | median/otsu threshold control on the 76-cell third-party grid, seed 0, scoring only | "the method is a label-free threshold rule" is shown on our model only | 1.5 h |
| **T** | E=32 on the two AST degradation cells, seeds 1-5 | the "E=32 reverses the AST degradations" claim rests on **one** seed (stage G2) | ~13 h |
| **R2** | the ASVspoof-independent third-party arm (WaveFake, commercial-TTS), seeds 3-9 | its deficit closure is the only pooled deficit result in the study that misses significance (n=48, 3 seeds) | ~14 h |

New files: `threshold_control_multi.py` (scoring only, reuses
`public_ckpt_tta.build_model/score`, verified to reproduce the pipeline's source
EER/AUC/acc exactly on `hf_w2v2_bisher/arabic`),
`analyze_threshold_control_multi.py`. Stages appended to `jobs.py`; no existing
file with committed results was modified.

### Results (filled as stages complete)

* **Stage S (done, 63 third-party cells, 9 checkpoints, 8 corpora, seed 0).**
  The median-threshold rule **beats** the gradient TTA method at breadth:
  median $75.6\%$ acc vs TTA $73.7\%$ (labelled oracle $75.9\%$), paired
  $p{=}0.041$, median $\ge$ TTA in $38/63$ cells. It recovers $133\%$ of TTA's
  accuracy gain over the shipped threshold. `otsu` is bad ($-5.4$ vs TTA).
  Reading: where adaptation degrades a checkpoint's ranking (the significant
  degradations), a threshold move cannot -- so the label-free rule is the safer
  of the two. Combined with the 70 our-model cells: **133 cells, median and TTA
  statistically indistinguishable** ($p{=}0.89$), both $\sim$1 pt under the
  oracle. This is now the paper's framing: the gradient method is doing,
  expensively, what a one-line threshold move does at least as well. Per-target
  medians in `threshold_control_multi.csv`; analysis
  `analyze_threshold_control_multi.py`.
* **Stage T (done, hf_ast_asv19, E=32, seeds 1-5 + G2's seed 0).** Both AST
  degradations -- the two largest significant ones in the study at E=4 -- reverse
  cleanly at E=32:
  - `in_the_wild`: E=4 gain $-3.78 \to$ E=32 gain $+7.24$, **6/6 seeds positive**,
    p=0.031 (the n=6 Wilcoxon floor). E=32 beats E=4 by +11.0, 6/6.
  - `dataset2`: E=4 gain $-2.14 \to$ E=32 gain $+1.86$, **6/6 positive**, p=0.031.
    E=32 beats E=4 by +4.0, 6/6.
  The E-defect claim no longer rests on one seed. `analyze_stage_t.py`.
* **Stage R2:** _pending_
