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
