# Public-checkpoint TTA, extended to five corpora (2026-08-26/27 overnight run)

Extends `PUBLIC_CKPT_NOTES.md` (In-the-Wild only) to every EER target plus
ASVspoof2021-DF. Four third-party checkpoints we did not train, five corpora,
seeds 0-2 for `ours`, seed 0 for `tent` / `st_only` / inductive.

Artifacts: `results_public_ckpt_multi.csv` (188 rows),
`public_ckpt_multi_summary.csv` (paired source/adapted cells),
`fig_auc_gain_public.png`, `queue_log.txt`, `run_log_public_ckpt_multi.txt`.
Runners: `make_target_manifests.py`, `run_public_ckpt_multi.sh`,
`run_gpu_queue.sh`, `analyze_public_ckpt_multi.py`.

**Why it was run.** `main_icassp.tex` carries a precondition claim -- adaptation
pays where the source model's ranking already transfers -- resting on
*four corpora scored with one model*, i.e. an ecological correlation
(r = +0.86, n = 4, p = 0.14; within each corpus the sign reverses). The paper
itself says settling it "needs more corpora, not more seeds". This run varies
the **model** as well as the corpus, at no training cost.

---

## 1. The headline: the precondition claim does not survive

Across **43 (model, corpus, seed) cells** from four independent checkpoints:

| outcome measure | r | p |
|---|---|---|
| absolute EER gain (pts) | **-0.20** | 0.21 |
| relative EER reduction | +0.11 | 0.49 |
| AUC change | -0.19 | 0.22 |
| acc@0.5 change | +0.14 | 0.38 |

Nothing. And it is not a headroom artefact: absolute EER-point gain is
mechanically bounded by source EER (r = -0.99 between them), which is why the
three unbounded measures are reported alongside -- they are flat too.

Worse for the claim, the **within-corpus** slopes (same clips, different
detectors -- the axis one model on four corpora cannot produce at all) are
significantly *negative* on two corpora:

| corpus | n | r | p |
|---|---|---|---|
| in_the_wild | 12 | **-0.97** | 0.000 |
| arabic | 12 | **-0.83** | 0.001 |
| dataset2 | 12 | +0.52 | 0.086 |
| asvspoof2021df | 4 | -0.50 | 0.50 |
| asvspoof2019 | 3 | -0.58 | 0.61 |

Within-model slopes disagree in sign too (+0.60, -0.93, +0.34, -0.55). Our own
model's +0.86 is not reproduced by any third-party model.

Collapsing the three DeepFense checkpoints -- they differ *only* by training
seed, so counting them as three independent models overstates n -- gives the
honest n = 7 over two model families: **r = -0.53**.

**Reading.** The negative within-corpus slopes are mechanistically sensible:
on a fixed corpus, the better a model already ranks, the less headroom
adaptation has. That is a *headroom* law, not a *ranking-transfer* law, and it
is the opposite of what the paper's precondition predicts.

## 2. The replacement claim, and it is strong

The precondition that *does* hold is not about ranking quality but about
calibration headroom. Define, per cell,

    deficit = (100 - EER) - acc@0.5

i.e. the balanced accuracy attainable at the EER-optimal threshold, minus what
the model actually delivers at its shipped 0.5 threshold: ranking it *has* but
cannot cash in. Computed before and after adaptation from each row's own
numbers, so the two share no term.

| | value |
|---|---|
| mean deficit **before** | 11.84 pts (range -1.2 to 34.4) |
| mean deficit **after** | **1.47 pts** (range -1.2 to 13.5) |
| reduced in | **33/43** cells |
| Wilcoxon signed-rank | **p = 5.9e-07** |
| corr(deficit_before, deficit_after) | **-0.01** |

That last row is the load-bearing one: **the endpoint does not depend on the
starting point.** Adaptation lands near the attainable operating point whether
the source arrived 34 points short or 1 point short. `deepfense_s42` on ITW:
deficit 34.37 -> **-0.37**.

This is the paper's own thesis -- ranking transfers, the threshold does not --
stated as a measurable, and now verified on four detectors we did not train at
p ~ 1e-6, versus the retired AUC precondition's p = 0.14 on four corpora and
one model.

It also *explains* the negative AUC correlation in S1: better-ranking models
tend to arrive better calibrated, so they have less to fix. The EER gain is
small precisely where the threshold was already about right.

**A statistic to avoid.** `corr(deficit, acc_gain) = +0.94` looks spectacular
and is partly circular: `deficit` and `acc_gain` both contain `-src_acc`, so a
badly calibrated source inflates both by construction. Report the
before/after comparison above, never that correlation.

## 3. What else survived

Calibration repair is robust and model-independent. At seed 0 (n = 13 cells):

* acc@0.5 improved in **11/13** cells, mean **+12.0 points**, max +37.1
* AUC improved in **11/13**
* EER improved in only **9/13**, mean +0.78 pts

The cleanest cell in the whole grid is `ssl_aasist_wavefake` on ASVspoof2019:
EER **0.47 -> 0.47** (exactly zero change, at ceiling) with acc@0.5
**89.5 -> 99.5**. Ranking untouched, operating point repaired -- the paper's
actual thesis, isolated, on someone else's model.

So the grid **kills the secondary claim and strengthens the primary one**. The
manuscript currently blurs them: it leads on EER improvement and carries the
AUC precondition as a live hypothesis. The defensible framing leads on
threshold repair.

## 4. Tent is model-dependent, not reliably catastrophic

`main_icassp.tex` calls entropy-minimisation TTA "unpredictably catastrophic".
Across 17 third-party cells: Tent is worse than source in **10/17**, but
collapses past 40% EER in only **2/17**. Worst case +22.33 EER pts
(`deepfense_s42`, arabic 32.46 -> 52.32; also dataset2 43.00 -> 60.88 and ITW
14.74 -> 37.07). On `ssl_aasist_wavefake` Tent is *harmless to mildly helpful*
on all five corpora (22.31->22.79, 4.61->4.20, 3.32->3.32, 0.50->0.34).

"Unpredictable" holds. "Catastrophic" is a minority outcome concentrated on the
weakest checkpoint. This confirms and sharpens the earlier two-checkpoint
finding rather than contradicting it.

## 5. The consistency term does almost nothing on third-party models

`ours` beats `st_only` in 10/17 cells, but the deltas are noise-sized (mostly
+/-0.2 EER; best -1.02, worst +0.62). On our own model the two components are
complementary; on released checkpoints, **self-training does the work and the
consistency term is roughly neutral**. Worth stating plainly rather than
reporting `ours` alone.

## 6. Inductive check passes

Adapt on half the pool, evaluate on the disjoint half: improved in **12/17**
cells, mean **+0.60** EER pts. No memorisation blow-up -- the transductive
gains are not an artefact of adapting and evaluating on the same clips.

## 7. Gains are strongly pool-dependent (open, partially tested)

The same checkpoint on the same corpus gives very different gains depending on
the pool:

| checkpoint | full ITW pool (31,779, 63/37) | balanced pool (6,000, 50/50) |
|---|---|---|
| `deepfense_s42` | 16.41 -> 10.10 (**+6.31**) | 14.74 -> 12.64 (**+2.10**) |
| `ssl_aasist_wavefake` | 5.04 -> 4.08 (+0.96) | 4.61 -> 4.48 (+0.13) |
| `deepfense_s240` | 8.41 -> 7.88 (+0.53) | 6.84 -> 6.82 (+0.02) |
| `deepfense_s2` | 9.59 -> 9.06 (+0.53) | 8.60 -> 8.29 (+0.31) |

Every gain shrinks. The paper's most quotable third-party number (+6.31) is the
full-pool figure.

**Isolation run: it is pool SIZE, not prevalence.** A size-matched
natural-prevalence pool (`manifest_tgt_itwnatural_seed0.csv`: 6,000 clips at the
full pool's 0.3718 fake rate) reproduces the *balanced* 6,000-clip result, not
the full-pool one:

| checkpoint | FULL 31,779 @63/37 | NAT 6,000 @63/37 | BAL 6,000 @50/50 |
|---|---|---|---|
| `deepfense_s42` | **+6.31** | +1.64 | +2.10 |
| `ssl_aasist_wavefake` | +0.96 | +0.42 | +0.13 |
| `deepfense_s2` | +0.53 | +0.28 | +0.31 |
| `deepfense_s240` | +0.53 | -0.09 | +0.02 |

NAT ~ BAL << FULL on every checkpoint. So the `q=0.3` prevalence sensitivity is
*not* what drives it here, which is mildly surprising given that is the known
weak spot -- holding prevalence fixed and shrinking the pool loses the gain just
as completely as rebalancing does.

The calibration channel is more robust than the EER channel to this: on the
natural 6k pool acc@0.5 still moves +42.6, +18.8, +16.2, -3.4 points.

### It is not pool size either. **E=4 is undertrained.**

"Size" was confounded with *update count*: four epochs over 31,779 clips is 5.3x
the gradient steps of four over 6,000. Rerunning the 6k natural pool at **E=21**
(the matched update count) does not merely recover the full-pool gain, it
**exceeds it on every checkpoint**:

| checkpoint | FULL 31,779 E=4 | NAT 6,000 E=4 | NAT 6,000 **E=21** |
|---|---|---|---|
| `deepfense_s42` | +6.31 | +1.64 | **+7.68** (14.44 -> 6.76) |
| `ssl_aasist_wavefake` | +0.96 | +0.42 | **+1.47** (4.85 -> 3.38) |
| `deepfense_s2` | +0.53 | +0.28 | +0.57 |
| `deepfense_s240` | +0.53 | -0.09 | +0.27 |

acc@0.5 for `s42`: 38.4 -> 94.0. Cost is ~43 min/cell versus ~8 at E=4.

**This reaches past the third-party arm.** Every pool in the main table is
4,447-6,000 clips adapted at E=4, so the published gains are plausibly
*understated across the board*. The defensible framing is not "tune E" -- which
invites the how-did-you-tune-without-labels objection the paper rightly avoids
-- but "**E should be a fixed number of gradient steps, not a fixed number of
epochs**", since epochs-at-fixed-pool-size silently makes the update budget a
function of pool size. That reformulation is label-free.

**Boundary test: E=21 is safe on badly-ranked pools.** The worry was that extra
self-training on bad pseudo-labels would accelerate collapse (the 2021-DF /
Protocol A mechanism). It does not:

| target | checkpoint | src AUC | gain E=4 | gain E=21 | AUC after E=21 | acc src -> E=21 |
|---|---|---|---|---|---|---|
| dataset2 | `s42` | 0.585 | +0.21 | -0.49 | 0.584 (flat) | 46.1 -> 57.8 |
| dataset2 | `ssl` | 0.734 | +1.06 | +0.88 | 0.736 | 66.9 -> 67.6 |
| arabic | `s42` | 0.734 | +3.18 | **+8.07** | 0.734 -> **0.836** | 53.9 -> 76.1 |
| arabic | `ssl` | 0.858 | -0.27 | +0.16 | 0.863 | 59.3 -> 76.9 |

On a near-chance pool (`s42`/dataset2, AUC 0.585) the model neither improves nor
degrades -- AUC 0.585 -> 0.584, adaptation is simply inert. Where there is
ranking to exploit it pays hugely (`s42`/arabic: +8.07 EER, and AUC itself rises
0.734 -> 0.836, so this is not only threshold repair).

Note the two cells that share source AUC 0.734 and gain +8.07 vs +0.88: source
AUC still fails to predict, consistent with S1. What separates them is the
calibration deficit of S2 (`s42`/arabic starts at acc 53.9, `ssl`/dataset2 at
66.9).

**Status:** E=21 helps or is neutral in 3/4 weak-pool cells and never collapses,
but this is one seed per cell and E was chosen to match an update count, not
tuned. It is a measurement, not yet a protocol change.

**Meanwhile, do not quote the +6.31 without saying which pool it is on.**

## 8. ASVspoof2021-DF: high scores are real, but the cell is not clean

All four checkpoints score 3.3-5.0% EER (AUC ~0.99) on the balanced 2021-DF
pool, against our own model's 26.33% (Protocol A). That looked artefactual, so
three hypotheses were tested on `ssl_aasist_wavefake` and all three failed:

| hypothesis | test | result |
|---|---|---|
| FFmpeg fallback correlates with label | AUC on natively-decoded clips only | 0.9942 vs 0.9938 full -- unchanged |
| tile-repeat padding leaks duration | AUC on clips >= 4 s (never padded) | 0.9993 -- *better*, not worse |
| duration itself is the cue | duration-only AUC; corr(score, dur) within class | 0.59; +0.004 real / +0.083 fake |

Decoder route splits 7.0% of fakes vs 3.4% of reals -- too small to drive 0.99.
So the result stands. For context, `main_icassp.tex`'s own Limitations note
that specialised AASIST-class systems reach **2-3% EER** on 2021-DF, so 3.3% is
inside the published range, not above it.

**However:** for the three DeepFense checkpoints, ASVspoof2021-DF is **not
corpus-disjoint** from their training data. The DF track is built from
ASVspoof2019 LA lineage (shared bonafide sources, overlapping spoofing systems,
re-encoded with codecs). Those three cells are *condition shift*, not clean
cross-corpus transfer, and must be labelled as such. Only the
`ssl_aasist_wavefake` (WaveFake/LJSpeech) cell is a clean transfer.

## 9. Guards and fixes this run forced

* **`--mode signcheck` was testing calibration, not polarity.** It decided the
  score-column question by `mean(score|fake) > mean(score|real)`, which failed
  both DeepFense checkpoints on Arabic (mean P(fake) 0.999 on reals vs 0.990 on
  fakes) *while AUC was 0.736* -- ranking fine, model merely saturated. A flip
  inverts ranking, so AUC is the correct discriminator. Now three-valued: PASS /
  FAIL / INCONCLUSIVE, the last for corpora where the model sits at chance and
  no test on that corpus can decide polarity either way. Re-verified it still
  passes both originally-validated ITW cases.
* **FFmpeg fallback added to `load_clip`.** Mandatory for 2021-DF; verified on a
  known-failing clip (rms 0.10, not silence). Failures propagate rather than
  becoming zeros.
* **In-domain guard.** DeepFense x asvspoof2019 is refused: our asvspoof2019
  pool is ASVspoof2019 LA *train*, exactly what they were fitted on.
* **`analyze_public_ckpt_multi.py` family filter.** `results_ext.csv` holds two
  families (`xlsr`, `rawnet2lite`); grouping without filtering turned
  ASVspoof2019's 5.03% source EER into 26.82%. Now asserted against the four
  headline source EERs in `main_icassp.tex`.
* **Pool-size gate.** Every rebuilt target pool is asserted against the `n`
  recorded in `results_ext.csv` (5580 / 4447 / 6000 / 5570, all matched).
  2021-DF has no prior run to match, and the gate says so rather than implying
  otherwise.

## 10. Side results

* **Phase-1 on the repaired decoder** (`results_phase1_postfix_decoder.csv`):
  AttentiveSpecCNN **4.34% EER / 90.98 bal-acc**, against the published
  4.81 / 93.84. Same range; no sign the published number was inflated by
  zero-filled audio. This does not *prove* the original training saw good audio
  -- it shows the result reproduces with audio that is definitely good.
  Classical baselines also reproduce (MFCC+RF 11.33% EER).
* **`train.py` clobbers `results.csv`.** It writes a Phase-1
  baseline-vs-CNN table over what was the Phase-3 TTA pilot table -- a different
  experiment. Restored from git; the new table lives in
  `results_phase1_postfix_decoder.csv`. `train.py` should be given an `--out`
  for its comparison table before it is run again.
* **dataset2 has a partial rate/label confound**: every real clip is 24 kHz,
  while 294 of 2,173 fakes are 16/22.05 kHz. It largely washes out once
  everything is resampled to 16 kHz, but it affects our own dataset2 numbers
  too and deserves a sentence.
