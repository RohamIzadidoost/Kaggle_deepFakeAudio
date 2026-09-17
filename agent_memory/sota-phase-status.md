---
name: sota-phase-status
description: "Post-paper \"beat SOTA\" phase — compute constraints, the 5-seed grid result, and the Protocol A plan"
metadata: 
  node_type: memory
  type: project
  originSessionId: c7d6f656-ee0b-4dbe-b6da-6a4cf7966d06
  modified: 2026-07-30T10:29:46.580Z
---

As of 2026-07-30 the TTA paper is written and compiles locally. Work has shifted
to making a *defensible* SOTA claim. Plan at
`~/.claude/plans/lets-work-more-to-spicy-stonebraker.md` (Phases 0-4).
Extends [[darad-status]].

**Compute changed: cloud H200 budget is EXHAUSTED.** Local RTX 3080 (10 GB) only
from here on. Target venue ICASSP 2027 (deadline ~Sept/Oct 2026). Consequence: the
GPU-resident-cache pattern from the 33 GB MIG slice does not fit locally — use
pinned-CPU pools with per-batch transfer, or per-fold build/free.

**LaTeX now works locally** (installed texlive-xetex, texlive-lang-arabic,
fonts-vazirmatn, texlive-publishers, texlive-latex-extra, texlive-science,
latexmk). `latexmk -pdf main.tex` and `latexmk -xelatex main_fa.tex` both compile.
This supersedes every older note saying "needs an Overleaf compile". A Persian
translation `main_fa.tex` exists (article + xepersian + Vazirmatn; `hyperref` needs
`hyperfootnotes=false` or it blows the input stack under RTL).

**5-seed grid landed (190 rows in results_ext.csv; seeds 0-4, except arabic 0-3).**
Two things changed the paper's claims:
- **Tent has a latent catastrophic failure, not just variance.** ASVspoof2019
  per-seed EER: 4.16 / 3.91 / 3.04 / **49.52** / 3.42 → mean 12.81 ± 20.52. The
  paper's sentence that Tent "is competitive on ASVspoof2019" is now FALSE and
  must be corrected.
- AUC-vs-gain correlation strengthened from r=+0.41 (n=12) to **r=+0.568 (n=19)**.
- LibriSpeech-TTS null result replicated at 5 seeds (better in only 1/5 inductive),
  so it is a demonstrated property, not an unresolved question.

**Why the user has never had a literature-comparable number, and how to get one:**
`data/dataset_1/ASVspoof2021_DF_eval_part{00,01,02}` plus the official CM keys are
on disk, and `eval_protocol.py::score_official_df()` was written but wired into
nothing. Verified published 2021-DF references: Wav2DF-TSL 1.95%, AASIST-classifier
2.87%, Wav2Vec2-AASIST 8.54%, challenge-era DF top-1 ~15.6%.

Two gotchas found there, both silent:
- The DF eval set on disk is only **458,871 of 611,829 trials** (`part03` absent),
  but the shortfall is *uniform* — every codec splits exactly 75/25, label balance
  3.70% vs 3.69% bonafide, phase distribution preserved. So pooled EER over what
  remains is unbiased and reportable with a footnote; downloading part03 is optional.
- `score_official_df` accepted a `phase` argument and **ignored it**, silently
  pooling eval+progress+hidden. The literature number is the `eval` partition alone
  (533,928 trials). Fixed, and `load_df_keys` now returns `(label, phase)`.

**Standing decision (2026-07-30): do NOT edit main.tex yet.** Phase 1 (RawBoost
source retraining) would invalidate every table again, so the user chose to wait
for the Protocol A gate before doing one single refresh. Regenerate tables with
`python summarize_ext.py [--latex]` — never hand-transcribe from a log.

**Phase 0 gate result (2026-07-31), Protocol A complete (`results_protocol_a.csv`):**

| Row | EER | AUC |
|---|---|---|
| Source (2019-LA only) | 26.33% | .870 |
| Naive TTA (`ours`) | 42.45% | .686 — catastrophic harm |
| Prior-aware TTA (`ours_prior`) | 32.58% | .820 — harm, much reduced |

Two distinct, now-separated problems, not one:

1. **Naive TTA's confident-tail quantiles assume ~50/50 target balance.** DF eval
   is 97.5% spoof / 2.5% bonafide. `Q=0.3` pseudo-labels the bottom 30% of the pool
   as "confident real" regardless of true prevalence, so most of that bucket is
   mislabeled spoof — self-training then teaches the model spoof is real. This is
   the "PROTOCOL BLIND SPOT" flagged in [[darad-status]] months ago, finally
   observed with real numbers (previous probes only went to 90/10 skew; DF eval is
   97.5/2.5).
   **Fixed** in `protocol_a.py`: `estimate_prevalence()` (2-component GMM on the
   score histogram, NOT `mean(scores)` — that assumes calibration, which the
   paper's own thesis says doesn't transfer; GMM exploits rank/separation instead).
   Validated against real withheld labels: true 2.47%, estimated 2.63%.
   `adapt(..., prior_aware=True)` splits the total `2×Q` confident-label budget by
   estimated prevalence instead of 50/50 between tails. Backward-compatible by
   construction: reduces to the original symmetric behavior at prevalence=0.5, so
   the balanced main-grid leave-one-corpus-out results are provably unaffected.
   Recovered 61% of the EER damage, 73% of the AUC damage.
2. **Residual harm (32.58% still worse than source's 26.33%) is a second, separate
   cause: the source model itself is badly overfit** (train loss bottomed at
   0.0001 on 2019-LA-only, no RawBoost). Symptom: confident fraction climbed
   monotonically across adaptation epochs (84%→79%→84%→90%) instead of
   stabilizing, and the disjoint-eval threshold pinned to 1.0 — scores saturating.
   Prior-aware quantiles fix *allocation*, not degenerate *score quality* from an
   overfit source model.

**Conclusion driving next step:** this is exactly why Phase 1 (RawBoost source
retraining, `rawboost_gpu.py` already built+verified) is the correct next action —
not further TTA tuning. Expect it to both raise the 26.33% source baseline and let
TTA help instead of hurt, since the two failure modes are now known to be separable.

**Files**: `protocol_a.py` (source train/adapt/score + both TTA modes + resume
logic keyed on already-recorded method names in the results CSV — do not delete
`results_protocol_a.csv` rows if only adding a new method), `ckpt_protocol_a.pt`
(reusable, do not retrain by accident — the script skips training if this exists).

**Phase 3 complete (2026-07-31): ASDG head-to-head on the main grid** (`results_asdg.csv`,
12 folds, 4 targets x 3 seeds, same hyperparameters/source-pool composition as the
main grid and DANN baseline). Implemented `ASDGLoss` in `losses.py` as the deliberate
symmetric counterpart to the project's own asymmetric `RealAnchoredContrastiveLoss`
-- differs only in adding a fake-fake cross-domain repulsion term (self-test proves
byte-identical real-real term, disagree only on clustered-fake geometry: anchor=0.0
vs asdg=1.0). Pipeline: `asdg_pipeline.py`.

Result: `ours` (TTA) beats ASDG on 3/4 targets; ASDG beats DANN on 3/4 but beats
plain source-only on only 1/4 (barely, ASVspoof2019); ASDG shows a real instability
signature on In-the-Wild (per-seed 13.90/19.78/28.05, monotonic drift, std 7.11).
This is the real data behind `main.tex`'s existing but previously-untested Related
Work claim that the method is "differentiated from ASDG and shown to beat it."

**Standing decision still in force: main.tex not yet updated.** By this point three
independent things now need folding into one refresh: (1) 5-seed grid + corrected
Tent claim + strengthened r=+0.568 correlation, (2) Protocol A section (source-only
26.33% vs published SOTA, naive-TTA collapse + diagnosed cause + prior-aware partial
fix, RawBoost retrain not improving EER -- an honest limitations addition), (3) this
ASDG table. Regenerate main-grid tables with `summarize_ext.py`; ASDG/DANN tables are
small enough to hand-write from `results_asdg.csv`/`results_dann.csv` directly.
