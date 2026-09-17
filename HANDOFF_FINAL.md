# Final handoff — 2026-09-17

Written when the working machine was about to be abandoned. `PROJECT_LOG.md`
holds the research narrative; this file holds only what the conversation held
and the repository did not.

## 1. Where the work lives

| Piece | Location | Re-obtainable? |
|---|---|---|
| Code, manuscript, score arrays, audit outputs | git (`icassp-breadth`) | yes, if pushed |
| `ckpt_ext/` — the 12 source checkpoints the runs reuse | `deepfake_backup_20260907.zip` only | **no** |
| `data/` — 240 GB of corpora | neither | yes, public downloads |
| `public_ckpt/` — 16 GB third-party weights | neither | yes, HF (see below) |
| Agent memory notes | `agent_memory/` in this repo | n/a |

The zip contains `ckpt_ext/`, `ckpt/`, `ckpt_pilot/`, `protocols/`, `folds/`
and a Sep-14 copy of `scores_protocol_a_public/`. It contains **no `.git`, no
source code and no manuscript**, and it predates all work after 2026-09-14.
Git and the zip are complements; neither alone is a backup.

Third-party weights: `huggingface.co/DeepFense` (ASV19 Wav2Vec2 AASIST NoAug,
seeds 2/42/240) and `huggingface.co/ash56/ssl-aasist` (WaveFake-trained
SSL-AASIST). Corpora: ASVspoof2021-DF eval, In-the-Wild, WaveFake,
ASVspoof2019-LA.

## 2. Submission status

`main_icassp.tex` is finished: 4 content pages + a references page, no overfull
boxes, retitled *Confident-Tail Test-Time Adaptation Moves the Threshold, Not
the Ranking, in Audio Deepfake Detection*. The ICASSP 2027 deadline is
2026-09-16 AoE, i.e. 2026-09-17 11:59 UTC. **As far as this repository records,
the paper was never submitted.** If it was, nothing here notes it.

## 3. What the paper claims, and how to check it in one command

    python audit_icassp.py          # needs numpy, pandas, scipy, scikit-learn

It verifies every saved score array against its historical AUC and accuracy,
recomputes every reported metric, rebuilds the tables, and prints the two
headline results. No GPU, no audio decoding, no training. `make_figure.py`
redraws the figure from those outputs; `bootstrap_eer_ci.py` recomputes the
confidence intervals.

Headline, over 79 adaptation runs (4 checkpoints, 2 corpora, spoof prevalence
0.01–0.972, budgets 0.02–0.50, 3 seeds on 8 cells):

1. Adaptation is an operating-point procedure. The attainable accuracy ceiling
   shifts by a median 0.12 points; the gap between that ceiling and accuracy at
   threshold 0.5 shifts by 14.87. No run reverses that (0/79).
2. Accuracy rises **iff** the post-adaptation gap ends below the source's:
   79/79.
3. Where it lands is ordered by `Delta = sum_c (b_c - pi_c)^+`, the fraction of
   the pool the counting bound forces to be mislabeled — not by purity. 12 of
   14 strata that vary the budget; 7 of 7 whose source threshold was well
   placed.
4. Necessary, not sufficient: on a balanced DF pool with `Delta = 0` and
   measured purity 1.000, adaptation still costs 9.6 ± 0.8 accuracy points.

Every prose number in the manuscript was verified against the audited arrays.

## 4. Open items

- **Never submitted, if that was intended.** Nothing else blocks it.
- The reviewer's deeper asks that were *not* closed: a label-free rule for
  choosing the budget (Delta needs a prior estimate, and BBSE fails at low
  prevalence — that failure is itself a reported result); exact-count tail
  selection rather than quantile thresholds; and tuned reproductions of the
  published TTA baselines, which are currently controlled variants in a
  footnote.
- `results_protocol_a_public_itw.csv.bak_20260916` is a scratch backup that got
  committed; it is redundant with git history and can be removed.

## 5. Traps this repository has caught before

Collected in `agent_memory/repo-hazards.md` and worth re-reading before any
rerun: a 45 GB auto-download path, roughly eight copies of the TTA loop, and a
resume guard that silently skips completed cells. Two more found on 2026-09-16:
a CUDA OOM that kills one cell while the shell loop continues without `set -e`,
and `audit_icassp.py` having hardcoded the adaptation-pool size, which silently
dissolved the disjoint split on every resampled pool.
