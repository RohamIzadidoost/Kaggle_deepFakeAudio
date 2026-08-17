# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Research codebase for an audio-deepfake-detection paper. It re-evaluates a
published method ("Unmasking the Fake", Gujjar et al. 2024) under a fair,
speaker-disjoint, EER-based protocol, then goes further: a proposed
AttentiveSpecCNN detector, a cross-corpus generalization study, a failed
train-time domain-generalization method (DA-RAD), and a working test-time
adaptation (TTA) method. `PROJECT_LOG.md` is the authoritative narrative of
what was tried, what worked, what failed, and why — read it before making
research-direction changes; don't re-litigate decisions already recorded
there. `README.md` has the Phase-1 (baseline vs. AttentiveSpecCNN) pipeline
usage; `RUN_GUIDE.md` documents the overnight TTA notebook pipeline.

## Environment

A venv lives in `env/`. Activate it before running anything:

```bash
source env/bin/activate
```

Pinned versions matter (numpy/pandas ABI compatibility, librosa/numba
removed to avoid a conflict) — see `requirements.txt`. Don't casually bump
these.

## Commands

### Phase 1 pipeline (reproduce baseline, train/eval AttentiveSpecCNN)

```bash
python build_manifest.py --root data --out manifest.csv          # merge 3 datasets, parse labels + speaker IDs
python build_balanced_subset.py --manifest manifest.csv --out manifest_balanced.csv
python split_manifest.py --manifest manifest_balanced.csv --out_dir splits   # speaker-disjoint train/val/test
python paper_baseline.py                                          # classical MFCC+GNB/NMF reproduction
python train.py --epochs 15                                       # train AttentiveSpecCNN, reports EER
python explain.py --n 6                                           # Grad-CAM + attention explainability figures
python cross_dataset.py --epochs 10                                # leave-one-source-out generalization study
```

`train.py` key flags: `--epochs`, `--batch_size` (default 64), `--lr`
(default 1e-3), `--workers` (default 8), `--seed` (default 42), `--out`
(checkpoint path, default `attentive_spec_cnn.pt`).

### Phase 3 pipeline (DA-RAD / TTA, overnight notebook)

The heavy leave-one-corpus-out TTA experiments run from
`overnight_pipeline.ipynb` (or its exported `overnight_pipeline.py`), not
from individual scripts — it decodes all audio once into a single fp16 GPU
tensor (no DataLoader/workers, since shared-memory workers previously
crashed) and writes results incrementally to `results.csv` so an interrupted
run still leaves usable data. See `RUN_GUIDE.md` for the exact
notebook-restart sequence (there's a required kernel restart after the
install cell) and the `SMOKE` toggle for a ~3 min dry run before a multi-hour
`SMOKE=False` run. Watch progress with:

```bash
tail -f run_log.txt
```

Individual pieces of that pipeline also exist as standalone modules:
`darad_dataset.py` (dataset for the DA-RAD/TTA experiments),
`augment.py` (RawBoost-style augmentation), `losses.py` (domain-adversarial /
real-anchor losses — validated as harmful, kept for the ablation), `sampler.py`,
`eval_protocol.py`, `trainer.py`, `tta.py` (the validated TTA method),
`repair_tta.py`, `cross_dataset.py`. Shell wrappers: `run_ablation.sh`,
`run_gonogo.sh`, `run_tta_confirm.sh`, `run_tta_validation.sh`.

### Adaptive q/E (Phase 4, pre-GPU validated, not yet run at scale)

For a cloud box with no data, `MONDAY_HANDOFF.md` is the end-to-end runbook
(uploads, downloads, gates, what to report).

```bash
python test_adaptive_tta.py          # 36 CPU checks, no GPU needed
python verify_reduction.py           # MUST pass before trusting any adaptive result
python adaptive_pipeline.py          # smoke (tiny subsets, *_smoke.csv outputs)
ADAPTIVE_SMOKE=0 python adaptive_pipeline.py   # real: 4 targets x seeds 0-2
```

`adaptive_tta.py` holds the schedules as **pure numpy** (no CUDA, no globals) so
they can be validated on synthetic scores without a GPU — that is how three
proposed mechanisms were killed before spending GPU time (see `PROJECT_LOG.md`
§11; the confounded ones are kept as *recorded negative results* in the test
file, don't resurrect them). `adaptive_pipeline.py` copies sections 1–5 of
`extended_pipeline.py` verbatim so target pools are identical, reuses the 12
`ckpt_ext/` source checkpoints, and never trains or downloads.
`verify_reduction.py` asserts the adaptive loop with its switches off is
*bitwise identical* to the published `adapt()` — if it fails, every
adaptive-vs-fixed comparison is confounded by the refactor.

There is no formal test suite or linter configured in this repo, apart from
`test_adaptive_tta.py` / `verify_reduction.py` above, which are plain scripts.

## Architecture

### Data flow (Phase 1)

`build_manifest.py` walks `data/dataset_1` (ASVspoof 2021), `data/dataset_2`
(LibriSpeech real + TTS fakes), and `data/dataset_3` (MLAAD v5), producing a
single labelled `manifest.csv` with a parsed speaker column. Because the raw
data is ~97% fake, `build_balanced_subset.py` draws a class-balanced,
generator-diverse subset via water-filling. `split_manifest.py` then does a
**speaker-disjoint** StratifiedGroupKFold split (`splits/train.csv`,
`val.csv`, `test.csv`) — no speaker ever crosses a split boundary, which is
the leakage fix over the original paper's random k-fold. `deepfake_dataset.py`
turns manifest rows into log-mel spectrograms for `AttentiveSpecCNN`
(`model.py`), with an ffmpeg fallback decoder — `libsndfile` silently fails
to decode a large fraction of ASVspoof FLACs, and without the fallback
roughly half of the real class silently became zeros, teaching the model a
"silence → real" shortcut. `metrics.py` computes EER, balanced accuracy, and
per-class reports; that's the field-standard number everything is judged on,
not raw accuracy.

`model.py`'s `AttentiveSpecCNN` is a CNN over log-mel spectrograms with a
learnable **temporal attention pooling** layer (in place of the paper's
static MFCC→GNB→NMF chain) — it weights time frames by how much they
contribute to the real/fake decision, which doubles as the explainability
signal (`explain.py` renders it plus Grad-CAM over conv feature maps).
Output logits use column order `[real=0, fake=1]`, matching `LABEL_TO_IDX`.

### Phase 2/3 (DA-RAD, TTA — the research contribution)

`models/` holds the SSL-based architecture: `ssl_frontend.py` (fine-tuned
XLS-R encoder — this fine-tuning, not the novel losses, is what actually
produced the big EER win), `lcnn.py`, `dg_model.py` (domain-generalization
model wrapper used for the DA-RAD ablation). `losses.py` implements the
domain-adversarial (gradient-reversal) and "Real-Anchored" contrastive
losses from DA-RAD — **the ablation in `PROJECT_LOG.md` §4 shows every one of
these hurts results** (Real-Anchor collapses the model to chance); they're
kept in the codebase for the ablation table, not as something to build on.

The method that actually works is **test-time adaptation** (`tta.py`,
orchestrated via `trainer.py` and the overnight notebook): on unlabeled
target-corpus audio, iteratively (a) score all clips with the source model,
(b) take the most-confident top/bottom 30% of scores as pseudo-fake/real
labels (`q=0.3`, so 60% of the pool is labelled and the middle 40% ignored),
(c) self-train only the top LayerNorm parameters on those
pseudo-labels, (d) enforce prediction consistency under a channel
perturbation. This relies on the finding that a detector's **AUC/ranking
transfers cross-corpus even when its decision threshold doesn't** — see
`PROJECT_LOG.md` §5. Naive entropy-minimization TTA (Tent) collapses to ~49%
EER on this data; the self-training + consistency structure is what stabilizes
it. Reported numbers are transductive (adapt+eval on the same unlabeled
pool) by convention for TTA, with an inductive (disjoint adapt/eval split)
check reported alongside as an anti-memorization sanity check — preserve that
distinction if you add new result reporting.

### Protocols and folds

`folds/` and `protocols/` hold pre-computed train/test CSVs and checkpoints
for specific experimental protocols (e.g. `protoA_*`, `protoB_itw_*`) used by
the cross-corpus and ablation runs — these are experiment artifacts, not
something to regenerate casually, since results in `results.csv` /
`ablation_results.txt` / `gonogo_results.txt` / `tta_results.txt` are keyed
to them.

### Outputs

Trained checkpoints live at the repo root (`attentive_spec_cnn.pt`) and in
`ckpt/` / `protocols/` (per-seed, per-target source models, reused rather
than retrained). Result tables are CSVs at the repo root
(`results.csv`, `cross_dataset_results.csv`, `baseline_results.csv`,
`ablation_results.txt`, `tta_results.txt`, `gonogo_results.txt`) plus figures
(`fig_*.png`) referenced from `main.tex` / `report_tables.tex` (the IEEE
paper source) and `results_report.docx`.
