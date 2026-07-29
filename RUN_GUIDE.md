# Overnight Run Guide

`overnight_pipeline.ipynb` — one notebook, leave-and-forget. Produces the paper's
cross-corpus table, ablation, inductive check, hyperparameter sensitivity, and figure.

## Steps

1. Upload `overnight_pipeline.ipynb` and `kaggle.json` to the work folder.
2. Kernel: **Python 3 (ipykernel)**.
3. Run cell 1 (install) → **restart the kernel** → continue.
4. Leave `SMOKE = True`, Run All. Takes ~3 min end-to-end and validates the whole flow.
5. Set `SMOKE = False`, Kernel → **Restart & Run All**, then leave.

Watch from a terminal any time: `tail -f run_log.txt`

## Outputs

| file | contents |
|---|---|
| `results.csv` | one row per (seed, target, method, setting) — written as each finishes |
| `sweep.csv` | TTA hyperparameter sensitivity (q, epochs, lambda) |
| `ckpt/source_<target>_seed<n>.pt` | trained source models (reused, not retrained) |
| `fig_score_dist.png` | the score-distribution figure for the paper |
| `run_log.txt` | timestamped progress log |

The summary cells print the mean±std table, the inductive results, and a
ready-to-paste LaTeX table.

## What it runs

Leave-one-corpus-out: for each held-out target (In-the-Wild, Arabic), train on the
other corpora, then compare **source-only / Tent / self-training-only / ours
(ST + consistency)** — 3 seeds each — plus an inductive check (adapt on one half of
the target, evaluate on the disjoint half). Then the hyperparameter sweep.

Expect roughly 2–4 h for the grid and ~1 h for the sweep.

## Design notes (why this version is faster and better than the last one)

- **All audio is decoded once into a single fp16 tensor on the GPU** (~4 GB). No
  DataLoader, no workers, no `/dev/shm` — which is what broke before — and the GPU
  stops idling at ~16% waiting on I/O.
- **Random 3 s crops** out of 4 s cached audio: train-time augmentation that the
  previous notebook was missing.
- **All eight `dataset_2` TTS generators** are used (the old glob caught only two),
  and source fakes are water-filled across corpora and generators so no single
  generator dominates.
- Fine-tunes the **top 4** XLS-R layers (the validated setting).
- Everything is checkpointed and results are appended incrementally, so an
  interrupted night still leaves usable results.

## Knobs

At the top of the config cell: `SEEDS`, `TARGETS`, `SOURCE_EPOCHS`, `TTA_EPOCHS`,
`BATCH`, `SOURCE_PER_CLASS`, `TARGET_PER_CLASS`, `MAX_PER_CORPUS_CLASS`.

- To add leave-one-out folds for the other corpora:
  `TARGETS = ["in_the_wild", "arabic", "dataset2", "asvspoof2019"]`
- If GPU memory is tight, lower `BATCH` (64 → 32).
- To re-run adaptation without retraining sources, keep the `ckpt/` folder.

## Gotchas already handled

numpy/pandas ABI clash (pinned numpy 1.26.4, librosa/numba removed), Kaggle CLI not
on PATH (uses the Python API), shared-memory crashes (no workers at all), fp16
GradScaler overflow (bf16, with an automatic fp32 fallback if the GPU lacks bf16).
