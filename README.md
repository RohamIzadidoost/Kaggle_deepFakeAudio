# Ranking Transfers, Thresholds Don't

Unsupervised test-time adaptation for cross-corpus audio deepfake detection.
This repo has two independent studies sharing infrastructure:

1. **The paper** (`main_icassp.tex` / `main.tex`) — a leave-one-corpus-out study
   showing a fine-tuned SSL detector's *ranking* (AUC) transfers across corpora
   even when its decision threshold doesn't, and that this enables a stable
   unsupervised test-time adaptation (TTA) method. This is the main content of
   this README.
2. **An earlier, separate baseline study** — a leakage-free re-evaluation of a
   published method ("Unmasking the Fake", Gujjar et al. 2024) plus an
   explainable `AttentiveSpecCNN` detector. Documented in
   [§10](#10-earlier-work-attentivespeccnn-baseline-study); it shares no code
   path with the paper above.

`PROJECT_LOG.md` is the authoritative narrative of what was tried, what
worked, what failed, and why. `CLAUDE.md` has the terse command reference.
This README is the ordered, step-by-step path from a clean checkout to every
number and figure in the paper.

## Headline results (Table 2, 10 seeds/target, leave-one-corpus-out)

| Method | ASVspoof2019 | LibriSpeech-TTS | In-the-Wild | Arabic |
|---|---|---|---|---|
| Source-only | 5.03 / .989 | 33.54 / .727 | 12.78 / .945 | 22.50 / .843 |
| Tent | 23.47±21.96 / .776 | 35.44±3.20 / .674 | 27.34±11.21 / .735 | 47.35±5.03 / .527 |
| Self-training only | 4.20 / .987 | **33.39** / .709 | 11.55 / .950 | 22.19 / .846 |
| **Ours (TTA)** | **3.47** / .994 | 33.74 / .727 | **11.33** / .958 | **20.99** / .866 |
| Oracle (target-supervised) | 0.89 / .999 | 13.93 / .939 | 4.42 / .992 | 9.50 / .968 |

EER% / AUC. Full comparison (DANN, ASDG, RawNet2Lite, significance) below.

---

## 1. Setup

```bash
python -m venv env && source env/bin/activate
pip install -r requirements.txt
```

Pinned versions matter (numpy/pandas ABI compatibility, librosa/numba removed
to avoid a conflict) — don't casually bump `requirements.txt`.

## 2. Data

| Local folder | Source | Fetch |
|---|---|---|
| `data/asvspoof2019_LA/` | ASVspoof 2019 LA | `python download_data.py --root data` (Kaggle API) |
| `data/in_the_wild/` | In-the-Wild | same command |
| `data/dataset_2/` | LibriSpeech real + TTS fakes | Kaggle: `adarshsingh0903/audio-deepfake-detection-dataset` |
| `data/dataset_3/` | MLAAD v5 (38 languages) | Kaggle: `trapka/mlaadthe-multi-languagaudioanti-spoofing-dataset` |
| `data/arabic_arad/` | Arabic deepfakes (ArAD) | `python download_arabic.py` (HuggingFace `datasets`) |

Kaggle auth: `~/.kaggle/kaggle.json`. Both scripts are idempotent — they skip
a corpus whose expected files already exist. Verify everything landed before
burning GPU time:

**MLAAD path note.** `extended_pipeline.py`/`asdg_pipeline.py` (§3–4, the
paper's pipeline) look for MLAAD at `data/mlaad/**/meta.csv`, not
`data/dataset_3/` (the Phase 1 path, §10). If you already have `data/dataset_3/`
from Phase 1, symlink instead of re-downloading 42 GB:
```bash
ln -s dataset_3/Mlaad_v5/mlaad_v5 data/mlaad   # from repo root, relative to data/
```

```bash
python verify_downloads.py
```

## 3. Main cross-corpus grid — Table 2 (`tab:main`), the Discussion's per-seed claims

`extended_pipeline.py` trains a source model per (target, seed) on the other
three corpora + a 38-language MLAAD subset, then evaluates source-only, Tent,
BN-only, self-training-only, ours (self-training + consistency), and an
oracle. It's a jupytext script (`# %%` cells) meant to be read top to bottom,
config via constants near the top rather than CLI flags.

```bash
# Smoke first: confirms the pipeline runs end-to-end (single fold, ~15 min).
# Edit the PILOT flag near the top of extended_pipeline.py to True (it's the
# default), then:
python extended_pipeline.py

# Full grid, all 4 targets x seeds 0-9 (edit PILOT = False, or use the env
# override which does this for you and selects which seeds to run):
EXT_SEEDS=0,1,2,3,4,5,6,7,8,9 python extended_pipeline.py
```

Appends incrementally to `results_ext.csv` with a per-`(seed, target)` resume
guard — safe to re-run after an interruption, it repeats no completed work.
**~24 GPU-hours** for the full 10-seed grid on one ≥24 GB GPU (measured: 23h
wall on an H200 MIG 1g.35gb slice). `ckpt_ext/` (not committed) caches
trained source models per `(target, seed)` so re-running adaptation alone
doesn't retrain sources.

```bash
tail -f logs/run_log_ext.txt   # progress, once running
```

### RawNet2Lite backbone ablation

Isolates whether cross-corpus viability comes from SSL pretraining or the TTA
step (the paper's "does the backbone matter?" result):

```bash
python improve_rawnet2lite.py    # -> results_rawnet2lite_v2.csv
```

## 4. DANN / ASDG baselines — Table 3 (`tab:dann`)

Both prior domain-generalisation baselines live in one script (they share a
training loop), trained to convergence with the held-out target's unlabeled
inputs available — strictly more privileged access than the TTA method has.

```bash
# Smoke: set SMOKE = True at the top of asdg_pipeline.py first.
python asdg_pipeline.py

# Real run (SMOKE = False), all 4 targets, seeds 0-2 by default:
python asdg_pipeline.py
# more seeds without editing SEEDS in the file:
ASDG_SEEDS=0,1,2,3,4,5,6,7,8,9 python asdg_pipeline.py
```

Writes `results_dann.csv` and `results_asdg.csv`. DANN costs ~30 min/fold,
ASDG ~7 min/fold — DANN for the full 10-seed x 4-target grid is the most
expensive single item after the main grid (~20 GPU-hours).

## 5. Statistical significance

Reads `results_ext.csv` (and the DANN/ASDG CSVs) — no GPU needed, seconds of
CPU:

```bash
python stats_tests.py    # -> stats_results.csv
```

Produces: per-target and pooled Wilcoxon signed-rank tests (source vs. ours,
ours vs. Tent, vs. self-training-only, vs. DANN/ASDG), and 10k-resample
bootstrap 95% CIs on per-target mean EER.

## 6. Figures

```bash
python make_auc_gain_scatter.py   # results_ext.csv -> fig_auc_gain.png (Fig. 2/4)
python make_perlang_figure.py     # results_perlang.csv -> fig_perlang.png
```

`make_auc_gain_scatter.py` needs nothing beyond `results_ext.csv`.
`results_perlang.csv` and `results_divergence.csv` (the per-language recall
and proxy-𝒜-distance analyses, §III of the paper) come from
`analysis_pipeline.py`, which needs the `cloud_bundle/` pool dumps written
during the main grid run (`source_pool_<target>_seed0.csv` /
`target_pool_<target>_seed0.csv`) — not just `results_ext.csv`:

```bash
python analysis_pipeline.py
# writes analysis_res/results_{divergence,perlang,dann}.csv incrementally;
# the paper-facing copies at repo root are pulled out after the run:
cp analysis_res/results_perlang.csv results_perlang.csv
cp analysis_res/results_divergence.csv results_divergence.csv
```

`embedding_analysis.py` produces the companion embedding-geometry figures
(`fig_embedding_geometry.png`, `fig_tsne_multitarget.png`) from the same
`cloud_bundle/` pool dumps.

## 7. Unattended cloud session (optional, matches how the 10-seed results were actually produced)

The grid above ran across two GPU sessions via priority-ordered job-queue
supervisors rather than one script invocation, because it doesn't fit in one
sitting. If reproducing on a fresh cloud box:

```bash
nohup python tuesday_runner.py > logs/tuesday.out 2>&1 &
tail -f logs/run_log_tuesday.txt
# then, once that session's budget is exhausted:
nohup python wednesday_runner.py > logs/wednesday.out 2>&1 &
```

Each supervisor works a time-budgeted queue against `results_ext.csv` /
`results_dann.csv` / `results_asdg.csv`'s resume guards, so interrupting and
resuming (even switching machines) loses no completed work. `MONDAY_HANDOFF.md`
is the from-scratch runbook for a cloud box with no local data or repo.

## 8. Rebuilding the paper PDF

```bash
latexmk -pdf main_icassp.tex   # ICASSP submission (4pp + refs)
latexmk -pdf main.tex          # long/journal version (11pp)
```

Every number in both traces to a committed CSV per the steps above —
regenerating figures/tables does not require recompiling from source unless
you've re-run a pipeline step.

## 9. What's deliberately *not* reproduced

- **The hyperparameter sweep** (`sweep_reseeded.py` → `sweep_reseeded.csv`,
  `main.tex` Table 4/`tab:sweep`). The paper explicitly reports it "as it
  stood rather than re-running it" — a 2-target, seed-0 artifact predating
  the 4-target extension, kept for provenance, not meant to be regenerated.
- **The DA-RAD ablation** (`run_ablation.sh` → `ablation_results.txt`,
  `losses.py`, `trainer.py`) — the train-time domain-adversarial + real-anchor
  approach that motivated the switch to TTA (§"What did not work"). Historical
  evidence, not a numbered paper table; safe to run (`bash run_ablation.sh`,
  needs `protocols/protoB_*.csv`) if you want to reproduce that finding too.
- **Adaptive q/E and Protocol A** (`adaptive_pipeline.py`, `protocol_a.py`,
  `MONDAY_HANDOFF.md`) — a validated-on-CPU mechanism explored for a possible
  follow-up, explicitly excluded from this paper (`ICASSP_PLAN.md` §3, P3).

---

## 10. Earlier work: AttentiveSpecCNN baseline study

A separate, earlier project sharing this repo's data infrastructure: a
leakage-free re-evaluation of a published audio-deepfake method under a fair,
speaker-disjoint, EER-based protocol, plus an explainable `AttentiveSpecCNN`
detector. Not part of the paper above — no code path overlap.

| Approach | EER | Balanced acc | AUC |
|---|---|---|---|
| **AttentiveSpecCNN** | **4.81%** | **93.84%** | **0.99** |
| Best reproduced classical baseline (RF on MFCC) | 11.33% | 87.51% | 0.96 |
| Reference paper's headline (GNB on MFCC) | 26.69% | 73.05% | 0.81 |

```bash
# Uses data/dataset_1 (ASVspoof 2021), data/dataset_2, data/dataset_3 — see §2.
python build_manifest.py --root data --out manifest.csv          # merge 3 datasets, parse labels + speaker IDs
python build_balanced_subset.py --manifest manifest.csv --out manifest_balanced.csv
python split_manifest.py --manifest manifest_balanced.csv --out_dir splits   # speaker-disjoint split
python paper_baseline.py                                          # classical MFCC+GNB/NMF reproduction
python train.py --epochs 15                                       # train AttentiveSpecCNN, reports EER
python explain.py --n 6                                           # Grad-CAM + attention explainability figures
python cross_dataset.py --epochs 10                                # leave-one-source-out generalization study
```

A leave-one-source-out study (`cross_dataset.py`) shows this in-domain
success does **not** transfer to a fully unseen dataset (AUC collapses to
~0.4–0.56) — the finding that motivated the cross-corpus TTA study above.

| File | Purpose |
|---|---|
| `build_manifest.py` | Merge 3 datasets → `manifest.csv`; parse labels + speaker IDs |
| `build_balanced_subset.py` | Class-balanced, generator-diverse subset via water-filling |
| `split_manifest.py` | Speaker-disjoint train/val/test split (StratifiedGroupKFold) |
| `deepfake_dataset.py` | PyTorch dataset: audio → log-mel; robust decoder (soundfile + ffmpeg fallback) |
| `metrics.py` | EER + balanced accuracy + per-class report |
| `paper_baseline.py` | Reproduction of the paper's MFCC → GNB+NMF pipeline |
| `model.py` | `AttentiveSpecCNN` — CNN + temporal attention pooling |
| `train.py` | Training loop; reports EER on the test set |
| `explain.py` | Grad-CAM + temporal-attention explanation figures |
| `cross_dataset.py` | Leave-one-source-out generalisation study |

### Note on data decoding

`libsndfile` silently fails to decode a large fraction of the ASVspoof FLACs
(they are valid; `ffmpeg` reads them). `deepfake_dataset.load_audio` falls
back to an ffmpeg-based decoder so these files are not silently turned into
silence — without it, ~half of the real class became zeros and the model
learned a "silence → real" shortcut.
