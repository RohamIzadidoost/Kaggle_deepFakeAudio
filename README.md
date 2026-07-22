# Explainable Deep Learning for Deepfake Voice Detection

A leakage-free re-evaluation of a published audio-deepfake method, and an
explainable deep-learning detector that improves on it. Built on three merged
public datasets (ASVspoof 2021, a LibriSpeech/TTS set, and MLAAD v5).

## TL;DR results (speaker-disjoint test set, Equal Error Rate — lower is better)

| Approach | EER | Balanced acc | AUC |
|---|---|---|---|
| **AttentiveSpecCNN (this work)** | **4.81%** | **93.84%** | **0.99** |
| Best reproduced classical baseline (RF on MFCC) | 11.33% | 87.51% | 0.96 |
| Reference paper's headline (GNB on MFCC) | 26.69% | 73.05% | 0.81 |

A leave-one-source-out study shows this in-domain success does **not** transfer to
a fully unseen dataset (AUC collapses to ~0.4–0.56), quantifying a generalisation
gap that raw in-domain accuracy hides.

## Context

The reference paper ("Unmasking the Fake", *IEEE Access* 2024) reports 99.93%
accuracy. That figure comes from a small, single dataset with random k-fold
splits and pre-split oversampling (both leak information), and reports no EER.
Here we reproduce its pipeline as a fair baseline and evaluate everything under a
**speaker-disjoint** protocol with **EER**, the anti-spoofing field standard.

## Datasets (not included — download separately)

Place each under `data/` as shown, then run the pipeline below.

| Local folder | Source |
|---|---|
| `data/dataset_1/` | [ASVspoof 2021](https://www.kaggle.com/datasets/mohammedabdeldayem/avsspoof-2021) |
| `data/dataset_2/` | [Audio Deepfake Detection dataset](https://www.kaggle.com/datasets/adarshsingh0903/audio-deepfake-detection-dataset) |
| `data/dataset_3/` | [MLAAD v5](https://www.kaggle.com/datasets/trapka/mlaadthe-multi-languagaudioanti-spoofing-dataset) |

## Setup

```bash
python -m venv env && source env/bin/activate
pip install -r requirements.txt
```

## Pipeline

```bash
# 1. Merge the three datasets into one labelled manifest (adds a speaker column)
python build_manifest.py --root data --out manifest.csv

# 2. Draw a class-balanced, source-diverse subset (data is ~97% fake)
python build_balanced_subset.py --manifest manifest.csv --out manifest_balanced.csv

# 3. Speaker-disjoint train/val/test split (no speaker crosses splits)
python split_manifest.py --manifest manifest_balanced.csv --out_dir splits

# 4. Reproduce the paper's classical baseline (MFCC + GNB/NMF classifiers)
python paper_baseline.py

# 5. Train the proposed model (CNN + temporal attention) and evaluate with EER
python train.py --epochs 15

# 6. Explainability figures (Grad-CAM + attention) for sample clips
python explain.py --n 6

# 7. Cross-dataset generalisation (leave-one-source-out)
python cross_dataset.py --epochs 10
```

## Files

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
| `main.tex`, `report_tables.tex` | IEEE-format write-up (Overleaf-ready) |
| `results_report.docx` | Word version of the report |
| `*.csv` | Result tables |
| `attentive_spec_cnn.pt` | Trained model checkpoint |

## Note on data decoding

`libsndfile` silently fails to decode a large fraction of the ASVspoof FLACs
(they are valid; `ffmpeg` reads them). `deepfake_dataset.load_audio` falls back to
an ffmpeg-based decoder so these files are not silently turned into silence —
without it, ~half of the real class became zeros and the model learned a
"silence → real" shortcut.
