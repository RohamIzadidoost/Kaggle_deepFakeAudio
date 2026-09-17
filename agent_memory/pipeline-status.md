---
name: pipeline-status
description: "State of the deepfake-audio data pipeline (manifest build, split, dataset loader)"
metadata: 
  node_type: memory
  type: project
  originSessionId: c7d6f656-ee0b-4dbe-b6da-6a4cf7966d06
  modified: 2026-07-22T06:44:05.820Z
---

Data pipeline in /home/general/Desktop/deepfake_project works end-to-end as of
2026-07-22:
- build_manifest.py — scans data/dataset_1|2|3, writes manifest.csv (cols:
  filepath,label,dataset_source,generator,extra). Fixed a bug: load_mlaad now
  builds paths from the actual model_dir + filename instead of trusting the CSV's
  embedded lang segment (~1000 MLAAD rows had wrong lang e.g. zh vs zh-cn →
  nonexistent paths). 0 missing files across all 614,764 rows after fix.
- build_manifest.py now also emits a namespaced `speaker` column: asv:<spkid>
  (from trial_metadata col 0), mlaad:<reader> (parsed from original_file), ds2:<ls
  speaker> for real / ds2g:<stem> for dataset_2 fakes. 2198 distinct speakers.
- build_balanced_subset.py — carves a class-balanced, source-diverse subset
  (manifest_balanced.csv, 38,502 rows = 19,251 real + 19,251 fake, fake spread
  over 91 generators + all 3 sources via water-filling). Configurable --ratio.
- split_manifest.py — REWRITTEN: now SPEAKER-DISJOINT via StratifiedGroupKFold on
  the speaker column (no speaker crosses train/val/test), asserts disjointness.
  Note: val can land on very few speakers (~14) because a few ASVspoof speakers
  own thousands of files each — honest tradeoff of group splitting.
- metrics.py — compute_eer + evaluation_report (EER, balanced acc, AUC, per-class
  F1). EER is the field-standard metric the paper never reported.
- deepfake_dataset.py — PyTorch Dataset: load_audio → mono → resample 16k →
  pad/crop 4s (center crop when random_crop=False for eval) → log-mel (80 mels).
  compute_class_weights for weighted loss.
- CRITICAL FIX (2026-07-22): libsndfile/soundfile FAILS to decode ~55% of ASVspoof
  real FLACs and ~44% of ASVspoof fake FLACs ("flac decoder lost sync") even though
  the files are valid (ffprobe/ffmpeg read them fine). They were being silently
  zero-filled → ~48% of REAL samples were pure silence vs ~20% of fake → model
  learned a "silence => real" shortcut. Fixed via load_audio() in
  deepfake_dataset.py: soundfile first, then librosa(sr=None, ffmpeg) fallback,
  which recovers 100% of failures. paper_baseline.py uses the same load_audio.
  Anytime audio "fails to load", suspect this, NOT genuinely corrupt files.

Next planned steps: (1) paper baseline MFCC→GNB+NMF on balanced set, (2) the DL
model (CNN + attention + Grad-CAM explainability). See [[project-goal]].

Env: venv at ./env, python 3.14, torch 2.13+cu130, torchaudio 2.11, soundfile,
pandas 3, sklearn 1.9. Untitled.ipynb is a broken early draft — ignore it.
