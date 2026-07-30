# Project Log — Generalizable Audio Deepfake Detection

A running record of the idea, decisions, dead-ends, and validated results, so the
work is understandable on any machine. (Companion to the code in this repo.)

---

## 1. Assignment & goal

Course/research project (Manipal University Jaipur). Original task: read the IEEE
Access paper *"Unmasking the Fake: Machine Learning Approach for Deepfake Voice
Detection"* (Gujjar et al., 2024), merge & clean three deepfake-audio datasets, and
build a deep-learning model that improves on it. This evolved into a real research
paper on **cross-corpus generalization**.

**Authors:** Roham Izadidoost, Dr. Sumit Srivastava, Shweta Sharma (Manipal University Jaipur).

---

## 2. Datasets

| Local name | Source | Role |
|---|---|---|
| dataset_1 | ASVspoof 2021 DF (Kaggle) | in-domain / OOD test |
| dataset_2 | Adarsh audio-deepfake (LibriSpeech real + TTS fakes) | source |
| dataset_3 | MLAAD v5 (multilingual TTS, ~90 generators) | source/target (fake-only) |
| asvspoof2019_LA | ASVspoof 2019 LA (train/dev/eval) | **source training** (official protocol) |
| in_the_wild | In-the-Wild (31,779 clips, real+fake) | **OOD test** (English) |
| **Arabic ArAD** | HuggingFace DeepFake-Audio-Rangers (~1.65 GB, 19.5k) | **OOD test (Arabic)** — planned, for multilingual |

Pipeline: `build_manifest.py` → `build_balanced_subset.py` → `split_manifest.py`
(speaker-disjoint) → `deepfake_dataset.py` / `darad_dataset.py`. Metrics: EER via
`metrics.py` / `eval_protocol.py`.

---

## 3. Phase 1 — Reproduce & beat the paper (done)

- **The paper's 99.93% accuracy is not credible** — leaky k-fold, SMOTE-before-split,
  internally inconsistent metrics, no EER. Reproduced its MFCC→GNB/NMF pipeline on a
  proper **speaker-disjoint** split: best classical baseline = RF on MFCC **11.33% EER**.
- Built **AttentiveSpecCNN** (log-mel CNN + temporal attention): **4.81% EER / 93.8%
  balanced acc** in-domain — clearly beats the classical baseline.
- **Critical data bug found & fixed:** `libsndfile` silently failed to decode ~half
  the ASVspoof FLACs → they became silence → a "silence = real" shortcut. Fixed with
  an ffmpeg fallback (`deepfake_dataset.load_audio`). Numbers above are post-fix.
- **Cross-dataset finding (the seed of the paper):** leave-one-source-out collapsed to
  AUC ~0.42–0.56 (below chance). In-domain success did **not** transfer — models learn
  recording-domain shortcuts, not synthesis artifacts.

---

## 4. Phase 2 — DA-RAD (the idea that FAILED, and why it matters)

**DA-RAD** = Domain-Adversarial Real-Anchored Detection: fine-tuned XLS-R SSL encoder +
three train-time domain-generalization losses (RawBoost aug + gradient-reversal
domain-adversarial + a "Real-Anchored" contrastive loss).

**Result: negative.** Ablation (test = held-out In-the-Wild):

| Config | EER |
|---|---|
| XLS-R baseline | 30.97% (frozen) / **9.42% (fine-tuned)** |
| + RawBoost | worse |
| + GRL (adversarial) | worse / unstable |
| + Real-Anchor (full DA-RAD) | **collapsed to 50% (chance)** |

Every DG component hurt; the adversarial GRL was catastrophically unstable. **The real
win was simply fine-tuning the SSL encoder** (31% → 9.4% EER), not the novel losses.
Lesson: adversarially deleting "domain" fights the classifier and destroys signal.

---

## 5. Phase 3 — Test-Time Adaptation (the idea that WORKED)

**Pivot:** instead of train-time invariance (can't anticipate an unseen corpus), adapt
the model **at test time** on the unlabeled target audio.

**Enabling insight:** cross-corpus, a detector's **ranking (AUC) transfers even when its
threshold doesn't** (source AUC stayed 0.966 on unseen In-the-Wild while accuracy@0.5
fell to ~55%). So the model's most-confident guesses on new data are trustworthy.

**Method (final, validated):** on the unlabeled target corpus, iteratively (a) score all
clips, (b) take the confident top/bottom ~15% as pseudo-fake/real, (c) self-train the
top LayerNorm params on those pseudo-labels, (d) enforce prediction consistency under a
channel perturbation. No labels used in adaptation (labels only for final EER).

> Note: the original "Real-Anchored" loss was **dropped** — the ablation showed it
> *hurts* (again). Winning recipe = **rank-transfer self-training + channel consistency**.
> Vanilla entropy-minimization TTA (Tent) **collapsed** (49% EER) — our structure is
> what makes it stable.

---

## 6. Final validated results

Source model = fine-tuned XLS-R; adaptation is unsupervised on the target corpus.

| Target corpus | Source EER | After TTA (3 seeds) | AUC |
|---|---|---|---|
| **In-the-Wild** (large domain gap) | 9.42% | **6.41% ± 0.11%** | 0.985 |
| **ASVspoof2021-DF** (small gap) | 5.50% | **5.29% ± 0.03%** | 0.989 |

- **Stable** across seeds (std < 0.15). Gain is **larger where the domain gap is larger**
  (coherent with the mechanism).
- **Inductive check** (adapt on In-the-Wild split A, eval on *disjoint* split B):
  9.88% → **7.48%** — the improvement holds on unseen clips, so it is **not** transductive
  memorization.

**Methodology honesty (must be explicit in the paper):** adaptation uses unlabeled target
*inputs*, never labels. Most numbers are **transductive** (adapt+eval on the same
unlabeled pool) — standard for the TTA literature (cf. Tent) — reported alongside the
**inductive** result as the anti-memorization check.

---

## 7. Contribution (paper framing)

An **unsupervised test-time adaptation** method that reliably improves **cross-corpus**
deepfake detection, built on the insight that ranking transfers across corpora even when
calibration doesn't. Simple, stable, label-free, and it works where both train-time
domain-generalization (DA-RAD) and naive TTA (Tent) fail. Multilingual OOD evidence
(English In-the-Wild + Arabic ArAD + multilingual MLAAD) is the plan for the full paper.

---

## 8. Compute request (3-day sprint, Custom Compute Resource)

| Resource | Request | Minimum |
|---|---|---|
| GPU memory | **24 GB** | 16 GB |
| CPU cores | **16** | 8 |
| RAM | **64 GB** | 32 GB |
| Storage | **150 GB** | 100 GB |
| GPUs | 1 | 1 |

Basis: measured ~4 GB GPU for the method (24 GB = headroom for big batches + a
bigger-model ablation); audio decode + ffmpeg + RawBoost is CPU-heavy (16 workers);
storage = all datasets (~65 GB incl. Arabic) + env (~20 GB) + working (~10 GB) +
download/extract headroom (~25 GB). Total active compute ≈ 30 GPU-hours → fits 3 days.

**3-day plan:** Day 1 — download/merge (incl. Arabic) + train source (3 seeds). Day 2 —
baselines + TTA across 3–4 OOD targets × 3 seeds + ablations. Day 3 — explainability +
write the paper.

---

## 9. Key files

`build_manifest.py`, `build_balanced_subset.py`, `split_manifest.py`,
`deepfake_dataset.py`, `darad_dataset.py`, `sampler.py`, `losses.py`, `augment.py`,
`metrics.py`, `eval_protocol.py`, `models/` (lcnn, ssl_frontend, dg_model),
`trainer.py`, `tta.py` (the working method), `explain.py`,
`main.tex` (IEEE paper), plus result files: `results.csv`, `cross_dataset_results.csv`,
`ablation_results.txt`, `gonogo_results.txt`, `tta_results.txt`.

---

## 10. Extended 4-target cloud run (`results_ext.csv`, `analysis_res/`)

Ran leave-one-corpus-out over 4 EER-capable targets (ASVspoof2019, LibriSpeech-TTS
"dataset2", In-the-Wild, Arabic ArAD), MLAAD (38 languages, fake-only) folded into
every source pool, batch=32/lr=2e-4 (corrected from an earlier under-trained
batch=64/lr=1e-4 config), 3 seeds, plus new baselines: `bn_only` (BatchNorm-style
recalibration — a clean no-op for XLS-R since it uses LayerNorm, confirms the
architectural reason our method must use gradient-based adaptation), `oracle`
(supervised-on-target upper bound), `RawNet2Lite` (from-scratch non-SSL backbone,
collapses to ~chance on every target — confirms SSL pretraining, not TTA, is what
makes the source model viable cross-corpus at all).

**Headline result changed from "TTA helps everywhere" to "TTA helps on 3/4
targets, is EER-neutral on the 4th."** On LibriSpeech-TTS (source AUC only
~0.72, the weakest of the 4 targets — the others are ≥0.85), self-training alone
still helps (-0.57 EER) but adding the consistency term *hurts* (+0.59), net
EER-neutral vs source. This is the first target where the consistency term is a
net negative — previously (2-target pilot) it looked uniformly beneficial. Do
not re-litigate this as a bug to fix by re-tuning λ per target; it's reported as
an honest scope limit of the confidence-based mechanism (weak source ranking →
unreliable pseudo-labels → consistency amplifies the unreliability).

**DANN comparison (`results_dann.csv`)**: trained a real DANN baseline (not just
the earlier DA-RAD pilot) to convergence, 8 epochs, ~30 min/target-seed (~3x our
TTA cost), with privileged access to target inputs during training. It still
underperforms our method on 3/4 targets and is the least seed-stable method
tested (ASVspoof2019 EER ranges 3.2–10.4% across seeds). Second independent
piece of evidence (after DA-RAD) that train-time domain-adversarial training
doesn't help here.

**Domain divergence (`results_divergence.csv`)**: proxy 𝒜-distance and a linear
domain-origin probe both *increase slightly* after adaptation on all 4 targets —
even on targets where EER improves a lot. This rules out "adaptation works by
erasing domain identity" and is independent support for the ranking-calibration
mechanism (re-shaping the decision boundary, not domain-invariance). Also
explains post-hoc why DANN underperforms: it optimizes for a quantity
(domain separability) that isn't actually the bottleneck.

**Per-language MLAAD recall (`results_perlang.csv`)**: 6 languages held out of
every source pool entirely (mt, sl, hu, hr, fi, lt). Unseen-language fake
recall (93.3%) ≈ seen-language recall (93.9%) — no meaningful unseen-language
generalization gap, aside from one outlier (Maltese/`mt` on In-the-Wild, 63%).

All of the above is now written into `main.tex` (expanded to 10+ pages: new
main results table over 4 targets/6 methods, gain-decomposition table, backbone
table, DANN table, divergence table, per-language figure, expanded
discussion/limitations, broader-impact paragraph). Source data:
`results_ext.csv`, `results_dann.csv`, `results_divergence.csv`,
`results_perlang.csv` at repo root; full bundle incl. checkpoints was unzipped
from `results_bundle.zip` (not committed — `ckpt_ext/` is large). New
citations added to `main.tex` for DANN, AdaBN, Ben-David 𝒜-distance theory, and
the In-the-Wild dataset paper — **unverified against the original sources,
flagged for a citation check before submission**.
