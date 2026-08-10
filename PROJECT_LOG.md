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
clips, (b) take the confident top/bottom 30% as pseudo-fake/real (`q=0.3`), (c) self-train the
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

---

## 11. Adaptive `q` and `E` (2026-08-07 — pre-GPU validation phase)

**Motivation.** Every TTA number so far uses one hardcoded config — `q=0.3`,
`λ=0.3`, `E=4` — with no recorded provenance (`tta.py:114-119` uses the literal
`0.3` for all three auxiliary weights). `main.tex` Limitation 1 already concedes
it is not uniformly optimal, and Limitation 6 shows a fixed symmetric `q` drives
Protocol A from 26.33% to 42.45% EER. Goal: make the knobs adapt from unlabeled
target audio alone.

**Three proposed mechanisms were tested and killed before any GPU time.** This
is the main content of this phase — do not resurrect them without new evidence.

| mechanism | verdict |
|---|---|
| λ gated by pseudo-label churn under augmentation | **Confounded.** Moves λ by 0.017 across the entire AUC range .60–.99, but by 0.165 across augmentation magnitude alone. It reads `augment()`'s own gain/noise constants, not the corpus. |
| `select_q_max` (pick q from churn) | **Permanently saturated.** Returns the grid minimum q=0.05 on real audio for every corpus — 13/128 clips labelled vs 78 at the published q=0.3. Same confound. |
| E stopping on epoch-to-epoch label churn | **Degenerate.** Measured *exactly* 0.0000 every epoch of every arm: labels come from quantile cuts, so they only move if the ranking reorders, and self-training sharpens the boundary without reordering. Fired at `E_MIN` every time and measured **worse than the fixed-E control (10.16 vs 9.38 EER)**. |

**A second finding, worth a paragraph in the paper.** The `tail_gap` statistic
from `lambda_selector.py` correlates with Δ₂ at **r=+0.74, p=0.006** (n=12).
`main.tex:939-956` reports this as a failure because a reliability proxy needs
r<0 — but the magnitude is real and *the sign is the result*: high tail
separation means saturated scores, where MSE-against-stop-gradient freezes an
already-overconfident boundary. That is a **calibration** property, exactly
where `main.tex:952-956` predicted a working signal would have to live. For
contrast, **source AUC — which requires labels — predicts Δ₂ worse** (r=−0.397,
p=0.20).

It still does not cash out, which is why **λ stays fixed at 0.3**: the whole gap
between always-λ=0.3 and a *per-point oracle* is **0.24 EER pooled**, and
leave-one-target-out threshold selection lands inside that gap (better on
In-the-Wild, worse on ASVspoof2019) and flips sign with the threshold grid
resolution. Twelve (target, seed) points cannot settle a 0.24 EER question.

**What the method actually does now.** λ fixed at 0.3; q and E adapt:

* **q** — curriculum ramp from `Q_START=0.1` to the published ceiling `Q_CAP=0.3`,
  with the budget split asymmetrically by a BIC-guarded prevalence estimate
  (`2q·π`, `2q·(1−π)`, the `protocol_a.py:414` formula). The BIC guard is new:
  shrink back to π=0.5 when a 2-component score fit does not decisively beat a
  1-component one — the estimator's recorded failure returned 0.458 against a
  true 0.025 on the RawBoost model. At π=0.5 the split is *identically* the
  published symmetric behaviour, so on balanced pools only the ramp acts.
* **E** — run to 8 (the sweep's better value) with stopping on **`score_shift`**
  (mean per-clip |Δscore| between epochs), which decays cleanly on real data
  (0.051 → 0.037 → 0.022 → 0.009) where label churn was flat at zero. The rule
  is deliberately a **safety valve, not an efficiency measure** — every swept
  point says more epochs help, so it is biased toward spending the full budget,
  and divergence must be *sustained* (two consecutive rises above 1.5× the
  running minimum) after a single-epoch blip cut a healthy run short.
* **Collapse guard** — revert to the pre-adaptation model if the score
  distribution degenerates. Insurance against the Tent failure mode that
  `main.tex:460-479` describes as undetectable in advance.

**Priorities were inverted relative to effect size**, which is why λ was dropped
rather than fought for: q on skewed pools is worth ~16 EER (Protocol A), E=4→8
is worth 1.2–1.6 (sweep), λ selection ≤0.5 and fragile.

**Files.** `adaptive_tta.py` (pure numpy, no CUDA — that is what made the
pre-GPU validation possible), `test_adaptive_tta.py` (36 CPU checks, all
passing, including the confound as a *recorded negative result* so it is not
rediscovered), `adaptive_pipeline.py` (sections 1–5 copied programmatically from
`extended_pipeline.py` so target pools are provably identical; new adaptive loop
+ grid), `verify_reduction.py`. `protocol_a.py` gains an `adaptive=True` arm.

**`verify_reduction.py` is the load-bearing gate**: with both switches off,
`adapt_adaptive` is **bitwise identical** to the published `adapt()`. If that
ever fails, every adaptive-vs-fixed comparison is confounded by the refactor
rather than the schedules. Run it before trusting any new result.

**Not yet run.** The real grid (4 targets × seeds 0–2, reusing the 12 `ckpt_ext`
checkpoints, no source training) and the Protocol A arm both await the 35 GB
GPU. Smoke-run EERs are on 128 clips (~0.8% per clip) and carry **no signal** —
the smoke validates mechanism, not results.

**Trap for whoever runs this next.** `extended_pipeline.py`'s data cell starts a
**45 GB MLAAD download** on any machine where `data/mlaad` is missing (hit once,
killed at 1.4 GB). `adaptive_pipeline.py` asserts corpora are present instead,
and omits MLAAD entirely — it only ever fed source training, which that file
does not do, and dropping it leaves the target pools bit-identical.
