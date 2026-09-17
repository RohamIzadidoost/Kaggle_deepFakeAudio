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
the In-the-Wild dataset paper — ~~unverified against the original sources~~
**VERIFIED 2026-08-18, see §12**.

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

## 12. Citation verification (2026-08-18) — closes R6

`ICASSP_PLAN.md` R6 listed unverified citations as the one **Blocker**
("a fabricated reference is the one failure mode that ends a paper regardless of
merit"). All 11 `\bibitem` entries in `main_icassp.tex` were checked against
DBLP / ACL Anthology / ISCA Archive / IEEE Xplore / PMLR / arXiv.

> **Correction (same day).** The first pass asserted `main.tex` carried an
> *identical* bibliography. It does not — that was assumed, not checked, and
> `main.tex` had three defects `main_icassp.tex` did not: `xlsrsls` missing its
> venue, `crossdomain` **missing its entire author list** (exactly the defect R6
> named), and `itw` misspelling Diekmann as "Dieckmann". All three are now fixed.
> Verify each file's bibliography separately; they drift.

**Result: no fabricated references, no wrong author lists, no wrong venues or
years. Zero corrections required.** Specifically confirmed:

| key | verified against | status |
|---|---|---|
| `wav2vec2aug` | ISCA Odyssey 2022 archive; arXiv:2202.12233 | ok |
| `xlsrsls` | ACM MM 2024, DOI 10.1145/3664647.3681345 | ok (Qishan Zhang, Shuangbing Wen, Tao Hu) |
| `aasist` | ICASSP 2022 | ok (Jung, Heo, Tak, Shim, Chung, Lee, Yu, Evans) |
| `rawnet2` | ICASSP 2021, pp. 6369–6373 | ok (Tak, Patino, Todisco, Nautsch, Evans, Larcher) |
| `asdg` | IEEE TIFS **vol. 19, pp. 344–358**, 2024 | ok (Xie, Cheng, Wang, Ye) |
| `crossdomain` | EMNLP 2024, `2024.emnlp-main.286` | ok — all 7 authors correct |
| `tent` | ICLR 2021 | ok (D. Wang, Shelhamer, Liu, Olshausen, Darrell) |
| `dann` | ICML 2015, PMLR v37, pp. 1180–1189 | ok |
| `adabn` | arXiv:1603.04779 | ok (Yanghao Li, Naiyan Wang, Jianping Shi, Jiaying Liu, Xiaodi Hou) |
| `bendavid` | *Machine Learning* **79(1–2):151–175**, 2010 | ok — the `no. 1--2` is right |
| `itw` | Interspeech 2022, pp. 2783–2787 | ok — all 5 authors correct |

Two notes, neither an error:

* `crossdomain` previously had no author list (the reason R6 called it out); it
  now carries all seven and they are correct.
* `asdg` is cited as 2024, which is the **issue** year (TIFS vol. 19). Its DOI
  carries 2023 because it was published online-first in October 2023. Citing 2024
  is standard and matches DBLP.

Optional polish, not required for correctness: `asdg` could gain
`vol.~19, pp.~344--358` and `rawnet2`/`itw` their page ranges. Several entries use
"et al." — fine under the ICASSP page budget.

**Addendum — two citations added later the same day** (commits `9ed54fe`,
`e9032ab`, from a parallel session) were verified after the fact:

* `shiftyspeech` — **correct**. arXiv:2502.05674 is titled *"ShiftySpeech: A
  Large-Scale Synthetic Speech Dataset with Distribution Shifts"* with **nine**
  authors including Lin Zhang. Note the `ash56/ssl-aasist` model card still shows
  the paper's **older** title ("Less is More for Synthetic Speech Detection in
  the Wild") and an eight-author list omitting Lin Zhang — the card is stale, the
  bibitem is right. Cite arXiv, not the card.
* `deepfensehf` — **correct and appropriately scoped**: the DeepFense repos ship
  released weights with no paper and no model card, which the entry states
  outright rather than inventing a reference.

Lesson for the next citation pass: R6 is not a one-time gate. Any new `\bibitem`
added after a verification sweep is unverified again by definition.

---

## 13. ICASSP breadth pass + autonomous strengthening (2026-08-27 → 09-07)

Branch `icassp-breadth`. Full narrative and every number: `FINDINGS_AUTONOMOUS_RUN.md`
(Addenda 1–5). Summary of what changed the paper's claims:

**The headline narrowed, and firmed.** Across 369 third-party cross-corpus cells
(9 released checkpoints, 5 architectures incl. an Audio Spectrogram Transformer,
10 corpora) the method **closes the calibration deficit** `(100−EER) − acc@0.5`
from 7.29 → 1.96 points (p = 2.2e−17, 248/369). Its effect on **ranking** is an
order of magnitude smaller: null (p = 0.52) on the 209 ASVspoof-lineage cells,
+0.9 EER (p = 0.003) once the two lineage-independent corpora are pooled in —
driven entirely by WaveFake, where the deficit is largest. The paper leads with
threshold-repair, not EER, and reports the ranking effect split this way.

**Claims retired / reframed:**
- The `r = +0.86` source-AUC precondition changes sign under resampling
  (−0.20 at 43 cells, −0.02 at 257) — retired, not restated.
- "Improves In-the-Wild EER on all four public checkpoints" was true only of that
  single-corpus arm; at breadth 7/12 up, 5/12 down, 4 significant degradations.
- `E` counts epochs, so the update budget scales with pool size — reported as a
  protocol defect. E=32 more than doubles the Arabic gain (10/10 seeds,
  p = 0.002) but degrades the weakest-ranking target; matched-budget control
  (stage K) shows Tent/st_only do not benefit from the 8× budget.

**Mechanism nailed down (Addendum 3 + stage S):** a one-line label-free rule —
threshold at the median score — is statistically indistinguishable from the full
gradient method across 133 cells (recovers 94% on 70 our-model cells; *beats* TTA
by 1.9 pts, p = 0.04, on 63 third-party cells, where adaptation can degrade
ranking and a threshold move cannot). Both land ~1 pt under a labelled oracle.
Skewing the pool away from
class balance breaks both together (50/70/90% fake → acc gain
+7.6/−3.5/−26.5, ranking flat). **The method is a threshold rule targeting a
balanced-prior operating point.** Actionable precondition: the target pool must
be roughly class-balanced — checkable without labels, unlike the AUC story.

**The positive EER story survives, conditionally (Addendum 4):** on two corpora
sharing no lineage with the source pool (WaveFake, 2024 commercial TTS) the
method improves EER by +2.5 / +2.9 points (10 seeds, p = 0.002), *larger* than
any original target. The benefit is contingent on calibration headroom, which
lineage-relatedness predicts — ASVspoof2021 relatives (LA/DF/PA) arrive
well-thresholded and adaptation only drifts them off.

**Own model at 10 seeds (Addendum 1):** 28 source checkpoints trained locally
(~6.6 min each, fidelity-gated against the cloud run — see `FIDELITY_NOTE.md`),
so the paper's own method is no longer capped at n = 3. Published config
replicates: Arabic +1.41 (p = 0.008), ASVspoof2019 +1.13 (p = 0.008),
In-the-Wild +1.12 (nominal), dataset2 null.

**Autonomous GPU pass (2026-09-07, Addendum 5):** queue was drained; three new
value-ordered stages queued (S: median control at breadth; T: E=32 on the AST
degradation cells at 5 seeds; R2: the independent-corpora third-party arm at 10
seeds). New files `threshold_control_multi.py`, `analyze_threshold_control_multi.py`;
no committed-result file modified. GPU cost of the whole branch: ~101 h + this
pass, 0 failures.

**R2 reviewer vulnerabilities (2026-09-09, Addendum 6).** A reviewer summary
flagged two downgrade risks. Both now addressed:
- **Median-rule equivalence.** The median-threshold rule matches the full method
  *on average*, but on ASVspoof2019 (strong source ranking) gradient TTA raises
  ROC-AUC within every one of the six attack families — 30/30 attack×seed cells,
  +1.1 EER, p = 1.9e−9 — which no threshold move can do. Null on dataset2
  (source AUC 0.71). So the equivalence holds where the method reduces to
  re-calibration; given a usable ranking it adds real separability
  (`r2_vuln1_subpop.py`, stage U).
- **Class-balance precondition.** BBSE (`p̂ = M⁻¹q`, M the source confusion
  matrix) estimates target prevalence label-free to within 1.4% at 50–95% skew;
  re-pointing the threshold at the p̂ quantile matches the labelled
  accuracy-optimal threshold to within 0.4 pts (a score GMM does not: 28% error
  at 90% fake), robust to moderate calibration drift. The threshold is
  recoverable at any prior; the self-training pseudo-labels stay noisy under
  severe skew (`r2_vuln2_prevalence_cpu.py`, CPU only). A parallel opus session
  had already killed the score-shape strategies (GMM/Otsu/k-means/FreeMatch/
  prototype) — BBSE is different because it reads source error structure, not
  the target score distribution.

**Dynamic q + Protocol A (2026-09-10, Addendum 7).** The full V2 answer — not
just recover the eval threshold (BBSE) but make the *self-training* work on
skewed pools. `adapt_dynq()` = published `adapt()` with the confident-tail
budget split by a BBSE prevalence estimate (byte-identical to `adapt()` on a
balanced pool). Controlled skew sweep, ten seeds: fixed `q=0.3` degrades EER as
the pool skews (In-the-Wild 11.0→13.3→16.4 at 70/90/95% fake); dynq reverses it
on the ranking-capable targets (ASVspoof2019 5.34→3.52, In-the-Wild 13.30→11.61
at 90% fake, p≤0.006), flat on the AUC-0.71 target. Protocol A (official 97%-spoof
DF eval): the guarded adaptive variant never reproduces the naive collapse
(42.46 EER) — reverts to source on 2/3 seeds, reaches 24.4 vs source 26.33 on
1/3 — but does not reliably beat source; on the RawBoost source everything fails
(that model predicts 46% of a 97.5%-spoof pool as real). New: `adapt_dynq`,
`_bbse_pi_fake` in adaptive_pipeline.py; `bbse` / `ours_bbse_adaptive` arms +
`PROTOA_SEED`/`PROTOA_ARMS` in protocol_a.py.

**Framing rebalanced (2026-09-10).** The abstract/intro/conclusion had drifted
to leading with "the mechanism is threshold repair" — too self-deprecating for a
paper whose own-model generalisation result (LOCO +0.88 EER p<1e-4;
lineage-independent corpora +2.7 EER p<1e-5) is solid and unchanged. Rewrote to
lead with "improves cross-corpus generalisation" and use operating-point repair
as the *mechanism and scope map*, not the headline verdict. Same evidence.

**Paper state:** `main_icassp.tex` compiles at 4 content pages + refs-only page
5, no overfull boxes. `PROJECT_LOG` §6 ("Final validated results") is superseded
by this section and the FINDINGS doc — the 9.42% / 5.50% numbers there are the
pre-breadth cloud run. `main.tex` (the 11-page full version) was NOT updated on
this branch and still carries the pre-breadth claims.

## 2026-09-14 — User-requested ICASSP review correction

Revised `main_icassp.tex` around the confident-tail counting constraint and
recomputed all short-paper tables from 89 cached score arrays. The original
short paper is preserved in `manuscript_audit/original/`; the detailed correction
and remaining experiments are documented in `manuscript_audit/README.md`.

The major audit findings were a phase-denominator mix-up (533,928 eval trials,
not 611,829), near-one scores described incorrectly as exact ties, and the
nearest-ROC-point EER approximation understating error on coarse/tied scores.
`metrics.compute_eer` now interpolates the ROC crossing and has CPU regression
tests. Historical CSVs are unchanged and still use the old estimator. Added
exact oracle threshold diagnostics and frozen-source median/BBSE controls.

The short-paper claims now distinguish damage prevention from source improvement,
report balanced accuracy and disjoint results, qualify the prior-estimation and
purity arguments, identify baseline variants accurately, and avoid significance
claims from reused checkpoints/data. The earlier 369-cell deficit/guard claims
are archived, not included in the revised short paper. No neural training or
inference was performed, and no files were pushed or submitted.

## 2026-09-17 — The operating-point law (ICASSP rewrite)

Reviewer pass demanded EER/AUC/BA leadership, a low-prevalence sweep, disjoint
win counts with intervals, and a figure. Doing it produced a different paper.

**What the audit exposed first.** Balanced accuracy contradicts the accuracy
headline: symmetric q=0.3 on DF-42 "collapses" accuracy 98.78→61.80 while BA
*rises* 78.35→80.35. BBSE's accuracy restoration lowers BA on 2 of 3
checkpoints. The 97.22% always-spoof baseline makes DF accuracy nearly
uninformative. Leading with the accuracy collapse was a liability.

**Three GPU queues (63 symmetric-budget runs total, 0 crashes after the OOM
fix).** Low-prevalence sweep on In-the-Wild (pi 0.01–0.10); a dose-response
sweep in q at pi=0.01 and 0.97 plus large budgets (q up to 0.50) at native
prevalence; a second checkpoint (DF-42); and the same grid on a second corpus
(ASVspoof2021-DF resampled to pi 0.01/0.10/0.50/0.90). All predictions
pre-registered in `PREREGISTERED_low_prevalence_predictions.md` and
`PREREGISTERED_excess_dose_response.md` before launch; 3 of 7 first-round
predictions were falsified, including "small q must break where the bound says
it must" and "BBSE stays safe down to pi=0.02" (it estimates 36–42% spoof on
1–10% spoof pools).

**The result.** Confident-tail adaptation is an operating-point procedure:
the attainable accuracy ceiling shifts by a median 0.12 points while the gap
between that ceiling and accuracy at threshold 0.5 shifts by 15.86, with 0/63
exceptions. Accuracy rises **iff** the post-adaptation gap ends below the
source's — 63/63. Where it lands is ordered by the contamination fraction
`Delta = sum_c (b_c - pi_c)^+` implied by the counting bound (12/14 strata that
vary the budget; 7/7 of those whose source threshold was well placed), not by
pseudo-label purity. Neither purity nor q orders outcomes: q=0.50 at pi=0.37
gains +7.5 BA while q=0.15 at pi=0.01 gains +2.6, and on a balanced DF pool
with Delta=0 and purity exactly 1.000 adaptation still costs 8.6 accuracy
points — the sharpest available statement that the bound is necessary, not
sufficient.

**Repo changes.** `audit_icassp.py` gained the contamination analysis, the
correct 8,000-clip adaptation budget for resampled pools (it was hardcoded at
20,000, which silently merged the adaptation pool into the evaluation pool and
dropped the disjoint split), a self-check of the reconstructed disjoint mask
against the run-time row, and acceptance of either EER estimator (runs after
2026-09-14 record the interpolated one). New: `bootstrap_eer_ci.py` (paired
stratified bootstrap), `make_figure.py`, `run_skew_low*.sh`, `run_excess*.sh`.
`main_icassp.tex` is 4 pages + references, 0 overfull boxes, every prose number
verified against the audited arrays.

## 2026-09-17 (afternoon) — Reviewer correction, and a batch-normalisation bug

**Reject, on the framing of the central result.** The objection was that an
oracle-accuracy decomposition was being presented as evidence of ranking
preservation. Checked, and correct on all three counts:

- At skewed prevalence the "attainable ceiling" sits only 1.7--3.0 points above
  the always-majority rule, and within 1 point in 26 of 79 runs. Its stability
  was largely a property of the metric, not evidence about the ranking.
- The 79/79 "adaptation helps iff the gap falls" was arithmetic:
  `d_acc = d_oracle - d_gap`, so the biconditional follows once the oracle term
  is small.
- Delta had only been validated within prevalence, where it is a monotone
  transform of q and the two cannot be separated.

The manuscript now makes the ranking claim directly and without the
decomposition: accuracy moves a median 15.14 points while EER moves 0.38 and
AUC 0.12, accuracy moving more than ten times as far as EER in 66 of 79 runs.
Delta survives a proper across-strata test that it had never been given: pooled
Spearman +0.75 against +0.36 for q; pairs from different strata matched on
Delta but far apart in q differ by a median 7.8 gap points, against 28.9 for
pairs matched on q. Figure rebuilt on both. 4 pages + references, 19/19 changed
numbers re-verified.

**View pilot (PILOT_VIEWS_NOTES.md).** Views built from offset crop + gain +
noise made the frozen teacher *worse*. Isolating each transformation showed why:
crops and gain cost nothing (-3.3% to -0.1% relative EER), additive noise costs
+80.8% at sigma=0.005 and +32.3% at sigma=0.001. The existing consistency term
perturbs with exactly that gain+noise, so it demands agreement against a view
whose own ranking is 84% worse -- a mechanism for its null ablation.

**A float16 claim, made and retracted.** The pilot's view 0 did not reproduce
the cached pipeline score (4.856% against 5.060%), and the float16 decode
buffer looked like the cause. It was not: `PUBA_FP32=1` gives 5.058%.

**The real cause, and it is serious.** torchaudio's `_Wav2Vec2Model.forward`
runs `layer_norm(waveforms, waveforms.shape)` on the batched tensor, taking the
statistics over batch *and* time, so every clip is normalised by its
neighbours' loudness; upstream fairseq normalises per utterance before
batching, so the two agree only at batch size one. Forward hooks put the
divergence at the first convolution, i.e. in the input. One clip scores 0.2029
alone, 0.2031 among copies of itself, 0.3449 among 32 distinct clips; mean
shift 0.034 over 4,000 clips, maximum 0.73, and it persists with autocast off.
Per-utterance normalisation takes batch dependence from 0.606 to 0.025 (bf16)
and 0.00095 (fp32), and is wired in as opt-in `PUBA_PERSAMPLE_NORM=1` so the
cached arrays keep their meaning.

Consequences still open: every `ssl_aasist` score depends on manifest order,
batch size and pool composition. The manuscript's arms share one batching, so
its comparisons hold, but the prevalence intervention changes pool composition
and therefore scores independently of adaptation, and that confound is
unquantified. The multi-view teacher gain (-4.6% relative EER) was measured
with view 0 and the view mean under different batch statistics and is therefore
unverified; a corrected re-run was in flight when work stopped
(`pilot_views_itw_fixed.npz`).

Also fixed: `protocol_a_public.py`'s resume guard treated a float32 rerun as
satisfied by the float16 row, so audio precision is now part of the run key and
the cached-score filename.
