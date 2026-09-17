---
name: darad-status
description: DA-RAD paper implementation progress (the ICASSP/Interspeech generalization paper)
metadata: 
  node_type: memory
  type: project
  originSessionId: c7d6f656-ee0b-4dbe-b6da-6a4cf7966d06
  modified: 2026-07-29T12:25:04.219Z
---

Building DA-RAD (Domain-Adversarial Real-Anchored Detection) per the approved plan
at ~/.claude/plans/lets-work-more-to-spicy-stonebraker.md. Goal: conference paper
showing SSL detectors learn recording-domain shortcuts; DA-RAD (RawBoost aug +
source-adversarial GRL + asymmetric Real-Anchored contrastive loss) improves
cross-corpus EER. See [[project-goal]], [[cross-dataset-results]].

Task list tracked via TaskCreate (#1-#9). DONE + self-tested (no external deps):
- losses.py — GradientReversal/DomainAdversarialLoss + RealAnchoredContrastiveLoss
  (asymmetric: anchor real cross-domain, margin-push fakes, fakes unconstrained).
- augment.py — RawBoost (LnL convolutive / ISD impulsive / SSI coloured), algos 1-5.
- sampler.py — DomainBalancedBatchSampler (multi-domain + both-class batches; handles
  fake-only corpora like MLAAD).
- eval_protocol.py — official DF EER from CM keys (parses the real keys: 611,829
  trials) + cross-corpus matrix summary (gap/avg/worst).

ALSO DONE (2026-07-23): models/ (lcnn, ssl_frontend, dg_model — full DA-RAD assembly
grad-flows), darad_dataset.py (raw-waveform + RawBoost + domain), trainer.py
(config-driven, AMP, grad-accum, GRL ramp, DG losses, saves head-only checkpoint).
Data downloaded+verified: ASVspoof2019-LA (train 25380/dev 24844/eval 71237, exact
official counts) + In-the-Wild (31779 = 19963 real/11816 fake) via download_data.py.
XLS-R-300M weights cached. build_manifest extended → manifest_all.csv (768,004 rows,
5 corpora). Unit tests pass for every module.

HARNESS VALIDATED: trainer.py with real frozen XLS-R-300M + head, Protocol A
(train 2019-LA 5k balanced subset → test 2021-DF eval 4k), reduced run reached
best val EER 16.15% / AUC 0.92 in 6 epochs on the RTX 3080. Confirms the full
pipeline (data→SSL→losses→AMP→EER) works; full-scale run would approach lit ~5%.
Gotchas fixed: AMP dtype (run frozen encoder autocast-OFF fp32), lazy XLS-R load
must move to input device, save trainable-only (not 1.2GB).

FIRST DA-RAD ABLATION = NEGATIVE RESULT (2026-07-23, Protocol B, test=held-out
In-the-Wild, 5 epochs, 9.7k train, FROZEN XLS-R): EER XLS-R 30.97% → +RawBoost
32.65% → +GRL 35.58% → +DA-RAD(anchor) 45.05%; AUC 0.76→0.56. Every DG component
made cross-corpus WORSE, monotonically. Prime suspect: DG losses act only on the
tiny 256-d head while XLS-R is FROZEN — you can't make frozen features
domain-invariant with a small adversarial head, so GRL/anchor just fight the
classifier and destroy signal. Fix to try next: LoRA/partial-unfreeze the encoder
so the representation itself can adapt; also down-weight λ_grl/λ_anchor, more
epochs/data. Do NOT trust DA-RAD until re-run with an adaptable encoder.
Results in ablation_results.txt + protocols/abl_*.log.

GO/NO-GO with ADAPTABLE encoder (2026-07-23, top-4 XLS-R layers fine-tuned via
bf16, lr 1e-4, Protocol B, test=held-out In-the-Wild, 4 epochs, gonogo_results.txt):
- FINE-TUNING FIXES THE BASELINE: frozen XLS-R 31% EER → fine-tuned XLS-R 9.42%
  EER (AUC 0.966). Confirms the frozen-encoder diagnosis; harness now hits
  literature-competitive numbers. This is the real, honest win — and it's just
  standard SSL fine-tuning, NOT the DG novelty.
- DA-RAD DG LOSSES DON'T BEAT IT: anchor-only best 10.88% (WORSE than 9.42%); full
  DA-RAD best 13.58% and CATASTROPHICALLY unstable — GRL drove it to 50% EER / AUC
  0.50 (pure chance) by epoch 4. => Clear NO-GO for DA-RAD as the contribution.
Bugs fixed to get here: partial-unfreeze (top-K layers in SSLFrontend), lazy encoder
must be materialised before optimizer, and fp16 GradScaler overflows on fine-tune
grads → switched trainer to bf16 autocast (no scaler).

*** BREAKTHROUGH (2026-07-23): Real-Manifold TTA WORKS ***
After DA-RAD (train-time DG) failed, pivoted to Test-Time Adaptation. Take the
frozen fine-tuned XLS-R source model (protocols/ft_xlsr.pt), adapt UNSUPERVISED to
held-out In-the-Wild (labels only for eval), adapting LayerNorm affine params of
top-4 layers + head. Results (tta.py, tta_results.txt, transductive on 4k balanced):
- source-only:            EER 9.42%  AUC 0.966
- vanilla Tent (ent-min): EER 49.45% AUC 0.506  (COLLAPSED — anti-example)
- Real-Manifold TTA:      EER 6.80%  AUC 0.984  <-- 28% relative EER cut, AUC UP
Method = self-train on confident-RANKED pseudo-labels (works because ranking/AUC
transfers cross-corpus even when threshold doesn't) + pull pseudo-real cluster
compact to a real prototype + channel-consistency, with anti-collapse (confident
tails only, LN-only params, pseudo-fake push). AUC improved => RANKING improved,
not just calibration (the key scientific bar). This is the first GENUINE positive.

VALIDATION DONE (tta_results.txt): the TTA win is REAL and STABLE, but the winning
recipe DROPS our real-anchor.
- multi-seed full: 6.80 / 6.83 / 7.17% EER (mean 6.93±0.20) vs source 9.42% — stable.
- INDUCTIVE (adapt on ITW-A, eval on DISJOINT ITW-B): 9.88 -> 7.48% EER — the gain
  holds on unseen samples, so it's NOT transductive memorization. 
- ABLATION (seed0): self-train-only 7.13; +consistency 6.37 (BEST); +anchor 7.37
  (WORSE); full(st+anchor+cons) 6.80. => self-training on rank-transferred confident
  pseudo-labels is the core driver, channel-CONSISTENCY adds real value, and the
  REAL-ANCHOR does NOT help (slightly hurts) — same as it failed in DA-RAD. DROP IT.
- vanilla Tent collapsed (49%) — our method's stability vs entropy-min is a key point.

REAL contribution/novelty now = the INSIGHT that ranking/AUC transfers cross-corpus
even when the threshold doesn't (source AUC 0.966 on unseen ITW), which makes
confident-tail pseudo-labels reliable and enables STABLE self-training TTA where
entropy-min (Tent) collapses. Method = "rank-transfer self-training + channel
consistency" TTA. Best ~6.4% EER on In-the-Wild (from 9.4% source).
CONFIRMED (tta_results.txt, run_tta_confirm.sh, 2026-07-23): no-anchor config
(self-train + consistency) across 3 seeds x 2 OOD targets:
- In-the-Wild:        source 9.42% -> adapted 6.41±0.11% EER, AUC 0.985±0.001
- ASVspoof2021-DF:    source 5.50% -> adapted 5.29±0.03% EER, AUC 0.989±0.0002
Both TIGHT across seeds (std <0.15 EER) -- stable, not luck. DF gain is smaller
because DF is closer to source training distribution (less domain gap to close) --
coherent with the method's mechanism (helps more where the gap is bigger).
Combined with the earlier inductive check (adapt on ITW-A, eval disjoint ITW-B:
9.88->7.48%, proving it's not memorizing eval clips), this is now a validated,
multi-corpus, multi-seed, inductively-real result. READY to write up / scale up.

METHODOLOGY NOTE (important, discussed with user): TTA uses UNLABELED target
inputs during adaptation (self-training on the model's own confident pseudo-labels),
NEVER target labels -- labels only used post-hoc for EER. Most reported numbers are
TRANSDUCTIVE (adapt+eval on the same unlabeled pool) which is standard for TTA
literature (cf. Tent) but must be stated explicitly in the paper alongside the
inductive result, not hidden. This is the key methodological honesty point for
reviewers -- don't let it get lost.

NEXT: this is now solid enough to (a) write up as the paper's core result, and/or
(b) scale up on better GPUs for a full multi-target, larger-scale version.

ARABIC + CLOUD NOTEBOOK (2026-07-27): Professor approved; adding Arabic (ArAD,
HF DeepFake-Audio-Rangers/Arabic_Audio_Deepfake) for MULTILINGUAL. Extracted 19,561
wavs via download_arabic.py (decode=False -> raw bytes, no torchcodec) to
data/arabic_arad/<split>/<real|fake>/; build_manifest has load_arabic_arad;
manifest_all.csv now 6 corpora / 787,565 rows (54,267 real / 733,298 fake; Arabic
2,570 real / 16,991 fake, ~3s clips). Cloud compute spec given to prof: GPU 24GB /
16 CPU / 64GB RAM / 150GB storage / 1 GPU / 3 days.
Deliverable: cloud_pipeline.ipynb (+ jupytext source cloud_pipeline.py) — clean
self-contained notebook, smoke-tested end-to-end on local data. Distilled WINNING
method only (fine-tune XLS-R top-4 -> TTA self-train+consistency, NO anchor/GRL):
download (kaggle 2019-LA + In-the-Wild, HF Arabic) -> manifest -> source train ->
TTA -> cross-corpus EER table (In-the-Wild + Arabic). Needs ~/.kaggle/kaggle.json
on the cloud. Uses mp.set_start_method("fork") for DataLoader-worker portability. Notebook now
also runs the Tent baseline so its output table is source / Tent / ours per target.
CLOUD ENV: Manipal JupyterLab (jupyter.muj.edu.in). Kernel "Python 3 (ipykernel)"
= Python 3.10.12 (USE THIS, not the 3.8 kernel). Hardened notebook: robust install
(matches torchaudio to preinstalled torch to avoid ABI mismatch), env-check cell 0
(prints torch/torchaudio/CUDA/bf16, asserts XLS-R bundle), and amp() helper that
uses bf16 if torch.cuda.is_bf16_supported() else fp32 fallback (T4/V100-safe).
Re-smoke-tested end-to-end. Needs kaggle.json uploaded to the cloud.

PAPER DRAFT DONE: main.tex fully rewritten around the TTA story, title "Ranking
Transfers, Thresholds Don't". Real numbers in tables (source 9.42->ours 6.41 ITW,
Tent 49.45 collapse, ablation, inductive 9.88->7.48). Arabic row is a placeholder
to fill from the cloud run. Authors: Roham, Dr. Sumit Srivastava, Shweta Sharma.
Insight FIGURE (fig_score_dist.png, make_figure.py, from real scores): logit-axis
score distributions on unseen In-the-Wild, source vs TTA. KILLER supporting data:
source AUC 0.966 but acc@0.5 only 61% (78% of REAL clips scored as fake, median
real score 0.996!) -> ranking transfers, threshold doesn't; TTA -> acc@0.5 93%.
Structurally validated (no local LaTeX); needs one Overleaf compile + upload the png.

OVERNIGHT NOTEBOOK v2 (2026-07-28): overnight_pipeline.ipynb (+ .py jupytext),
replaces cloud_pipeline. Built after the first cloud run gave a WEAK source model
(ITW EER 21.96 / AUC 0.850 vs local 9.42 / 0.966) and TTA therefore couldn't help.
ROOT CAUSES FOUND + FIXED:
1. cloud AudioDS always CENTER-cropped -> no train-time augmentation. Now caches 4s
   and takes RANDOM 3s crops.
2. dataset_2 glob only caught seedtts_files+xTTS (1,199 fakes). It actually has 8
   generators (FlashSpeech/NaturalSpeech3/OpenAI/PromptTTS2/VALLE/VoiceBox/seedtts/
   xTTS) = 2,173 fakes. Now globs all subfolders + tracks `generator`.
3. naive balanced() made fakes ~95% ASVspoof. Now water-fills fakes across
   (corpus, generator).
4. n_finetune was bumped 4->6 without evidence; back to validated 4.
VERIFIED LOCALLY (RTX3080, reduced 5 epochs/2000 per class/bs16): source ITW EER
10.90 AUC 0.959 (vs cloud 21.96/0.850) -> the fixes work. Full run 6.2 min.
KEY ARCHITECTURAL WIN: all audio decoded ONCE into a single fp16 GPU tensor
(17.5k clips = 2.25GB, 7 seconds to build) -> NO DataLoader/workers at all. This
kills the /dev/shm crashes AND the ~16% GPU utilisation (was I/O bound).
Grid = leave-one-corpus-out x 3 seeds x {source, tent, st_only, ours} + inductive
check + TTA hyperparameter sweep (q/epochs/lambda). Incremental results.csv +
sweep.csv + ckpt/ + run_log.txt so an interrupted night keeps its work.

IMPORTANT OBSERVATION to verify at scale: with the BETTER source model, TTA's gain
shrank (10.90 -> 10.30/10.60 EER, AUC 0.959->0.964) vs the earlier 9.42->6.41.
CORRECTION (2026-07-28): I first blamed this on "calibration headroom" — that was
WRONG. EER/AUC are threshold-free and invariant to monotonic score transforms, so
a calibration fix CANNOT change them. The earlier ITW gain (AUC 0.966->0.985) was a
genuine RANKING improvement. The acc@0.5 61%->93% was the calibration part. Real
candidate causes for the smaller gain: different (augmented, generator-diverse)
source training, reduced scale, TTA_EPOCHS 3 vs 4, smaller target pool, single seed.
Resolve with the full multi-seed run; do not repeat the calibration explanation.

PROTOCOL BLIND SPOT (flagged by the RePAIR-TTA doc, must address): our target pools
are built class-BALANCED by sample_target(), and TTA uses symmetric q=0.3 tails.
So the symmetric-prior assumption is always satisfied in our evaluation and we have
NEVER tested the failure mode where target fake prevalence != 50%. A reviewer will
ask. Cheap decisive test: rebuild target pools at fake prevalence 10/30/50/70/90%
and re-run the EXISTING st_cons method.

RePAIR-TTA PILOT TESTED (2026-07-28, repair_tta.py, repair_results.csv,
repair_run.log; local RTX3080, ITW target, source ckpt from overnight_pipeline,
single seed, 1400-clip pools). Implemented K=4-view logit mean/variance,
reliability w=|2r-1|*exp(-v/median(v)), reliability-gated pairwise softplus
ranking loss + consistency. VERDICT: DO NOT ADOPT the pairwise objective.
- Condition A FAILED: balanced pool EER source 10.71 / st_cons 10.71 (AUC .967) /
  repair 12.71 (AUC .946) / repair_nocons 10.86. Pairwise is WORSE, and pairwise+
  consistency is worse than pairwise alone => the two losses interact badly.
- Condition B only PARTIAL: EER delta vs source by fake prevalence
  0.1: repair +0.68, st_cons +0.68 | 0.3: repair +0.72, st_cons -0.27
  0.5: repair +2.00, st_cons 0.00  | 0.7: repair -0.16, st_cons +0.34
  0.9: repair +1.35, st_cons +2.06
  So repair IS better than st_cons in the fake-heavy regime (0.7, 0.9) as the doc
  predicted, but loses everywhere else.
- BUT THE BIG FINDING: source-only is best or tied at 3 of 5 prevalences. With the
  fixed (augmented + generator-diverse) source model, NEITHER adaptation method
  convincingly beats doing nothing. Note EER noise at n=1400 is ~±1 point, so
  most gaps here are not significant.
- WHAT IS VALIDATED: the reliability signal itself. Pseudo-label accuracy by
  reliability decile rose monotonically 0.84/0.79/0.77/0.79/0.82/0.92/0.94/0.96/
  0.96/0.99 and pair purity among class-spanning pairs was 0.999 (9.1% same-class
  pairs). So perturbation-stability gating DOES predict label correctness -- it is
  the pairwise LOSS that fails to exploit it. If revisiting, use w to weight/select
  within the existing st_cons cross-entropy instead of adding a ranking loss.

BUG HIT ON THE FIRST CLOUD OVERNIGHT RUN (2026-07-28) — MY DESIGN FLAW: the
SMOKE=True validation run wrote a throwaway 1-epoch/64-clip model to
ckpt/source_in_the_wild_seed0.pt; the real SMOKE=False run then RESUMED that
poisoned checkpoint instead of training, so seed0/in_the_wild reported chance
(EER 50.38 / AUC 0.501) and its TTA rows are meaningless. Only that one fold was
affected (smoke only covered seed0 + in_the_wild); arabic + seeds 1,2 trained
normally. FIXED in overnight_pipeline: SUFFIX = "_smoke" if SMOKE else "" applied
to CKPT_DIR / results / sweep / run_log / scores npz, so smoke can never poison a
real run. If a run looks like chance, CHECK FOR A "loaded ckpt/..." LINE FIRST.

*** FIRST COMPLETE CLOUD GRID (2026-07-28, results_cloud.csv) ***
3 seeds x {in_the_wild, arabic} x {source,tent,st_only,ours} + inductive.
Config: SOURCE_EPOCHS=12, BATCH=64, SOURCE_PER_CLASS=5000, TARGET_PER_CLASS=3000.
TRANSDUCTIVE EER mean+-std (AUC):
  in_the_wild: source 16.09+-0.33 (.920) | tent 33.99+-2.49 (.660) |
               st_only 14.87+-0.47 (.925) | ours 14.70+-0.55 (.934)
  arabic:      source 28.44+-1.73 (.785) | tent 36.30+-11.59 (.661) |
               st_only 28.50+-2.45 (.780) | ours 27.98+-2.12 (.793)
INDUCTIVE (adapt half, eval disjoint half):
  in_the_wild: source 16.40 -> ours 14.99 (-1.41, 3/3 seeds)  <-- REAL, replicated
  arabic:      source 28.25 -> ours 28.23 (-0.01, 2/3)        <-- NO inductive gain
PAIRED per-seed (neg = better):
  ITW  ours vs source -1.39 (3/3) | st_only vs source -1.22 (3/3) | ours vs st_only -0.18 (2/3)
  ARAB ours vs source -0.47 (3/3) | st_only vs source +0.05 (1/3) | ours vs st_only -0.52 (3/3)
READ: (1) Tent collapses in 6/6 folds — strongest, most robust result we have.
(2) TTA works on In-the-Wild, transductive AND inductive, 3/3 seeds. Modest (~1.4
EER, 8.7% rel) but consistent. (3) Arabic: marginal transductively, ZERO
inductively — do not oversell multilingual. (4) Nice mechanism nugget: on the
harder-shift target (Arabic, source AUC .785) plain self-training FAILS (1/3)
and only the consistency term rescues it (ours vs st_only 3/3); on the easier
target (ITW, AUC .920) self-training alone already gets most of the gain.
CAVEAT: n=3 so nothing reaches p<.05 — report "consistent across 3 seeds",
mean+-std, NOT "significant". And the 12-epoch baseline is probably overtrained
(local 5-epoch got ITW source 10.90 vs 16.09 here), so the gain may shrink
against a properly trained source. THE 6-EPOCH TEST IS NOW THE PRIORITY.

*** FINAL 6-EPOCH GRID COMPLETE + INSERTED INTO PAPER (2026-07-28) ***
results_final.csv / sweep_final.csv (from cloud H200, leave-one-corpus-out,
SOURCE_EPOCHS=6, BATCH=64, 3 seeds x {in_the_wild, arabic}). This SUPERSEDES the
12-epoch run as the paper's headline numbers (main.tex fully updated).
TRANSDUCTIVE (mean+-std, 3 seeds):
  ITW:    source 15.86+-2.84/.922 | Tent 43.65+-9.41/.564 | ours 13.17+-1.63/.945
          (17.0% relative, 3/3 seeds improved, paired mean -2.69)
  Arabic: source 28.93+-0.92/.784 | Tent 43.82+-7.18/.579 | ours 27.18+-0.97/.805
          (6.0% relative, 3/3 seeds improved, paired mean -1.75)
INDUCTIVE: ITW 15.72->14.27 (3/3, -1.46) | Arabic 28.53->27.85 (2/3, -0.68, weaker)
ABLATION (mechanism finding, now the paper's nicest result): on ITW (smaller gap,
src AUC .922) self-training ALONE gets most of the gain (15.86->14.62); on Arabic
(larger gap, src AUC .784) self-training alone does NOTHING (28.93->28.89, 0/3
improved) and consistency is what rescues it (->27.18). I.e. consistency isn't
just an add-on, it's what keeps adaptation stable as the shift grows.
Tent collapses on BOTH targets, ALL 6 folds (EER>43%, AUC .56-.58) -- the most
robust finding in the whole project.
CAVEAT (now written into the paper's Discussion): ITW source EER has high seed
variance (13.1-18.8%, std 2.84) under this batch=64 recipe -- NOT explained by
epoch count (retracted the overtraining theory: 6ep was WORSE than 12ep at same
batch, so epochs isn't the driver of the local-vs-cloud gap; batch size 64 vs
local's 16, same unscaled LR 1e-4, is the more likely cause -- UNRESOLVED, flagged
as future work, not blocking since ordering ours<source<<Tent holds in every seed
regardless).
SWEEP findings (noisy since RNG isn't reseeded between sweep points -- trends
trustworthy, absolute values less so): q=0.4>0.3 default, epochs=8>4 default,
both showed clean monotonic improvement -- CURRENT DEFAULTS (q=0.3,epochs=4) ARE
NOT OPTIMAL, room to improve if more compute time available later.

fig_score_dist.png UPDATED (2026-07-29): got the real file from
~/Downloads/fig_score_dist(1).png, copied into project root, matches the caption
exactly (source AUC.899/acc76%/EER18.8%, ours AUC.934/acc85%/EER14.6%, seed0/ITW).

LOCAL EXTENSIONS (2026-07-29): user is running MORE cloud (baselines/languages)
in a SEPARATE chat; THIS chat does local-only analysis using cloud_bundle/ (the
zip user sent: all 6 leave-one-corpus-out ckpts + results.csv + sweep.csv +
exact source/target pool CSVs for seed0, paths 100% resolve locally).
- embedding_analysis.py: t-SNE + linear domain-origin probe on frozen embeddings,
  source vs ITW target, before/after adaptation (500+500 clips, real ckpt).
  GOTCHA: adapt() batch=64 (cloud value) OOM-killed (exit 137) on the local 10GB
  3080 (desktop compositor already holds ~937MB) -- dropped to bs=16, worked
  (peak ~3.6GB). RESULT: domain probe 80.3%+-1.9% -> 78.3%+-2.7% (NOT a strong
  domain-invariance effect, within 1 std). BUT the t-SNE itself is a genuinely
  good finding: source/target embeddings intermix WITHIN each real/fake cluster
  in BOTH panels (before AND after) -- i.e. the representation is already
  organised by CLASS not domain, which is the mechanistic reason ranking
  transfers cross-corpus. Wrote this up honestly (not oversold as
  domain-invariance) as new main.tex subsection "Embedding geometry: class
  organises the space, not domain" + Fig fig_embedding_geometry.png, and used it
  to give a post-hoc explanation for why GRL/domain-adversarial (the failed
  DA-RAD approach) had little useful signal to act on. main.tex re-validated
  (balanced envs/braces).
- sweep_reseeded.py: WRITTEN but not yet run. Fixes the RNG-continuity bug in the
  cloud's sweep.csv (each config there shared one unbroken RNG stream so
  adjacent points' absolute EER isn't trustworthy, only the monotonic trend is).
  Reseeds torch.manual_seed(0) fresh before each config, bs=16 (learned from the
  OOM above). NEXT: run this locally.
- Still on the local-extension menu (not started): reliability/pair-purity
  diagnostics on the REAL final ckpts (extends the repair_tta.py pilot which
  used a throwaway ckpt), attention/saliency example-clip visualization.

RESEEDED SWEEP CORRECTS A WRONG CONCLUSION (2026-07-29, sweep_reseeded.csv, local):
the cloud sweep did NOT reseed RNG between points, making its lambda column
misleading. Properly reseeded (ITW source 18.73):
  q:      0.1->14.87  0.2->14.70  0.3->14.25(best)  0.4->14.47
  epochs: 2->15.38    4->14.25    8->13.03(best)
  lambda: 0.0->15.75  0.3->14.25  1.0->13.98
  arabic: q 0.1->25.13  0.2->23.05(best)  0.3->24.74
=> lambda=0 (no consistency) is 1.5 EER WORSE on ITW, not better as the cloud
sweep suggested. This CONFIRMS the consistency term matters on ITW too, not just
Arabic. q=0.3 optimal for ITW, q=0.2 for Arabic; epochs=8 > 4. USE THE RESEEDED
NUMBERS in the paper, not the cloud sweep.

TWO NEW CLOUD NOTEBOOKS BUILT + SMOKE-TESTED (2026-07-29):
1. extended_pipeline.ipynb -- full leave-one-corpus-out over 4 EER targets
   (asvspoof2019, dataset2, in_the_wild, arabic) x 3 seeds, MLAAD (38 langs,
   fake-only) added to every source pool for language/generator diversity, plus
   3 new comparison points: bn_only (AdaBN-style, no gradient), oracle
   (supervised on target = upper bound), RawNet2Lite (from scratch, no SSL).
   batch 32 / lr 2e-4 (the batch64/lr1e-4 combo was undertrained).
   PILOT RESULT (seed0/ITW, on cloud): source 14.15 / tent 27.07(collapse) /
   bn_only 14.15 / st_only 11.83 / ours 11.17 / oracle 4.65 / RawNet2Lite 37.27.
   inductive 14.43->12.67. MLAAD held-out-language fake-recall 95.55%.
   ~25 min/fold, peak 10.9GB. NOTE: bn_only == source EXACTLY, because the model
   uses LayerNorm (no running stats to recompute) -- a real finding for the paper,
   showing normalisation-statistics adaptation alone does nothing here.
2. analysis_pipeline.ipynb -- run AFTER the extended grid (reuses ckpt_ext/).
   A: proxy A-distance d_A=2(2acc-1) source-vs-target in embedding space, BEFORE
      and AFTER TTA, per target + correlation with the EER gain (Ben-David).
   B: DANN baseline (GRL on source-vs-target discriminator, lambda ramp), 3 seeds
      -- so the paper can answer "why not DANN?" with numbers. NOTE we already
      have 2 prior DANN/GRL failures, expect it to lose; that IS the result.
   C: per-language MLAAD fake-recall, all 38 langs, split seen/unseen in training.
   D: t-SNE source vs target embeddings before/after adaptation.
   Writes results_divergence.csv / results_dann.csv / results_perlang.csv.
BUGS FOUND+FIXED while smoke-testing these (all would have wasted cloud hours):
 - mlaad_pool/mlaad_heldout computed BEFORE pool.reset_index -> stale indices ->
   CUDA index-out-of-bounds in the first fit().
 - RawNet2Lite block-3 stride 2 vs Identity shortcut -> shape mismatch.
 - record() appended different column sets to ONE csv under the first header ->
   silently misaligned columns. Now one csv per part.
 - pandas 3: pd.to_numeric(errors="ignore") removed.
 - pandas 3: groupby().apply() DROPS the grouping column -- this silently emptied
   the per-language lookup (0/38 languages scored). Use explicit loops instead.
   (Same class of bug bit cap_per_class earlier -- never use groupby().apply()
   here.)

sweep_reseeded.py RUN + INSERTED INTO PAPER (2026-07-29). Fixed the RNG-continuity
bug (each config now gets fresh torch.manual_seed(0) before adapting). Full
results (seed0, sweep_reseeded.csv), both targets vs default q=0.3/E=4/lam=0.3:
  ITW:    source 18.73/.900 | q:.1=14.87 .2=14.70 [.3=14.25 default] .4=14.47
          | E: 2=15.38 [4=14.25] 8=13.03(best) | lam: 0=15.75 [.3=14.25] 1.0=13.98
  Arabic: source 28.56/.781 | q:.1=25.14 .2=23.05(best) [.3=24.74 default] .4=24.51
          | E: 2=26.99 [4=24.74] 8=23.12(best) | lam: 0=25.46 [.3=24.74 best] 1.0=25.28
CLEAN FINDINGS (trustworthy now): (1) MORE EPOCHS ALWAYS HELPS on both targets
(E=8 beats E=4 by 1.2-1.6 EER pts) -- our E=4 default was a conservative,
compute-matched choice, not tuned optimum -- flagged as future-work headroom.
(2) lam=0 (no consistency) is the WORST config on both targets -- confirms the
ablation finding with a second, independent sweep-based check. (3) optimal q is
TARGET-DEPENDENT (0.3 for ITW, 0.2 for Arabic/bigger-shift) -- kept one fixed
q=0.3 for the whole paper for fair cross-target comparison, noted this in text.
Written into main.tex as new subsection "Hyperparameter sensitivity" (Sec V-C,
after Ablation) + Table tab:sweep. main.tex re-validated (balanced envs/braces,
xcolor loaded for \rowcolor/\colorbox).

GIT: 25 files staged (not committed) 2026-07-29 -- main.tex, all new
scripts/figures/results, plus analysis_pipeline.py/ipynb + extended_pipeline.py/
ipynb (from the user's PARALLEL cloud-extension chat, writing to the same repo).
cloud_bundle/ (1.2GB, the unzipped cloud checkpoints) excluded via new
.gitignore rule -- local scratch only, not repo content.

REMAINING FOR PAPER: (1) main.tex needs one Overleaf compile to sanity check
(structurally validated locally at every edit, but no local LaTeX compiler) --
upload main.tex + fig_score_dist.png + fig_embedding_geometry.png together.
(2) reliability/pair-purity diagnostics on real ckpts + saliency figures, if
more local-extension time remains. (3) whatever the parallel cloud chat produces
(baselines/more languages/more comparisons) still needs merging into main.tex
once that session reports back.

*** OPEN QUESTION THAT NOW DOMINATES EVERYTHING *** (RESOLVED — see above)
Does TTA still beat source-only once the source model is properly trained? The
earlier 9.42->6.41 win was on a weaker/less-augmented source. Tonight's multi-seed
overnight_pipeline grid must answer this with error bars. If TTA does NOT beat
source-only significantly, the paper CANNOT claim a TTA method contribution and
should pivot to the analysis/benchmark framing (cross-corpus shortcut is real;
what actually helps is fine-tuning + augmentation + generator diversity; popular
TTA/DG add-ons do not). Do not write the TTA claim until the seeds are in.

STRATEGIC READ (updated): DA-RAD (train-time DG) is dead, but the TTA pivot is a
live, promising method-paper contribution. Was: the method-paper bet is NOT supported. The publishable, TRUE story is the analysis framing:
cross-corpus shortcut is real; frozen-SSL + naive DG losses fail/hurt; simply
fine-tuning the SSL encoder closes most of the gap (31%→9%); DG add-ons don't help.
Recommend pivoting to that framing before spending the offered bigger-GPU budget.
Caveat: single-seed, reduced (9.7k, 4 epochs) — not proof DA-RAD can NEVER work,
but no signal to justify large compute yet.

Protocol manifests in protocols/. REMAINING (tasks #8,#9, the multi-GPU-hour part):
port baselines RawNet2/AASIST/RawGAT-ST/ASDG + pooled + SSL+RawBoost; build Protocol
B leave-one-corpus-out manifests + run full DA-RAD vs baselines grid, 3 seeds,
bootstrap CIs, ablations; explainability (Grad-CAM/IG, t-SNE/UMAP, domain-probe);
rewrite paper. See [[cross-dataset-results]].
