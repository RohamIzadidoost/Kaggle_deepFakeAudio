# Our TTA on public checkpoints — port, gates, and run plan

Implements `HANDOFF_PUBLIC_CKPT.md`. Status: **ported and gated on CPU; GPU grid
armed and waiting** for the GPU to free up.

## What this answers

Every number in the paper so far compares our source model against our baselines
on our protocol. This runs the *unmodified published TTA config* on third-party
released weights, on In-the-Wild, so the contribution stops depending on our own
baseline being the only comparator.

## Checkpoints

Both are wav2vec2/XLS-R-300M + AASIST, and both build their front-end through
fairseq upstream — which will not co-install against this repo's pinned
torch 2.13 / torchaudio 2.11. Resolved by **route (a), weight remap**.

| key | source | trained on | ITW seen? |
|---|---|---|---|
| `ssl_aasist_wavefake` | [ash56/ssl-aasist](https://huggingface.co/ash56/ssl-aasist) (Garg et al. 2025, [arXiv:2502.05674](https://arxiv.org/abs/2502.05674)) | WaveFake / LJSpeech, HiFiGAN vocoder | no |
| `deepfense_w2v2_aasist_s{2,42,240}` | [DeepFense/ASV19_Wav2Vec2_AASIST_NoAug_Seed*](https://huggingface.co/DeepFense/ASV19_Wav2Vec2_AASIST_NoAug_Seed42) | ASVspoof2019 LA train | no |

Two *independent training corpora* is a deliberate upgrade over the handoff's
single-checkpoint plan: if the gain reproduces on both, it is not a quirk of one
model's score distribution.

**Note on the handoff's first choice.** It named Tak et al.'s `SSL_Anti-spoofing`
release. Its weights are on Google Drive; the two HF checkpoints above are the
same SSL-AASIST architecture, are programmatically downloadable, and one of them
(DeepFense) is trained on ASVspoof2019-LA as the handoff wanted. The port applies
unchanged to Tak's `.pth` if it is ever wanted.

## The port

`vendor_aasist.py` → `aasist_backend.py`. The AASIST back-end is kept **verbatim**
(pure PyTorch); only the front-end is swapped for
`torchaudio.pipelines.WAV2VEC2_XLSR_300M`, loaded through torchaudio's own
authoritative fairseq key mapping
(`torchaudio.models.wav2vec2.utils.import_fairseq._convert_state_dict`, which is
a pure state-dict function and needs no fairseq installed).

The surgery is textual and every anchor is asserted, so an upstream edit fails
loudly instead of silently vendoring something different.

## Gates (all passed on CPU, before any GPU time)

1. **Strict weight load.** 429 fairseq keys → 421 tensors (the 8 dropped are
   pretraining-only: `mask_emb`, `quantizer.*`, `project_q.*`, `final_proj.*`),
   loaded `strict=True`. Back-end: 245 keys, **0 missing, 0 unexpected**.
2. **Not silently the stock model.** All 421 front-end tensors differ from a
   fresh `WAV2VEC2_XLSR_300M` init — these are genuinely the fine-tuned weights.
3. **Score polarity (the silent killer).** On 120 known-label ITW clips:

   | column | EER | AUC |
   |---|---|---|
   | `col0` (declared `fake_col`) | **5.00%** | 0.9861 |
   | `col1` | 95.00% | 0.0139 |

   Exactly the `100 − x` flip the handoff warned about. Both public checkpoints
   use `[spoof=0, bonafide=1]`, the opposite of this repo's `[real=0, fake=1]`,
   so `P(fake)` is **column 0**. Confirmed from DeepFense's `config.yaml`
   (`label_map: bonafide: 1, spoof: 0`) and from ash56's own inference snippet
   (it scores `batch_out[:, 1]` as the bonafide score). Re-verified on GPU before
   the grid runs.
4. **Non-empty trainable set.** `tta.py`'s `select_tta_params` hardcodes our
   module path and would select **nothing** on a foreign tree — a silent no-op
   that reads as "adaptation didn't help". Replaced with a generic search for the
   largest `nn.ModuleList` of LayerNorm-bearing blocks; asserts non-zero. Finds
   24 blocks, selects the top 4 → **18 tensors / 16,706 params** (0.005% of
   315.9M), logged in every result row.
5. **End-to-end mechanism smoke** (64 clips, CPU): loss 0.0914 → 0.0212 over 4
   epochs, accuracy 81.25% → 90.62%, EER/AUC unchanged. The model demonstrably
   adapts and improves calibration without reordering a tiny well-separated pool.
   Per `PROJECT_LOG.md`, smoke validates mechanism, **not results**.

## Deliberate deviations, and why

* **Crop length 64,600 / 64,000 samples, not our 3 s.** Matched to each
  checkpoint's own training length (upstream pads by tile-repeat; we replicate
  that exactly). Using our 3 s convention would handicap the source model and
  make the comparison worthless.
* **BatchNorm frozen during adaptation.** The AASIST back-end contains
  `BatchNorm2d`. A plain `model.train()` would update BN running statistics on
  target audio — that is **AdaBN**, a different mechanism, and it would confound
  the measurement of ours. All BN modules are forced to `eval()`, so the only
  things that adapt are the top-4 LayerNorms and the classifier.
* **Head = `out_layer` only.** The closest analogue to our published
  "top LayerNorms + classifier head". Adapting the whole graph back-end would be
  a different, much larger method.
* **Fixed `q = 0.3`.** The handoff says do not improvise, so the adaptive-`q`
  work (`PROJECT_LOG.md` §11) is deliberately *not* used here. ITW is 63/37
  real/fake, mild enough that the symmetric split is not obviously wrong — but
  this is the known weak spot on skewed pools and is the first thing to try if
  the fixed-`q` result disappoints.

## Config — identical to every `ours` row in `results_ext.csv`

`Q=0.3, LAMBDA_CONS=0.3, TTA_LR=1e-4, TTA_EPOCHS=4, N_FINETUNE=4`, anchor **off**
(`PROJECT_LOG.md` §4: it collapses the model to chance). Verified equal to
`extended_pipeline.py` programmatically.

## Reference numbers — what can and cannot be claimed

Neither checkpoint has a published **single-model In-the-Wild EER**. Garg et al.
report **5.70% on ITW for an ensemble**; DeepFense publishes no model card (its
checkpoint records a 0.0017 best dev metric on ASVspoof2019).

So the handoff's "reproduce the published number" gate **cannot be run as
written**, and no claim of reproducing a published ITW figure should be made.
Correctness rests instead on gates 1–4, which are stronger evidence of a faithful
port than an EER coincidence would be. The full-pool source EER should land in a
plausible neighbourhood of 5.70%; a value near 50% or near `100 − x` means stop.

## Running it

`./run_public_ckpt.sh` waits for the other project's sweep (PID 50253) to exit
**and** for GPU memory to actually come back, then runs the grid in value order,
appending to `results_public_ckpt.csv` after every run. One failure does not stop
the rest, and it falls back 16 → 8 → 4 on OOM.

```bash
tail -f run_log_public_ckpt.txt
```

Order: sign checks → ash56 source (full ITW) → ash56 `ours` → DeepFense `ours`
→ inductive → `st_only` → **Tent** → seed variance.

Tent is the load-bearing contrast: `PROJECT_LOG.md` §5 records that naive entropy
minimisation collapses to ~chance here. If it collapses on a third-party
checkpoint while ours improves, the *structure* — not our source model — is doing
the work.

## If the gain does not reproduce

Report it. Per the handoff, a null sharpens the preconditions story: it would
mean the gain depends on properties of our own source model's score distribution
rather than on the target corpus alone, which is what the `tail_gap` /
calibration finding (r=+0.74, p=0.006, n=12) already suspects. Note that this
checkpoint's scores are strongly saturated (mean P(fake) on fakes ≈ 1.000), which
is exactly the regime §11 predicts MSE-consistency handles worst.
