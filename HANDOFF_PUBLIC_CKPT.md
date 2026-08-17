# Handoff: TTA on a published checkpoint (In-the-Wild)

Run this on the 3080 box (has `data/`, `ckpt_ext/`, `protocols/`, all result CSVs).

## Why this experiment exists

The paper's weakest point for ICASSP is not the effect size — it is that every
number compares our own source model against our own baselines on our own
leave-one-corpus-out protocol. The one protocol with published comparators
(Protocol A, ASVspoof2021-DF) is where the method *regresses* (26.33% → 42.46%
EER, `results_protocol_a.csv`), and that regression is already documented with
its mechanism in `PROJECT_LOG.md` §11 and `main.tex` Limitation 2.

A reviewer will write "no comparison to state of the art". The cheapest honest
answer is not to build a better source model — it is to show **our TTA improves
somebody else's published detector**, on a standard benchmark, with their
released weights. TTA costs no training, so this is plumbing plus one adaptation
run, not a new model.

Target claim: *"Applied to a public released detector without retraining, our
adaptation reduces In-the-Wild EER from X to Y."* That single sentence converts
the contribution from "our method on our protocol" to "a label-free wrapper that
improves the field's models."

## The experiment

1. Load a **public, released** audio-deepfake checkpoint trained on
   ASVspoof2019-LA (never on In-the-Wild).
2. Score In-the-Wild with it, unmodified → **source EER**. This number must
   reproduce, within reason, whatever the checkpoint's authors or a third party
   report for it on In-the-Wild.
3. Run our published TTA config on the unlabeled In-the-Wild pool → **adapted
   EER**.
4. Report both, plus the transductive/inductive pair, over seeds 0–2 minimum
   (10 seeds if it is cheap, to match the rest of the paper).

Labels are used only for scoring. Never for adaptation.

## Checkpoint choice

**First choice: Tak et al., `SSL_Anti-spoofing`** (wav2vec2/XLS-R front-end +
AASIST back-end, trained on ASVspoof2019-LA, released DF-track weights). It is
the canonical public SSL anti-spoofing checkpoint and its architecture is
LayerNorm-rich, so our parameter-selection rule transfers unchanged.

*Verify the repo URL, exact checkpoint filename, and any reported In-the-Wild
number from the actual source before citing anything.* Do not take the values in
this document on faith — `PROJECT_LOG.md` already flags an unverified citation as
the one failure mode that ends a paper regardless of merit.

**Known blocker.** That checkpoint builds its XLS-R front-end through `fairseq`.
This repo is pinned to `torch==2.13.0` / `torchaudio==2.11.0` (see
`requirements.txt`) and fairseq will not co-install against those. Two ways out,
in order of preference:

- **(a) Weight remap.** Load their `.pth`, remap the XLS-R encoder state-dict
  keys onto `torchaudio.pipelines.WAV2VEC2_XLSR_300M.get_model()`, keep their
  back-end module as-is. Mechanical, and it has a built-in correctness gate: if
  the remap is wrong, the reproduced source EER will be nowhere near the
  published one. **Do not proceed past a failed gate** — a silently broken port
  makes every adapted number meaningless.
- **(b) Separate venv.** Build a throwaway env with the old torch fairseq wants,
  score there, and pass scores back as a CSV. Uglier, but isolates the
  dependency mess from the pinned env. Acceptable if (a) stalls.

**Fallback: AASIST (clovaai)** — pure PyTorch, weights ship in the repo, no
fairseq. Cheaper to stand up, but it is not a transformer: it has no LayerNorm
stack for our rule to select, so you would have to adapt BatchNorm affine
params instead. That is a *different method*, not our method on a new model, and
it weakens the claim. Use it only if both fairseq routes fail, and state the
substitution explicitly in any write-up.

## Published config to apply — do not improvise

From `extended_pipeline.py` (this is the exact config behind every `ours` row in
`results_ext.csv`; `adapt()` at line 425, `set_tta_params()` at line 415):

```
Q            = 0.3        # confident top/bottom quantile -> pseudo-labels
LAMBDA_CONS  = 0.3        # consistency weight
TTA_LR       = 1e-4       # Adam
TTA_EPOCHS   = 4
N_FINETUNE   = 4          # top-4 transformer layers' LayerNorms
BATCH        = 32         # lower to 12-16 if the crop length grows to 4s
loss         = self-training (CE on confident pseudo-labels) + LAMBDA_CONS * MSE consistency
```

Trainable set: LayerNorm affine params of the top-4 transformer layers, plus the
classifier head modules. Everything else frozen.

**The anchor loss stays OFF.** `tta.py` still exposes `use_anchor`, and
`PROJECT_LOG.md` §4 records that it collapses the model to chance. The published
`ours` is self-training + consistency only.

## Traps, in the order they will bite

1. **Score column order — the silent killer.** Our code takes
   `softmax(logits,1)[:,1]` as P(fake) (`LABEL_TO_IDX` is `[real=0, fake=1]`).
   Public repos are not consistent here; several use `[spoof, bonafide]` so
   column 1 is *bonafide*. Get this backwards and EER comes out as `100 − x`
   with no error raised. **Check the sign before anything else**: score a handful
   of known-real and known-fake clips and confirm fakes get the higher score.
   The reproduced source EER in step 2 is the second line of defence — if it
   lands near `100 − published`, you have found the flip.
2. **Hardcoded module path.** `select_tta_params()` in `tta.py` walks
   `model.encoder.model.model.encoder.transformer.layers[-4:]`, which is our
   torchaudio wrapper's path and will not exist on a foreign model. Replace with
   a generic walk that collects `nn.LayerNorm` modules from the top-4
   transformer blocks of whatever module tree the checkpoint has. Assert the
   count of selected parameter tensors is non-zero — a silent empty selection
   makes TTA a no-op that looks like "adaptation didn't help".
3. **Crop length.** We use 3.0 s crops (`CROP_SEC` in `extended_pipeline.py:65`).
   Published SSL anti-spoofing checkpoints are usually trained at 4 s
   (64,600 samples). Match *their* training length, not ours, or the source EER
   will be unfairly bad and the whole comparison is worthless. Halve the batch
   to fit — the consistency term runs a second forward pass, so activation
   memory is roughly doubled.
4. **Forward signature.** `adapt()` expects `model(wav) -> (logits, emb)`. Most
   public models return logits alone, or `(hidden, logits)`. Wrap, don't edit
   `adapt()`.
5. **In-the-Wild must be genuinely unseen** by the checkpoint. Confirm from its
   training description that it is ASVspoof2019-LA only. If it saw ITW, this
   experiment says nothing.

## Gates before any number is believed

- Sign check on known-label clips (trap 1).
- Reproduced source EER is in the neighbourhood of the published value for that
  checkpoint on ITW. If it is not, stop and fix the port.
- Non-empty trainable-parameter set (trap 2), logged as a count.
- Adapted-vs-source compared on the **same clip set**, same crop length, same
  scoring code path.

This mirrors the `verify_reduction.py` discipline already in the repo: prove the
harness is unconfounded before trusting what it measures.

## What to report back

Append to a new `results_public_ckpt.csv` with columns matching
`results_ext.csv`'s style (`seed,target,method,setting,family,eer,auc,acc,n,minutes`),
using `family` = the checkpoint's name. Report:

- source EER/AUC (unmodified public checkpoint on ITW)
- ours EER/AUC, transductive
- ours EER/AUC, inductive (adapt/eval split disjoint) — the anti-memorization check
- st_only EER/AUC, so the gain decomposition survives on this model too
- the published reference EER you reproduced against, and its source

Also log wall-clock and the trainable-parameter count.

## Expected cost

In-the-Wild is ~31.8k clips. `logs/run_log_protocol_a.txt` shows ~235 clips/s
scoring on this hardware, so one full pass is ~2.3 min. Four adaptation epochs
(each a score pass plus a train pass) plus final scoring lands well under an
hour per seed. Three seeds is an evening. Memory is the only real constraint —
drop `BATCH` until the 4 s crops with a doubled forward pass fit.

## If it works

The main-results table gains a row that no reviewer can dismiss as
self-refereed, and `main.tex`'s "Not a state-of-the-art detector" limitation
gets to keep its honesty while the contribution stops depending on our own
baseline being the only comparator.

## If it does not

That is also publishable, and it sharpens the preconditions story: it would mean
the gain depends on properties of *our* source model's score distribution rather
than on the target corpus alone — which is exactly the mechanism
`PROJECT_LOG.md` §11 already suspects from the `tail_gap` / calibration finding
(r=+0.74, p=0.006, n=12). Report it either way; do not bury a null.
