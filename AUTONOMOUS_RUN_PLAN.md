# Autonomous GPU run, 2026-08-27 -> 08-31

Mandate: ~90 GPU-hours, no supervision until Monday. Goal is defensible ICASSP
results, enough that the remaining work is writing rather than computing.

Live status:

```bash
python orchestrator.py --status        # per-job progress
tail -f queue_master.log               # what is running now
```

## What the run has to fix

Three weaknesses were left open by the 2026-08-26 overnight run
(`PUBLIC_CKPT_MULTI_NOTES.md`):

1. **Too few model families.** The four "third-party" checkpoints were really
   two: three DeepFense checkpoints differ only by training seed. A claim about
   "detectors in general" cannot rest on two.
2. **Too few corpora.** Five, and the retired AUC precondition needed corpora
   rather than seeds.
3. **Depth.** One seed per cell in the E experiments, and E=21 was never swept,
   so nothing could be tested for significance and the curve shape was unknown.
4. **Never tested on our own model.** Every E result came from other people's
   checkpoints. Only our model's numbers appear in the paper's main table.

## What was added before launching

**Four new model families (9 checkpoints, 5 families).** Weights and config only
-- no `*.py` was downloaded, so nothing fetched is executed.

| checkpoint | architecture | trained on | provenance |
|---|---|---|---|
| `ssl_aasist_wavefake` | XLS-R + AASIST | WaveFake/LJSpeech | documented |
| `deepfense_*_s{2,42,240}` | XLS-R + AASIST | ASVspoof2019 | documented, 3 siblings |
| `hf_xlsr_gustking` | w2v2-XLSR + seq-cls | undocumented | partial |
| `hf_xlsr_stafford` | w2v2-XLSR + seq-cls | commercial TTS vendors | documented (sibling of gustking) |
| `hf_w2v2_mothecreator` | w2v2-base + seq-cls | undocumented | **unknown** |
| `hf_w2v2_bisher` | w2v2-base + seq-cls | undocumented | **unknown** |
| `hf_ast_asv19` | **AST** (spectrogram transformer) | ASVspoof2019 | documented |

`provenance` is load-bearing: a cross-corpus claim requires knowing the model
never saw the target. Rows from `unknown`-provenance checkpoints are a separate
arm and must not carry a cross-corpus headline.

AST matters disproportionately -- it consumes a **mel spectrogram, not a
waveform**. If our parameter-selection rule (top-4 blocks' LayerNorms + head)
works there too, the method is architecture-agnostic in a way four
XLS-R+AASIST variants could never demonstrate.

**Two new corpora, both already on disk and unused:**

* `asvspoof2021la` -- 163,114 spoof / 18,452 bonafide, 67 speakers.
* `asvspoof2021pa` -- **replay** attacks: a real voice played through a speaker
  and re-recorded. A different detection problem from synthetic speech, so every
  model here should sit near chance. Deliberately included as the extreme-OOD
  arm: it is where we find out whether adaptation is *inert* or *harmful* when
  there is no ranking to exploit. Never pooled with the synthetic-speech targets.

Seven corpora total, ten seeds of pools built for each.

## Bugs this setup phase caught

* **OOM from a GPU-resident cache.** `adaptive_pipeline.py` was written for a
  35 GB cloud slice and pins the whole 33,597-clip waveform cache on the GPU
  (4.30 GB). On a 10 GB card every eval-only arm ran and every adaptation arm
  died, which reads as "the experiment failed" rather than "it never fit".
  Fixed with `CACHE_ON_CPU=1` (pinned host memory, batches moved per step) --
  the same solution `public_ckpt_tta.build_cache` already documents.
  `verify_reduction.py` still reports **bitwise identical**, so the fix changed
  where tensors live and not what is computed.
* **Head selection silently missed on foreign trees.** `set_tta_params` resolved
  heads with `getattr(model, name)`, which finds AASIST's root-level `out_layer`
  but *not* the HF wrappers' nested `model.classifier`. Left unfixed it would
  have adapted LayerNorms only on five of nine checkpoints -- a different method,
  reported as the same one. Now matched by module-name suffix.
* **GPU contention.** An interactive sign-check launched beside the E-sweep
  caused the OOM above. All GPU work now goes through `orchestrator.py`, which
  owns the card and runs strictly one job at a time.

## Gates passed before any result is believed

* `ASTFbank` reproduces `ASTFeatureExtractor` **exactly** (max|diff| = 0.00e+00
  at 2 s, 4 s and 11 s, covering both the pad and the truncate branches). A
  mismatched front-end would be indistinguishable from "AST does not transfer".
* Every new checkpoint scores strongly on the corpus it was *trained* on
  (AUC 0.90-0.9998). A model that cannot recognise its own training
  distribution has a broken port.
* Polarity confirmed empirically on In-the-Wild for all five, by AUC rather than
  class means. Source AUC there spans 0.62-0.94, which is the range needed to
  test the precondition at all.
* Every rebuilt pool is size-gated against `results_ext.csv`; the three new
  corpora state openly that their gate is by construction, with no prior run to
  match.

## Stage order, and why

Ordered so that losing the GPU at any point leaves the most valuable work done.

| stage | what | est |
|---|---|---|
| **B** | port gates for the 5 new checkpoints | 0.3 h |
| **A** | **our own model's E curve** (E=8/16/32/64, seeds 0-2) | 21.5 h |
| **C** | breadth: 9 checkpoints x their valid corpora, E=4, seed 0 | 9.4 h |
| **D** | depth: seeds 1-9 on a 15-cell core (5 families x 3 corpora) | 27.0 h |
| **E** | Tent + `st_only` on the new checkpoints | 10.4 h |
| **F** | the replay arm | 1.8 h |

Stage B first because it costs 20 minutes and decides whether C-F mean anything.
Stage A next because it is the only experiment that can change the paper's main
table.

Ten seeds are restricted to a core subset by necessity: ten seeds on every cell
is ~92 GPU-hours by itself. The split is principled -- pooled claims (the
calibration-deficit result) are strengthened by breadth, which stage C buys at
seed 0; only per-cell significance needs depth, and n=10 is the minimum at which
a paired Wilcoxon can reach p<0.05 at all (2/2^10; at n=5 the floor is 0.0625).

## Decision points

* **After stage A seed 0** -- the E curve shape. Plateau => "spend a fixed step
  budget" is a label-free protocol fix. Peak-then-decline => locating the peak
  needs target labels and the finding stays an observation. This determines
  whether a stage G (matched-E across the grid) is worth queueing.
* **After stage C** -- whether the calibration-deficit result holds across five
  model families and seven corpora, and whether the retired AUC precondition
  stays retired with ~50 cells instead of 43.
