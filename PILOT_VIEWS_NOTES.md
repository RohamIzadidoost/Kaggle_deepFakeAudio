# Pilot: do acoustic views carry better ranking information?

Run 2026-09-17, before building any relative-supervision adaptation. The
question is the one that decides whether that direction is worth the
complexity: if a frozen multi-view teacher already delivers the improvement,
adaptation has not earned its place.

Scripts: `pilot_views.py` (score N clips under K views), `pilot_transforms.py`
(score one clip under each transformation in isolation), `analyse_views.py`.
Labels are used only for diagnosis; nothing in a method would be allowed them.

## 1. The first view family was unusable, and that was informative

Views built as *offset crop + gain + additive noise*, all at once, 10,000
In-the-Wild clips, WF checkpoint:

| | EER | AUC |
|---|---|---|
| standard leading crop | 4.85% | 0.9859 |
| the seven perturbed views | 8.73–9.13% | 0.965–0.967 |
| **mean over 8 views** | **7.22%** | 0.9768 |

Averaging made ranking *worse* by 48.9% relative. Bundling three
transformations meant this could not say which was at fault, so each was
measured alone (`pilot_transforms.py`, same 10,000 clips):

| transformation | EER | rel. vs standard |
|---|---|---|
| standard crop | 4.906% | — |
| crop @25% / @50% / @100% | 4.744 / 4.901 / 4.823% | −3.3% / −0.1% / −1.7% |
| gain only, U(0.7,1.3) | 4.891% | −0.3% |
| noise only, sigma=0.001 | 6.490% | **+32.3%** |
| noise only, sigma=0.005 | 8.870% | **+80.8%** |
| gain + noise (what the existing TTA perturbs with) | 9.032% | **+84.1%** |

**Temporal crops and gain are free; additive noise destroys the evidence.**

This has a consequence for the existing method, not just the new one. Its
consistency term enforces agreement between a clean clip and one perturbed by
exactly this gain+noise, i.e. against a view whose own ranking is 84% worse.
That is a mechanism for the manuscript's otherwise puzzling finding that
removing consistency changes nothing (3.98% vs 4.06% EER).

## 2. With crop-only views, the teacher does gain — modestly

Eight crops, no noise, same 10,000 clips:

- every view on its own: 4.74–4.97% EER, AUC ~0.9856. All preserve evidence.
- single standard view: 4.838% EER / 0.9858 AUC
- **mean over 8 views: 4.587% EER / 0.9868 AUC, −5.2% relative**
- paired stratified bootstrap, 2,000 replicates: **−0.250 EER points,
  95% CI [−0.461, −0.005]**. Excludes zero, but only just.
- monotone in K: −0.5% at 3 views, −1.8% at 4, −2.9% at 6, −5.2% at 8. Not
  saturated at 8.

## 3. Cross-view agreement is a sharp reliability signal

Over label-discordant pairs, with crop-only views:

| | share of pairs | ordering accuracy |
|---|---|---|
| all 8 views agree | 98.4% | **99.22%** |
| views split | 1.6% | **64.95%** |
| single view, for reference | — | 98.58% |

Agreement separates reliable from unreliable orderings almost perfectly, which
is what a pairwise weight would need. The gain over single-view supervision is
real but small in absolute terms: +0.64 points of pair precision for discarding
1.6% of pairs.

## 4. What this sets as the bar

The comparison that matters is no longer against the frozen source. It is
against the **frozen multi-view teacher at 4.587% EER**, which costs only extra
inference. A pairwise adaptation has to beat that, not 4.838%.

Open, in order:
1. Confirm the teacher gain on the full In-the-Wild pool and on a second corpus
   and checkpoint; the CI above is too wide to build on
   (`run_pilot_confirm.sh`).
2. If confirmed, test whether pairwise adaptation from unanimous-view
   preferences beats the teacher, and separately whether the loss change alone
   (single-view preferences) does anything.
3. Views here vary only the temporal window. Whether a richer view family
   (codec, reverberation, band-limiting) preserves evidence is unmeasured, and
   the transformation table above is the right way to screen each one before it
   enters a view set.

## 5. Confirmation on the full pool and a second corpus

Crop-only views, eight of them, `run_pilot_confirm.sh`:

| | single view | 8-crop mean | relative | paired bootstrap 95% CI |
|---|---|---|---|---|
| In-the-Wild, WF, all 31,779 | 4.856% | **4.634%** | −4.6% | [−0.361, −0.110] pts |
| ASVspoof2021-DF, DF-2, 30,000 | 4.195% | **3.505%** | **−16.4%** | [−1.181, −0.118] pts |

Both exclude zero; the DF interval is wide because a 30,000-clip sample of a
97.15%-spoof partition holds only ~855 bona fide clips. Agreement behaves the
same on both: unanimous pairs 98.4% / 96.6% of all, ordered correctly 99.24% /
99.97%, against 98.58% / 99.30% for a single view.

## 6. RETRACTED: the float16 buffer was not the cause

An earlier version of this section claimed the `float16` decode buffer cost
4.0% relative EER. That was wrong. Re-running the pipeline with `PUBA_FP32=1`
gives 5.058% against float16's 5.060% -- no effect. The float16 re-score does
reproduce the cached scores bit-exactly, so the pipeline itself is accounted
for, but precision was not what separated it from the pilot's view 0.

## 7. What actually separated them: scores depend on the batch

The pilot scored a flattened clips-by-views tensor, so a batch held 4 clips x 8
views; the pipeline scores 32 distinct clips. Those give different scores for
the same waveform. Localised with forward hooks, the divergence appears at the
very first convolution, meaning the *input* differs, and the cause is in
torchaudio:

```python
# torchaudio _Wav2Vec2Model.forward
if self.normalize_waveform:
    waveforms = nn.functional.layer_norm(waveforms, waveforms.shape)
```

`waveforms.shape` is `(batch, time)`, so the mean and variance are taken over
batch *and* time together: every clip is normalised by its neighbours'
loudness. Upstream fairseq applies the same normalisation per utterance, before
batching, so the two agree only at batch size one. Measured on In-the-Wild, one
clip scores 0.2029 alone, 0.2031 in a batch of 32 copies of itself, and 0.3449
in a batch of 32 distinct clips; across 4,000 clips the mean shift is 0.034 and
the maximum 0.73. It is not numerical: it persists with autocast disabled.

Normalising per utterance before the call removes it. Batch-of-1 against
batch-of-32 goes from a 0.606 maximum difference to 0.025 under bf16 autocast
and **0.00095** in float32, so what remains is only autocast arithmetic.
Wired in as `PUBA_PERSAMPLE_NORM=1`, deliberately opt-in: every cached score
array and every audited number came from the batched path, and changing it
silently would make those artefacts mean something else.

### What this affects

- Any score from the `ssl_aasist` path depends on manifest order, batch size,
  and pool composition. The manuscript's arms all share a batching, so its
  *comparisons* are consistent, but its numbers are not reproducible under a
  different batch size or pool.
- The prevalence intervention changes pool composition, therefore batch
  composition, therefore scores, independently of adaptation. That confound has
  to be quantified before the prevalence results can be trusted at face value.
- The pilot's multi-view result is suspect for the same reason: view 0 and the
  view mean were scored under different batch statistics, so the -4.6% teacher
  gain may be an artefact. It is being re-measured under the fix.

## 8. Where that leaves the proposed direction

The honest comparison table on native In-the-Wild, WF:

| | EER | needs target pool? | needs training? |
|---|---|---|---|
| published source (fp16) | 5.060% | no | no |
| fp32 input | 4.856% | no | no |
| fp32 + 8-crop teacher | 4.634% | no | no |
| tail adaptation, q=0.3 (fp16) | 4.55% | yes | yes |

A free teacher gets most of the way to what the adaptation achieves. The
adaptation row is not yet measured under matched precision, which
`run_fp32_match.sh` fixes; until it is, the comparison is not fair to either
side. If the matched adaptation row lands near 4.63%, the existing method's
ranking benefit is entirely reproducible by test-time augmentation, and the
case for relative-supervision adaptation has to rest on beating that teacher,
not on beating the source.
