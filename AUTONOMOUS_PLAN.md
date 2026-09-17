# Autonomous re-validation plan (2026-09-17 to 2026-09-20)

Written when the author left for three days. No human is available, so this
file records the plan, the decision rules, and the stop conditions, and is the
recovery point if the working context is lost.

## Why the whole evidence base is being re-run

torchaudio's `_Wav2Vec2Model.forward` normalises the waveform with
`layer_norm(waveforms, waveforms.shape)` on the *batched* tensor, so every clip
is normalised by its neighbours' loudness. Upstream fairseq normalises per
utterance before batching; the two agree only at batch size one. Consequences:

- Any score from the `ssl_aasist` path depends on manifest order, batch size
  and **pool composition**.
- The prevalence intervention changes pool composition, so it changes scores
  independently of adaptation. Measured from cached scores: between the
  pi=0.97 and pi=0.01 pools, shared clips shift by mean 0.019, with 4.8%
  moving more than 0.1. Small next to the paper's 30-point effects, but a
  reviewer who learns of it will ask for corrected numbers, and they would be
  right to.

`PUBA_PERSAMPLE_NORM=1` normalises per utterance and reduces batch dependence
from 0.606 to 0.00095 (float32). Every stage below runs with it set. The
normalisation mode is part of the resume key and the cached-score filename, so
corrected and legacy runs coexist and neither satisfies the other.

## Corrected-run configuration (settled 2026-09-17 15:30)

`PUBA_PERSAMPLE_NORM=1 PUBA_FP32=1`, with normalisation applied at decode time.

Getting here cost a false start worth recording. Applying the normalisation
inside the model OOMed every adaptation cell at 9.07 GiB while the source cells
passed at 4.3 GiB, and no isolated probe reproduced it: a faithful
forward+backward with the consistency term measured 5.00 GiB with and without
the patch. The bisect that settled it was a real pipeline cell run under legacy
normalisation, which completed at 6.6 GiB. Moving the normalisation to decode
time -- where upstream fairseq does it anyway, per utterance before batching --
removed the interaction and returned a real adaptation step to 4.99 GiB.

Float32 audio came with it: normalisation amplifies the waveform about
twelvefold, and this model is sensitive enough that storing the amplified
signal in the float16 buffer moves 3 of 512 clips by more than 0.01. On raw
audio the same comparison showed nothing (5.058 against 5.060% EER), so the
effect belongs to normalised input specifically.

Lesson for the remaining stages: probe results that disagree with the pipeline
are not evidence about the pipeline. Bisect against a real cell.

## Stages

Each stage is one long-running script. After each, commit and push, analyse,
and only then launch the next; a stage that dies part-way can be relaunched and
the resume guard will cost only the missing cells.

| stage | what | script | approx |
|---|---|---|---|
| A | In-the-Wild prevalence grid, 8 priors x 5 arms | `run_revalidate_itw.sh` | ~7 h |
| B | ASVspoof2021-DF prevalence grid, 4 priors x 4 arms | to write | ~6 h |
| C | DF anchor on the full available eval, 6 arms | to write | ~5 h |
| D | remaining checkpoints on native In-the-Wild | to write | ~2 h |
| E | three seeds on the cells the claims rest on | to write | ~3 h |
| F | rebuild audit, tables, figure; rewrite; verify every number | — | — |

## Decision rules

- **If the corrected grid reproduces the qualitative result** (accuracy moves
  far more than EER; Delta orders the post-adaptation threshold gap across
  strata), replace every number in the manuscript with its corrected value and
  add the normalisation bug as a methodological finding. That is the strongest
  position available: the result survives a correction that a reviewer would
  otherwise have raised.
- **If Delta no longer orders outcomes across strata**, the contamination claim
  goes. What remains is the ranking-versus-decision decomposition, which does
  not depend on Delta, plus the bug and the view-screening results. Rewrite
  around those rather than defending a dead claim.
- **If the ranking-versus-decision result also fails**, report that the earlier
  result was an artefact of batched normalisation. That is publishable as a
  negative methodological finding and is the honest outcome.
- **Never** report legacy and corrected numbers interchangeably. The corrected
  path is the one that matches upstream; legacy numbers stay only in the log.

## Independent results already established, not affected by the bug

These were measured within a single scoring pass and hold either way:

1. View screening: temporal crops and gain cost nothing (-3.3% to -0.1%
   relative EER); additive noise costs +80.8% at sigma=0.005 and +32.3% at
   sigma=0.001. The existing consistency term perturbs with exactly that
   gain+noise, so it demands agreement against a view whose own ranking is 84%
   worse -- a mechanism for its null ablation.
2. The frozen multi-view teacher, re-measured under the fix: 4.723% -> 4.435%
   EER on the full In-the-Wild pool, -6.1% relative, 95% CI [-0.403,-0.169].
   Against the published pipeline number of 5.060% the stack is -12.4%
   relative, with no target labels, no prior estimate and no adaptation.
3. Cross-view agreement is a reliability signal: unanimous pairs are ordered
   correctly 99.1% of the time against 60% for split pairs.

## Stop conditions

- Do not start a stage that cannot finish inside the remaining window.
- Do not push a manuscript whose numbers have not been re-verified against the
  audit outputs by `audit_icassp.py`.
- If a stage fails twice for the same reason, record it here and move on rather
  than looping.
