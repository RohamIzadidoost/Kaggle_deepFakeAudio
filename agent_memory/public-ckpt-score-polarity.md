---
name: public-ckpt-score-polarity
description: "Public anti-spoofing checkpoints use [spoof=0, bonafide=1] — the opposite of this repo — and getting it wrong yields 100-x EER silently; test polarity by AUC, never by class means"
metadata:
  node_type: memory
  type: reference
  originSessionId: 89f368c0-53d3-4c15-b46c-9f9d2cf6633a
  modified: 2026-08-26T09:41:26.917Z
---

This repo is `LABEL_TO_IDX = [real=0, fake=1]`, so `P(fake) = softmax(logits,1)[:,1]`.
**Public audio-anti-spoofing checkpoints overwhelmingly use the opposite**,
`[spoof=0, bonafide=1]`, making column 1 the *real* score. Verified for both
`ash56/ssl-aasist` (its own inference snippet scores `batch_out[:,1]` as the
bonafide score, and its `calculate_EER` treats bonafide as the target class) and
`DeepFense/ASV19_*` (`config.yaml`: `label_map: bonafide: 1, spoof: 0`).

Measured on 120 In-the-Wild clips with `ash56/ssl-aasist`:
`col0` → EER 5.00% / AUC 0.9861; `col1` → EER 95.00% / AUC 0.0139.

**Why:** the flip raises no error — it returns exactly `100 − x` EER and
`1 − AUC`, so a wrong-polarity run looks like a plausible failed experiment
rather than a bug.

**Test polarity by AUC, not by comparing class means (2026-08-26).** The
original `--mode signcheck` decided polarity by `mean(score|fake) >
mean(score|real)`. That is a *calibration* test, and calibration is exactly what
does not transfer across corpora. Extending the arm to new targets, it failed
both DeepFense checkpoints on Arabic — mean P(fake) 0.999 on reals vs 0.990 on
fakes — while AUC was 0.736, i.e. ranking worked fine and the model was merely
saturated. Deciding on the means would have discarded a real cross-corpus result
as a porting bug. A flip inverts ranking (AUC → 1 − AUC), so AUC is the correct
discriminator. Note the third outcome: on a corpus where the model sits at chance
(DeepFense on dataset2, AUC 0.47 vs 0.53), *no* test on that corpus can establish
polarity — the gate must say "inconclusive, use a corpus where it ranks" rather
than pass or fail. `SIGNCHECK_MARGIN = 0.10` sets that band.

**How to apply:** never take a foreign checkpoint's column order on faith. Run
`public_ckpt_tta.py --mode signcheck` on a corpus where the model actually ranks,
and read the AUC line, not the means. Also note that the pseudo-label targets in
the TTA CE loss must be mapped into the checkpoint's column order, not ours.

Related: [[public-ckpt-tta-phase]], [[repo-hazards]], [[validate-before-gpu]].
