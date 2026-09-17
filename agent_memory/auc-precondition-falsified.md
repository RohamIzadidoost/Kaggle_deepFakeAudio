---
name: auc-precondition-falsified
description: "The ICASSP paper's 'source AUC predicts adaptation gain' claim was falsified on 43 third-party cells; the replacement claim is calibration-deficit recovery"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0735564f-c5f1-4ff5-9780-4f382309a388
  modified: 2026-08-27T02:26:33.280Z
---

The ICASSP manuscript's precondition claim — adaptation pays where the source
model's ranking already transfers (r = +0.86, n = 4 corpora, one model, p = 0.14)
— **does not survive** extension to four third-party checkpoints (2026-08-27).

Across 43 (model, corpus, seed) cells: r = **−0.20** (p = 0.21) for absolute EER
gain, and flat on every headroom-controlled measure (relative reduction +0.11,
AUC change −0.19, acc@0.5 change +0.14). Two within-corpus slopes are
significantly *negative* — In-the-Wild r = −0.97 (p < 0.001), Arabic r = −0.83
(p = 0.001). Collapsing the three DeepFense checkpoints (they differ only by
training seed) gives the honest n = 7 over two model families: r = −0.53.

**The replacement claim, which is much stronger.** Define
`deficit = (100 − EER) − acc@0.5` — attainable balanced accuracy at the
EER-optimal threshold minus what the model delivers at its shipped 0.5
threshold. Adaptation drives it 11.84 → **1.47** points, reduced in 33/43 cells,
Wilcoxon **p = 5.9e-07**, and critically **corr(deficit_before, deficit_after)
= −0.01**: the endpoint does not depend on the starting point. This is the
paper's own "ranking transfers, threshold does not" thesis stated as a
measurable.

**Why:** the negative AUC correlation is a headroom law, not a ranking law —
better-ranking models arrive better calibrated, so they have less to fix.

**How to apply:** lead the paper on threshold repair, retire or heavily qualify
the AUC precondition. Do **not** report `corr(deficit, acc_gain) = +0.94` — it
is partly circular (both terms contain −src_acc); report the before/after
comparison instead.

Related: [[public-ckpt-tta-phase]], [[tent-does-not-always-collapse]],
[[tta-components-are-complementary]], [[validate-before-gpu]].
