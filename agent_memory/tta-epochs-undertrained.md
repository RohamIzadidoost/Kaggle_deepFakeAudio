---
name: tta-epochs-undertrained
description: "TTA's E=4 is an epoch count, so the update budget silently scales with pool size — E=21 on a 6k pool beats the full-pool result"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0735564f-c5f1-4ff5-9780-4f382309a388
  modified: 2026-08-27T02:26:46.380Z
---

The published TTA config uses `TTA_EPOCHS = 4`. Because that is an *epoch*
count, the number of gradient steps is a function of pool size — 4 epochs over
In-the-Wild's full 31,779 clips is 5.3x the updates of 4 epochs over a
6,000-clip pool. Every pool in the ICASSP main table is 4,447–6,000 clips.

Measured 2026-08-27 on four third-party checkpoints, In-the-Wild at natural
prevalence, 6,000 clips, E=21 (the matched update count). It does not merely
recover the full-pool gain, it **exceeds** it:

| checkpoint | FULL 31,779 E=4 | NAT 6,000 E=4 | NAT 6,000 E=21 |
|---|---|---|---|
| deepfense_s42 | +6.31 | +1.64 | **+7.68** (14.44 → 6.76) |
| ssl_aasist_wavefake | +0.96 | +0.42 | **+1.47** |

Boundary-tested on badly-ranked pools: **no collapse**. On a near-chance pool
(deepfense_s42 / dataset2, AUC 0.585) adaptation is simply inert, AUC 0.585 →
0.584. Where ranking exists it pays hugely (deepfense_s42 / arabic: +8.07 EER,
AUC 0.734 → 0.836). Cost ~43 min/cell vs ~8 at E=4.

**Why it matters:** the paper's published gains are plausibly understated across
the board, and this was invisible because pool sizes are similar within the main
table — only comparing against the 31,779-clip ITW arm exposed it.

**How to apply:** frame as "**E should be a fixed number of gradient steps, not
a fixed number of epochs**" — that is label-free and avoids the
how-did-you-tune-without-target-labels objection that tuning E would invite. Not
yet a protocol change: one seed per cell, and E was set to match an update
count, not swept.

Related: [[auc-precondition-falsified]], [[public-ckpt-tta-phase]],
[[adaptive-tta-phase]].
