---
name: tent-does-not-always-collapse
description: "Tent's collapse is model-dependent — catastrophic on one public checkpoint, mildly helpful on another, but it degrades AUC on both"
metadata: 
  node_type: memory
  type: project
  originSessionId: 89f368c0-53d3-4c15-b46c-9f9d2cf6633a
  modified: 2026-08-18T08:08:54.677Z
---

Measured 2026-08-18 on two public checkpoints, In-the-Wild, all 31,779 clips,
seed 0, transductive. **An earlier version of this note said "Tent does not
collapse" — that was based on one checkpoint and was wrong as a general claim.**

| checkpoint | source EER/AUC | Tent EER/AUC |
|---|---|---|
| ash56 (WaveFake) | 5.04% / 0.9856 | 4.52% / **0.9790** |
| DeepFense s42 (ASV19) | 16.41% / 0.9178 | **49.80% / 0.5020** |

On DeepFense s42 Tent collapses to **49.80% EER at AUC 0.502** — pure chance,
almost exactly the "~49%" that `PROJECT_LOG.md` §5 reports. So the §5 claim
**reproduces on a third-party model**, but it is model-dependent: on ash56 Tent
instead gives a modest EER gain.

**The unifying signature is AUC, not EER.** Tent degraded the ranking on *both*
models — mildly on ash56 (−0.0066, masked by a recalibration gain that makes EER
look better) and totally on s42 (−0.4158). Our method improved AUC on both.

**`main.tex` does NOT need changing — verified 2026-08-18.** Line 69 already
scopes it correctly: Tent "collapses to chance on Arabic in every seed and is
highly volatile on In-the-Wild, including an unpredictable full collapse on an
otherwise-easy target (ASVspoof2019: four of ten seeds at or near chance)". These
results are an *instance* of that documented volatility — corroboration, not
falsification. Only `PROJECT_LOG.md:90`'s informal "Tent collapsed (49% EER)" is
unscoped. Two earlier versions of this note wrongly said the paper needed a
rewording; do not re-raise that.

The genuinely new part is the AUC invariant: Tent degraded the ranking on **both**
models, even on ash56 where EER improved. The paper does not state that. Add it as
a new observation, not a correction.

See [[public-ckpt-tta-phase]], [[tta-components-are-complementary]].
