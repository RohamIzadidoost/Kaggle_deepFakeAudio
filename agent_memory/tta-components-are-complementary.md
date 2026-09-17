---
name: tta-components-are-complementary
description: The full ablation inverts across checkpoints — neither self-training nor consistency alone is sufficient; the two-component structure is load-bearing
metadata: 
  node_type: memory
  type: project
  originSessionId: 89f368c0-53d3-4c15-b46c-9f9d2cf6633a
  modified: 2026-08-18T08:09:03.378Z
---

Full 2x2 ablation on two public checkpoints, In-the-Wild, 31,779 clips, seed 0
(2026-08-18). Deltas are EER vs that model's own source.

| | ash56 (src 5.04%, AUC 0.9856) | DeepFense s42 (src 16.41%, AUC 0.9178) |
|---|---|---|
| self-training only | −0.10 (AUC +0.0006) | **−4.42** (AUC +0.0328) |
| consistency only | −0.68 (AUC +0.0019) | **+12.24** (AUC **−0.1240**) |
| both (`ours`) | **−0.96** (AUC +0.0044) | **−6.31** (AUC +0.0433) |

**The decomposition inverts.** On ash56 consistency does most of the work and
self-training almost none. On DeepFense the opposite: self-training carries it
and **consistency alone is actively destructive** (+12.24 EER, AUC collapses to
0.79 — the same failure mode as Tent, at lower magnitude).

**Why:** on both models the combination beats either component alone, and on
ash56 it beats their *sum* (−0.96 vs −0.78). Self-training's pseudo-labels
anchor the objective; without that anchor the consistency term is free to drift
toward a degenerate solution.

**How to apply:** do not claim "consistency carries ~89% of the gain" — that was
an ash56-only inference and it reverses on the other checkpoint. The supported
claim is that **neither component is sufficient and the pairing is necessary**,
which is a stronger justification for the method's design than either component
being dominant. Any single-checkpoint ablation of this method should be assumed
non-transferable until repeated on a second model.

**This contradicts nothing in the paper.** `main.tex`'s `tab:gaindecomp` measures
Δ₂ = ST-only → ours, i.e. consistency as a *marginal addition* to self-training;
it never measures consistency **alone**. `cons_only` is a new cell, and it
supports the existing two-component argument rather than undermining it.

One open question worth a look: `main.tex:530-533` attributes consistency's
failure on LibriSpeech-TTS to weak ranking (source AUC 0.727), but consistency
alone was destructive here at source AUC **0.918**. Different quantities, so not a
contradiction — but "low source AUC" may not be the whole mechanism.

See [[tent-does-not-always-collapse]], [[public-ckpt-tta-phase]].
