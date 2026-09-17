---
name: repo-hazards
description: "Traps in this repo — a 45GB auto-download, eight copies of the TTA loop, and a resume guard that silently skips everything"
metadata: 
  node_type: memory
  type: project
  originSessionId: a7e7d7d9-af89-4792-b7a8-336e8fb579a1
  modified: 2026-08-10T03:59:07.431Z
---

Things that will bite you in `~/Desktop/deepfake_project`:

- **`extended_pipeline.py`'s data cell starts a 45 GB MLAAD download** on any
  machine where `data/mlaad` is missing. Hit this once and killed it at 1.4 GB.
  `adaptive_pipeline.py` asserts corpora are present instead of fetching.
- **Its resume guard keys on `(seed, target, setting)` only**, so re-running it
  against an existing `results_ext.csv` silently skips all 20 folds no matter
  what methods you added. Never add a method arm by editing that file — clone
  the scaffolding into a new driver, as the repo already does several times.
- **Roughly eight near-duplicate copies of the TTA `adapt()` loop exist**
  (`tta.py`, `repair_tta.py`, `overnight_pipeline.py`, `extended_pipeline.py`,
  `sweep_reseeded.py`, `sweep2_pipeline.py`, `protocol_a.py`, `analysis_*`).
  They differ in real ways (epochs, batch, whether q/lambda are parameters or
  globals). Always check which one produced the numbers you're comparing to.
- **The notebook-derived pipelines run a pip cell at import** that uninstalls
  librosa/numba and pins numpy. Guard it before importing such a file.
- Local hardware is an **RTX 3080 (10 GB)**; the published numbers came from an
  H200 MIG 1g.35gb slice. `ckpt_ext/` has seeds 0–2 only — the cloud run's
  seeds 3–4 checkpoints were never saved.

See [[adaptive-tta-phase]].
