---
name: public-ckpt-tta-phase
description: "Phase applying our TTA to third-party public checkpoints on In-the-Wild, to answer \"no comparison to SOTA\""
metadata: 
  node_type: memory
  type: project
  originSessionId: 89f368c0-53d3-4c15-b46c-9f9d2cf6633a
  modified: 2026-08-17T12:00:03.217Z
---

Started 2026-08-17. Implements `HANDOFF_PUBLIC_CKPT.md`: run our published TTA
config on **somebody else's released detector** so the contribution stops
depending on our own baseline being the only comparator.

Two public checkpoints ported, both wav2vec2/XLS-R-300M + AASIST, deliberately
from **two different training corpora** so a gain on both can't be a quirk of one
score distribution:
- `ash56/ssl-aasist` — WaveFake/LJSpeech HiFiGAN (NOT ASVspoof2019, despite the
  handoff assuming Tak et al.'s 19LA checkpoint)
- `DeepFense/ASV19_Wav2Vec2_AASIST_NoAug_Seed{2,42,240}` — ASVspoof2019 LA

**The fairseq blocker was solved by route (a).** `torchaudio.models.wav2vec2.
utils.import_fairseq._convert_state_dict` is a *pure state-dict function* — it
needs no fairseq installed. Feed it the `ssl_model.model.*` / `frontend.model.*`
sub-dict and it maps 429 fairseq keys → 421 torchaudio tensors, strict-loadable.
This is the reusable trick for any fairseq wav2vec2 checkpoint in this repo.

**Neither checkpoint has a published single-model In-the-Wild EER** (Garg et al.
report 5.70% only for an *ensemble*; DeepFense ships no model card). So the
handoff's "reproduce the published number" gate cannot be run as written — do not
claim reproduction of a published ITW figure. Correctness rests on the strict
load + polarity + non-empty-trainable-set gates instead.

Files: `public_ckpt_tta.py` (harness), `vendor_aasist.py` → `aasist_backend.py`,
`build_itw_manifest.py`, `run_public_ckpt.sh`, `PUBLIC_CKPT_NOTES.md`.
Results land in `results_public_ckpt.csv`.

See [[public-ckpt-score-polarity]], [[validate-before-gpu]], [[sota-phase-status]].
