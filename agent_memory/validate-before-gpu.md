---
name: validate-before-gpu
description: "Working style — falsify mechanisms cheaply before spending scarce GPU, and report negatives straight rather than salvaging them"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a7e7d7d9-af89-4792-b7a8-336e8fb579a1
  modified: 2026-08-10T03:59:34.386Z
---

On this project, build the cheap falsification test *before* the expensive run,
and when it fails, say so plainly and redirect rather than rescuing the original
plan.

**Why:** GPU access here is scarce and bursty (cloud allocation already ran out
mid-experiment once; local is a 10 GB RTX 3080 with rented time booked days
ahead). A mechanism that can be killed by a two-minute CPU test on synthetic
scores must never consume a booked GPU day. In this phase that pattern killed
three proposed mechanisms — two before any GPU time, one in a 60-second smoke
run — and each would otherwise have burned hours and produced an
uninterpretable result.

The user asked for adaptive `q` *and* `lambda`, was shown evidence that lambda
had ≤0.5 EER of fragile headroom while `q`/`E` had 1.2–16, and chose to redirect.
Presenting effect sizes and letting them re-aim worked better than delivering
the literal request.

**How to apply:**
- Write signal/schedule logic as pure numpy with no CUDA and no module globals,
  so it is testable without a GPU. That single choice is what made the early
  kills possible.
- Validate any new label-free signal against results the repo *already* has
  (`results_ext.csv` gives per-(target,seed) Δ₂) before trusting it.
- Watch for metrics that respond to a constant you chose rather than to the data
  — that was the flaw in all three killed mechanisms.
- Keep failed mechanisms as recorded negative-result tests, matching how the repo
  already keeps `losses.py` for validated-harmful losses.
- Correct your own overstated claims immediately and specifically; a result that
  flips with a threshold-grid resolution is not a result.

See [[adaptive-tta-phase]] and [[repo-hazards]].
