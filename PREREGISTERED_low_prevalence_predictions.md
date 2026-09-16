# Pre-registered predictions: low-spoof-prevalence intervention

Written 2026-09-16 09:45 IST, **before launching any arm** of this sweep and
before inspecting any of its output. Extends the sweep pre-registered in
`PREREGISTERED_skew_predictions.md` (which covered P(fake) = 0.90/0.95/0.97) into
the deployment-relevant direction: pools where spoof is the *minority* class.

Checkpoint `ssl_aasist_wavefake` (WF), In-the-Wild audio, resampled to a target
P(fake). Same weights, same recipe, same audio as the high-skew sweep. The
resampler is real-limited here, so it keeps all 19,963 bona fide clips and thins
the 11,816 spoof clips.

| target P(fake) | spoof kept | N | trivial baseline (all-bona-fide) |
|---|---|---|---|
| 0.10 | 2,218 | 22,181 | 90.00% |
| 0.05 | 1,051 | 21,014 | 95.00% |
| 0.02 | 407 | 20,370 | 98.00% |
| 0.01 | 202 | 20,165 | 99.00% |

Adaptation pool is fixed at 8,000 clips at every prevalence (as in the high-skew
sweep), so the number of optimization steps is identical across the grid:
4 epochs x ceil(8000/32) = 1,000 steps. Only the class mix changes. The
evaluation pool size still varies with prevalence; that remains a limitation.

## The constraint changes class

Bound (Eq. 2): purity(C_c) <= min(1, pi_c / b_c). In the high-skew sweep the
**bona fide** tail was the constrained one. Below 50% spoof it is the **spoof**
tail: bound_spoof = min(1, pi / q). The bona fide bound is 1.000 everywhere here
since 1 - pi >= 0.90 > 0.30 >= q.

| target P(fake) | bound q=0.02 | bound q=0.10 | bound q=0.30 |
|---|---|---|---|
| 0.10 | 1.000 | 1.000 | 0.333 |
| 0.05 | 1.000 | 0.500 | 0.167 |
| 0.02 | 1.000 | 0.200 | 0.067 |
| 0.01 | **0.500** | 0.100 | 0.033 |

## Predictions

1. **q=0.30 degrades at every prevalence in this grid** (bound <= 0.333
   throughout), worst at P(fake)=0.01.
2. **q=0.10 is intact at P(fake)=0.10 (bound 1.000) and degrades from 0.05
   onward** (0.500 -> 0.200 -> 0.100).
3. **The sharpest test: q=0.02 crosses from feasible to infeasible between
   P(fake)=0.02 (bound 1.000) and P(fake)=0.01 (bound 0.500).** The current
   manuscript presents small q as the prior-free safe fallback. If the bound
   governs, that fallback should itself break at P(fake)=0.01 and hold at 0.02
   and above. This is a prediction of a *crossing at a named point*, not a
   monotone trend, so it is the most falsifiable cell in the sweep.
4. **Failure signature is FPR, not FNR.** Contaminating the spoof tail with bona
   fide clips teaches the model to call real audio fake. Predicted damage shows
   up as false alarms on the bona fide majority. (At high skew the mirror
   signature held: q=0.30 at 0.97 gave FNR 29.5% with FPR 1.6%.)
5. **Arms that violate the bound fall below the trivial all-bona-fide
   baseline**; the unchanged source does not.
6. **EER degrades less than accuracy**, as at high skew (q=0.30 at P(fake)=0.97:
   accuracy 98.94 -> 65.42 but EER 5.26 -> 6.02). Purity governs pseudo-labels,
   and EER is threshold-free.
7. **BBSE stays feasible while its estimate holds.** It allocates
   q_c = 2q * pi_hat_c, so a correct estimate gives a tiny spoof tail. Predicted
   safe down to P(fake)=0.02; at 0.01 the estimator's clipping floor and source
   confusion-matrix error may dominate, so no directional prediction there.

## What would falsify the bound

- q=0.02 at P(fake)=0.01 performing like q=0.02 at P(fake)=0.10 (no crossing).
- Any non-monotone relation between the bound and accuracy across the 4x3 grid.
- Degradation carried by FNR rather than FPR, which would mean the damage is not
  tail contamination by the majority class.
