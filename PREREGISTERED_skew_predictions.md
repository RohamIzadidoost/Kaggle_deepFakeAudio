# Pre-registered predictions: In-the-Wild prevalence intervention

Written before any result of this sweep was seen (run launched 2026-09-14
09:37; this file written 09:38, first arm still adapting). Checkpoint
`ssl_aasist_wavefake`, In-the-Wild audio, resampled to a target P(fake).
Only the class mix changes -- same audio, same weights, same recipe.

Bound (Eq. 2): purity_real <= min(1, pi_real/q_real).
Symmetric rule (Eq. 3) sound only while q <= min(pi, 1-pi).

| target P(fake) | pi_real | min(pi,1-pi) | bound q=0.02 | q=0.10 | q=0.30 |
|---|---|---|---|---|---|
| 0.372 | 0.628 | 0.372 | 1.000 | 1.000 | 1.000 |
| 0.9 | 0.100 | 0.100 | 1.000 | 1.000 | 0.333 |
| 0.95 | 0.050 | 0.050 | 1.000 | 0.500 | 0.167 |
| 0.97 | 0.030 | 0.030 | 1.000 | 0.300 | 0.100 |

## Predictions

1. **q=0.02 stays safe at every skew** (bound 1.000 even at P(fake)=0.97,
   where pi_real=0.030 > 0.02). Accuracy high, adaptation neutral-to-helpful.
2. **q=0.30 degrades monotonically with skew**, tracking its bound
   (1.000 -> 0.333 -> 0.167 -> 0.100), and should collapse at 0.97 as DF did.
3. **q=0.10 sits between**, intact to P(fake)=0.90 (bound 1.000) and
   degrading from 0.95 (0.500) onward.
4. **Accuracy monotone in the bound across the 3x3 grid**, as in the DF
   sweep (Spearman +1.000 there).
5. **The prior-split arm's fate depends on the estimate, not the budget.**
   This checkpoint arrives on ITW with a +16.7 deficit, so BBSE is already
   +0.23 off at natural prevalence; by the identifiability argument the
   error should grow with skew, so prior-split should NOT track its own
   (correct-prior) bound of 1.000.

## What would falsify the bound

Accuracy at q=0.30 / P(fake)=0.97 comparable to q=0.02 at the same skew;
or a non-monotone bound-vs-accuracy relationship across the grid. Either
would mean pseudo-label purity is not what governs this failure.
