# Pre-registered: is it the excess, or is it just "large q"?

Written 2026-09-16 ~13:45 IST, before launching any run below and before seeing
any of its output. Extends `PREREGISTERED_low_prevalence_predictions.md`.

## The problem with the current evidence

Across the 31 completed cells, the quantity

    D = sum_c (b_c - pi_c)^+          (for exact symmetric tails, D = (q - min(pi,1-pi))^+)

orders the outcomes: every cell with D <= 0.089 improves balanced accuracy
(19/21 positive, 18 by >4 points), every cell with D >= 0.204 fails to improve
by more than +2.0, and 8 of those 10 are harmful. No cell lies between.

But D and q are confounded in that sample. Of the 10 high-D cells, 10 use
q=0.30; of the 21 low-D cells, 17 use q<=0.10. A reader can reasonably say the
result is nothing but "q=0.3 hurts and q=0.02 helps", with D a relabelling.

The one place the two accounts already disagree is native In-the-Wild
prevalence, where q=0.30 has D=0 and helps (+12.7 to +32.7 BA on three
checkpoints). That is 4 cells. The runs below separate the accounts deliberately.

## Experiment B (decisive, run first)

WF on In-the-Wild resampled to its own native prior (PUBA_SKEW=0.3718, so the
pool is the native 31.8k clips but the run uses the same 8,000-clip adaptation
budget as every other resampled cell). Sweep q in {0.30, 0.40, 0.45, 0.50}:

| q | D | excess-account prediction | "large q is bad" prediction |
|---|---|---|---|
| 0.30 | 0.000 | safe, dBA >= +5 | harmful |
| 0.40 | 0.028 | safe, dBA >= +5 | harmful |
| 0.45 | 0.078 | safe, dBA >= +5 | harmful |
| 0.50 | 0.128 | inside the empty gap; locates the transition | harmful |

q=0.50 labels the entire pool; it is the largest symmetric budget that exists.
q=0.45 is half again the budget that collapses accuracy to 51.9% at pi=0.01.

**This is the falsification test.** If q=0.40 and q=0.45 are harmful here
(dBA < 0), D is not the governing quantity and the finding reduces to "large
budgets hurt". If they are safe, q alone cannot explain the outcomes, because
the same q=0.45 is far past the failure point at every skewed prevalence.

## Experiment A (dose-response through the gap)

WF on In-the-Wild at pi=0.01 (constrained tail is spoof) and pi=0.97
(constrained tail is bona fide), 8,000-clip adaptation:

    pi=0.01, min(pi,1-pi)=0.0109:  q = 0.06, 0.11, 0.15, 0.19, 0.23
                                   D = 0.049, 0.099, 0.139, 0.179, 0.219
    pi=0.97, min(pi,1-pi)=0.030 :  q = 0.08, 0.13, 0.17, 0.21, 0.25
                                   D = 0.050, 0.100, 0.140, 0.180, 0.220

Predictions:
1. dBA declines with D on both sides and crosses +5 somewhere in D in
   [0.089, 0.204], turning the empty gap into a measured transition.
2. The crossing occurs at **approximately the same D on both sides**, even
   though the contaminated class is spoof at pi=0.01 and bona fide at pi=0.97.
   A class-specific mechanism would put the two crossings at different D.
3. The matched-D pairs (0.05/0.05, 0.10/0.10, ...) agree to within a few
   points of balanced accuracy.

## Experiment C (does the barrier depend on the checkpoint?)

DF-42 on In-the-Wild, which arrives with AUC 0.9171 and 39.08% accuracy, a far
weaker ranking than WF's 0.985. pi in {0.01, 0.97}, q in {0.02, 0.15, 0.30}.
Prediction: the same ordering by D, with the transition in the same interval.
If the ranking quality matters more than D, this checkpoint should fail
everywhere, including at D=0.

## What would falsify the account outright

- Experiment B: q=0.45 harmful at native prevalence.
- Experiment A: no transition within the gap, or two clearly different
  crossing points for the two prevalences.
- Experiment C: outcomes unordered by D on a second checkpoint.

## Experiment E (second corpus, added 2026-09-16 18:05, before any result of A/B/C was read)

Everything above runs on In-the-Wild audio. `PUBA_SKEW` resamples any corpus,
so the same grid runs on the ASVspoof2021-DF eval partition with DF-2: a
different corpus, a different checkpoint, and one regime the study has never
covered, a balanced pool.

DF has 11,136 bona fide and 389,299 spoof available, so resampling below the
native 0.972 keeps all bona fide clips and thins spoof:

| target pi | N | min(pi,1-pi) | D at q=.02 | q=.15 | q=.30 |
|---|---|---|---|---|---|
| 0.50 | 22,272 | 0.500 | 0 | 0 | 0 |
| 0.90 | 111,360 | 0.100 | 0 | 0.05 | 0.20 |
| 0.10 | 12,373 | 0.100 | 0 | 0.05 | 0.20 |
| 0.01 | 11,248 | 0.010 | 0.01 | 0.14 | 0.29 |

Adaptation is 8,000 clips everywhere, as in the rest of the grid.

Predictions:
1. **q=0.30 on the balanced pool (D=0) is safe**, despite being the budget that
   collapses accuracy at pi=0.972 on this same checkpoint and corpus. This is
   the within-corpus, within-checkpoint version of Experiment B's test: only the
   prevalence differs.
2. pi=0.90 and pi=0.10 are mirror images at every q, since both have
   min(pi,1-pi)=0.100 and therefore identical D, while the contaminated class
   is the opposite one. Their dBA should agree more closely with each other
   than either does with pi=0.01 or pi=0.50.
3. q=0.15 at pi=0.01 has D=0.14, inside the empty gap, and should land between
   the two clusters.
4. The dBA ordering by D holds on this corpus as it did on In-the-Wild.

Note: at pi=0.01 the disjoint remainder holds only ~3.2k clips with ~32 spoof,
so its EER there is noisy and the balanced-accuracy comparison on the full pool
is the one to read.
