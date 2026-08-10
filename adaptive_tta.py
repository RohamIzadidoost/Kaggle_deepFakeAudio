"""Label-free adaptive schedules for the TTA knobs q, lambda and E.

Why this exists: every TTA number in the paper uses one hardcoded configuration
(q=0.3, lambda=0.3, E=4) and none of the three has recorded provenance -- in
tta.py:114-119 the literal 0.3 is the weight of all three auxiliary loss terms,
which is what an inherited default looks like rather than a tuned one. main.tex
Limitation 1 already concedes that this fixed point is not uniformly optimal
(lambda=0 is better on LibriSpeech-TTS, q=0.2 better on Arabic) and that a
practitioner without target labels cannot correct for it.

A previous attempt to fix this (lambda_selector.py) used confident-tail
*separation* as the reliability proxy and failed in the wrong direction
(r=+0.74 against the true delta_2, where r<0 was needed): LibriSpeech-TTS, the
one target where consistency hurts, has the *highest* tail separation of any
target. The paper's own diagnosis is that a working signal "would need to
measure something closer to calibration correctness than to confidence".

This module supplies that signal: **pseudo-label churn**, the rate at which a
clip's confident-tail assignment flips under the same channel augmentation the
consistency loss uses. Churn measures instability, not confidence, so it is not
the quantity that already failed -- a target can be very confident and very
unstable at the same time, which is exactly the LibriSpeech-TTS regime.

Everything here is pure numpy: no CUDA, no torch, no module-level state, no
pipeline imports. That is deliberate -- it makes the whole signal unit-testable
on synthetic score arrays on a CPU, which is what let this be validated before
any GPU time was spent on it (see test_adaptive_tta.py).

No target labels are read anywhere in this module.
"""

import numpy as np

# Defaults. LAMBDA_MAX and Q_CAP are the paper's published values, kept as the
# *ceiling* of each schedule so that the low-churn regime reproduces the fixed
# configuration exactly rather than drifting away from it.
LAMBDA_MAX = 0.3
Q_CAP = 0.3
Q_START = 0.1
Q_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)
CHURN_TAU = 0.5          # normalised-churn budget for select_q_max; see below
E_MIN, E_MAX = 2, 8
STOP_TOL = 0.002         # mean per-clip score movement below this = converged
DIVERGE_RATIO = 1.5      # score movement rising by more than this = diverging

_EPS = 1e-12


# ------------------------------------------------------------------ tail labels
def pseudo_labels(s, lo_frac, hi_frac):
    """Confident-tail pseudo-labels: -1 ignore, 0 pseudo-real, 1 pseudo-fake.

    Deliberately byte-for-byte the same rule as extended_pipeline.py:417-420,
    including the assignment *order* (real first, then fake, so an overlap
    resolves to fake). The adaptive path has to reduce exactly to the published
    path when the schedules are switched off, and that reduction is only
    checkable if the tail rule is shared rather than reimplemented.
    """
    s = np.asarray(s, dtype=np.float64)
    pl = np.full(len(s), -1, dtype=np.int64)
    if lo_frac > 0:
        pl[s <= np.quantile(s, lo_frac)] = 0
    if hi_frac > 0:
        pl[s >= np.quantile(s, 1.0 - hi_frac)] = 1
    return pl


def tail_cuts(s, lo_frac, hi_frac):
    """The absolute score thresholds implied by the two tail fractions."""
    s = np.asarray(s, dtype=np.float64)
    lo = np.quantile(s, lo_frac) if lo_frac > 0 else -np.inf
    hi = np.quantile(s, 1.0 - hi_frac) if hi_frac > 0 else np.inf
    return float(lo), float(hi)


# ----------------------------------------------------------------- churn signal
def view_churn(s, s_aug, lo_frac, hi_frac):
    """Fraction of confident clips whose tail assignment flips under augmentation.

    Returns (c_view, c_null).

    The augmented scores are thresholded at the *clean* view's cuts, not at
    their own quantiles. That matters: re-quantiling s_aug would renormalise
    away exactly the drift we are trying to detect, since any monotone shift of
    the whole score distribution leaves its own quantiles unchanged.

    c_null is the churn an uninformative ranking would produce -- the value
    c_view would take if s_aug were independent of s. It is computed here in
    closed form as the exact expectation of the permutation null (permuting
    s_aug and averaging would converge to this, at a cost and with sampling
    noise, for no benefit):

        E[churn | independence] = P(s_aug > lo) for a pseudo-real clip
                                  P(s_aug < hi) for a pseudo-fake clip

    Normalising by c_null is what keeps the lambda gate free of hand-picked
    thresholds -- "flips half as often as chance" is a scale-free statement,
    whereas "flips less than 12% of the time" would just relocate the arbitrary
    constant we are trying to eliminate.
    """
    s = np.asarray(s, dtype=np.float64)
    s_aug = np.asarray(s_aug, dtype=np.float64)
    pl = pseudo_labels(s, lo_frac, hi_frac)
    lo, hi = tail_cuts(s, lo_frac, hi_frac)

    pl_aug = np.full(len(s), -1, dtype=np.int64)
    if lo_frac > 0:
        pl_aug[s_aug <= lo] = 0
    if hi_frac > 0:
        pl_aug[s_aug >= hi] = 1

    conf = pl >= 0
    n_conf = int(conf.sum())
    if n_conf == 0:
        # No confident set at all: nothing for the consistency term to act on,
        # so report maximal instability and let the gate close.
        return 1.0, 1.0

    c_view = float((pl_aug[conf] != pl[conf]).mean())

    n_real = int((pl == 0).sum())
    n_fake = int((pl == 1).sum())
    p_lo = float((s_aug <= lo).mean()) if lo_frac > 0 else 0.0
    p_hi = float((s_aug >= hi).mean()) if hi_frac > 0 else 0.0
    c_null = (n_real * (1.0 - p_lo) + n_fake * (1.0 - p_hi)) / max(n_conf, 1)
    return c_view, float(c_null)


def epoch_churn(pl_prev, pl_now):
    """Flip rate between two epochs' pseudo-labels, over jointly-confident clips.

    Restricted to clips confident in *both* epochs on purpose. The q ramp
    changes the size of the confident set by construction, so counting
    ignore <-> confident transitions would report the schedule's own behaviour
    as instability and stop the run early for the wrong reason.

    NOT a convergence signal, despite looking like one -- see `score_shift`.
    Kept for the trace, where a non-zero value is a genuine warning sign.
    """
    pl_prev, pl_now = np.asarray(pl_prev), np.asarray(pl_now)
    both = (pl_prev >= 0) & (pl_now >= 0)
    if not both.any():
        return 1.0
    return float((pl_prev[both] != pl_now[both]).mean())


def score_shift(s_prev, s_now):
    """How much the score distribution moved between epochs: mean |delta|.

    This replaces pseudo-label churn as the stopping signal. Label churn turned
    out to be degenerate in practice -- measured at exactly 0.0000 on every
    epoch of a real run -- because the labels come from *quantile* cuts, so they
    only move if the ranking reorders. Self-training mostly sharpens the
    decision boundary without reordering, which leaves the labels bit-identical
    while the model is still very much changing. Stopping on that fires at
    E_MIN every time, which measured worse than not adapting adaptively at all
    (EER 10.16 vs 9.38 fixed).

    Scores are in [0, 1], so this is directly interpretable as "the average clip
    moved this far".
    """
    s_prev, s_now = np.asarray(s_prev, dtype=np.float64), np.asarray(s_now, dtype=np.float64)
    return float(np.abs(s_now - s_prev).mean())


# ---------------------------------------------------------------------- lambda
def lambda_from_churn(c_view, c_null, lam_max=LAMBDA_MAX):
    """Gate the consistency weight by how stable the confident tails are.

        lambda = lam_max * clip(1 - c_view / c_null, 0, 1)

    A ranking that survives the augmentation keeps the published lam_max; one
    that flips at chance rate gets lambda -> 0, which degrades the method
    gracefully into st_only rather than into a new failure mode.

    The pre-registered prediction this must satisfy, checkable against
    results_ext.csv before any new number is believed: LibriSpeech-TTS
    (source AUC .718, delta_2 = +0.51, consistency hurts in 5/5 seeds) must
    show high churn here, and Arabic (delta_2 = -1.23) must show low churn.
    """
    if c_null <= _EPS:
        return 0.0
    return float(lam_max * np.clip(1.0 - c_view / c_null, 0.0, 1.0))


# -------------------------------------------------------------------- prevalence
def estimate_prevalence_guarded(scores, min_prior=0.02, max_prior=0.98,
                                bic_margin=0.0):
    """Label-free P(real) from the score distribution, with a trust guard.

    Returns (pi_real, bic_delta, shrunk).

    The estimator itself is the one validated in protocol_a.py:291 -- a
    2-component GMM on the score histogram, reading off the low-scoring
    component's mixture weight (same n_components=2, random_state=0, n_init=3,
    same [min_prior, max_prior] clip, so the un-shrunk path returns the same
    number). It reads mixture weight rather than mean(scores) because the
    latter assumes calibration, which is the exact property this paper's thesis
    says does NOT transfer cross-corpus.

    Reimplemented here rather than imported because protocol_a.py runs torch and
    CUDA setup at module scope, and this module is deliberately CPU-only and
    dependency-free so it can be tested without a GPU.

    The guard is new. On the DF eval pool the estimator is excellent (true
    2.47%, estimated 2.63%), but on the RawBoost-retrained model it returned
    0.458 against a true 0.025 -- an 18x error that silently drove a whole
    protocol into the ground (main.tex:1070-1078). The failure mode is fitting
    two components to what is really one blob, so: if a 2-component fit does not
    beat a 1-component fit on BIC, we do not trust the split and fall back to
    pi=0.5, i.e. to the symmetric budget the published method already uses.
    Falling back to current behaviour is a safe default in a way that a
    confidently wrong prior is not.
    """
    from sklearn.mixture import GaussianMixture

    x = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    g2 = GaussianMixture(n_components=2, random_state=0, n_init=3).fit(x)
    g1 = GaussianMixture(n_components=1, random_state=0, n_init=3).fit(x)
    # BIC is lower-is-better, so a positive delta means 2 components won.
    bic_delta = float(g1.bic(x) - g2.bic(x))

    if bic_delta <= bic_margin:
        return 0.5, bic_delta, True

    means, weights = g2.means_.flatten(), g2.weights_
    pi_real = float(weights[np.argmin(means)])
    return float(np.clip(pi_real, min_prior, max_prior)), bic_delta, False


def tail_budget(q, pi_real):
    """Split the total confident-label budget 2q by estimated class prevalence.

    Same formula as protocol_a.py:414. At pi_real=0.5 this is identically
    (q, q) -- the symmetric split the published method uses -- so on the
    balanced leave-one-corpus-out grid the prevalence machinery is a no-op and
    only the q ramp changes anything. It bites on skewed pools, where a fixed
    symmetric q=0.3 labels the bottom 30% "confident real" regardless of the
    pool's true balance.
    """
    lo_frac = 2.0 * q * pi_real
    hi_frac = 2.0 * q * (1.0 - pi_real)
    return float(np.clip(lo_frac, 0.0, 1.0)), float(np.clip(hi_frac, 0.0, 1.0))


# --------------------------------------------------------------------------- q
def select_q_max(s, s_aug, pi_real=0.5, grid=Q_GRID, cap=Q_CAP, tau=CHURN_TAU):
    """Largest tail fraction whose pseudo-labels are still stable under a view.

    NOT USED BY THE METHOD -- retained for the analysis table only.

    It inherits the confound documented on `lambda_from_churn`: `view_churn`
    responds to the augmentation magnitude, so on real target audio this returns
    the grid minimum (0.05) regardless of the corpus, labelling 13 of 128 clips
    where the published q=0.3 labels 78. A brake that is always fully engaged is
    not a brake. The method uses `q_max = Q_CAP` and lets the curriculum ramp and
    the prevalence split do the adapting.

    Widening the tails necessarily reaches into the ambiguous middle, so churn
    grows with q; we walk the grid upward and stop at the first q whose
    normalised churn exceeds tau, returning the last q that passed. Stopping at
    the first failure (rather than taking the max over all passing q) is the
    conservative reading and keeps one noisy grid point from re-opening the
    budget.

    tau is not a free parameter in the usual sense: at normalised churn tau=0.5
    the lambda gate has already halved the consistency weight, so this is "stop
    widening once the tails are half-trusted", tied to the lambda mechanism
    rather than chosen independently.

    `cap` is the hard clamp at the published q=0.3. Without it a target whose
    scores happen to look stable could expand the budget past anything the
    paper has evidence for; the sweep only ever covered q <= 0.4, on 2 of 4
    targets at 1 seed.
    """
    best = None
    for q in grid:
        lo_frac, hi_frac = tail_budget(q, pi_real)
        c_view, c_null = view_churn(s, s_aug, lo_frac, hi_frac)
        if c_null <= _EPS or c_view / c_null > tau:
            break
        best = q
    if best is None:
        best = min(grid)          # nothing passed: fall back to the tightest tail
    return float(min(best, cap))


def q_schedule(epoch, n_epochs, q_max, q_start=Q_START):
    """Linear curriculum ramp from a strict high-purity tail up to q_max.

    Early epochs are where confirmation bias is cheapest to acquire and most
    expensive to undo -- the model is still miscalibrated on the target, so a
    wide tail buys a lot of wrong labels that later epochs then reinforce.
    Starting at q_start=0.1 and widening only as the model settles is the
    standard curriculum answer to that.

    `n_epochs` should be E_MAX even when adaptive stopping may cut the run
    short, so that the ramp's shape does not depend on when it happens to stop.
    If q_max came back below q_start the ramp is flat at q_max rather than
    descending into it.
    """
    q_max = float(q_max)
    lo = min(float(q_start), q_max)
    if n_epochs <= 1:
        return q_max
    t = min(max(epoch, 0), n_epochs - 1) / (n_epochs - 1)
    return float(lo + (q_max - lo) * t)


# ------------------------------------------------------------------- stopping
def should_stop(shift_history, epoch, min_epochs=E_MIN, max_epochs=E_MAX,
                tol=STOP_TOL, diverge_ratio=DIVERGE_RATIO):
    """Label-free stopping rule on epoch-to-epoch score movement.

    Returns (stop, reason). `shift_history` holds `score_shift` values, oldest
    first; `epoch` is the 0-based index of the epoch just finished.

    E_MAX=8 rather than the paper's 4 because the sweep shows E=8 beats E=4 on
    both swept targets and main.tex:693-696 calls E=4 "a conservative,
    compute-matched choice, not a tuned optimum" -- a budget decision, not a
    result. Since every measured point says more epochs help, this rule is
    deliberately biased toward running the full budget: it is a safety valve for
    a run that has stopped moving or started running away from itself, not an
    efficiency optimisation trying to shave epochs off a healthy run. Stopping
    early is the behaviour that has actually been observed to hurt here.
    """
    if epoch + 1 >= max_epochs:
        return True, "max_epochs"
    if epoch + 1 < min_epochs or not shift_history:
        return False, ""
    if shift_history[-1] <= tol:
        return True, "converged"
    # Divergence must be *sustained*: two consecutive rises AND a level well
    # above the best epoch so far. A single-epoch comparison fired on a
    # 0.0091 -> 0.0168 blip in a smoke run, cutting a healthy run short at
    # epoch 6 -- and stopping early is the failure mode that has actually cost
    # EER here, so the rule should need convincing.
    h = shift_history
    if len(h) >= 3 and h[-1] > h[-2] > h[-3] and h[-1] > diverge_ratio * min(h):
        return True, "diverging"
    return False, ""


def collapsed(s, min_std=1e-3, min_tail_gap=1e-3, q=0.1):
    """Has the score distribution degenerated? Returns (collapsed, reason).

    Insurance against the failure main.tex:460-479 describes as undetectable in
    advance: Tent looked like every other run right up until its score
    distribution collapsed, giving 49.52% EER on a seed indistinguishable from
    four that were fine. We cannot predict that, but we can notice it happening
    and hand back the pre-adaptation model, which turns a catastrophic run into
    a no-op run.
    """
    s = np.asarray(s, dtype=np.float64)
    if not np.isfinite(s).all():
        return True, "non_finite_scores"
    if float(s.std()) < min_std:
        return True, "score_std_collapse"
    lo, hi = float(np.quantile(s, q)), float(np.quantile(s, 1.0 - q))
    if hi - lo < min_tail_gap:
        return True, "tails_merged"
    return False, ""
