"""CPU-only checks for adaptive_tta.py. Run: python test_adaptive_tta.py

Plain asserts and prints rather than pytest -- the repo has no test runner
configured, and the point of this file is that the whole adaptive signal can be
validated on synthetic score arrays before spending any GPU time on it.

Each scenario states the property being checked and why it matters.
"""

import numpy as np

import adaptive_tta as A

RNG = np.random.default_rng(0)
_fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not cond:
        _fails.append(name)


def two_gauss(n, mu0, mu1, sd, clip=True):
    """Balanced two-class score pool; class 0 low, class 1 high."""
    s = np.concatenate([RNG.normal(mu0, sd, n // 2), RNG.normal(mu1, sd, n - n // 2)])
    return np.clip(s, 0, 1) if clip else s


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)


# ---------------------------------------------------------------------------
print("\n[1] Churn responds correctly to a destroyed ranking (sanity floor)")
s = two_gauss(4000, 0.05, 0.95, 0.03)
s_aug = RNG.permutation(s)          # independent of s by construction = the null
c, c0 = A.view_churn(s, s_aug, 0.3, 0.3)
check("churn ~ null", abs(c / c0 - 1.0) < 0.15, f"c={c:.4f} null={c0:.4f}")
check("lambda ~ 0", A.lambda_from_churn(c, c0) < 0.05)

# ---------------------------------------------------------------------------
print("\n[2] NEGATIVE RESULT: churn cannot gate lambda -- it measures the")
print("    augmentation constant, not the target's ranking quality.")
# This was the originally planned lambda mechanism. It does not work, and this
# check pins down why so the finding is not rediscovered later.
#
# Sweeping true AUC over the entire range that matters, at fixed augmentation,
# barely moves lambda; sweeping the augmentation magnitude at FIXED AUC moves it
# a great deal. The gate would therefore be reading augment()'s gain/noise
# settings -- a constant we chose -- rather than anything about the corpus.
from scipy.stats import norm
sd = 0.15


def pool_at_auc(target_auc, n=6000):
    d = norm.ppf(target_auc) * sd * np.sqrt(2)
    return two_gauss(n, 0.5 - d / 2, 0.5 + d / 2, sd)


lam_by_auc = []
for target_auc in (0.60, 0.72, 0.80, 0.86, 0.94, 0.99):
    s = pool_at_auc(target_auc)
    s_aug = np.clip(s + RNG.normal(0, 0.03, len(s)), 0, 1)
    lam_by_auc.append(A.lambda_from_churn(*A.view_churn(s, s_aug, 0.3, 0.3)))

s = pool_at_auc(0.86)
lam_by_noise = []
for nz in (0.01, 0.03, 0.10, 0.30):
    s_aug = np.clip(s + RNG.normal(0, nz, len(s)), 0, 1)
    lam_by_noise.append(A.lambda_from_churn(*A.view_churn(s, s_aug, 0.3, 0.3)))

span_auc = max(lam_by_auc) - min(lam_by_auc)
span_noise = max(lam_by_noise) - min(lam_by_noise)
print(f"    lambda across AUC .60->.99 (fixed aug): span {span_auc:.4f}  {[round(v,3) for v in lam_by_auc]}")
print(f"    lambda across aug .01->.30 (fixed AUC): span {span_noise:.4f}  {[round(v,3) for v in lam_by_noise]}")
check("confound is real: aug magnitude dominates AUC", span_noise > 5 * span_auc,
      f"{span_noise:.4f} vs {span_auc:.4f}")
check("AUC signal is negligible in absolute terms", span_auc < 0.05,
      f"span {span_auc:.4f} of a 0..{A.LAMBDA_MAX} range")

# ---------------------------------------------------------------------------
print("\n[4] Prevalence: 97.5% skew recovered, budget goes asymmetric")
n = 8000
s = np.clip(np.r_[RNG.normal(0.05, 0.05, int(0.025 * n)),
                  RNG.normal(0.90, 0.08, n - int(0.025 * n))], 0, 1)
pi, bic, shrunk = A.estimate_prevalence_guarded(s)
lo_f, hi_f = A.tail_budget(0.3, pi)
check("pi_real near true 0.025", abs(pi - 0.025) < 0.02, f"pi={pi:.4f} (true 0.025)")
check("not shrunk (BIC favours 2 components)", not shrunk, f"bic_delta={bic:.1f}")
check("pseudo-real budget shrinks", lo_f < 0.05, f"lo_frac={lo_f:.4f} hi_frac={hi_f:.4f}")

# ---------------------------------------------------------------------------
print("\n[5] Guard: a single blob must shrink back to the symmetric split")
s = np.clip(RNG.normal(0.5, 0.12, 6000), 0, 1)
pi, bic, shrunk = A.estimate_prevalence_guarded(s)
check("shrunk to pi=0.5", shrunk and pi == 0.5, f"pi={pi:.4f} bic_delta={bic:.1f} shrunk={shrunk}")
lo_f, hi_f = A.tail_budget(0.3, pi)
check("budget reduces to symmetric (q, q)", (lo_f, hi_f) == (0.3, 0.3), f"({lo_f}, {hi_f})")

# ---------------------------------------------------------------------------
print("\n[6] tail_budget at pi=0.5 is a no-op vs the published symmetric split")
for q in (0.1, 0.2, 0.3):
    check(f"tail_budget({q}, 0.5) == ({q}, {q})", A.tail_budget(q, 0.5) == (q, q))

# ---------------------------------------------------------------------------
print("\n[7] pseudo_labels reproduces extended_pipeline.py:417-420 exactly")
s = two_gauss(3000, 0.2, 0.8, 0.2)
Q = 0.3
lo, hi = np.quantile(s, Q), np.quantile(s, 1 - Q)
ref = np.full(len(s), -1, dtype=np.int64)
ref[s <= lo] = 0
ref[s >= hi] = 1
check("identical to the published tail rule", np.array_equal(A.pseudo_labels(s, Q, Q), ref))

# ---------------------------------------------------------------------------
print("\n[8] select_q_max: bounded by the clamp, tightens under heavy perturbation")
# Inherits the same confound as [2] -- it responds to perturbation magnitude, not
# to ranking quality -- so it is NOT used to pick q from ranking quality. It is
# kept only as a widening brake: under a violent perturbation the budget must not
# stay wide open. The clamp at Q_CAP is what actually bounds it.
s = two_gauss(4000, 0.05, 0.95, 0.03)
s_aug = np.clip(s + RNG.normal(0, 0.02, len(s)), 0, 1)
q_clean = A.select_q_max(s, s_aug)
s2 = two_gauss(4000, 0.45, 0.55, 0.2)
s2_aug = RNG.permutation(s2)          # ranking fully destroyed
q_noisy = A.select_q_max(s2, s2_aug)
check("clean data reaches the cap", q_clean == A.Q_CAP, f"q_max={q_clean}")
check("destroyed ranking tightens the budget", q_noisy < q_clean, f"q_max={q_noisy}")
check("never exceeds the 0.3 clamp", max(q_clean, q_noisy) <= A.Q_CAP)

# ---------------------------------------------------------------------------
print("\n[9] q ramp")
qs = [A.q_schedule(e, 8, q_max=0.3) for e in range(8)]
check("starts at Q_START", abs(qs[0] - A.Q_START) < 1e-9, f"{qs[0]:.3f}")
check("ends at q_max", abs(qs[-1] - 0.3) < 1e-9, f"{qs[-1]:.3f}")
check("monotone non-decreasing", all(b >= a - 1e-12 for a, b in zip(qs, qs[1:])))
flat = [A.q_schedule(e, 8, q_max=0.05) for e in range(8)]
check("flat when q_max < Q_START", max(flat) - min(flat) < 1e-12, f"{flat[0]:.3f}")

# ---------------------------------------------------------------------------
print("\n[10] stopping rule (on score movement, not label churn)")
check("respects E_MIN", A.should_stop([0.0], epoch=0)[0] is False)
check("fires on converged", A.should_stop([0.05, 0.001], epoch=1)[1] == "converged")
check("fires on sustained divergence", A.should_stop([0.02, 0.05, 0.09, 0.20], epoch=3)[1] == "diverging")
# The exact blip that cut a healthy smoke run short at epoch 6.
check("ignores a single-epoch blip",
      A.should_stop([0.0505, 0.0370, 0.0223, 0.0091, 0.0168], epoch=4)[0] is False)
check("caps at E_MAX", A.should_stop([0.5] * 8, epoch=A.E_MAX - 1)[1] == "max_epochs")
check("keeps going while still moving", A.should_stop([0.20, 0.15], epoch=1)[0] is False)
# The regression this rule exists to avoid: a healthy run must NOT stop at E_MIN.
# Label churn measured exactly 0.0 every epoch of a real run, which fired
# "converged" immediately and cost 0.8 EER against the fixed-E control.
check("healthy run reaches full budget",
      all(not A.should_stop([0.02] * (e + 1), epoch=e)[0] for e in range(1, A.E_MAX - 1)))

print("\n[10b] score_shift is the non-degenerate signal")
s_a = two_gauss(2000, 0.2, 0.8, 0.15)
check("identical scores -> zero shift", A.score_shift(s_a, s_a) == 0.0)
moved = np.clip(s_a + 0.05, 0, 1)
check("a real move is detected", A.score_shift(s_a, moved) > 0.01, f"{A.score_shift(s_a, moved):.4f}")
# and the contrast that motivated the switch: quantile labels can be bit-identical
# while the scores underneath have moved substantially.
sharper = np.clip((s_a - 0.5) * 1.4 + 0.5, 0, 1)      # same ranking, sharper boundary
same_labels = np.array_equal(A.pseudo_labels(s_a, 0.3, 0.3), A.pseudo_labels(sharper, 0.3, 0.3))
check("labels frozen while scores move", same_labels and A.score_shift(s_a, sharper) > 0.01,
      f"labels identical={same_labels}, shift={A.score_shift(s_a, sharper):.4f}")

# ---------------------------------------------------------------------------
print("\n[11] epoch_churn ignores confident-set size changes from the ramp")
pl_a = np.array([0, 0, -1, -1, 1, 1])
pl_b = np.array([0, 0, 0, 1, 1, 1])      # ramp widened the set, no flips
check("widening alone is not churn", A.epoch_churn(pl_a, pl_b) == 0.0)
pl_c = np.array([1, 0, -1, -1, 1, 1])    # one real->fake flip out of 4 shared
check("a real flip is counted", abs(A.epoch_churn(pl_a, pl_c) - 0.25) < 1e-9)

# ---------------------------------------------------------------------------
print("\n[12] collapse guard")
check("constant scores flagged", A.collapsed(np.full(1000, 0.5))[1] == "score_std_collapse")
check("healthy scores not flagged", A.collapsed(two_gauss(1000, 0.1, 0.9, 0.05))[0] is False)
check("nan flagged", A.collapsed(np.r_[np.nan, RNG.random(999)])[1] == "non_finite_scores")

# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
if _fails:
    print(f"{len(_fails)} FAILED: {_fails}")
    raise SystemExit(1)
print("all checks passed")
