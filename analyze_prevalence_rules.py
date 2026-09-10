"""How label-free operating-point rules behave as the target prior moves.

Context (ICASSP_REVIEW_AND_PLAN.md, W1): the manuscript's strongest
self-criticism is that a one-line median-threshold rule recovers ~94% of the
accuracy gain, which reads as "the gradient method is unnecessary". Two things
are wrong with reading it that way, and this script measures the second:

  1. A threshold cannot change EER or AUC -- both are computed from the score
     ROC. The median rule's EER is *identical* to source-only, so it cannot
     explain a single number in the main table. It is a control for the
     accuracy claim, not a rival method. (No experiment needed; it is a
     definition.)
  2. The median rule was only ever evaluated on artificially balanced pools.
     Every cell in threshold_control_multi.csv sits at pos_rate 0.489-0.539,
     which is exactly where the median IS the optimal threshold. This script
     sweeps the target prior on real score distributions and shows what the
     rule does once the pool looks like deployment.

Rules compared, all label-free except the oracle:
    shipped   -- tau = 0.5, what the checkpoint ships with
    median    -- tau = median(scores); assumes a balanced pool
    bbse      -- tau = the (1 - pi_hat) quantile, pi_hat from black-box shift
                 estimation; assumes only that the source confusion matrix is
                 measurable on the checkpoint's own training data
    oracle    -- the labelled accuracy-optimal threshold (upper bound)

    python analyze_prevalence_rules.py scores_protocol_a_public [--corpus df2021]
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

RNG = np.random.RandomState(0)


def resample_prior(y, s, prior, n, rng):
    """Draw a pool of size n with P(fake) = prior, with replacement."""
    idx_f = np.where(y == 1)[0]
    idx_r = np.where(y == 0)[0]
    n_f = int(round(n * prior))
    n_r = n - n_f
    if n_f > len(idx_f) or n_r > len(idx_r):
        take_f = rng.choice(idx_f, n_f, replace=True)
        take_r = rng.choice(idx_r, n_r, replace=True)
    else:
        take_f = rng.choice(idx_f, n_f, replace=False)
        take_r = rng.choice(idx_r, n_r, replace=False)
    sel = np.concatenate([take_f, take_r])
    return y[sel], s[sel]


def bbse_prior(M, s, tau=0.5):
    """p = M^{-T} q, clipped to the simplex. M[y, yhat] from source data."""
    q = np.array([np.mean((s >= tau) == 0), np.mean((s >= tau) == 1)])
    try:
        p = np.linalg.solve(M.T, q)
    except np.linalg.LinAlgError:
        return 0.5
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(p[1] / p.sum())


def acc(y, s, tau):
    yh = (s >= tau).astype(int)
    return float((yh == y).mean()) * 100, float(balanced_accuracy_score(y, yh)) * 100


def rules(y, s, M):
    out = {}
    out["shipped"] = acc(y, s, 0.5)
    out["median"] = acc(y, s, float(np.median(s)))
    if M is not None:
        pi = bbse_prior(M, s)
        out["bbse"] = acc(y, s, float(np.quantile(s, 1 - pi)))
        out["_pi_hat"] = pi
    grid = np.quantile(s, np.linspace(0.001, 0.999, 400))
    best = max(grid, key=lambda t: (s >= t).astype(int).__eq__(y).mean())
    out["oracle"] = acc(y, s, float(best))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores_dir")
    ap.add_argument("--corpus", default="df2021")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    os.environ["PUBA_CORPUS"] = a.corpus
    import protocol_a_public as PA
    ev = PA.build_eval()
    y_all = ev.label.values.astype(int)
    true_prior = float(y_all.mean())
    print(f"{a.corpus}: {len(ev)} clips, true P(fake) = {true_prior:.4f}\n")

    priors = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, round(true_prior, 4)]
    rows = []
    for f in sorted(os.listdir(a.scores_dir)):
        if not f.endswith("__source.npy"):
            continue
        name = f[: -len("__source.npy")]
        s_all = np.load(os.path.join(a.scores_dir, f))
        if len(s_all) != len(y_all):
            print(f"  !! {name}: {len(s_all)} scores vs {len(y_all)} labels -- skipping")
            continue
        mpath = os.path.join(a.scores_dir, f"{name}__M.npy")
        M = np.load(mpath) if os.path.exists(mpath) else None
        if M is None:
            print(f"  {name}: no source confusion matrix cached "
                  f"({name}__M.npy) -- BBSE row omitted")
        for prior in priors:
            agg = {}
            for r in range(a.reps):
                rng = np.random.RandomState(1000 * r + 7)
                y, s = resample_prior(y_all, s_all, prior, a.n, rng)
                for k, v in rules(y, s, M).items():
                    agg.setdefault(k, []).append(v)
            row = dict(ckpt=name, prior=prior)
            for k, v in agg.items():
                if k == "_pi_hat":
                    row["pi_hat"] = round(float(np.mean(v)), 4)
                    row["pi_err"] = round(float(np.mean(v)) - prior, 4)
                else:
                    row[f"{k}_acc"] = round(float(np.mean([x[0] for x in v])), 2)
                    row[f"{k}_bacc"] = round(float(np.mean([x[1] for x in v])), 2)
            rows.append(row)
    df = pd.DataFrame(rows)
    if not len(df):
        print("no score files found")
        return
    out = a.out or f"prevalence_rules_{a.corpus}.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {out}")

    cols = [c for c in ("shipped_bacc", "median_bacc", "bbse_bacc", "oracle_bacc")
            if c in df.columns]
    print("\nmean BALANCED accuracy by target prior (the metric a majority-class "
          "predictor cannot win):")
    print(df.groupby("prior")[cols].mean().round(2).to_string())


if __name__ == "__main__":
    main()
