"""ICASSP R2 vulnerability 2 -- the class-balance precondition, CPU falsification.

The stage-P skew sweep (results_adaptive.csv, setting=skew0.7/skew0.9) showed the
published method AND the median-threshold rule both collapse when the target pool
is not ~50/50: raw-accuracy gain over source goes +6.4 / -3.5 / -26.5 at
50 / 70 / 90 % fake. Deepfakes in the wild are a minority class, so a reviewer
asks whether the method is usable at all off the balanced grid.

Both failures share one mechanism: the median predicts exactly 50 % positive and
the symmetric q=0.3 tails label the bottom 30 % "confident real" regardless of the
true prior. This script tests, on CPU, whether estimating the target prevalence
label-free and re-pointing BOTH knobs at it recovers the skewed case without
hurting the balanced one.

  Strategy A -- 2-component GMM on the target scores (adaptive_tta.estimate_prevalence_guarded)
  Strategy B -- BBSE: p_target = M^{-1} q, M the source-val confusion matrix
  control    -- oracle prevalence (upper bound on any prevalence-based rule)

Threshold rules compared (raw + balanced accuracy):
  shipped 0.5 | median (0.5-quantile) | prev-quantile at each prevalence estimate
Pseudo-label quality compared:
  symmetric q=0.3 tails vs adaptive_tta.tail_budget(0.3, pi_hat) asymmetric tails

Data: the real saved In-the-Wild source scores (scores_in_the_wild_source.npz,
2000 clips, balanced, labelled) plus synthetic logit-normal score ensembles
spanning source AUC {0.65,0.75,0.85,0.95} -- the range the four LOCO targets and
the third-party checkpoints actually show.

No GPU, no torch. Writes r2_vuln2_prevalence_cpu.csv and prints the analysis.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

import adaptive_tta as AT

RNG = np.random.default_rng(0)
SKEWS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
AUCS = [0.65, 0.75, 0.85, 0.95]
N_POOL = 6000            # matches the LOCO target-pool size
N_BOOT = 40              # bootstrap resamples per (source, skew)
Q = 0.30


# ----------------------------------------------------------------- score models
def synth_scores(auc, n, rng):
    """Class-conditional logit-normal scores with a given ROC-AUC.

    s = sigmoid(z), z ~ N(+mu, 1) for fake, N(-mu, 1) for real. For unit-variance
    Gaussians AUC = Phi(2 mu / sqrt(2)) = Phi(mu sqrt(2)), so mu = Phi^{-1}(auc)/sqrt(2).
    n is the per-class count of the balanced generating pool.
    """
    mu = stats.norm.ppf(auc) / np.sqrt(2.0)
    z_fake = rng.normal(+mu, 1.0, n)
    z_real = rng.normal(-mu, 1.0, n)
    s = 1.0 / (1.0 + np.exp(-np.concatenate([z_real, z_fake])))
    y = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    return y, s


def load_itw():
    d = np.load("scores_in_the_wild_source.npz")
    return d["y"].astype(int), d["s"].astype(float)


# ------------------------------------------------------------------- estimators
def bbse_prevalence(s_cal, y_cal, s_tgt, tau=0.5):
    """Black-Box Shift Estimation of P_target(fake).

    M[y, yhat] = P_src(yhat | y) from a labelled calibration split; q = observed
    target prediction histogram; p = M^{-1} q, clipped to a simplex.
    """
    yhat_cal = (s_cal >= tau).astype(int)
    M = np.zeros((2, 2))
    for y in (0, 1):
        m = y_cal == y
        if m.sum() == 0:
            return np.nan
        M[y, 0] = np.mean(yhat_cal[m] == 0)
        M[y, 1] = np.mean(yhat_cal[m] == 1)
    # rows index true class, columns predicted; we need   q_hat = M^T p
    q_hat = np.array([np.mean((s_tgt >= tau) == 0), np.mean((s_tgt >= tau) == 1)])
    try:
        p = np.linalg.solve(M.T, q_hat)
    except np.linalg.LinAlgError:
        return np.nan
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return float(p[1] / p.sum())


def gmm_prevalence(s_tgt):
    pi_real, bic_delta, shrunk = AT.estimate_prevalence_guarded(s_tgt)
    return 1.0 - pi_real, bic_delta, shrunk


# ----------------------------------------------------------------------- accuracy
def raw_acc(y, s, t):
    return 100.0 * np.mean((s >= t).astype(int) == y)


def bal_acc(y, s, t):
    p = (s >= t).astype(int)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return 100.0 * np.mean(p == y)
    return 100.0 * 0.5 * (np.mean(p[pos] == 1) + np.mean(p[neg] == 0))


def oracle_t(y, s, fn):
    cand = np.quantile(s, np.linspace(0.01, 0.99, 199))
    accs = [fn(y, s, t) for t in cand]
    return float(cand[int(np.argmax(accs))])


def pl_purity(y, s, lo_frac, hi_frac):
    """Fraction of confident-tail pseudo-labels that are correct, + coverage."""
    pl = AT.pseudo_labels(s, lo_frac, hi_frac)
    real_sel, fake_sel = pl == 0, pl == 1
    pr = np.mean(y[real_sel] == 0) if real_sel.sum() else np.nan
    pf = np.mean(y[fake_sel] == 1) if fake_sel.sum() else np.nan
    return pr, pf, int(real_sel.sum()), int(fake_sel.sum())


# ---------------------------------------------------------------------- one cell
def make_skewed(y, s, skew, rng, n_pool=N_POOL):
    """Resample (with replacement) to a pool that is `skew` fraction fake."""
    idx_f = np.where(y == 1)[0]
    idx_r = np.where(y == 0)[0]
    n_f = int(round(n_pool * skew))
    n_r = n_pool - n_f
    take = np.concatenate([rng.choice(idx_r, n_r, replace=True),
                           rng.choice(idx_f, n_f, replace=True)])
    rng.shuffle(take)
    return y[take], s[take]


def run_source(name, y0, s0):
    rows = []
    # a fixed labelled calibration split for BBSE: half of the balanced pool
    cal_rng = np.random.default_rng(12345)
    perm = cal_rng.permutation(len(y0))
    cal = perm[: len(y0) // 2]
    s_cal, y_cal = s0[cal], y0[cal]

    for skew in SKEWS:
        for b in range(N_BOOT):
            rng = np.random.default_rng((hash((name, skew, b)) & 0xFFFFFFFF))
            y, s = make_skewed(y0, s0, skew, rng)
            auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan

            pi_gmm, bic_delta, shrunk = gmm_prevalence(s)
            pi_bbse = bbse_prevalence(s_cal, y_cal, s, tau=0.5)
            pi_pred = float(np.mean(s >= 0.5))
            pi_true = skew

            rules = {
                "shipped": 0.5,
                "median": float(np.median(s)),
                "prev_oracle": float(np.quantile(s, 1.0 - pi_true)),
                "prev_gmm": float(np.quantile(s, np.clip(1.0 - pi_gmm, 0.01, 0.99))),
                "prev_bbse": float(np.quantile(s, np.clip(1.0 - pi_bbse, 0.01, 0.99)))
                if np.isfinite(pi_bbse) else np.nan,
                "prev_predrate": float(np.quantile(s, np.clip(1.0 - pi_pred, 0.01, 0.99))),
            }
            row = dict(source=name, auc_src=round(auc, 4), skew=skew, boot=b,
                       pi_true=pi_true, pi_gmm=round(pi_gmm, 4),
                       pi_bbse=round(pi_bbse, 4) if np.isfinite(pi_bbse) else np.nan,
                       pi_predrate=round(pi_pred, 4),
                       bic_delta=round(bic_delta, 2), gmm_shrunk=bool(shrunk))
            for rn, t in rules.items():
                row[f"racc_{rn}"] = round(raw_acc(y, s, t), 2) if np.isfinite(t) else np.nan
                row[f"bacc_{rn}"] = round(bal_acc(y, s, t), 2) if np.isfinite(t) else np.nan
            row["racc_orateN"] = round(raw_acc(y, s, oracle_t(y, s, raw_acc)), 2)
            row["bacc_orateN"] = round(bal_acc(y, s, oracle_t(y, s, bal_acc)), 2)

            # pseudo-label quality: symmetric vs prevalence-split tails
            pr_s, pf_s, nr_s, nf_s = pl_purity(y, s, Q, Q)
            lo_g, hi_g = AT.tail_budget(Q, 1.0 - pi_gmm)   # tail_budget takes pi_real
            pr_g, pf_g, nr_g, nf_g = pl_purity(y, s, lo_g, hi_g)
            pi_b = pi_bbse if np.isfinite(pi_bbse) else pi_true
            lo_b, hi_b = AT.tail_budget(Q, 1.0 - pi_b)
            pr_b, pf_b, nr_b, nf_b = pl_purity(y, s, lo_b, hi_b)
            row.update(plpur_real_sym=round(pr_s, 4), plpur_fake_sym=round(pf_s, 4),
                       pln_real_sym=nr_s, pln_fake_sym=nf_s,
                       plpur_real_gmm=round(pr_g, 4) if np.isfinite(pr_g) else np.nan,
                       plpur_fake_gmm=round(pf_g, 4) if np.isfinite(pf_g) else np.nan,
                       pln_real_gmm=nr_g, pln_fake_gmm=nf_g,
                       plpur_real_bbse=round(pr_b, 4) if np.isfinite(pr_b) else np.nan,
                       plpur_fake_bbse=round(pf_b, 4) if np.isfinite(pf_b) else np.nan,
                       pln_real_bbse=nr_b, pln_fake_bbse=nf_b)
            rows.append(row)
    return rows


def main():
    all_rows = []
    y_itw, s_itw = load_itw()
    all_rows += run_source("itw_real", y_itw, s_itw)
    for auc in AUCS:
        y0, s0 = synth_scores(auc, 6000, np.random.default_rng(int(auc * 1000)))
        all_rows += run_source(f"synth_auc{auc}", y0, s0)

    df = pd.DataFrame(all_rows)
    df.to_csv("r2_vuln2_prevalence_cpu.csv", index=False)
    print(f"wrote r2_vuln2_prevalence_cpu.csv  ({len(df)} rows)\n")

    # ---- 1. do the estimators track true prevalence? -----------------------
    print("=== 1. prevalence estimation error (|pi_hat - pi_true|, mean over boots) ===")
    df["ae_gmm"] = (df.pi_gmm - df.pi_true).abs()
    df["ae_bbse"] = (df.pi_bbse - df.pi_true).abs()
    df["ae_predrate"] = (df.pi_predrate - df.pi_true).abs()
    est_tab = df.groupby("skew")[["ae_gmm", "ae_bbse", "ae_predrate"]].mean().round(3)
    print(est_tab)
    for est in ("pi_gmm", "pi_bbse", "pi_predrate"):
        d = df.dropna(subset=[est])
        r, p = stats.pearsonr(d[est], d.pi_true)
        print(f"  corr({est}, pi_true) = {r:+.3f}  (p={p:.1e}, n={len(d)})")

    # ---- 2. threshold-rule raw accuracy by skew ---------------------------
    print("\n=== 2. RAW accuracy by threshold rule and skew (mean over sources+boots) ===")
    rules = ["shipped", "median", "prev_oracle", "prev_gmm", "prev_bbse",
             "prev_predrate", "orateN"]
    acc_tab = df.groupby("skew")[[f"racc_{r}" for r in rules]].mean().round(1)
    acc_tab.columns = rules
    print(acc_tab)

    print("\n=== 3. BALANCED accuracy by threshold rule and skew ===")
    bacc_tab = df.groupby("skew")[[f"bacc_{r}" for r in rules]].mean().round(1)
    bacc_tab.columns = rules
    print(bacc_tab)

    # ---- 4. paired test: prev_gmm vs median, per skew --------------------
    print("\n=== 4. prev_gmm vs median  (raw acc, paired Wilcoxon over sources x boots) ===")
    for skew in SKEWS:
        d = df[df["skew"] == skew]
        diff = d.racc_prev_gmm - d.racc_median
        try:
            w, p = stats.wilcoxon(d.racc_prev_gmm, d.racc_median)
        except ValueError:
            p = np.nan
        print(f"  skew {skew}:  median {d.racc_median.mean():5.1f}  "
              f"prev_gmm {d.racc_prev_gmm.mean():5.1f}  "
              f"delta {diff.mean():+5.1f}  p={p:.1e}  win {int((diff>0).sum())}/{len(d)}")

    print("\n=== 4b. prev_oracle vs median (upper bound on any prevalence rule) ===")
    for skew in SKEWS:
        d = df[df["skew"] == skew]
        diff = d.racc_prev_oracle - d.racc_median
        print(f"  skew {skew}:  delta {diff.mean():+5.1f}  "
              f"(median {d.racc_median.mean():5.1f} -> oracle-prev {d.racc_prev_oracle.mean():5.1f})")

    # ---- 5. does the balanced case get hurt? ----------------------------
    print("\n=== 5. balanced-case (skew=0.5) sanity: all rules should coincide ===")
    d = df[df["skew"] == 0.5]
    for r in rules:
        print(f"  racc_{r:14s} {d[f'racc_{r}'].mean():5.1f}   bacc {d[f'bacc_{r}'].mean():5.1f}")

    # ---- 6. pseudo-label purity: symmetric vs prevalence-split ----------
    print("\n=== 6. pseudo-label purity (real-tail), symmetric q=0.3 vs GMM-prevalence tails ===")
    pp = df.groupby("skew").agg(
        pur_real_sym=("plpur_real_sym", "mean"),
        n_real_sym=("pln_real_sym", "mean"),
        pur_real_gmm=("plpur_real_gmm", "mean"),
        n_real_gmm=("pln_real_gmm", "mean"),
        pur_real_bbse=("plpur_real_bbse", "mean"),
        n_real_bbse=("pln_real_bbse", "mean"),
        pur_fake_sym=("plpur_fake_sym", "mean"),
        pur_fake_bbse=("plpur_fake_bbse", "mean"),
    ).round(3)
    print(pp)


if __name__ == "__main__":
    main()

