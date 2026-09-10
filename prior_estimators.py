"""Label-free target-prior estimators, compared on real cached detector scores.

BBSE needs a labelled sample of the checkpoint's own training distribution to
build the confusion matrix M. That is usually available to a deployer, but not
always -- three of the nine third-party checkpoints in this study have
undocumented training data and are excluded from the BBSE arm for exactly that
reason. So it is worth knowing whether a source-data-free estimator does as
well.

Estimators:
  bbse   p = M^{-T} q                            (needs labelled source data)
  sld    Saerens-Latinne-Decaestecker EM         (needs only the SOURCE PRIOR)
  gmm    2-component Gaussian mixture weight     (needs nothing)
  meanp  mean predicted probability              (needs nothing; assumes
                                                  calibration, which is the one
                                                  property this paper says does
                                                  NOT transfer)

    python prior_estimators.py
"""
import numpy as np


def bbse(M, s, tau=0.5):
    q = np.array([np.mean((s >= tau) == 0), np.mean((s >= tau) == 1)])
    try:
        p = np.linalg.solve(M.T, q)
    except np.linalg.LinAlgError:
        return 0.5
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(p[1] / p.sum())


def sld(s, pi_source, iters=200, tol=1e-8):
    """EM prior estimation (Saerens et al. 2002).

    Re-weights the source posteriors by the current prior ratio, renormalises,
    and takes the mean as the next prior. Needs only `pi_source`, the class
    balance of the training set, which a model card states even when the corpus
    itself is not shipped.
    """
    p0 = np.clip(np.stack([1 - s, s], 1), 1e-6, 1 - 1e-6)
    pi_s = np.array([1 - pi_source, pi_source])
    pi = pi_s.copy()
    for _ in range(iters):
        w = p0 * (pi / pi_s)[None, :]
        w = w / w.sum(1, keepdims=True)
        new = w.mean(0)
        if np.abs(new - pi).max() < tol:
            pi = new
            break
        pi = new
    return float(pi[1])


def gmm(s, seed=0):
    from sklearn.mixture import GaussianMixture
    g = GaussianMixture(2, random_state=seed, n_init=3).fit(s.reshape(-1, 1))
    return float(g.weights_[int(np.argmax(g.means_.flatten()))])


def meanp(s):
    return float(s.mean())


if __name__ == "__main__":
    import os
    import pandas as pd
    import protocol_a_public as PA
    from analyze_prevalence_rules import resample_prior

    ev = PA.build_df_eval()
    y_all = ev.label.values.astype(int)
    la19 = PA.build_la19_train()
    pi_source = float(la19.label.mean())     # ASVspoof2019-LA train is 89.8% spoof
    print(f"DF eval {len(ev)} clips, true P(fake)={y_all.mean():.4f}; "
          f"source prior (2019-LA train) = {pi_source:.4f}\n")

    rows = []
    for f in sorted(os.listdir("scores_protocol_a_public")):
        if not f.endswith("__source.npy"):
            continue
        name = f[: -len("__source.npy")]
        s_all = np.load(f"scores_protocol_a_public/{f}")
        if len(s_all) != len(y_all):
            continue
        mp = f"scores_protocol_a_public/{name}__M.npy"
        M = np.load(mp) if os.path.exists(mp) else None
        for prior in (0.5, 0.7, 0.9, 0.95, float(round(y_all.mean(), 4))):
            est = {"bbse": [], "sld": [], "gmm": [], "meanp": []}
            for r in range(5):
                rng = np.random.RandomState(1000 * r + 7)
                _, s = resample_prior(y_all, s_all, prior, 20000, rng)
                if M is not None:
                    est["bbse"].append(bbse(M, s))
                est["sld"].append(sld(s, pi_source))
                est["gmm"].append(gmm(s))
                est["meanp"].append(meanp(s))
            row = dict(ckpt=name, prior=prior)
            for k, v in est.items():
                if v:
                    row[k] = round(float(np.mean(v)), 4)
                    row[f"{k}_err"] = round(float(np.mean(v)) - prior, 4)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("prior_estimators_df2021.csv", index=False)
    print(df.to_string(index=False))
    print("\nmean |error| over all priors and checkpoints:")
    print(df[[c for c in df.columns if c.endswith("_err")]].abs().mean().round(4).to_string())
