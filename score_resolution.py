"""Score-resolution diagnostics for every cached arm.

Why this exists: on the official ASVspoof2021-DF eval, Tent returns an EER of
4.06% -- apparently better than the 4.51% source model it started from. It is
not. Tent saturates 98.07% of the 400,435 scores to exactly 1.0, leaving 1,247
distinct values in the whole pool; the "EER" is then a property of how the ROC
breaks ties, and AUC (which cannot be rescued by tie-breaking) falls .9923 ->
.9765. Reporting EER alone would have recorded an entropy-minimisation collapse
as an improvement.

    python score_resolution.py [scores_dir] [--corpus df2021]
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores_dir", nargs="?", default="scores_protocol_a_public")
    ap.add_argument("--corpus", default="df2021")
    a = ap.parse_args()
    os.environ["PUBA_CORPUS"] = a.corpus
    import protocol_a_public as PA
    from sklearn.metrics import roc_auc_score
    from metrics import compute_eer

    ev = PA.build_eval()
    y = ev.label.values.astype(int)
    rows = []
    for f in sorted(os.listdir(a.scores_dir)):
        if not f.endswith(".npy") or "__M" in f:
            continue
        s = np.load(os.path.join(a.scores_dir, f))
        if len(s) != len(y):
            continue
        ckpt, arm = f[:-4].split("__", 1)
        eer, _ = compute_eer(y, s)
        # ties at the numerical ceiling/floor are where entropy minimisation
        # hides: they cost AUC but can leave EER looking healthy
        rows.append(dict(
            ckpt=ckpt, arm=arm, n=len(s),
            eer=round(eer * 100, 3), auc=round(float(roc_auc_score(y, s)), 4),
            n_unique=int(len(np.unique(s))),
            pct_at_max=round(float(np.mean(s >= 1 - 1e-6)) * 100, 2),
            pct_at_min=round(float(np.mean(s <= 1e-6)) * 100, 2),
            largest_tie_pct=round(float(pd.Series(s).value_counts().iloc[0]
                                        / len(s)) * 100, 2)))
    df = pd.DataFrame(rows).sort_values(["ckpt", "arm"])
    out = f"score_resolution_{a.corpus}.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {out}")
    bad = df[df.largest_tie_pct > 50]
    if len(bad):
        print("\n!! arms whose scores are >50% a single tied value -- their EER "
              "is a tie-breaking artefact and must not be read as a ranking:")
        print(bad[["ckpt", "arm", "eer", "auc", "largest_tie_pct", "n_unique"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
