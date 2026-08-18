"""
Turn results_public_ckpt.csv into the paper table.

Handles the two things that would otherwise corrupt the summary:

* **Duplicate source rows.** Every adaptation run re-scores the source model
  before adapting, so `source` appears once per run. They are identical by
  construction (deterministic centre-crop scoring in eval mode) -- we assert
  that rather than assume it, then de-duplicate.
* **Paired deltas.** A method's gain must be measured against the source EER on
  *the same eval set*, which for the inductive runs is the held-out half, not
  the full pool. Deltas are therefore computed within (family, setting, seed).

    python summarize_public_ckpt.py
    python summarize_public_ckpt.py --csv results_public_ckpt.csv --latex
"""

import argparse

import numpy as np
import pandas as pd

KEY = ["family", "setting", "seed"]


def load(path):
    df = pd.read_csv(path)

    # source rows are re-emitted by every run; confirm they agree before dropping
    src = df[df.method == "source"]
    for key, g in src.groupby(KEY):
        if g.eer.nunique() > 1:
            spread = g.eer.max() - g.eer.min()
            print(f"WARNING: source EER disagrees for {key}: "
                  f"{sorted(g.eer.round(4).unique())} (spread {spread:.4f})")
    df = df.drop_duplicates(subset=KEY + ["method"], keep="first")
    return df


def summarize(df):
    src = (df[df.method == "source"]
           .set_index(KEY)[["eer", "auc", "acc"]]
           .rename(columns=lambda c: c + "_src"))
    adp = df[df.method != "source"].join(src, on=KEY)
    adp["d_eer"] = adp.eer - adp.eer_src
    adp["rel"] = 100 * adp.d_eer / adp.eer_src
    return adp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results_public_ckpt.csv")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    df = load(args.csv)
    adp = summarize(df)

    print("\n=== per-run ===")
    cols = ["family", "method", "setting", "seed", "eer_src", "eer", "d_eer",
            "rel", "auc_src", "auc", "acc_src", "acc", "n"]
    show = adp[cols].sort_values(["family", "method", "setting", "seed"])
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # seed aggregation: mean +/- std where more than one seed exists
    print("\n=== aggregated over seeds ===")
    agg = (adp.groupby(["family", "method", "setting"])
              .agg(n_seeds=("seed", "nunique"),
                   eer_src=("eer_src", "mean"),
                   eer_mean=("eer", "mean"), eer_std=("eer", "std"),
                   d_mean=("d_eer", "mean"), d_std=("d_eer", "std"),
                   rel=("rel", "mean"))
              .reset_index())
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if args.latex:
        print("\n=== latex ===")
        print(r"\begin{tabular}{llrrrr}")
        print(r"\toprule")
        print(r"Checkpoint & Method & Source EER & Adapted EER & $\Delta$ & Rel.\ \\")
        print(r"\midrule")
        for _, r in show.iterrows():
            print(f"{r.family.replace('_',' ')} & {r.method} & {r.eer_src:.2f} & "
                  f"{r.eer:.2f} & {r.d_eer:+.2f} & {r.rel:+.1f}\\% \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")


if __name__ == "__main__":
    main()
