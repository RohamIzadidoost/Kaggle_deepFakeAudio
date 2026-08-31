"""Turn the multi-target public-checkpoint grid into the AUC-vs-gain evidence.

The ICASSP paper's precondition claim -- adaptation pays where the source model's
ranking already transfers -- currently rests on four corpora scored with ONE
model (ours). That makes the +0.86 correlation *ecological*: it is a relation
between corpus means, and within each corpus the sign reverses (-0.77, -0.64,
-0.64, +0.12). A reviewer can fairly say four points cannot carry a precondition.

This script re-tests the same relation on a grid that varies the *model* as well
as the corpus, using four third-party checkpoints we did not train. That buys
three things one model on four corpora cannot:

  1. more points, from independent training corpora and independent authors;
  2. **within-model, across-corpus** slopes -- one per checkpoint. If the
     relation holds inside a single fixed model as the corpus changes, it is no
     longer only a between-cluster artifact;
  3. **within-corpus, across-model** slopes -- one per corpus. If adaptation
     pays more for whichever model ranks better *on the same clips*, the
     precondition is about ranking, not about the corpus being easy.

Reports all three, and does not hide the disagreements between them.

    python analyze_public_ckpt_multi.py
    python analyze_public_ckpt_multi.py --fig fig_auc_gain_public.png
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

OURS_LABEL = "xlsr_ours"


def load_pairs(path, pool_note):
    """Pair every `source` row with the `ours` row from the same run."""
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df.setting == "transductive"]
    keys = ["seed", "target", "family"]
    src = (df[df.method == "source"].groupby(keys, as_index=False)
           .agg(src_eer=("eer", "mean"), src_auc=("auc", "mean"),
                src_acc=("acc", "mean"), n=("n", "max")))
    ada = (df[df.method == "ours"].groupby(keys, as_index=False)
           .agg(ada_eer=("eer", "mean"), ada_auc=("auc", "mean"),
                ada_acc=("acc", "mean")))
    out = src.merge(ada, on=keys, how="inner")
    out["pool"] = pool_note
    return out


def load_ours(path="results_ext.csv", family="xlsr"):
    """Our own source model's four corpora, for side-by-side comparison.

    `results_ext.csv` holds TWO families: `xlsr` (the real source model) and
    `rawnet2lite` (the non-SSL backbone baseline, which never reaches a ranking
    worth adapting). Averaging over both inflates source EER wildly -- it turned
    ASVspoof2019's 5.03% into 26.82% -- while the `ours` rows are xlsr-only and
    so still looked right. Filter the family explicitly; do not group without it.
    """
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[(df.setting == "transductive") & (df.family == family)]
    src = (df[df.method == "source"].groupby(["seed", "target"], as_index=False)
           .agg(src_eer=("eer", "mean"), src_auc=("auc", "mean"),
                src_acc=("acc", "mean"), n=("n", "max")))
    ada = (df[df.method == "ours"].groupby(["seed", "target"], as_index=False)
           .agg(ada_eer=("eer", "mean"), ada_auc=("auc", "mean"),
                ada_acc=("acc", "mean")))
    out = src.merge(ada, on=["seed", "target"], how="inner")
    # sanity: these must reproduce the manuscript's headline source EERs
    # (ASVspoof2019 5.03, In-the-Wild 12.78, Arabic 22.50, dataset2 33.54)
    ref = {"asvspoof2019": 5.03, "in_the_wild": 12.78, "arabic": 22.50, "dataset2": 33.54}
    means = out.groupby("target").src_eer.mean()
    for tgt, want in ref.items():
        if tgt in means and abs(means[tgt] - want) > 0.5:
            raise SystemExit(
                f"results_ext.csv source EER for {tgt} is {means[tgt]:.2f}, expected "
                f"~{want} from main_icassp.tex. Wrong family filter or wrong file.")
    out["family"] = OURS_LABEL
    out["pool"] = "extended_pipeline target pool"
    return out


def corr(x, y):
    """Pearson r with n, and Spearman as the rank-robust companion."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return dict(n=len(x), r=np.nan, p=np.nan, rho=np.nan, p_rho=np.nan)
    r, p = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    return dict(n=len(x), r=r, p=p, rho=rho, p_rho=p_rho)


def fmt(c):
    if np.isnan(c["r"]):
        return f"n={c['n']:<3} (too few points / no variance)"
    return (f"n={c['n']:<3} r={c['r']:+.2f} (p={c['p']:.3f})   "
            f"rho={c['rho']:+.2f} (p={c['p_rho']:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi", default="results_public_ckpt_multi.csv")
    ap.add_argument("--itw", default="results_public_ckpt.csv",
                    help="the published full-pool In-the-Wild arm")
    ap.add_argument("--include_itw_fullpool", action="store_true",
                    help="also include the 31,779-clip ITW rows (different pool, "
                         "so off by default to keep one pool rule per plot)")
    ap.add_argument("--fig", default="fig_auc_gain_public.png")
    ap.add_argument("--out", default="public_ckpt_multi_summary.csv")
    args = ap.parse_args()

    pub = load_pairs(args.multi, "extended_pipeline target pool")
    if args.include_itw_fullpool:
        pub = pd.concat([pub, load_pairs(args.itw, "full ITW pool (31,779)")])
    if pub.empty:
        raise SystemExit(f"no rows in {args.multi} yet -- has the grid run?")

    pub["gain"] = pub.src_eer - pub.ada_eer          # positive = adaptation helped
    pub["auc_gain"] = pub.ada_auc - pub.src_auc
    pub["acc_gain"] = pub.ada_acc - pub.src_acc      # calibration at the 0.5 threshold
    pub["rel_gain"] = pub.gain / pub.src_eer         # fraction of source EER removed

    ours = load_ours()
    if not ours.empty:
        ours["gain"] = ours.src_eer - ours.ada_eer
        ours["auc_gain"] = ours.ada_auc - ours.src_auc
        ours["acc_gain"] = ours.ada_acc - ours.src_acc

    pd.set_option("display.width", 200)
    print("=" * 92)
    print("PER-CELL RESULTS  (public checkpoints, transductive, positive gain = EER reduced)")
    print("=" * 92)
    show = pub.sort_values(["family", "src_auc"])[
        ["family", "target", "seed", "n", "src_eer", "ada_eer", "gain", "src_auc", "ada_auc", "auc_gain"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    print()
    print("=" * 92)
    print("THE RELATION: does source AUC predict the EER gain?")
    print("=" * 92)

    print("\n[A] Pooled over every (model, corpus) cell -- the headline n-extension")
    print("    " + fmt(corr(pub.src_auc, pub.gain)))

    print("\n[A2] The same test on outcome measures that are NOT bounded by headroom.")
    print("     Absolute EER-point gain is mechanically limited by how much EER")
    print("     there was to remove: a cell at 0.47% source EER cannot gain 3 points")
    print("     however well it ranks. Since source EER and source AUC are strongly")
    print("     coupled, [A] partly measures that bound rather than the hypothesis.")
    print("     Relative reduction and AUC/accuracy change do not share the bound:")
    print(f"     src_auc vs relative EER reduction   " + fmt(corr(pub.src_auc, pub.rel_gain)))
    print(f"     src_auc vs AUC change               " + fmt(corr(pub.src_auc, pub.auc_gain)))
    print(f"     src_auc vs acc@0.5 change           " + fmt(corr(pub.src_auc, pub.acc_gain)))
    print(f"     (for reference, src_auc vs src_eer  " + fmt(corr(pub.src_auc, pub.src_eer)) + ")")

    print("\n[B] WITHIN MODEL, across corpora (one row per checkpoint).")
    print("    This is the axis the paper's own n=4 already covers, but now")
    print("    repeated on models we did not train:")
    for fam, g in pub.groupby("family"):
        print(f"    {fam:<28} " + fmt(corr(g.src_auc, g.gain)))
    if not ours.empty:
        per_target = ours.groupby("target", as_index=False).agg(
            src_auc=("src_auc", "mean"), gain=("gain", "mean"))
        print(f"    {OURS_LABEL + ' (corpus means)':<28} " + fmt(corr(per_target.src_auc, per_target.gain)))

    print("\n[C] WITHIN CORPUS, across models (one row per corpus).")
    print("    Same clips, different detectors -- isolates ranking quality from")
    print("    corpus difficulty, which the paper's n=4 cannot do at all:")
    for tgt, g in pub.groupby("target"):
        print(f"    {tgt:<28} " + fmt(corr(g.src_auc, g.gain)))

    print("\n[D] Direction check, model-agnostic:")
    helped = (pub.gain > 0).sum()
    print(f"    adaptation reduced EER in {helped}/{len(pub)} (model, corpus) cells")
    hi = pub[pub.src_auc >= 0.90]
    lo = pub[pub.src_auc < 0.90]
    for name, g in (("source AUC >= 0.90", hi), ("source AUC <  0.90", lo)):
        if len(g):
            print(f"    {name}: mean gain {g.gain.mean():+.2f} EER pts "
                  f"({(g.gain > 0).sum()}/{len(g)} improved), "
                  f"mean AUC change {g.auc_gain.mean():+.4f}")

    print("\n[F] Ranking vs. calibration, separated.")
    print("    EER and AUC measure ranking; acc@0.5 measures whether the decision")
    print("    threshold is in the right place. The paper's claim is that transfer")
    print("    breaks calibration while leaving ranking largely intact, so a cell")
    print("    with ~0 EER gain but a large acc@0.5 gain is the thesis, not a null:")
    for _, r in pub.sort_values("acc_gain", ascending=False).iterrows():
        print(f"    {r.family:<28} {r.target:<14} "
              f"EER {r.src_eer:6.2f} -> {r.ada_eer:6.2f} ({r.gain:+5.2f})   "
              f"acc@0.5 {r.src_acc:5.1f} -> {r.ada_acc:5.1f} ({r.acc_gain:+5.1f})")
    print(f"    mean acc@0.5 change {pub.acc_gain.mean():+.1f} pts "
          f"({(pub.acc_gain > 0).sum()}/{len(pub)} improved); "
          f"mean EER change {pub.gain.mean():+.2f} pts")

    if not ours.empty:
        print("\n[E] Our own source model on the same four corpora, for reference:")
        ref = ours.groupby("target", as_index=False).agg(
            src_eer=("src_eer", "mean"), ada_eer=("ada_eer", "mean"),
            gain=("gain", "mean"), src_auc=("src_auc", "mean"))
        print(ref.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    pub.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        markers = {"arabic": "o", "dataset2": "s", "in_the_wild": "^", "asvspoof2019": "D"}
        colors = {f: c for f, c in zip(sorted(pub.family.unique()),
                                       plt.cm.viridis(np.linspace(0.1, 0.85, pub.family.nunique())))}
        for (fam, tgt), g in pub.groupby(["family", "target"]):
            ax.scatter(g.src_auc, g.gain, marker=markers.get(tgt, "o"), s=70,
                       color=colors[fam], edgecolor="k", linewidth=0.5,
                       label=None, zorder=3)
        # within-model trend lines: the axis that answers the ecological objection
        for fam, g in pub.groupby("family"):
            if len(g) >= 2:
                o = g.sort_values("src_auc")
                ax.plot(o.src_auc, o.gain, "-", color=colors[fam], alpha=0.55,
                        linewidth=1.4, label=fam, zorder=2)
        if not ours.empty:
            pt = ours.groupby("target", as_index=False).agg(
                src_auc=("src_auc", "mean"), gain=("gain", "mean"))
            ax.scatter(pt.src_auc, pt.gain, marker="*", s=260, color="crimson",
                       edgecolor="k", linewidth=0.6, label="ours (XLS-R), corpus mean", zorder=4)
        ax.axhline(0, color="k", linewidth=0.8, linestyle=":")
        ax.set_xlabel("source AUC on the target pool (ranking that survives transfer)")
        ax.set_ylabel("EER gain from adaptation (pts, + = better)")
        ax.set_title("Adaptation gain vs. source ranking, across models and corpora")
        ax.legend(fontsize=7.5, loc="best", framealpha=0.9)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.fig, dpi=200)
        print(f"wrote {args.fig}  (marker = corpus, colour = checkpoint)")
    except Exception as e:
        print(f"(figure skipped: {type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
