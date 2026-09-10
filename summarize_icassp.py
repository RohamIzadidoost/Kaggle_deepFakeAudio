"""Collect every result produced by the ICASSP strengthening pass into the
tables the manuscript needs. Read-only; no GPU.

    python summarize_icassp.py
"""
import os

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)


def hdr(t):
    print(f"\n{'='*78}\n{t}\n{'='*78}")


def wilcoxon(a, b):
    try:
        from scipy.stats import wilcoxon as w
        return float(w(a, b).pvalue)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------- published
def published_protocol_a():
    hdr("Published ASVspoof2021-DF pooled EER, for the anchor column")
    print(pd.DataFrame([
        ("Wav2DF-TSL", 1.95, "reported"),
        ("SSL + SLS classifier", 2.87, "reported"),
        ("Wav2Vec2-AASIST + RawBoost", 8.54, "reported"),
        ("ASVspoof2021 challenge DF top-1", 15.64, "challenge"),
        ("our own Protocol-A source model", 26.33, "results_protocol_a.csv"),
    ], columns=["system", "EER%", "source"]).to_string(index=False))


# ---------------------------------------------------------------- A1 / A2
def public_ckpt(corpus, path):
    if not os.path.exists(path):
        print(f"  ({path} not yet written)")
        return None
    try:
        d = pd.read_csv(path)
    except Exception:
        # written by the earlier append-blind record(); parse it positionally
        import protocol_a_public as PA
        d = PA._read_ragged(path)
    if "smoke" in d:
        d = d[d.smoke.astype(str).str.lower() != "true"]
    off = d[d.setting == "official_eval"]
    hdr(f"{corpus}: published checkpoints, whole benchmark")
    cols = ["ckpt", "method", "eer", "auc", "acc", "acc_median_rule", "deficit", "n"]
    extra = [c for c in ("pi_fake_hat", "pi_fake_true", "lo_frac", "hi_frac")
             if c in off.columns]
    print(off[cols + extra].to_string(index=False))

    piv = off.pivot_table(index="ckpt", columns="method",
                          values=["eer", "acc"], aggfunc="first")
    print("\npaired deltas vs source (positive = better):")
    rows = []
    for m in [x for x in off.method.unique() if x != "source"]:
        sub = off[off.method.isin(["source", m])].pivot_table(
            index="ckpt", columns="method", values=["eer", "acc"], aggfunc="first")
        if ("eer", m) not in sub or ("eer", "source") not in sub:
            continue
        de = (sub[("eer", "source")] - sub[("eer", m)]).dropna()
        da = (sub[("acc", m)] - sub[("acc", "source")]).dropna()
        rows.append(dict(method=m, n=len(de),
                         d_eer=round(float(de.mean()), 3),
                         d_acc=round(float(da.mean()), 2),
                         eer_better=int((de > 0).sum()),
                         p_eer=round(wilcoxon(sub[("eer", "source")].dropna(),
                                              sub[("eer", m)].dropna()), 4)))
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    # the median rule, which cannot move EER at all
    src = off[off.method == "source"]
    print(f"\nlabel-free threshold rules on the SOURCE model "
          f"({corpus}, true P(fake) implied by the pool):")
    print(src[["ckpt", "acc", "acc_median_rule"]].assign(
        median_minus_shipped=lambda x: (x.acc_median_rule - x.acc).round(2)
    ).to_string(index=False))
    return d


# ---------------------------------------------------------------- baselines
def baselines():
    if not os.path.exists("results_adaptive.csv"):
        return
    d = pd.read_csv("results_adaptive.csv")
    d = d[d.setting == "transductive"]
    fam = ["source", "ours_fixed", "tent", "shot", "eta", "eata", "sar"]
    d = d[d.method.isin(fam)]
    if d.method.nunique() < 3:
        print("\n  (modern TTA baselines not yet run)")
        return
    hdr("Modern TTA baselines vs ours, leave-one-corpus-out (mean over seeds)")
    piv = d.pivot_table(index="method", columns="target", values="eer", aggfunc="mean")
    piv = piv.reindex([m for m in fam if m in piv.index])
    print(piv.round(2).to_string())
    print("\nAUC:")
    pa = d.pivot_table(index="method", columns="target", values="auc", aggfunc="mean")
    print(pa.reindex([m for m in fam if m in pa.index]).round(3).to_string())
    print("\nper-seed spread (std of EER), the stability claim:")
    ps = d.pivot_table(index="method", columns="target", values="eer", aggfunc="std")
    print(ps.reindex([m for m in fam if m in ps.index]).round(2).to_string())

    hdr("Paired, per (target, seed), ours vs each baseline")
    rows = []
    w = d.pivot_table(index=["target", "seed"], columns="method", values="eer")
    if "ours_fixed" not in w:
        return
    for m in [x for x in fam if x not in ("ours_fixed",)]:
        if m not in w:
            continue
        pair = w[["ours_fixed", m]].dropna()
        if not len(pair):
            continue
        rows.append(dict(baseline=m, n=len(pair),
                         ours=round(pair.ours_fixed.mean(), 2),
                         theirs=round(pair[m].mean(), 2),
                         ours_wins=int((pair.ours_fixed < pair[m]).sum()),
                         p=round(wilcoxon(pair.ours_fixed, pair[m]), 4)))
    print(pd.DataFrame(rows).to_string(index=False))


# ---------------------------------------------------------------- stop rule
def stop_rule():
    if not os.path.exists("stop_trace.csv"):
        print("\n  (stop traces not yet run)")
        return
    d = pd.read_csv("stop_trace.csv")
    hdr("Budget traces: does the ranking monitor see the collapse?")
    for variant in sorted(d.variant.unique()):
        s = d[d.variant == variant]
        agg = s.groupby("epoch")[["eer", "auc", "rho_src", "prior_gap", "conf"]].mean()
        keep = [e for e in (0, 1, 2, 4, 8, 16, 24, 32) if e in agg.index]
        print(f"\n{variant}:")
        print(agg.loc[keep].round(4).to_string())


def main():
    published_protocol_a()
    public_ckpt("ASVspoof2021-DF (official eval)", "results_protocol_a_public.csv")
    public_ckpt("In-the-Wild (all 31,779)", "results_protocol_a_public_itw.csv")
    baselines()
    stop_rule()
    for f in ("prevalence_rules_df2021.csv", "prevalence_rules_itw.csv"):
        if os.path.exists(f):
            hdr(f"Label-free threshold rules vs target prior ({f})")
            df = pd.read_csv(f)
            cols = [c for c in df.columns if c.endswith("_bacc") or c in
                    ("prior", "pi_hat", "pi_err")]
            print(df.groupby("prior")[
                [c for c in cols if c != "prior"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
