"""Does rank agreement with the source model predict TTA damage, without labels?

The paper's thesis is that the source model's RANKING is the asset test-time
adaptation exploits. If so, an adaptation that destroys the ranking has thrown
away its own precondition, and the destruction should be visible without target
labels -- by comparing the adapted scores' ORDER against the frozen source
model's order on the same pool.

This script tests that on the official ASVspoof2021-DF eval, across every
adaptation arm we ran (ours, the published symmetric-q configuration, Tent,
SHOT, ETA, SAR). rho is computable at deployment; the AUC change it is scored
against is not. If the two track, a deployer can decide whether to keep an
adapted model without ever labelling a target clip.

    python analyze_rank_monitor.py [--corpus df2021] [--tau 0.85]
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score

from metrics import compute_eer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="df2021")
    ap.add_argument("--scores_dir", default=None)
    ap.add_argument("--tau", type=float, default=0.85,
                    help="candidate keep/abort threshold on rho")
    ap.add_argument("--tau_sub", type=int, default=40000,
                    help="subsample for Kendall tau, which is O(n log n) but heavy")
    a = ap.parse_args()
    os.environ["PUBA_CORPUS"] = a.corpus
    import protocol_a_public as PA
    d = a.scores_dir or (f"scores_protocol_a_public"
                         + ("" if a.corpus == "df2021" else f"_{a.corpus}"))
    ev = PA.build_eval()
    y = ev.label.values.astype(int)
    rng = np.random.RandomState(0)
    sub = rng.choice(len(y), min(a.tau_sub, len(y)), replace=False)

    rows = []
    for ckpt in sorted({f.split("__")[0] for f in os.listdir(d) if f.endswith(".npy")}):
        sp = os.path.join(d, f"{ckpt}__source.npy")
        if not os.path.exists(sp):
            continue
        src = np.load(sp)
        if len(src) != len(y):
            continue
        auc_src = roc_auc_score(y, src)
        eer_src, _ = compute_eer(y, src)
        for f in sorted(os.listdir(d)):
            if not f.endswith(".npy") or "__M" in f or not f.startswith(ckpt + "__"):
                continue
            arm = f[:-4].split("__", 1)[1]
            s = np.load(os.path.join(d, f))
            if len(s) != len(y):
                continue
            auc = roc_auc_score(y, s)
            eer, _ = compute_eer(y, s)
            rows.append(dict(
                ckpt=ckpt, arm=arm,
                rho_src=round(float(spearmanr(s, src).statistic), 4),
                tau_b=round(float(kendalltau(s[sub], src[sub], variant="b").statistic), 4),
                largest_tie=round(float(pd.Series(s).value_counts().iloc[0] / len(s)), 4),
                auc=round(auc, 4), d_auc=round(auc - auc_src, 4),
                eer=round(eer * 100, 3), d_eer=round((eer_src - eer) * 100, 3)))
    df = pd.DataFrame(rows)
    if not len(df):
        print(f"no score dumps in {d}")
        return
    df = df.sort_values(["ckpt", "rho_src"])
    out = f"rank_monitor_{a.corpus}.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))

    adapt = df[df.arm != "source"]
    print(f"\nlabel-free monitor vs the labelled outcome ({len(adapt)} adapted arms):")
    for m in ("rho_src", "tau_b"):
        print(f"  {m:8s} vs d_auc  pearson {np.corrcoef(adapt[m], adapt.d_auc)[0,1]:+.3f}"
              f"   spearman {spearmanr(adapt[m], adapt.d_auc).statistic:+.3f}")

    print(f"\nas a keep/abort guard at rho >= {a.tau}:")
    kept = adapt[adapt.rho_src >= a.tau]
    dropped = adapt[adapt.rho_src < a.tau]
    print(f"  KEEP   {sorted(kept.arm)}  -> d_auc {list(kept.d_auc)}")
    print(f"  ABORT  {sorted(dropped.arm)}  -> d_auc {list(dropped.d_auc)}")
    harmful = set(adapt[adapt.d_auc < 0].arm)
    caught = harmful & set(dropped.arm)
    missed = harmful - set(dropped.arm)
    false_alarm = set(dropped.arm) - harmful
    print(f"  harmful arms (d_auc<0): {sorted(harmful)}")
    print(f"  caught {sorted(caught)}; missed {sorted(missed)}; "
          f"false alarms {sorted(false_alarm)}")
    print("\nCaveat: rho conflates two distinct failure modes -- genuine "
          "re-ordering (SHOT) and resolution collapse through score ties "
          "(Tent, ETA, whose largest_tie exceeds 0.93). Both are damage, so the "
          "guard is sound, but the mechanisms differ and the threshold here is "
          "fitted on these arms, not validated out of sample.")


if __name__ == "__main__":
    main()
