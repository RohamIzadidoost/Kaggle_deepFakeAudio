"""Do label-free monitors predict TTA damage, without ever seeing a target label?

Two monitors, and they catch DIFFERENT failures -- neither alone is sufficient:

  rho        Spearman correlation between the adapted scores and the FROZEN
             source model's scores on the same pool. Catches RANKING collapse.
             Tent and ETA destroy the ranking (rho 0.44, 0.31) while leaving the
             operating point untouched.
  prior gap  |predicted positive rate - pi_hat|, pi_hat estimated once from the
             frozen source model. Catches OPERATING-POINT collapse. The
             symmetric-q configuration leaves the ranking nearly intact
             (rho 0.82, which a rho-only guard would pass) and moves the
             predicted positive rate from 0.97 to 0.59.

  Honest note: on a well-separated checkpoint the source confusion matrix M is
  near identity, so pi_hat ~= the source model's own predicted positive rate and
  the second monitor reduces to *drift from the source rate*. Still the right
  quantity, still label-free -- but the source row's gap of 0.000 is true by
  construction, not a result.

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
    ap.add_argument("--gap", type=float, default=0.05,
                    help="candidate keep/abort threshold on the prior gap")
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
        acc_src = float(np.mean((src >= 0.5).astype(int) == y)) * 100
        mp = os.path.join(d, f"{ckpt}__M.npy")
        pi_hat = None
        if os.path.exists(mp):
            M = np.load(mp)
            q = np.array([np.mean((src >= 0.5) == 0), np.mean((src >= 0.5) == 1)])
            try:
                pv = np.clip(np.linalg.solve(M.T, q), 1e-4, 1 - 1e-4)
                pi_hat = float(pv[1] / pv.sum())
            except np.linalg.LinAlgError:
                pi_hat = None
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
                pred_rate=round(float(np.mean(s >= 0.5)), 4),
                prior_gap=(round(abs(float(np.mean(s >= 0.5)) - pi_hat), 4)
                           if pi_hat is not None else np.nan),
                auc=round(auc, 4), d_auc=round(auc - auc_src, 4),
                eer=round(eer * 100, 3), d_eer=round((eer_src - eer) * 100, 3),
                acc=round(float(np.mean((s >= 0.5).astype(int) == y)) * 100, 2),
                d_acc=round(float(np.mean((s >= 0.5).astype(int) == y)) * 100
                            - acc_src, 2)))
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

    print(f"\ncombined guard: keep iff rho >= {a.tau} AND prior gap <= {a.gap}")
    ok = (adapt.rho_src >= a.tau) & (adapt.prior_gap <= a.gap)
    # "harmful" = it lost ranking OR lost the operating point. Both matter, and
    # a guard scored only on AUC would call the symmetric-q accuracy collapse a
    # false alarm.
    harmful = (adapt.d_auc < -0.002) | (adapt.d_acc < -2.0)
    for lab, m in (("KEEP ", ok), ("ABORT", ~ok)):
        for _, r in adapt[m].iterrows():
            print(f"  {lab} {r.ckpt.split('_')[-1]:>5s}/{r.arm:<18s} "
                  f"rho {r.rho_src:.3f} gap {r.prior_gap:.3f} -> "
                  f"d_auc {r.d_auc:+.4f} d_acc {r.d_acc:+.2f}")
    tp = int((harmful & ~ok).sum()); fn = int((harmful & ok).sum())
    fp = int((~harmful & ~ok).sum()); tn = int((~harmful & ok).sum())
    print(f"  caught {tp}/{int(harmful.sum())} harmful, missed {fn}, "
          f"false alarms {fp}, correctly kept {tn}")
    print("\n  rho alone would have kept: "
          f"{sorted(adapt[(adapt.rho_src >= a.tau) & harmful].arm.tolist())}")
    print("  gap alone would have kept: "
          f"{sorted(adapt[(adapt.prior_gap <= a.gap) & harmful].arm.tolist())}")

    print("\nCaveats: rho conflates genuine re-ordering (SHOT) with resolution "
          "collapse through score ties (Tent, ETA, largest_tie > 0.93); both are "
          "damage, so the guard is sound, but the mechanisms differ. Both "
          "thresholds are fitted on these arms, not validated out of sample.")


if __name__ == "__main__":
    main()
