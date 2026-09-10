"""Fit and validate a LABEL-FREE stopping rule for the adaptation budget.

Context (ICASSP_REVIEW_AND_PLAN.md, W4): the manuscript concedes that E is a
mis-specified budget rather than a hyperparameter, and that its strongest
results come from E=32, chosen after seeing the labelled outcome. That is an
oracle, and a reviewer will say so. This script asks whether the budget can be
picked from unlabeled statistics alone.

Input: `stop_trace.csv`, written by adaptive_pipeline.adapt_traced -- one row
per (seed, target, variant, epoch) with the labelled outcome (eer/auc/acc) AND
four label-free monitors (rho_src, prior_gap, churn, conf). The monitors never
read the labels; the labels are here only so we can score, after the fact, what
each rule would have chosen.

Validation is leave-one-target-out: every threshold is chosen on the other
targets and applied to the held-out one, so no rule is scored on data that set
its own constant.

    python analyze_stop_rule.py [stop_trace.csv]
"""
import itertools
import sys

import numpy as np
import pandas as pd

TRACE = sys.argv[1] if len(sys.argv) > 1 else "stop_trace.csv"


def curves(df):
    """{(seed, target, variant): frame sorted by epoch}"""
    return {k: g.sort_values("epoch").reset_index(drop=True)
            for k, g in df.groupby(["seed", "target", "variant"])}


# ------------------------------------------------------------------ the rules
def rule_ranking(g, tau):
    """Stop at the last epoch whose ranking still agrees with the source model.

    The paper's thesis is that the source RANKING is the asset that transfers.
    An adaptation that destroys it has discarded its own precondition -- and a
    collapse (Tent: AUC .918 -> .502) is exactly a ranking collapse. Spearman
    rho against the frozen source scores is computable without labels.
    """
    ok = g.epoch[(g.rho_src >= tau)].values
    return int(ok.max()) if len(ok) else 0


def rule_prior(g, delta):
    """Stop at the last epoch whose predicted positive rate still matches the
    BBSE prior estimate. Drifting away from the estimated prior means the model
    is re-labelling the pool rather than re-calibrating on it."""
    ok = g.epoch[(g.prior_gap <= delta)].values
    return int(ok.max()) if len(ok) else 0


def rule_churn(g, c):
    """First epoch whose hard-prediction churn drops below c -- the obvious
    convergence signal, included so the principled monitors have to beat it."""
    ok = g.epoch[(g.epoch > 0) & (g.churn <= c)].values
    return int(ok.min()) if len(ok) else int(g.epoch.max())


def rule_conf(g, c):
    """Stop when mean confidence exceeds c. Entropy minimisation MAXIMISES this,
    so it should be the monitor that fails; it is the negative control."""
    ok = g.epoch[(g.conf >= c)].values
    return int(ok.min()) if len(ok) else int(g.epoch.max())


def rule_both(g, p):
    tau, delta = p
    return min(rule_ranking(g, tau), rule_prior(g, delta))


RULES = {
    "ranking-guard": (rule_ranking, [round(x, 3) for x in np.arange(0.50, 1.0, 0.02)]),
    "prior-guard": (rule_prior, [round(x, 3) for x in np.arange(0.02, 0.40, 0.02)]),
    "churn": (rule_churn, [round(x, 3) for x in np.arange(0.005, 0.20, 0.005)]),
    "confidence (neg. control)": (rule_conf, [round(x, 3) for x in np.arange(0.60, 0.999, 0.02)]),
    "ranking+prior": (rule_both, list(itertools.product(
        [round(x, 3) for x in np.arange(0.50, 1.0, 0.05)],
        [round(x, 3) for x in np.arange(0.02, 0.40, 0.04)]))),
}


def eer_at(g, ep):
    row = g[g.epoch == ep]
    if not len(row):
        row = g[g.epoch == g.epoch.max()]
    return float(row.eer.iloc[0])


def evaluate(cs, variant, rule_fn, param):
    """Mean EER over all (seed, target) folds if this rule with this param were used."""
    out = []
    for (seed, target, v), g in cs.items():
        if v != variant:
            continue
        out.append((target, eer_at(g, rule_fn(g, param))))
    return out


def main():
    df = pd.read_csv(TRACE)
    cs = curves(df)
    variants = sorted(df.variant.unique())
    targets = sorted(df.target.unique())
    print(f"{TRACE}: {len(df)} rows, {len(cs)} curves, "
          f"variants={variants}, targets={targets}, "
          f"epochs 0..{int(df.epoch.max())}\n")

    for variant in variants:
        sub = {k: g for k, g in cs.items() if k[2] == variant}
        if not sub:
            continue
        print(f"===== variant: {variant}  ({len(sub)} folds) =====")
        # reference points
        ref = {}
        for name, pick in [("source (E=0)", lambda g: 0),
                           ("published E=4", lambda g: min(4, int(g.epoch.max()))),
                           ("full budget", lambda g: int(g.epoch.max())),
                           ("ORACLE argmin", lambda g: int(g.loc[g.eer.idxmin(), "epoch"]))]:
            vals = [eer_at(g, pick(g)) for g in sub.values()]
            ref[name] = float(np.mean(vals))
        for name, v in ref.items():
            print(f"  {name:>22s}  mean EER {v:6.3f}")

        print(f"  {'rule':>22s}  {'LOTO EER':>9s}  {'vs E=4':>7s}  "
              f"{'vs oracle':>9s}   chosen thresholds")
        for rname, (fn, grid) in RULES.items():
            loto_vals, chosen = [], []
            for held in targets:
                # choose the threshold on the OTHER targets only
                best, best_v = None, np.inf
                for prm in grid:
                    v = [e for (t, e) in evaluate(sub, variant, fn, prm) if t != held]
                    if not v:
                        continue
                    m = float(np.mean(v))
                    if m < best_v:
                        best, best_v = prm, m
                if best is None:
                    continue
                chosen.append((held, best))
                loto_vals += [e for (t, e) in evaluate(sub, variant, fn, best) if t == held]
            if not loto_vals:
                continue
            m = float(np.mean(loto_vals))
            print(f"  {rname:>22s}  {m:9.3f}  {m - ref['published E=4']:+7.3f}  "
                  f"{m - ref['ORACLE argmin']:+9.3f}   "
                  f"{ {t: p for t, p in chosen} }")
        # per-target detail for the best principled rule
        print()
        for target in targets:
            g_ = [g for (s, t, v), g in sub.items() if t == target]
            if not g_:
                continue
            e0 = np.mean([eer_at(g, 0) for g in g_])
            e4 = np.mean([eer_at(g, min(4, int(g.epoch.max()))) for g in g_])
            ef = np.mean([eer_at(g, int(g.epoch.max())) for g in g_])
            eo = np.mean([eer_at(g, int(g.loc[g.eer.idxmin(), "epoch"])) for g in g_])
            rho_end = np.mean([g.rho_src.iloc[-1] for g in g_])
            print(f"    {target:>16s} n={len(g_)}  source {e0:6.2f}  E=4 {e4:6.2f}  "
                  f"full {ef:6.2f}  oracle {eo:6.2f}  rho_src(end) {rho_end:.3f}")
        print()


if __name__ == "__main__":
    main()
