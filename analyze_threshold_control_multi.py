"""Pair the label-free threshold baselines (threshold_control_multi.csv) against
the adapted third-party model's accuracy at 0.5 (results_public_ckpt_multi.csv).

Same logic as the our-model control in threshold_control.py, at breadth:
re-thresholding cannot change EER/AUC and adaptation was measured not to change
them either, so raw accuracy@0.5 is the only axis on which TTA and a threshold
move can differ. If the median rule tracks TTA here as it does on our own model
(recovers ~94%), "the method is a label-free threshold rule" rests on the same
breadth as the calibration-deficit result.

    python analyze_threshold_control_multi.py
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

tc = pd.read_csv("threshold_control_multi.csv")
res = pd.read_csv("results_public_ckpt_multi.csv")
res = res[res.setting == "transductive"]

# adapted acc@0.5, seed 0, deduped by mean over any repeated runs
ada = (res[res.method == "ours"].groupby(["family", "target"], as_index=False)
       .agg(acc_tta=("acc", "mean")))
ada = ada.merge(
    res[(res.method == "ours") & (res.seed == 0)][["family", "target"]].drop_duplicates(),
    on=["family", "target"])

m = tc.merge(ada, on=["family", "target"], how="inner")
# replay + in-domain-lineage cells are excluded from the pooled claim
m = m[~m.target.isin(["asvspoof2021pa", "in_the_wild_natural"])]

print(f"paired cells: {len(m)}  ({m.family.nunique()} checkpoints, {m.target.nunique()} corpora)\n")

for name, col in [("shipped 0.5", "acc_shipped"), ("otsu", "acc_otsu"),
                  ("median", "acc_median"), ("TTA @0.5", "acc_tta"),
                  ("oracle (labels)", "acc_oracle")]:
    print(f"  {name:16s} pooled mean acc  {m[col].mean():6.2f}")
print()

for name, col in [("otsu", "acc_otsu"), ("median", "acc_median"),
                  ("oracle", "acc_oracle")]:
    d = m[col] - m["acc_tta"]
    try:
        p = wilcoxon(m[col], m["acc_tta"]).pvalue
    except ValueError:
        p = float("nan")
    print(f"  {name:8s} vs TTA:  delta {d.mean():+.2f}  wins {(d > 0).sum()}/{len(d)}  p={p:.3g}")

gain_tta = m.acc_tta.mean() - m.acc_shipped.mean()
gain_med = m.acc_median.mean() - m.acc_shipped.mean()
print(f"\n  TTA recovers {gain_tta:+.2f} acc pts over shipped; "
      f"median recovers {gain_med:+.2f} ({100 * gain_med / gain_tta:.0f}% of TTA)")

# per-target
print("\n  per target:")
for tgt, g in m.groupby("target"):
    print(f"    {tgt:20s} n={len(g):2d}  shipped {g.acc_shipped.mean():5.1f}  "
          f"median {g.acc_median.mean():5.1f}  TTA {g.acc_tta.mean():5.1f}  "
          f"oracle {g.acc_oracle.mean():5.1f}")
