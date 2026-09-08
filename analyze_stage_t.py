"""Stage T: does E=32 reverse the two significant AST degradations, beyond seed 0?

hf_ast_asv19 on in_the_wild (-3.79 EER at E=4, the study's largest significant
degradation) and dataset2 (-2.14). Stage G2 flipped both positive at E=32 but at
seed 0 only. Stage T adds seeds 1-5. E=4 source/ours rows at seeds 0-9 already
exist (stage D).

    python analyze_stage_t.py
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

d = pd.read_csv("results_public_ckpt_multi.csv")
d = d[d.family == "hf_ast_asv19"]

for tgt in ["in_the_wild", "dataset2"]:
    g = d[d.target == tgt]
    e4 = g[g.setting == "transductive"]
    e32 = g[g.setting == "transductive_E32"]

    def paired(frame):
        s = frame[frame.method == "source"].groupby("seed").eer.mean()
        o = frame[frame.method == "ours"].groupby("seed").eer.mean()
        j = pd.DataFrame({"src": s, "ada": o}).dropna()
        j["gain"] = j.src - j.ada
        return j

    j4, j32 = paired(e4), paired(e32)
    print(f"\n=== hf_ast_asv19 / {tgt} ===")
    print(f"  E=4  (n={len(j4)}): gain {j4.gain.mean():+.2f}  seeds {sorted(j4.index)}")
    print(f"  E=32 (n={len(j32)}): gain {j32.gain.mean():+.2f}  seeds {sorted(j32.index)}")
    if len(j32) >= 6:
        p = wilcoxon(j32.src, j32.ada).pvalue
        print(f"  E=32 gain vs source: p={p:.4g}  ({(j32.gain > 0).sum()}/{len(j32)} positive)")
    common = j4.index.intersection(j32.index)
    if len(common) >= 6:
        diff = j32.loc[common, "gain"] - j4.loc[common, "gain"]
        p = wilcoxon(j32.loc[common, "gain"], j4.loc[common, "gain"]).pvalue
        print(f"  E=32 beats E=4 by {diff.mean():+.2f} on {len(common)} shared seeds, "
              f"p={p:.4g} ({(diff > 0).sum()}/{len(common)})")
