"""Stage R2: the ASVspoof-independent third-party arm at ten seeds.

WaveFake and 2024 commercial-TTS share no lineage with ASVspoof. Stage R ran
seeds 0-2 (underpowered: n=24/corpus, deficit closure p=0.24). R2 adds seeds
3-9. Deduped exactly as analyze_public_ckpt_multi.py (mean over repeated runs
per seed/target/family).

    python analyze_stage_r2.py
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

d = pd.read_csv("results_public_ckpt_multi.csv")
d = d[(d.setting == "transductive") &
      (d.target.isin(["hf_wavefake", "hf_commercialtts"]))]
keys = ["seed", "target", "family"]
src = (d[d.method == "source"].groupby(keys, as_index=False)
       .agg(se=("eer", "mean"), sa=("auc", "mean"), sc=("acc", "mean")))
ada = (d[d.method == "ours"].groupby(keys, as_index=False)
       .agg(ae=("eer", "mean"), aa=("auc", "mean"), ac=("acc", "mean")))
o = src.merge(ada, on=keys)
o["db"] = (100 - o.se) - o.sc
o["da"] = (100 - o.ae) - o.ac
o["egain"] = o.se - o.ae

print(f"total cells: {len(o)}  (seeds present: {sorted(o.seed.unique())})\n")
for tgt, g in o.groupby("target"):
    pe = wilcoxon(g.se, g.ae).pvalue if len(g) >= 6 else float("nan")
    pd_ = wilcoxon(g.db, g.da).pvalue if len(g) >= 6 else float("nan")
    print(f"  {tgt:16s} n={len(g):2d}  "
          f"EER {g.se.mean():5.2f}->{g.ae.mean():5.2f} ({g.egain.mean():+.2f}, p={pe:.3g})   "
          f"deficit {g.db.mean():5.2f}->{g.da.mean():5.2f} (p={pd_:.3g})")

pe = wilcoxon(o.se, o.ae).pvalue
pdd = wilcoxon(o.db, o.da).pvalue
print(f"\n  POOLED n={len(o)}  EER gain {o.egain.mean():+.2f} (p={pe:.3g})   "
      f"deficit {o.db.mean():.2f}->{o.da.mean():.2f} (p={pdd:.3g})")

print("\n  per checkpoint x corpus:")
for (f, t), g in o.groupby(["family", "target"]):
    print(f"    {f:26s} {t:16s} n={len(g):2d}  EER {g.egain.mean():+.2f}  "
          f"deficit {g.db.mean():5.2f}->{g.da.mean():5.2f}")
