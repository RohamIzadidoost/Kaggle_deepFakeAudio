"""ICASSP R2 vulnerability 1 -- the "one-line rule" equivalence, sub-population test.

A label-free median-threshold shift is statistically indistinguishable from the
full gradient TTA over 133 pooled cells. A reviewer asks why the gradient method
is needed. The median rule moves ONE global threshold across a possibly-distorted
score space; it cannot change any ROC-AUC. Gradient TTA adapts LayerNorm + head,
so it moves the embedding and CAN change per-sub-population separability.

This measures exactly that. For targets that carry a `generator` column
(ASVspoof2019: A01-A06; dataset2: 8 TTS systems) we compute, per generator
sub-population g, the ROC-AUC and EER on {all real clips} u {g's fakes}, before
and after adaptation. The paired test across sub-populations answers:

  * does adaptation raise per-sub-population AUC?  (something a threshold cannot)
  * or is per-sub-population AUC flat, i.e. the whole effect is the threshold?

Reuses adaptive_pipeline.adapt() (published config) and ckpt_ext/ source models.
Writes r2_vuln1_subpop.csv.

    SUBPOP_TARGETS=asvspoof2019,dataset2 SUBPOP_SEEDS=0,1,2 python r2_vuln1_subpop.py
"""
import os

os.environ.setdefault("ADAPTIVE_SMOKE", "0")
os.environ.setdefault("CACHE_ON_CPU", "1")
os.environ.setdefault("INCLUDE_MLAAD", "1")
os.environ.setdefault("NEW_TARGETS", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import adaptive_pipeline as AP

OUT = "r2_vuln1_subpop.csv"
TARGETS = os.environ.get("SUBPOP_TARGETS", "asvspoof2019,dataset2").split(",")
SEEDS = [int(x) for x in os.environ.get("SUBPOP_SEEDS", "0,1,2").split(",")]
MIN_G = 20  # skip a generator sub-population smaller than this


def eer(y, s):
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    return float(brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)) * 100


def subpop_metrics(y, s, gen, real_mask):
    """AUC/EER per generator: real clips vs that generator's fakes only."""
    rows = []
    s_real, y_real = s[real_mask], y[real_mask]
    for g in sorted(set(gen[~real_mask])):
        gm = (~real_mask) & (gen == g)
        if gm.sum() < MIN_G:
            continue
        ss = np.concatenate([s_real, s[gm]])
        yy = np.concatenate([y_real, y[gm]])
        rows.append(dict(generator=g, n_fake=int(gm.sum()),
                         auc=float(roc_auc_score(yy, ss)), eer=eer(yy, ss)))
    return rows


def main():
    dev = AP.DEVICE
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.target, prev.seed))

    rows = []
    for target in TARGETS:
        sub = AP.pool[AP.pool.corpus == target]
        if sub.empty:
            AP.log(f"skip {target}: not in pool")
            continue
        if sub.generator.nunique() < 2:
            AP.log(f"skip {target}: only {sub.generator.nunique()} generator(s)")
            continue
        for seed in SEEDS:
            ck = f"ckpt_ext/source_{target}_seed{seed}.pt"
            if not os.path.exists(ck):
                AP.log(f"skip {target} s{seed}: no {ck}")
                continue
            if (target, seed) in done:
                AP.log(f"skip {target} s{seed}: already in {OUT}")
                continue

            tgt = AP.sample_target(sub, AP.TARGET_PER_CLASS, seed)
            idx = AP.idx_of(tgt)
            gen = tgt.generator.to_numpy()
            lab = tgt.label.to_numpy()
            real_mask = (lab == "real") if lab.dtype == object else (lab == 0)

            model = AP.XLSRDetector(encoder_amp=AP.ENCODER_AMP).to(dev)
            model.load_state_dict(torch.load(ck, map_location=dev), strict=False)

            m0, y0, s0 = AP.metrics(model, idx)
            for r in subpop_metrics(y0, s0, gen, real_mask):
                rows.append(dict(target=target, seed=seed, phase="source",
                                 overall_auc=m0["auc"], overall_eer=m0["eer"], **r))

            model = AP.adapt(model, idx)                       # published config
            m1, y1, s1 = AP.metrics(model, idx)
            for r in subpop_metrics(y1, s1, gen, real_mask):
                rows.append(dict(target=target, seed=seed, phase="adapted",
                                 overall_auc=m1["auc"], overall_eer=m1["eer"], **r))

            del model
            torch.cuda.empty_cache()
            AP.log(f"{target} s{seed}: overall AUC {m0['auc']:.4f} -> {m1['auc']:.4f}  "
                   f"EER {m0['eer']:.2f} -> {m1['eer']:.2f}")
            pd.DataFrame(rows).to_csv(OUT, mode="a", index=False,
                                     header=not os.path.exists(OUT))
            rows = []

    AP.log("done")


if __name__ == "__main__":
    main()
