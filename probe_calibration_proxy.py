"""Is there a LABEL-FREE proxy for the calibration deficit?

The deficit `(100 - EER) - acc@0.5` predicts the adaptation gain far better than
source AUC does (r=+0.56, rho=+0.69 over 70 cells, against AUC's r=-0.12). But it
is computed from EER and accuracy, both of which need labels -- so as it stands it
explains results after the fact and cannot tell a deployer whether to adapt.

This scores each source checkpoint on its target pool and records statistics of
the score distribution ALONE. Nothing here touches a label; labels are read only
afterwards, to compute the deficit each proxy is being tested against.

Candidate proxies, weakest assumption last:

* `pred_rate`  -- fraction scored above the 0.5 threshold. Interpretable only
  against an expected prevalence, so it assumes the pool is roughly balanced
  (ours are, by construction). The weakest of the three assumptions.
* `near_thresh` -- fraction of mass in [0.4, 0.6]. A well-placed threshold sits
  in a low-density valley; a badly placed one cuts through a mode. Needs no
  prevalence assumption.
* `otsu_gap`   -- |t* - 0.5| where t* is the Otsu (between-class variance)
  split of the score histogram. This is the threshold the score distribution
  itself implies; how far the shipped 0.5 sits from it is a miscalibration
  signal that needs neither labels nor a prevalence assumption. The one worth
  hoping for.

    python probe_calibration_proxy.py
"""

import os

os.environ.setdefault("ADAPTIVE_SMOKE", "0")
os.environ.setdefault("CACHE_ON_CPU", "1")
os.environ.setdefault("INCLUDE_MLAAD", "1")
os.environ.setdefault("NEW_TARGETS", "1")

import numpy as np
import pandas as pd
import torch

import adaptive_pipeline as AP

OUT = "calibration_proxy.csv"


def otsu_threshold(s, bins=256):
    """Between-class-variance split of a 1-D score distribution. Label-free."""
    hist, edges = np.histogram(s, bins=bins, range=(0.0, 1.0))
    w = hist.astype(float) / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(w)
    w1 = 1.0 - w0
    m0 = np.cumsum(w * centers) / np.maximum(w0, 1e-12)
    mt = (w * centers).sum()
    m1 = (mt - np.cumsum(w * centers)) / np.maximum(w1, 1e-12)
    var_b = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.nanargmax(var_b))])


def main():
    rows = []
    for target in AP.EER_TARGETS + ["asvspoof2021la", "asvspoof2021df", "asvspoof2021pa"]:
        sub = AP.pool[AP.pool.corpus == target]
        if sub.empty:
            AP.log(f"  {target}: not in pool, skipping"); continue
        for seed in range(10):
            ckpt = f"ckpt_ext/source_{target}_seed{seed}.pt"
            if not os.path.exists(ckpt):
                continue
            tgt = AP.sample_target(sub, AP.TARGET_PER_CLASS, seed)
            idx = AP.idx_of(tgt)
            model = AP.XLSRDetector(encoder_amp=AP.ENCODER_AMP).to(AP.DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=AP.DEVICE), strict=False)
            m, y, s = AP.metrics(model, idx)
            del model; torch.cuda.empty_cache()

            s = np.asarray(s, dtype=float)
            t_star = otsu_threshold(s)
            rows.append(dict(
                target=target, seed=seed,
                # label-free
                pred_rate=float((s > 0.5).mean()),
                near_thresh=float(((s > 0.4) & (s < 0.6)).mean()),
                otsu=t_star, otsu_gap=abs(t_star - 0.5),
                score_mean=float(s.mean()), score_median=float(np.median(s)),
                score_std=float(s.std()),
                # labels used ONLY to build the target these proxies are tested against
                eer=m["eer"], auc=m["auc"], acc=m["acc"],
                deficit=(100 - m["eer"]) - m["acc"]))
            AP.log(f"  {target} s{seed}: pred_rate {rows[-1]['pred_rate']:.3f}  "
                   f"otsu {t_star:.3f}  deficit {rows[-1]['deficit']:+.2f}")
            pd.DataFrame(rows).to_csv(OUT, index=False)
    AP.log(f"wrote {OUT} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
