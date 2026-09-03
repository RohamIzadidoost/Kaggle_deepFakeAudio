"""Does test-time adaptation beat simply moving the threshold?

The study's two independent lines of evidence agree that the method repairs the
decision threshold (p=5e-20 on 196 third-party cells, p=9.7e-11 on our own model
over 70) and does NOT measurably improve ranking (EER p=0.81 and p=0.80). That
raises a question the paper never asks: if the benefit is entirely in where the
threshold sits, why run gradient descent to get it?

This is the control. For each source checkpoint it scores the target pool once and
evaluates accuracy at four thresholds:

  t=0.5      the shipped threshold -- what the source model actually delivers
  t=otsu     the between-class-variance split of the score histogram; LABEL-FREE
  t=median   the split that predicts 50% positive; label-free, but assumes a
             balanced pool (ours are, by construction)
  t=oracle   the accuracy-maximising threshold, chosen WITH labels -- an upper
             bound no label-free rule can beat, not a method

and compares them against the accuracy the *adapted* model reaches at 0.5,
already recorded in results_adaptive.csv.

The comparison is internally consistent by construction: re-thresholding cannot
change EER or AUC, and adaptation was measured not to change them either. So
accuracy is the only axis on which the two can differ.

Two outcomes, both worth having:
  * threshold-shifting matches adaptation -> the method's entire benefit is
    obtainable with no training, no backprop and no 16,706 parameters.
  * adaptation wins -> it does something a threshold cannot, which is a far
    sharper claim than the paper currently makes.

    python threshold_control.py
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

OUT = "threshold_control.csv"


def otsu_threshold(s, bins=256):
    hist, edges = np.histogram(s, bins=bins, range=(0.0, 1.0))
    w = hist.astype(float) / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(w); w1 = 1.0 - w0
    m0 = np.cumsum(w * centers) / np.maximum(w0, 1e-12)
    mt = (w * centers).sum()
    m1 = (mt - np.cumsum(w * centers)) / np.maximum(w1, 1e-12)
    return float(centers[int(np.nanargmax(w0 * w1 * (m0 - m1) ** 2))])


def raw_acc(y, s, t):
    """Raw accuracy at threshold t, in percent.

    Must match adaptive_pipeline.metrics exactly -- it records
    `accuracy_score(y, s >= 0.5)`, i.e. RAW accuracy with a >= comparison. The
    adapted model's accuracy is read from that file, so computing balanced
    accuracy here instead would compare two different quantities and inflate or
    deflate the comparison wherever a pool is not exactly 50/50 (dataset2 is
    2274/2173, arabic 2570/3000).
    """
    return float(100 * ((s >= t).astype(int) == y).mean())


def bal_acc(y, s, t):
    """Balanced accuracy: mean of TPR and TNR. Reported alongside raw, not
    instead of it, so the pool-imbalance sensitivity of raw accuracy is visible."""
    p = (s >= t).astype(int)
    pos = y == 1
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return float(100 * (p == y).mean())
    tpr = (p[pos] == 1).mean()
    tnr = (p[neg] == 0).mean()
    return float(100 * 0.5 * (tpr + tnr))


def oracle_threshold(y, s, fn):
    """Accuracy-maximising threshold under `fn`, chosen with labels. Upper bound only."""
    cand = np.unique(np.round(s, 4))
    if len(cand) > 2000:
        cand = np.quantile(s, np.linspace(0, 1, 2000))
    best_t, best_a = 0.5, -1.0
    for t in cand:
        a = fn(y, s, t)
        if a > best_a:
            best_a, best_t = a, float(t)
    return best_t, best_a


def main():
    rows = []
    targets = AP.EER_TARGETS + ["asvspoof2021la", "asvspoof2021df", "asvspoof2021pa"]
    for target in targets:
        sub = AP.pool[AP.pool.corpus == target]
        if sub.empty:
            continue
        for seed in range(10):
            ck = f"ckpt_ext/source_{target}_seed{seed}.pt"
            if not os.path.exists(ck):
                continue
            tgt = AP.sample_target(sub, AP.TARGET_PER_CLASS, seed)
            idx = AP.idx_of(tgt)
            model = AP.XLSRDetector(encoder_amp=AP.ENCODER_AMP).to(AP.DEVICE)
            model.load_state_dict(torch.load(ck, map_location=AP.DEVICE), strict=False)
            m, y, s = AP.metrics(model, idx)
            del model; torch.cuda.empty_cache()
            y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)

            t_otsu = otsu_threshold(s)
            t_med = float(np.median(s))
            t_or, a_or = oracle_threshold(y, s, raw_acc)
            t_orb, a_orb = oracle_threshold(y, s, bal_acc)
            rows.append(dict(
                target=target, seed=seed, eer=m["eer"], auc=m["auc"],
                # raw accuracy: the convention results_adaptive.csv records
                acc_shipped=raw_acc(y, s, 0.5),
                acc_otsu=raw_acc(y, s, t_otsu),
                acc_median=raw_acc(y, s, t_med),
                acc_oracle=a_or,
                # balanced, reported alongside
                bacc_shipped=bal_acc(y, s, 0.5),
                bacc_otsu=bal_acc(y, s, t_otsu),
                bacc_median=bal_acc(y, s, t_med),
                bacc_oracle=a_orb,
                t_otsu=t_otsu, t_median=t_med, t_oracle=t_or))
            r = rows[-1]
            AP.log(f"  {target} s{seed}: shipped {r['acc_shipped']:5.1f}  "
                   f"otsu {r['acc_otsu']:5.1f}  median {r['acc_median']:5.1f}  "
                   f"oracle {r['acc_oracle']:5.1f}")
            pd.DataFrame(rows).to_csv(OUT, index=False)
    AP.log(f"wrote {OUT} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
