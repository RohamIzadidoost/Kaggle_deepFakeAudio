"""Median/Otsu threshold control on the THIRD-PARTY breadth grid.

`threshold_control.py` asked, on our own model over 70 cells, whether the
method's whole benefit is obtainable by moving the decision threshold with no
training. Answer there: a median-score threshold recovers ~94% of the accuracy
gain. This script runs the identical control on the nine public checkpoints, so
the "the method is a label-free threshold rule" claim rests on the same breadth
(196+ cells, five architectures) as the calibration-deficit result it explains.

Scoring only -- no adaptation, no gradients. For each (checkpoint, corpus) cell
present in results_public_ckpt_multi.csv it scores the seed-0 target pool once
and records raw + balanced accuracy at four thresholds:

  0.5      the shipped threshold
  otsu     between-class-variance split of the score histogram   (label-free)
  median   the split predicting 50% positive; label-free, assumes a balanced
           pool -- the third-party pools are balanced by construction
  oracle   accuracy-maximising threshold, chosen WITH labels -- an upper bound

The adapted model's accuracy at 0.5 is already in results_public_ckpt_multi.csv
(`acc` column, method='ours'); analyze_threshold_control_multi.py pairs the two.

    python threshold_control_multi.py            # all cells, seed 0
    CELLS=hf_ast_asv19:in_the_wild python threshold_control_multi.py   # one cell
"""

import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import pandas as pd
import torch

import public_ckpt_tta as P

OUT = "threshold_control_multi.csv"
BATCH = 16
# replay is a different task (pooled separately everywhere); the *_natural
# variant is not part of any pooled claim. Both excluded, as in the findings.
SKIP_TARGETS = {"asvspoof2021pa", "in_the_wild_natural"}


def otsu_threshold(s, bins=256):
    hist, edges = np.histogram(s, bins=bins, range=(0.0, 1.0))
    w = hist.astype(float) / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(w)
    w1 = 1.0 - w0
    m0 = np.cumsum(w * centers) / np.maximum(w0, 1e-12)
    mt = (w * centers).sum()
    m1 = (mt - np.cumsum(w * centers)) / np.maximum(w1, 1e-12)
    return float(centers[int(np.nanargmax(w0 * w1 * (m0 - m1) ** 2))])


def raw_acc(y, s, t):
    return float(100 * ((s >= t).astype(int) == y).mean())


def bal_acc(y, s, t):
    p = (s >= t).astype(int)
    pos = y == 1
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return float(100 * (p == y).mean())
    return float(100 * 0.5 * ((p[pos] == 1).mean() + (p[neg] == 0).mean()))


def oracle_threshold(y, s, fn):
    cand = np.unique(np.round(s, 4))
    if len(cand) > 2000:
        cand = np.quantile(s, np.linspace(0, 1, 2000))
    best_t, best_a = 0.5, -1.0
    for t in cand:
        a = fn(y, s, float(t))
        if a > best_a:
            best_a, best_t = a, float(t)
    return best_t, best_a


def cells_from_results():
    d = pd.read_csv("results_public_ckpt_multi.csv")
    d = d[(d.setting == "transductive") & (d.method == "source")]
    return sorted(set(zip(d.family, d.target)))


def manifest_for(target):
    return f"manifest_tgt_{target}_seed0.csv"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open("threshold_control_multi.out", "a") as f:
        f.write(line + "\n")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.family, prev.target))
        log(f"resuming: {len(done)} cells already in {OUT}")

    want = os.environ.get("CELLS")
    cells = cells_from_results()
    if want:
        wl = {tuple(c.split(":")) for c in want.split(",")}
        cells = [c for c in cells if c in wl]

    rows = []
    for family, target in cells:
        if target in SKIP_TARGETS:
            continue
        if (family, target) in done:
            continue
        cfg = P.CHECKPOINTS[family]
        if target in cfg.get("trained_on_corpora", ()):
            log(f"skip {family}/{target}: in-domain")
            continue
        mf = manifest_for(target)
        if not os.path.exists(mf):
            log(f"skip {family}/{target}: no manifest {mf}")
            continue

        t0 = time.time()
        df = pd.read_csv(mf)
        try:
            model, _, _ = P.build_model(family, device)
            buf, y = P.build_cache(df, cfg["crop"], log=lambda *_: None)
            idx = torch.arange(len(df))
            s = P.score(model, buf, idx, cfg["fake_col"], BATCH, device)
        except Exception as e:                       # noqa: BLE001
            log(f"FAIL {family}/{target}: {e!r}")
            del_model = locals().get("model")
            if del_model is not None:
                del del_model
            torch.cuda.empty_cache()
            continue
        del model
        torch.cuda.empty_cache()

        y = np.asarray(y).astype(int)
        s = np.asarray(s, dtype=float)
        eer, _ = P.compute_eer(y, s)
        from sklearn.metrics import roc_auc_score
        t_otsu = otsu_threshold(s)
        t_med = float(np.median(s))
        _, a_or = oracle_threshold(y, s, raw_acc)
        _, a_orb = oracle_threshold(y, s, bal_acc)
        rows.append(dict(
            family=family, target=target, seed=0,
            n=len(y), pos_rate=float(y.mean()),
            eer=eer * 100, auc=float(roc_auc_score(y, s)),
            acc_shipped=raw_acc(y, s, 0.5), acc_otsu=raw_acc(y, s, t_otsu),
            acc_median=raw_acc(y, s, t_med), acc_oracle=a_or,
            bacc_shipped=bal_acc(y, s, 0.5), bacc_otsu=bal_acc(y, s, t_otsu),
            bacc_median=bal_acc(y, s, t_med), bacc_oracle=a_orb,
            t_otsu=t_otsu, t_median=t_med))
        r = rows[-1]
        log(f"{family}/{target}: shipped {r['acc_shipped']:5.1f}  otsu {r['acc_otsu']:5.1f}  "
            f"median {r['acc_median']:5.1f}  oracle {r['acc_oracle']:5.1f}  "
            f"({(time.time()-t0)/60:.1f} min)")
        pd.DataFrame(rows).to_csv(
            OUT, mode="a", index=False, header=not os.path.exists(OUT))
        rows = []

    log("done")


if __name__ == "__main__":
    main()
