"""Cache each published checkpoint's source confusion matrix M[y, yhat].

M is what black-box shift estimation needs, and it depends only on the
checkpoint and its own labelled training corpus -- not on the target pool. So it
is computed once here (a few thousand clips, ~30 s on GPU) and reused by
analyze_prevalence_rules.py for every target prior, instead of being recomputed
inside every adaptation arm.

    python cache_source_M.py [--n 4000] [--out scores_protocol_a_public]
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

import protocol_a_public as PA
import public_ckpt_tta as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", default="scores_protocol_a_public")
    ap.add_argument("--ckpts", default=",".join(PA.BBSE_SOURCE))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    pools = {}
    for kind in set(PA.BBSE_SOURCE.values()):
        d = PA.build_bbse_source(kind)
        if d is not None:
            pools[kind] = d
            print(f"source pool '{kind}': {len(d)} clips {d.label.value_counts().to_dict()}")

    for name in a.ckpts.split(","):
        kind = PA.BBSE_SOURCE.get(name)
        if kind not in pools:
            print(f"{name}: no labelled training corpus on disk -- skipped")
            continue
        dst = os.path.join(a.out, f"{name}__M.npy")
        if os.path.exists(dst):
            print(f"{name}: {dst} exists")
            continue
        cfg = P.CHECKPOINTS[name]
        model, _, _ = P.build_model(name, dev, log=lambda *x: None)
        bal = pd.concat([g.sample(min(len(g), a.n // 2), random_state=0)
                         for _, g in pools[kind].groupby("label")]).reset_index(drop=True)
        s = PA.stream_score(model, bal.path.tolist(), cfg["crop"], cfg["fake_col"],
                            tag=f"{name}/M ")
        yh = (s >= 0.5).astype(int)
        y = bal.label.values.astype(int)
        M = np.array([[np.mean(yh[y == 0] == 0), np.mean(yh[y == 0] == 1)],
                      [np.mean(yh[y == 1] == 0), np.mean(yh[y == 1] == 1)]])
        np.save(dst, M)
        print(f"{name}: M = {M.round(4).tolist()}  -> {dst}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
