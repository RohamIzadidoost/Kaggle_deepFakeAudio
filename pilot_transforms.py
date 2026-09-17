"""Which view transformation destroyed the spoofing evidence?

The first pilot built every non-standard view from an offset crop *and* a gain
*and* additive noise at once, and every one of them roughly doubled EER. That
tells us the bundle is unusable but not which ingredient is at fault, so this
script scores the same clips under each ingredient in isolation. A view family
worth building on must leave EER close to the standard crop.

Note that gain+noise is precisely the perturbation the existing consistency
term applies during adaptation, so its cost here is of independent interest.

    PILOT_N=10000 python pilot_transforms.py
"""
import os
import time

import numpy as np
import torch

import public_ckpt_tta as P
from pilot_views import full_waveform, log

N = int(os.environ.get("PILOT_N", "10000"))
CKPT = os.environ.get("PILOT_CKPT", "ssl_aasist_wavefake")
CORPUS = os.environ.get("PILOT_CORPUS", "itw")
SEED = int(os.environ.get("PILOT_SEED", "0"))
BATCH = int(os.environ.get("PILOT_BATCH", "32"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def window(x, crop, frac):
    """Crop at a fractional offset through the clip, wrapping if short."""
    if len(x) == 0:
        return np.zeros(crop, dtype=np.float32)
    if len(x) < crop:
        x = np.tile(x, int(crop / len(x)) + 2)
    off = int(round(max(len(x) - crop, 0) * frac))
    w = x[off:off + crop]
    return w if len(w) == crop else np.tile(x, 2)[off:off + crop]


TRANSFORMS = {
    "standard crop":            lambda x, c, r: window(x, c, 0.0),
    "crop @25%":                lambda x, c, r: window(x, c, 0.25),
    "crop @50%":                lambda x, c, r: window(x, c, 0.50),
    "crop @100%":               lambda x, c, r: window(x, c, 1.00),
    "gain only":                lambda x, c, r: window(x, c, 0.0) * r.uniform(.7, 1.3),
    "noise only s=0.005":       lambda x, c, r: window(x, c, 0.0) + r.normal(0, .005, c).astype(np.float32),
    "noise only s=0.001":       lambda x, c, r: window(x, c, 0.0) + r.normal(0, .001, c).astype(np.float32),
    "gain+noise (TTA's own)":   lambda x, c, r: window(x, c, 0.0) * r.uniform(.7, 1.3)
                                                + r.normal(0, .005, c).astype(np.float32),
}


def main():
    os.environ.setdefault("PUBA_CORPUS", CORPUS)
    import protocol_a_public as A
    pool = A.build_eval()
    if len(pool) > N:
        pool = pool.sample(N, random_state=SEED).sort_values("utt").reset_index(drop=True)
    log(f"pool {len(pool)} clips, spoof {pool.label.mean()*100:.2f}%")
    cfg = P.CHECKPOINTS[CKPT]
    crop, fake_col = cfg["crop"], cfg["fake_col"]
    model, _, _ = P.build_model(CKPT, DEVICE, log=log)
    model.eval()
    names = list(TRANSFORMS)
    T = len(names)
    scores = np.zeros((len(pool), T), dtype=np.float32)
    rng = np.random.RandomState(SEED)
    t0 = time.time()
    CH = 256
    for c0 in range(0, len(pool), CH):
        paths = pool.path.values[c0:c0 + CH]
        block = np.stack([np.stack([TRANSFORMS[nm](full_waveform(p), crop, rng)
                                    for nm in names]) for p in paths])
        n = len(paths)
        flat = torch.from_numpy(block.reshape(n * T, crop))
        got = []
        with torch.no_grad():
            for i in range(0, len(flat), BATCH):
                x = flat[i:i + BATCH].to(DEVICE).float()
                with P.amp_ctx(DEVICE):
                    got.append(torch.softmax(model(x)[0].float(), 1)[:, fake_col].cpu())
        scores[c0:c0 + n] = torch.cat(got).numpy().reshape(n, T)
        done = min(c0 + CH, len(pool))
        if done % 2048 == 0 or done == len(pool):
            log(f"  {done}/{len(pool)}  eta {(len(pool)-done)/(done/(time.time()-t0))/60:.1f} min")
    np.savez_compressed(f"pilot_transforms_{CORPUS}_{CKPT}.npz", scores=scores,
                        label=pool.label.values.astype(np.int64), names=np.array(names))
    from metrics import compute_eer
    from sklearn.metrics import roc_auc_score
    y = pool.label.values.astype(np.int64)
    base = 100 * compute_eer(y, scores[:, 0])[0]
    print(f"\n{'transformation':26s} {'EER%':>7} {'AUC':>8}  {'rel. EER vs standard':>21}")
    for t, nm in enumerate(names):
        e = 100 * compute_eer(y, scores[:, t])[0]
        print(f"{nm:26s} {e:7.3f} {roc_auc_score(y, scores[:, t]):8.4f}  {100*(e-base)/base:+20.1f}%")


if __name__ == "__main__":
    main()
