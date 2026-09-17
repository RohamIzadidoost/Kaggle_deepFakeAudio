"""Pilot: do multiple acoustic views carry better ranking information?

Before building any relative-supervision adaptation, three things have to be
true, and none of them needs training to check:

  1. Each view must preserve spoofing evidence. A view whose solo EER collapses
     has destroyed the thing we want ordered, however intelligible it sounds.
  2. Averaging scores over views must improve ranking over the single standard
     view. This is the frozen multi-view teacher. If adaptation later matches
     only this, it has not earned its complexity.
  3. Cross-view agreement on a pair must carry information about whether that
     pair is ordered correctly. The proposed loss weights pairs by agreement;
     if agreement does not separate correct from incorrect orderings, the
     weight has nothing to key on.

View 0 is the standard deterministic leading crop, so its column reproduces the
published scores exactly and acts as a self-check. Views 1..K-1 take windows at
different offsets and apply a per-clip gain and Gaussian noise from the same
perturbation family the existing consistency term uses.

Labels are used only for diagnosis here, never to build a view or a preference.

    PILOT_N=10000 PILOT_K=8 python pilot_views.py
"""
import os
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

import public_ckpt_tta as P

N = int(os.environ.get("PILOT_N", "10000"))
K = int(os.environ.get("PILOT_K", "8"))
CKPT = os.environ.get("PILOT_CKPT", "ssl_aasist_wavefake")
CORPUS = os.environ.get("PILOT_CORPUS", "itw")
SEED = int(os.environ.get("PILOT_SEED", "0"))
NOISE = float(os.environ.get("PILOT_NOISE", "0.005"))   # 0 disables additive noise
GAIN = (0.7, 1.3) if os.environ.get("PILOT_GAIN", "1") == "1" else None
BATCH = int(os.environ.get("PILOT_BATCH", "32"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.environ.get("PILOT_OUT", f"pilot_views_{CORPUS}_{CKPT}.npz")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def full_waveform(path):
    """P.load_clip's decode path, but returning the whole waveform."""
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        x, sr = P._ffmpeg_decode(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != P.SR:
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, P.SR).numpy()
    return x.astype(np.float32)


def views_of(x, crop, rng):
    """K windows of one clip: view 0 is the standard leading crop."""
    if len(x) == 0:
        x = np.zeros(crop, dtype=np.float32)
    if len(x) < crop:
        x = np.tile(x, int(crop / len(x)) + 1)[: crop * 2]
    span = max(len(x) - crop, 0)
    out = np.empty((K, crop), dtype=np.float32)
    out[0] = x[:crop]
    for v in range(1, K):
        off = int(round(span * v / max(K - 1, 1))) if span else 0
        w = x[off:off + crop]
        if len(w) < crop:                       # short clip: wrap around
            w = np.tile(x, int(crop / max(len(x), 1)) + 2)[off:off + crop]
        gain = rng.uniform(*GAIN) if GAIN else 1.0
        out[v] = w * gain
        if NOISE:
            out[v] += rng.normal(0, NOISE, crop).astype(np.float32)
    return out


def main():
    os.environ.setdefault("PUBA_CORPUS", CORPUS)
    import protocol_a_public as A          # reuses the exact eval manifest
    pool = A.build_eval()
    if len(pool) > N:
        pool = pool.sample(N, random_state=SEED).sort_values("utt").reset_index(drop=True)
    log(f"pool {len(pool)} clips, spoof {pool.label.mean()*100:.2f}%")

    cfg = P.CHECKPOINTS[CKPT]
    crop, fake_col = cfg["crop"], cfg["fake_col"]
    model, nf, nb = P.build_model(CKPT, DEVICE, log=log)
    model.eval()
    log(f"{CKPT}: {nf} front-end + {nb} back-end tensors, crop {crop}, fake_col {fake_col}")

    scores = np.zeros((len(pool), K), dtype=np.float32)
    rng = np.random.RandomState(SEED)
    t0 = time.time()
    CH = 256
    for c0 in range(0, len(pool), CH):
        paths = pool.path.values[c0:c0 + CH]
        block = np.stack([views_of(full_waveform(p), crop, rng) for p in paths])  # (n,K,crop)
        n = len(paths)
        flat = torch.from_numpy(block.reshape(n * K, crop))
        got = []
        with torch.no_grad():
            for i in range(0, len(flat), BATCH):
                x = flat[i:i + BATCH].to(DEVICE).float()
                with P.amp_ctx(DEVICE):
                    lg = model(x)[0]
                got.append(torch.softmax(lg.float(), 1)[:, fake_col].cpu())
        scores[c0:c0 + n] = torch.cat(got).numpy().reshape(n, K)
        done = min(c0 + CH, len(pool))
        if done % 1024 == 0 or done == len(pool):
            rate = done / (time.time() - t0)
            log(f"  {done}/{len(pool)} clips  {rate*K:.0f} view-passes/s  "
                f"eta {(len(pool)-done)/rate/60:.1f} min")
    np.savez_compressed(OUT, scores=scores, label=pool.label.values.astype(np.int64),
                        utt=pool.utt.values)
    log(f"wrote {OUT}  shape {scores.shape}")


if __name__ == "__main__":
    main()
