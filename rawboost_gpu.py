"""Batched-GPU RawBoost, numerically equivalent to `augment.py` per clip.

Why this exists: `augment.py`'s RawBoost is the reference implementation but it is
CPU/numpy and *strictly per-clip* (`__call__` does `.reshape(-1)`, so handing it a
batched tensor silently flattens B*L samples into one "clip" -- a wrong result, not
an error). `extended_pipeline.py` is deliberately DataLoader-free with a resident
audio cache, so there is nowhere to hook a per-clip CPU augmenter without either
a Python loop over the batch or reintroducing worker processes (which previously
exhausted shared memory on the MIG slice).

Cost split that makes this worth doing: per clip, coefficient generation is ~3 ms
(30 firwin + 6 freqz calls) while `signal.lfilter` over 64k samples with up to
~500 taps, six times, is ~60-180 ms. Filtering is ~90% of the work.

  * random parameter generation (notch coeffs, beta/idx/f_r, SNR, white noise)
    stays on CPU in numpy, drawn in the SAME ORDER as the reference -- this is
    what makes exact equivalence testable, and it is cheap
  * FIR filtering moves to batched grouped conv1d on GPU

Delay compensation. `augment._filter_fir` right-pads x by N=K+1, runs a causal
FIR, then trims N//2 from each end. Algebraically that is

    out[n] = sum_k b[k] * x[n + s - k],    s = (K+1)//2

i.e. a plain causal convolution followed by a left-shift of s. Filters have
per-clip random lengths, so kernels are zero-padded to Kmax (harmless -- appended
zero taps change nothing) and the per-clip shift is applied with a `torch.gather`,
the same pattern `_crop` uses for random crops.

Verify with `python rawboost_gpu.py` before using it in any experiment.
"""

import numpy as np
import torch
import torch.nn.functional as F

from augment import (ISD_additive_noise, LnL_convolutive_noise,  # noqa: F401 (self-test)
                     SSI_additive_noise, _gen_notch_coeffs, _rand_range)

DEFAULTS = dict(N_f=5, nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000,
                minCoeff=10, maxCoeff=100, minG=0, maxG=0, minBias=5, maxBias=20,
                P=10, g_sd=2, SNRmin=10, SNRmax=40)


# ------------------------------------------------------------------ helpers
def _fir_batch(x, kernels):
    """Batched FIR matching `augment._filter_fir` per clip.

    x: (B, L) float32 tensor. kernels: list of B 1-D numpy arrays, lengths may
    differ. Returns (B, L).
    """
    B, L = x.shape
    dev = x.device
    Kmax = max(len(k) for k in kernels)

    # zero-pad kernels to a common length, then flip: conv1d cross-correlates,
    # so w[j] = b[Kmax-1-j] turns it into a causal convolution.
    w = np.zeros((B, Kmax), dtype=np.float32)
    for i, k in enumerate(kernels):
        w[i, :len(k)] = k
    w = torch.from_numpy(w[:, ::-1].copy()).to(dev).unsqueeze(1)        # (B,1,Kmax)

    # left-pad Kmax-1 so out_full[m] == sum_k b[k] x[m-k]; right-pad Kmax so the
    # per-clip shift s (<= (Kmax+1)//2) can never read past the end.
    xpad = F.pad(x, (Kmax - 1, Kmax)).unsqueeze(0)                      # (1,B,·)
    out_full = F.conv1d(xpad, w, groups=B).squeeze(0)                   # (B,·)

    # per-clip left-shift by s = (K_i + 1)//2
    s = torch.tensor([(len(k) + 1) // 2 for k in kernels],
                     dtype=torch.long, device=dev).unsqueeze(1)
    idx = (torch.arange(L, device=dev).unsqueeze(0) + s).clamp(max=out_full.shape[1] - 1)
    return torch.gather(out_full, 1, idx)


def _norm_wav_batch(x, always):
    """Batched `augment._norm_wav`: divide by peak if `always` or peak > 1."""
    peak = x.abs().amax(dim=1, keepdim=True)
    if always:
        scale = torch.where(peak > 0, peak, torch.ones_like(peak))
    else:
        scale = torch.where(peak > 1, peak, torch.ones_like(peak))
    return x / scale


# ------------------------------------------------------------------ algorithms
def lnl_convolutive_batch(x, fs, p, bank=None):
    """Linear + non-linear multi-band convolutive noise (batched).

    Coefficients are drawn clip-major (all N_f filters for clip 0, then clip 1,
    ...) to match the reference's RNG consumption order, so a B=1 call is exactly
    reproducible against `augment.LnL_convolutive_noise` under the same seed.
    """
    B = x.shape[0]
    per_clip = []
    for _ in range(B):
        ks = []
        for i in range(p["N_f"]):
            if bank is not None:
                pool = bank[1] if i == 1 else bank[0]
                ks.append(pool[int(np.random.randint(len(pool)))])
                continue
            g_lo, g_hi = p["minG"], p["maxG"]
            if i == 1:                               # non-linear terms get a bias
                g_lo = p["minG"] - p["minBias"]
                g_hi = p["maxG"] - p["maxBias"]
            ks.append(_gen_notch_coeffs(p["nBands"], p["minF"], p["maxF"], p["minBW"],
                                       p["maxBW"], p["minCoeff"], p["maxCoeff"],
                                       g_lo, g_hi, fs))
        per_clip.append(ks)

    y = torch.zeros_like(x)
    for i in range(p["N_f"]):
        y = y + _fir_batch(torch.pow(x, i + 1), [ks[i] for ks in per_clip])
    y = y - y.mean(dim=1, keepdim=True)
    return _norm_wav_batch(y, False)


def isd_additive_batch(x, p):
    """Impulsive, signal-dependent additive noise (batched).

    y[idx] = x[idx] + g_sd*x[idx]*f_r  is  y = x * (1 + g_sd*f_r*mask), so the
    per-clip random index sets collapse into one (B, L) multiplier built on CPU.
    """
    B, L = x.shape
    mult = np.ones((B, L), dtype=np.float32)
    for i in range(B):
        beta = _rand_range(0, p["P"], 0)
        n = int(L * (beta / 100))
        idx = np.random.permutation(L)[:n]
        f_r = ((2 * np.random.rand(idx.shape[0])) - 1) * \
              ((2 * np.random.rand(idx.shape[0])) - 1)
        mult[i, idx] = 1.0 + p["g_sd"] * f_r
    y = x * torch.from_numpy(mult).to(x.device)
    return _norm_wav_batch(y, False)


def ssi_additive_batch(x, fs, p, bank=None):
    """Stationary, signal-independent coloured noise (batched)."""
    B, L = x.shape
    noise = torch.from_numpy(
        np.random.normal(0, 1, (B, L)).astype(np.float32)).to(x.device)
    if bank is not None:
        kernels = [bank[0][int(np.random.randint(len(bank[0])))] for _ in range(B)]
    else:
        kernels = [_gen_notch_coeffs(p["nBands"], p["minF"], p["maxF"], p["minBW"],
                                    p["maxBW"], p["minCoeff"], p["maxCoeff"],
                                    p["minG"], p["maxG"], fs) for _ in range(B)]
    noise = _norm_wav_batch(_fir_batch(noise, kernels), True)
    snr = torch.tensor([_rand_range(p["SNRmin"], p["SNRmax"], 0) for _ in range(B)],
                       dtype=torch.float32, device=x.device).unsqueeze(1)
    nx = x.norm(dim=1, keepdim=True)
    nn = noise.norm(dim=1, keepdim=True)
    noise = torch.where(nn > 0, noise / torch.where(nn > 0, nn, torch.ones_like(nn))
                        * nx / (10 ** (0.05 * snr)), noise)
    return x + noise


# ------------------------------------------------------------------ public API
def rawboost_batch(x, sr=16000, algo=4, bank=None, **overrides):
    """RawBoost over a batch of waveforms.

    x: (B, L) float tensor on any device. Returns (B, L), same dtype/device.
    algo: 1=LnL, 2=ISD, 3=SSI, 4=all three in series (reference default),
          5=LnL+ISD, 0=random one of 1..3, anything else=passthrough.
    bank: optional (plain_bank, biased_bank) from `build_coeff_bank` -- draws
          filters from a pre-generated pool instead of generating fresh ones.
          See `RawBoostGPU` for why.
    """
    p = {**DEFAULTS, **overrides}
    dt, single = x.dtype, x.dim() == 1
    if single:
        x = x.unsqueeze(0)
    z = x.float()

    if algo == 0:
        algo = int(np.random.randint(1, 4))
    if algo in (1, 4, 5):
        z = lnl_convolutive_batch(z, sr, p, bank=bank)
    if algo in (2, 4, 5):
        z = isd_additive_batch(z, p)
    if algo in (3, 4):
        z = ssi_additive_batch(z, sr, p, bank=bank)

    z = z.to(dt)
    return z.squeeze(0) if single else z


def build_coeff_bank(n, sr=16000, biased=False, **overrides):
    """Pre-generate `n` notch-filter coefficient vectors.

    `biased=True` uses the gain range LnL applies to its non-linear (i==1) term.
    """
    p = {**DEFAULTS, **overrides}
    g_lo, g_hi = ((p["minG"] - p["minBias"], p["maxG"] - p["maxBias"]) if biased
                  else (p["minG"], p["maxG"]))
    return [_gen_notch_coeffs(p["nBands"], p["minF"], p["maxF"], p["minBW"],
                             p["maxBW"], p["minCoeff"], p["maxCoeff"],
                             g_lo, g_hi, sr) for _ in range(n)]


class RawBoostGPU:
    """Stateful RawBoost that draws filters from a pre-generated coefficient bank.

    Motivation (measured, batch of 32 x 3 s): fresh generation costs 312 ms per
    batch, of which 159 ms is `_gen_notch_coeffs` on CPU (6 banks x 32 clips ->
    ~1000 `firwin` + ~200 `freqz` calls) and only 69 ms is the GPU convolution.
    Source training does ~794 batches/epoch, so fresh generation would add ~4 min
    per epoch -- about 11 hours across the full 20-fold leave-one-corpus-out grid.

    Sampling filters from a large fixed pool instead makes that cost ~zero. The
    augmentation is then "a random filter from a bank of `bank_size`" rather than
    "a freshly drawn filter"; with a few thousand filters the induced channel
    distribution is effectively the same for augmentation purposes, while the
    additive-noise and impulsive terms stay freshly random every call.

    Pass `bank_size=0` to fall back to exact reference-equivalent generation.
    """

    def __init__(self, sr=16000, algo=4, bank_size=4096, **overrides):
        self.sr, self.algo, self.overrides = sr, algo, overrides
        self.bank = None
        if bank_size:
            self.bank = (build_coeff_bank(bank_size, sr, False, **overrides),
                         build_coeff_bank(bank_size, sr, True, **overrides))

    def __call__(self, x):
        return rawboost_batch(x, sr=self.sr, algo=self.algo, bank=self.bank,
                              **self.overrides)


# ------------------------------------------------------------------ self-test
if __name__ == "__main__":
    import time

    from augment import RawBoost, _filter_fir

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    SR, L = 16000, 16000
    print(f"device={dev}  L={L}")
    rng = np.random.default_rng(0)

    # ---- 1. the filtering math, RNG-independent -----------------------------
    # This is the part that can silently misalign (delay compensation, Kmax
    # zero-padding, per-clip gather), so test it against the reference with
    # IDENTICAL kernels rather than relying on matched random draws.
    print("\n--- 1. _fir_batch vs augment._filter_fir, identical kernels ---")
    B = 5
    x = rng.standard_normal((B, L)).astype(np.float32) * 0.1
    np.random.seed(0)
    kern = [_gen_notch_coeffs(5, 20, 8000, 100, 1000, 10, 100, 0, 0, SR)
            for _ in range(B)]
    got = _fir_batch(torch.from_numpy(x).to(dev), kern).cpu().numpy()
    ref = np.stack([_filter_fir(x[i].astype(np.float64), kern[i])[:L] for i in range(B)])
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    corr = min(np.corrcoef(got[i], ref[i])[0, 1] for i in range(B))
    print(f"  kernel lengths {[len(k) for k in kern]}")
    print(f"  max rel err {rel:.3e}   min corr {corr:.8f}")
    assert rel < 1e-4 and corr > 0.99999, "FIR filtering diverged from reference"

    # ---- 2. end-to-end equivalence at B=1 ----------------------------------
    # At B=1 the batched path consumes the numpy RNG in exactly the reference's
    # order, so full per-algo equivalence is checkable. (At B>1 it cannot be:
    # the reference runs LnL->ISD->SSI per clip, we run each stage across the
    # whole batch, so the draw order necessarily differs. That is a difference
    # in which random perturbation each clip receives, not in correctness.)
    print("\n--- 2. end-to-end equivalence vs augment.RawBoost (B=1) ---")
    x1 = rng.standard_normal(L).astype(np.float32) * 0.1
    worst = 0.0
    for algo in (1, 2, 3, 4, 5):
        np.random.seed(1234)
        g = rawboost_batch(torch.from_numpy(x1).to(dev), sr=SR, algo=algo).cpu().numpy()
        np.random.seed(1234)
        r = RawBoost(sr=SR, algo=algo)(x1)
        rel = np.abs(g - r).max() / max(np.abs(r).max(), 1e-8)
        c = np.corrcoef(g, r)[0, 1]
        worst = max(worst, rel)
        print(f"  algo={algo}: max rel err {rel:.3e}   corr {c:.8f}")
        assert rel < 1e-3, f"algo {algo} diverged: rel err {rel:.3e}"
        assert c > 0.9999, f"algo {algo} corr too low: {c}"

    # ---- 3. contracts ------------------------------------------------------
    print("\n--- 3. contracts ---")
    xb = torch.from_numpy(rng.standard_normal((8, L)).astype(np.float32) * 0.1).to(dev)
    for algo in (0, 1, 2, 3, 4, 5, 99):
        o = rawboost_batch(xb, sr=SR, algo=algo)
        assert o.shape == xb.shape and o.dtype == xb.dtype and o.device == xb.device
        assert torch.isfinite(o).all(), f"algo {algo} produced non-finite values"
    assert rawboost_batch(xb[0], sr=SR, algo=4).shape == (L,)
    assert torch.allclose(rawboost_batch(xb, sr=SR, algo=99), xb)
    print("  shapes/dtype/device preserved, finite, algo=99 is identity")

    # each clip must get its OWN perturbation (a batching bug that broadcast one
    # clip's filter to all would show up as identical rows)
    o = rawboost_batch(xb.clone(), sr=SR, algo=4)
    deltas = (o - xb).abs().mean(dim=1)
    assert deltas.std() > 0, "all clips perturbed identically -- batching is broken"
    print(f"  per-clip perturbation differs (mean|delta| spread {deltas.std():.2e})")

    # ---- 4. throughput vs the CPU reference --------------------------------
    print("\n--- 4. throughput (the reason this module exists) ---")
    big = torch.from_numpy(rng.standard_normal((32, 48000)).astype(np.float32) * 0.1).to(dev)
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    rawboost_batch(big, sr=SR, algo=4)
    if dev == "cuda":
        torch.cuda.synchronize()
    gpu_s = time.time() - t

    cpu_ref = RawBoost(sr=SR, algo=4)
    npb = big[:4].cpu().numpy()
    t = time.time()
    for i in range(4):
        cpu_ref(npb[i])
    cpu_s = (time.time() - t) / 4 * 32
    print(f"  batch of 32 x 3s: gpu {gpu_s * 1000:.0f} ms   "
          f"cpu-reference {cpu_s * 1000:.0f} ms   speedup {cpu_s / gpu_s:.1f}x")

    # ---- 5. bank mode: must be much faster and statistically equivalent ----
    print("\n--- 5. coefficient-bank mode ---")
    rb = RawBoostGPU(sr=SR, algo=4, bank_size=1024)
    torch.cuda.synchronize() if dev == "cuda" else None
    t = time.time()
    for _ in range(5):
        ob = rb(big)
    torch.cuda.synchronize() if dev == "cuda" else None
    bank_s = (time.time() - t) / 5
    print(f"  bank  {bank_s * 1000:.0f} ms/batch   vs fresh {gpu_s * 1000:.0f} ms   "
          f"({gpu_s / bank_s:.1f}x faster, {cpu_s / bank_s:.0f}x vs CPU ref)")
    assert torch.isfinite(ob).all()

    # distribution of the induced perturbation should match fresh generation
    fresh = rawboost_batch(big, sr=SR, algo=4)
    df, db = (fresh - big).abs().mean().item(), (ob - big).abs().mean().item()
    print(f"  mean|delta|: fresh {df:.4f}  bank {db:.4f}  ratio {db / df:.3f}")
    assert 0.5 < db / df < 2.0, "bank mode perturbation magnitude is off-distribution"

    # bank must not leak into the exact path
    np.random.seed(1234)
    g = rawboost_batch(torch.from_numpy(x1).to(dev), sr=SR, algo=4).cpu().numpy()
    np.random.seed(1234)
    r = RawBoost(sr=SR, algo=4)(x1)
    assert np.abs(g - r).max() / np.abs(r).max() < 1e-3, "exact path regressed"
    print("  exact (bank=None) path still matches the reference")

    print(f"\nrawboost_gpu.py self-test passed (worst B=1 rel err {worst:.2e}).")
