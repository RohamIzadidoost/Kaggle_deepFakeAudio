"""
RawBoost recording-domain augmentation (Tak et al., ICASSP 2022).

Purpose in DA-RAD: perturb the *channel / recording* characteristics of the raw
waveform (convolutive colouration, impulsive device noise, stationary coloured
noise) so that "clean vs telephone/codec" stops being a proxy for real vs fake.
It touches recording conditions, not synthesis content, which is exactly the
confound our cross-corpus analysis exposed.

Three component algorithms combined per the reference:
  1. LnL_convolutive_noise  - linear + non-linear multi-band convolutive noise
  2. ISD_additive_noise      - impulsive, signal-dependent additive noise
  3. SSI_additive_noise      - stationary, signal-independent coloured noise

`algo` selects the combination (4 = all three in series, the default).
Operates on a 1-D float waveform; a torch-tensor wrapper is provided.
"""

import copy

import numpy as np
from scipy import signal


# --------------------------- primitives ---------------------------
def _rand_range(x1, x2, integer):
    y = float(np.random.uniform(low=x1, high=x2))   # python float scalar
    return int(y) if integer else y


def _norm_wav(x, always):
    peak = np.amax(abs(x))
    if peak == 0:
        return x
    if always or peak > 1:
        x = x / peak
    return x


def _gen_notch_coeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs):
    b = np.array([1.0])
    for _ in range(nBands):
        fc = _rand_range(minF, maxF, 0)
        bw = _rand_range(minBW, maxBW, 0)
        c = _rand_range(minCoeff, maxCoeff, 1)
        if c % 2 == 0:                      # firwin bandstop needs odd numtaps
            c += 1
        f1 = max(float(fc - bw / 2), 1e-3)
        f2 = min(float(fc + bw / 2), fs / 2 - 1e-3)
        taps = signal.firwin(c, [f1, f2], window="hamming", fs=fs)
        b = np.convolve(taps, b)
    G = _rand_range(minG, maxG, 0)
    _, h = signal.freqz(b, 1, fs=fs)
    b = (10 ** (G / 20)) * b / np.amax(abs(h))
    return b


def _filter_fir(x, b):
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), "constant")
    y = signal.lfilter(b, [1.0], xpad)
    y = y[N // 2: y.shape[0] - N // 2]
    return y


# --------------------------- algorithms ---------------------------
def LnL_convolutive_noise(x, fs, N_f, nBands, minF, maxF, minBW, maxBW,
                          minCoeff, maxCoeff, minG, maxG, minBias, maxBias):
    y = np.zeros_like(x)
    for i in range(N_f):
        g_lo, g_hi = minG, maxG
        if i == 1:                          # non-linear terms get a gain bias
            g_lo = minG - minBias
            g_hi = maxG - maxBias
        b = _gen_notch_coeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, g_lo, g_hi, fs)
        y = y + _filter_fir(np.power(x, i + 1), b)[: y.shape[0]]
    y = y - np.mean(y)
    return _norm_wav(y, 0)


def ISD_additive_noise(x, P, g_sd):
    y = copy.deepcopy(x)
    beta = _rand_range(0, P, 0)
    n = int(x.shape[0] * (beta / 100))
    idx = np.random.permutation(x.shape[0])[:n]
    f_r = ((2 * np.random.rand(idx.shape[0])) - 1) * ((2 * np.random.rand(idx.shape[0])) - 1)
    y[idx] = x[idx] + g_sd * x[idx] * f_r
    return _norm_wav(y, 0)


def SSI_additive_noise(x, fs, SNRmin, SNRmax, nBands, minF, maxF, minBW, maxBW,
                       minCoeff, maxCoeff, minG, maxG):
    noise = np.random.normal(0, 1, x.shape[0])
    b = _gen_notch_coeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs)
    noise = _norm_wav(_filter_fir(noise, b)[: x.shape[0]], 1)
    SNR = _rand_range(SNRmin, SNRmax, 0)
    nx, nn = np.linalg.norm(x), np.linalg.norm(noise)
    if nn > 0:
        noise = noise / nn * nx / (10 ** (0.05 * SNR))
    return x + noise


class RawBoost:
    """Callable RawBoost augmenter. Defaults follow the reference implementation."""

    def __init__(self, sr=16000, algo=4,
                 N_f=5, nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000,
                 minCoeff=10, maxCoeff=100, minG=0, maxG=0, minBias=5, maxBias=20,
                 P=10, g_sd=2, SNRmin=10, SNRmax=40):
        self.sr, self.algo = sr, algo
        self.lnl = dict(N_f=N_f, nBands=nBands, minF=minF, maxF=maxF, minBW=minBW, maxBW=maxBW,
                        minCoeff=minCoeff, maxCoeff=maxCoeff, minG=minG, maxG=maxG,
                        minBias=minBias, maxBias=maxBias)
        self.ssi = dict(nBands=nBands, minF=minF, maxF=maxF, minBW=minBW, maxBW=maxBW,
                        minCoeff=minCoeff, maxCoeff=maxCoeff, minG=minG, maxG=maxG,
                        SNRmin=SNRmin, SNRmax=SNRmax)
        self.P, self.g_sd = P, g_sd

    def _apply(self, x, algo):
        if algo == 1:
            return LnL_convolutive_noise(x, self.sr, **self.lnl)
        if algo == 2:
            return ISD_additive_noise(x, self.P, self.g_sd)
        if algo == 3:
            return SSI_additive_noise(x, self.sr, **self.ssi)
        if algo == 4:                        # all three in series
            x = LnL_convolutive_noise(x, self.sr, **self.lnl)
            x = ISD_additive_noise(x, self.P, self.g_sd)
            return SSI_additive_noise(x, self.sr, **self.ssi)
        if algo == 5:                        # convolutive + impulsive
            x = LnL_convolutive_noise(x, self.sr, **self.lnl)
            return ISD_additive_noise(x, self.P, self.g_sd)
        if algo == 0:                        # random among the three per call
            return self._apply(x, int(np.random.randint(1, 4)))
        return x                             # passthrough

    def __call__(self, wav):
        """wav: 1-D numpy float array or 1-D torch tensor. Returns same type/length."""
        is_torch = hasattr(wav, "detach")
        if is_torch:
            import torch
            arr = wav.detach().cpu().numpy().astype(np.float64).reshape(-1)
        else:
            arr = np.asarray(wav, dtype=np.float64).reshape(-1)

        n = arr.shape[0]
        out = self._apply(arr, self.algo)
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        if out.shape[0] != n:                # keep length identical to input
            out = np.pad(out, (0, max(0, n - out.shape[0])))[:n]

        if is_torch:
            import torch
            return torch.from_numpy(out).to(wav.device)
        return out


# --------------------------- self-test ---------------------------
if __name__ == "__main__":
    np.random.seed(0)
    fs = 16000
    t = np.arange(fs) / fs
    x = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.random.randn(fs)).astype(np.float32)

    for algo in [1, 2, 3, 4, 5]:
        rb = RawBoost(sr=fs, algo=algo)
        y = rb(x)
        assert y.shape == x.shape, f"algo {algo}: shape changed {y.shape} vs {x.shape}"
        assert np.isfinite(y).all(), f"algo {algo}: non-finite output"
        assert not np.allclose(y, x), f"algo {algo}: output identical to input"
        print(f"algo {algo}: out shape {y.shape}, "
              f"RMS in={np.sqrt((x**2).mean()):.3f} out={np.sqrt((y**2).mean()):.3f}")

    # torch round-trip
    import torch
    yt = RawBoost(sr=fs, algo=4)(torch.from_numpy(x))
    assert isinstance(yt, torch.Tensor) and yt.shape[0] == fs
    print("torch wrapper OK:", tuple(yt.shape), yt.dtype)
    print("augment.py self-test passed.")
