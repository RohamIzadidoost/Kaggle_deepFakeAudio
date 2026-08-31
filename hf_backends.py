"""Adapters that make HuggingFace audio classifiers speak `public_ckpt_tta.py`'s interface.

`score()` and `adapt()` call `model(wav)[0] -> logits` on a raw-waveform batch
`[B, T]`. HF audio classifiers want preprocessed inputs and return a
`SequenceClassifierOutput`, so each family gets a thin wrapper.

Two families are supported:

* `Wav2Vec2ForSequenceClassification` — takes the raw waveform, but its feature
  extractor declares `do_normalize=True`, i.e. per-utterance zero-mean/unit-var.
  Skipping that silently shifts every input off the distribution the head was
  fitted on, which reads as "this checkpoint is bad" rather than "we fed it
  wrong". Applied here, matching `Wav2Vec2FeatureExtractor`.

* `ASTForAudioClassification` — takes a **mel spectrogram**, not audio. This is a
  genuinely different input representation from everything else in the study,
  which is the point of including it: our parameter-selection rule (top-N blocks'
  LayerNorms + head) should be architecture-agnostic, and AST is the test of that.
  `ASTFbank` reimplements `ASTFeatureExtractor` on tensors so it can run on the
  GPU inside the adaptation loop; `verify_ast_fbank()` asserts it matches the
  real extractor numerically before any result is believed.

Nothing here needs to be differentiable w.r.t. the waveform: adaptation updates
LayerNorm affine parameters *inside* the model, so gradients never flow back
through feature extraction.
"""

import numpy as np
import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as ta_kaldi


class HFSeqClsWrapper(nn.Module):
    """Wav2Vec2ForSequenceClassification on raw audio -> (logits,)."""

    def __init__(self, model, do_normalize=True):
        super().__init__()
        self.model = model
        self.do_normalize = do_normalize

    def forward(self, wav):
        x = wav
        if self.do_normalize:
            # Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm, per utterance
            x = (x - x.mean(dim=1, keepdim=True)) / torch.sqrt(x.var(dim=1, keepdim=True) + 1e-7)
        return (self.model(input_values=x).logits,)


class ASTFbank(nn.Module):
    """Tensor reimplementation of ASTFeatureExtractor.

    Mirrors `_extract_fbank_features`: kaldi fbank with a hanning window and
    `num_mel_bins` bins, zero-padded (or truncated) to `max_length` frames, then
    normalised as `(x - mean) / (2 * std)`. Order matters -- upstream pads
    *before* normalising, so the padded region does not sit at zero afterwards.
    Replicated rather than corrected, because matching upstream is the point.
    """

    def __init__(self, num_mel_bins=128, max_length=1024, mean=-4.2677393, std=4.5689974):
        super().__init__()
        self.num_mel_bins, self.max_length = num_mel_bins, max_length
        self.mean, self.std = mean, std

    def forward(self, wav):
        feats = []
        for w in wav:
            fb = ta_kaldi.fbank(w.unsqueeze(0).float(), sample_frequency=16000,
                                window_type="hanning", num_mel_bins=self.num_mel_bins)
            n = fb.shape[0]
            if n < self.max_length:
                fb = nn.functional.pad(fb, (0, 0, 0, self.max_length - n))
            elif n > self.max_length:
                fb = fb[:self.max_length]
            feats.append(fb)
        x = torch.stack(feats)
        return (x - self.mean) / (self.std * 2)


class HFASTWrapper(nn.Module):
    """ASTForAudioClassification on raw audio -> (logits,)."""

    def __init__(self, model, fbank):
        super().__init__()
        self.model, self.fbank = model, fbank

    def forward(self, wav):
        with torch.no_grad():                    # feature extraction is not learned
            feats = self.fbank(wav)
        return (self.model(input_values=feats).logits,)


def verify_ast_fbank(path, device="cpu", n=4, tol=2e-3, log=print):
    """Assert ASTFbank matches the real ASTFeatureExtractor.

    A silently mismatched front-end would make a working checkpoint look broken
    and would be indistinguishable from "our method does not transfer to AST".
    Compared on random waveforms of several lengths, including one shorter than
    the padding window and one longer, since those exercise the pad and the
    truncate branches respectively.
    """
    from transformers import ASTFeatureExtractor

    fe = ASTFeatureExtractor.from_pretrained(path)
    mine = ASTFbank(num_mel_bins=fe.num_mel_bins, max_length=fe.max_length,
                    mean=fe.mean, std=fe.std).to(device)
    rng = np.random.RandomState(0)
    worst = 0.0
    for L in (16000 * 2, 16000 * 4, 16000 * 11)[:n]:
        w = rng.randn(L).astype(np.float32) * 0.1
        ref = fe(w, sampling_rate=16000, return_tensors="pt").input_values[0]
        got = mine(torch.from_numpy(w).unsqueeze(0).to(device))[0].cpu()
        d = (ref - got).abs().max().item()
        worst = max(worst, d)
        log(f"    ASTFbank vs ASTFeatureExtractor @ {L/16000:.0f}s: max|diff| = {d:.2e}")
    if worst > tol:
        raise RuntimeError(
            f"ASTFbank does not match ASTFeatureExtractor (max diff {worst:.3e} > {tol}). "
            f"Fix the front-end before trusting any AST result -- a wrong spectrogram "
            f"is indistinguishable from a checkpoint that does not transfer.")
    log(f"    ASTFbank matches upstream (worst {worst:.2e} <= {tol})")
    return worst


def build_hf(cfg, device, log=print):
    """Load an HF checkpoint and wrap it. Returns (wrapped_model, n_tensors)."""
    from transformers import ASTForAudioClassification, Wav2Vec2ForSequenceClassification

    path = cfg["path"]
    if cfg["kind"] == "hf_seqcls":
        m = Wav2Vec2ForSequenceClassification.from_pretrained(path)
        wrapped = HFSeqClsWrapper(m, do_normalize=cfg.get("do_normalize", True))
    elif cfg["kind"] == "hf_ast":
        verify_ast_fbank(path, device="cpu", log=log)
        m = ASTForAudioClassification.from_pretrained(path)
        from transformers import ASTFeatureExtractor
        fe = ASTFeatureExtractor.from_pretrained(path)
        wrapped = HFASTWrapper(m, ASTFbank(fe.num_mel_bins, fe.max_length, fe.mean, fe.std))
    else:
        raise ValueError(f"not an HF kind: {cfg['kind']}")

    n = sum(1 for _ in m.parameters())
    # The declared label order is a claim, not proof -- --mode signcheck still
    # verifies polarity empirically. Logged so a mismatch is visible in the run log.
    id2label = getattr(m.config, "id2label", None)
    log(f"    loaded {cfg['kind']} from {path}: {n} tensors, id2label={id2label}, "
        f"declared fake_col={cfg['fake_col']}")
    return wrapped.to(device).eval(), n
