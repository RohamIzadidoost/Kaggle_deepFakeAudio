"""
Self-supervised speech front-ends for DA-RAD.

`SSLFrontend` wraps a torchaudio wav2vec2/XLS-R bundle: it maps a raw 16 kHz
waveform (B, samples) to frame features (B, T', C). Weights (~1.2 GB for
XLS-R-300M) are fetched lazily on first use, optionally frozen (default) so only
a LoRA/head is trained on a 10 GB GPU.

`StubEncoder` is a tiny randomly-initialised stand-in with the same interface,
used to shape-test the whole DA-RAD assembly without downloading XLS-R.

Both expose `.out_dim` and `forward(waveform) -> (B, T', C)`.
"""

import torch
import torch.nn as nn


class StubEncoder(nn.Module):
    """Cheap frame encoder for tests: Conv1d over the raw waveform."""

    def __init__(self, out_dim=256, kernel=400, stride=160):
        super().__init__()
        self.out_dim = out_dim
        self.conv = nn.Conv1d(1, out_dim, kernel_size=kernel, stride=stride)

    def forward(self, waveform):          # (B, samples)
        x = waveform.unsqueeze(1)         # (B, 1, samples)
        x = self.conv(x)                  # (B, C, T')
        return x.transpose(1, 2)          # (B, T', C)


class SSLFrontend(nn.Module):
    """XLS-R/wav2vec2 front-end.

    n_trainable_layers = 0 -> fully frozen (features only).
    n_trainable_layers = K -> fine-tune the top K transformer layers (lower layers
    and the conv feature extractor stay frozen). Domain-generalization losses need
    an adaptable representation, so K>0 is required for DA-RAD to actually reshape
    the embedding; with K=0 the DG losses only perturb the downstream head and hurt.
    """

    def __init__(self, bundle_name="WAV2VEC2_XLSR_300M", n_trainable_layers=0, layer=-1):
        super().__init__()
        import torchaudio
        if not hasattr(torchaudio.pipelines, bundle_name):
            raise ValueError(f"unknown torchaudio bundle: {bundle_name}")
        self.bundle = getattr(torchaudio.pipelines, bundle_name)
        self.n_trainable_layers = n_trainable_layers
        self.layer = layer
        self.out_dim = getattr(self.bundle, "_params", {}).get("encoder_embed_dim", 1024)
        self.model = None                 # lazy — downloads weights on first use

    def _ensure(self, device):
        if self.model is None:
            self.model = self.bundle.get_model().to(device)
            for p in self.model.parameters():
                p.requires_grad_(False)
            if self.n_trainable_layers > 0:
                layers = self.model.model.encoder.transformer.layers
                for p in layers[-self.n_trainable_layers:].parameters():
                    p.requires_grad_(True)
            # eval() everywhere: disables dropout for stable features; grad still
            # flows to the (unfrozen) top layers' weights.
            self.model.eval()

    def forward(self, waveform):          # (B, samples) @ 16 kHz
        self._ensure(waveform.device)
        # frozen -> no graph; fine-tuning -> build graph (autograd only stores
        # activations from the first trainable layer upward, since lower frozen
        # outputs have requires_grad=False, so memory stays modest).
        grad_ctx = torch.enable_grad() if self.n_trainable_layers > 0 else torch.no_grad()
        with grad_ctx:
            # autocast OFF for the encoder: its conv feature extractor mixes half
            # input with float weights under AMP. The head still uses the caller's AMP.
            if waveform.is_cuda:
                with torch.amp.autocast("cuda", enabled=False):
                    feats, _ = self.model.extract_features(waveform.float())
            else:
                feats, _ = self.model.extract_features(waveform.float())
        return feats[self.layer]          # (B, T', C)


if __name__ == "__main__":
    enc = StubEncoder(out_dim=256)
    x = torch.randn(4, 48000)             # 3 s @ 16 kHz
    f = enc(x)
    print(f"StubEncoder out {tuple(f.shape)}  out_dim={enc.out_dim}")
    assert f.shape[0] == 4 and f.shape[2] == 256
    print("ssl_frontend.py (stub) self-test passed. "
          "Real XLS-R load is exercised later via the trainer (downloads weights).")
