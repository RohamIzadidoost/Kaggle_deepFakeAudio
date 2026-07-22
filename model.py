"""
AttentiveSpecCNN — the deep-learning detector for real-vs-fake audio.

Design rationale (the "reasonable, explainable, novel" brief):
  * Reasonable: a compact 2-D CNN over log-mel spectrograms — the standard,
    well-understood backbone for audio anti-spoofing, small enough to train on
    one GPU in minutes.
  * Novel (vs. the paper): the paper hand-engineers features with a static
    MFCC -> GNB -> NMF chain. We replace that with a *learnable* temporal
    ATTENTION pooling layer that decides, per time frame, how much each moment
    of audio contributes to the real/fake decision — a learned, data-driven
    generalisation of their fixed feature transform.
  * Explainable: the attention weights are a 1-D saliency over time (which
    moments drove the decision), and the conv feature maps support Grad-CAM over
    the time-frequency plane. Both are returned on request for visualisation.

Input : log-mel spectrogram (B, 1, n_mels, T)  from DeepfakeAudioDataset
Output: logits (B, 2)   with column order [real=0, fake=1]  (matches LABEL_TO_IDX)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> (optional) MaxPool."""

    def __init__(self, c_in, c_out, pool=(2, 2)):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(c_out)
        self.pool = nn.MaxPool2d(pool) if pool else None

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return self.pool(x) if self.pool is not None else x


class TemporalAttentionPool(nn.Module):
    """Attention pooling over the time axis.

    Input  (B, C, T)  ->  context (B, C) + attention weights (B, T).
    The weights are the model's per-frame saliency and are what we visualise.
    """

    def __init__(self, channels):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x):                     # x: (B, C, T)
        h = x.transpose(1, 2)                 # (B, T, C)
        attn = torch.softmax(self.score(h).squeeze(-1), dim=1)  # (B, T)
        context = torch.bmm(attn.unsqueeze(1), h).squeeze(1)    # (B, C)
        return context, attn


class AttentiveSpecCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=2, dropout=0.3):
        super().__init__()
        # normalise raw log-mel dB values (roughly -100..+40) to a stable range
        self.input_norm = nn.BatchNorm2d(1)

        self.features = nn.Sequential(
            ConvBlock(1, 16),      # -> (16, n_mels/2,  T/2)
            ConvBlock(16, 32),     # -> (32, n_mels/4,  T/4)
            ConvBlock(32, 64),     # -> (64, n_mels/8,  T/8)
            ConvBlock(64, 128),    # -> (128, n_mels/16, T/16)
        )
        self.attn_pool = TemporalAttentionPool(128)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, return_attn=False):
        x = self.input_norm(x)
        fmap = self.features(x)               # (B, C, F', T')
        self._last_fmap = fmap                # kept for Grad-CAM
        pooled_freq = fmap.mean(dim=2)        # collapse frequency -> (B, C, T')
        context, attn = self.attn_pool(pooled_freq)
        logits = self.head(context)
        if return_attn:
            return logits, attn
        return logits


if __name__ == "__main__":
    # shape smoke-test
    m = AttentiveSpecCNN()
    x = torch.randn(4, 1, 80, 401)
    logits, attn = m(x, return_attn=True)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"logits {tuple(logits.shape)}  attn {tuple(attn.shape)}  params {n_params:,}")
