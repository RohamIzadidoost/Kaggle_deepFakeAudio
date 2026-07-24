"""
LCNN (Light CNN with Max-Feature-Map) — a standard anti-spoofing baseline.

Operates on a log-mel / LFCC spectrogram (B, 1, F, T) and outputs (B, 2). MFM
activation halves the channel dimension by an element-wise max of two channel
groups, giving the compact, competitive front-end used widely in ASVspoof systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MFM(nn.Module):
    """Max-Feature-Map over channels: split in half, take element-wise max."""

    def forward(self, x):
        a, b = torch.chunk(x, 2, dim=1)
        return torch.max(a, b)


def _conv_mfm(c_in, c_out, k, s, p):
    # c_out is the channel count AFTER MFM, so the conv produces 2*c_out.
    return nn.Sequential(nn.Conv2d(c_in, 2 * c_out, k, s, p), MFM())


class LCNN(nn.Module):
    def __init__(self, n_classes=2, in_ch=1):
        super().__init__()
        self.features = nn.Sequential(
            _conv_mfm(in_ch, 32, 5, 1, 2), nn.MaxPool2d(2),
            _conv_mfm(32, 32, 1, 1, 0), nn.BatchNorm2d(32),
            _conv_mfm(32, 48, 3, 1, 1), nn.MaxPool2d(2), nn.BatchNorm2d(48),
            _conv_mfm(48, 48, 1, 1, 0), nn.BatchNorm2d(48),
            _conv_mfm(48, 64, 3, 1, 1), nn.MaxPool2d(2), nn.BatchNorm2d(64),
            _conv_mfm(64, 64, 1, 1, 0), nn.BatchNorm2d(64),
            _conv_mfm(64, 32, 3, 1, 1), nn.BatchNorm2d(32),
            _conv_mfm(32, 32, 1, 1, 0), nn.BatchNorm2d(32),
            _conv_mfm(32, 32, 3, 1, 1), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(32, 128), MFM_1d(), nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)     # (B, 32)
        return self.fc(x)


class MFM_1d(nn.Module):
    def forward(self, x):
        a, b = torch.chunk(x, 2, dim=1)
        return torch.max(a, b)


if __name__ == "__main__":
    m = LCNN()
    x = torch.randn(4, 1, 80, 401)
    y = m(x)
    n = sum(p.numel() for p in m.parameters())
    print(f"LCNN out {tuple(y.shape)}  params {n:,}")
    assert y.shape == (4, 2)
    print("lcnn.py self-test passed.")
