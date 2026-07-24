"""
DA-RAD detector: SSL/frame encoder -> attentive pooling -> shared embedding ->
binary real/fake head. The shared embedding is what the DA-RAD training losses
act on (source-adversarial GRL + Real-Anchored contrastive), so `forward` returns
both the class logits and the L2-normalisable embedding.

The domain-adversarial classifier and the anchor loss live in losses.py; this
module only produces the embedding they consume, keeping the model reusable as a
plain detector (embedding ignored) or inside the full DA-RAD objective.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentivePool(nn.Module):
    """Attention pooling over time for (B, T, C) -> (B, C) + weights (B, T)."""

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):                       # (B, T, C)
        attn = torch.softmax(self.score(x).squeeze(-1), dim=1)   # (B, T)
        ctx = torch.bmm(attn.unsqueeze(1), x).squeeze(1)         # (B, C)
        return ctx, attn


class DARADModel(nn.Module):
    def __init__(self, encoder, embed_dim=256, n_classes=2, dropout=0.3):
        super().__init__()
        self.encoder = encoder                  # exposes .out_dim, forward->(B,T,C)
        self.pool = AttentivePool(encoder.out_dim)
        self.proj = nn.Sequential(
            nn.Linear(encoder.out_dim, embed_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.classifier = nn.Linear(embed_dim, n_classes)
        self.embed_dim = embed_dim

    def forward(self, waveform, return_attn=False):
        feats = self.encoder(waveform)          # (B, T', C)
        ctx, attn = self.pool(feats)            # (B, C)
        emb = self.proj(ctx)                    # (B, embed_dim)  -> DA-RAD losses
        logits = self.classifier(emb)           # (B, n_classes)
        if return_attn:
            return logits, emb, attn
        return logits, emb


if __name__ == "__main__":
    # full DA-RAD assembly wired to the training losses, on a stub encoder
    from models.ssl_frontend import StubEncoder
    from losses import DomainAdversarialLoss, RealAnchoredContrastiveLoss

    torch.manual_seed(0)
    B, K = 12, 4                                # batch, n domains
    model = DARADModel(StubEncoder(out_dim=256), embed_dim=256)
    wav = torch.randn(B, 48000)
    labels = torch.randint(0, 2, (B,))
    domains = torch.randint(0, K, (B,))

    logits, emb = model(wav)
    assert logits.shape == (B, 2) and emb.shape == (B, 256)

    dom_loss = DomainAdversarialLoss(embed_dim=256, n_domains=K)
    anchor = RealAnchoredContrastiveLoss(margin=1.0)

    L = (F.cross_entropy(logits, labels)
         + 0.5 * dom_loss(emb, domains, lambd=1.0)
         + 0.3 * anchor(emb, labels, domains))
    L.backward()

    n = sum(p.numel() for p in model.parameters())
    print(f"DARADModel(stub) logits {tuple(logits.shape)} emb {tuple(emb.shape)} "
          f"params {n:,}")
    print(f"combined DA-RAD loss = {L.item():.4f}  (backward OK, grads flow)")
    print("dg_model.py self-test passed.")
