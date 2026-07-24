"""
DA-RAD training objectives.

Total loss:  L = L_BCE + lambda_grl * L_GRL + lambda_anchor * L_Anchor

Two custom pieces (the rest is standard cross-entropy on the real/fake head):

1. GradientReversalLayer / DomainAdversarialLoss
   A gradient-reversal layer feeds a domain (corpus/source) classifier. In the
   forward pass it is the identity; in the backward pass it multiplies the
   gradient by -lambda, so the encoder is trained to make embeddings
   *dataset-invariant* while the domain head still tries to predict the source.

2. RealAnchoredContrastiveLoss  (DA-RAD's primary novelty)
   ASYMMETRIC by design, and this is the key difference from ASDG-style
   aggregation/separation: we anchor ONLY the real class. Genuine speech from
   different corpora is pulled to a shared manifold (real-real attraction);
   fakes are pushed away from real by a margin but are left otherwise
   UNCONSTRAINED (no fake-fake clustering), because fakes are generator-specific
   and forcing them together is the wrong inductive bias.

All losses operate on L2-normalised embeddings so cosine similarity and
Euclidean distance are consistent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 1. Gradient reversal for source-adversarial training
# --------------------------------------------------------------------------
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd: float = 1.0):
    """Identity forward, gradient multiplied by -lambd on the backward pass."""
    return _GradReverse.apply(x, lambd)


class DomainAdversarialLoss(nn.Module):
    """Cross-entropy of a domain classifier placed behind a gradient-reversal layer.

    `lambd` typically ramps 0 -> 1 over training (GRL is unstable if applied at
    full strength from the start); the trainer sets it per-step.
    """

    def __init__(self, embed_dim: int, n_domains: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True),
            nn.Linear(embed_dim, n_domains),
        )

    def forward(self, embeddings, domain_labels, lambd: float = 1.0):
        logits = self.classifier(grad_reverse(embeddings, lambd))
        return F.cross_entropy(logits, domain_labels)


# --------------------------------------------------------------------------
# 2. Real-Anchored Contrastive Loss (primary novelty)
# --------------------------------------------------------------------------
class RealAnchoredContrastiveLoss(nn.Module):
    """Asymmetric contrastive alignment of genuine speech.

    Given L2-normalised embeddings, labels (1 = fake, 0 = real) and domain ids:
      * real-real term  : 1 - cos(e_i, e_j) for real i, j from DIFFERENT domains
                          (align genuine speech across corpora onto one manifold).
      * real-fake term  : hinge  max(0, margin - ||e_real - e_fake||)
                          (push fakes at least `margin` away from real).
    Fakes are never pulled toward each other. Cross-domain-only real pairs stop
    the loss from collapsing to a trivial within-corpus solution.

    Args:
        margin: separation margin in embedding distance units (default 1.0).
        cross_domain_only: if True (default), real-real attraction uses only
            pairs from different domains.
    """

    def __init__(self, margin: float = 1.0, cross_domain_only: bool = True):
        super().__init__()
        self.margin = margin
        self.cross_domain_only = cross_domain_only

    def forward(self, embeddings, labels, domains):
        e = F.normalize(embeddings, dim=1)
        labels = labels.view(-1)
        domains = domains.view(-1)
        real = labels == 0
        fake = labels == 1

        device = e.device
        zero = torch.zeros((), device=device)
        n_real = int(real.sum())
        if n_real == 0:                       # nothing to anchor in this batch
            return zero

        e_real = e[real]
        dom_real = domains[real]

        # ---- real-real attraction (cross-domain pairs) ----
        cos = e_real @ e_real.t()             # (R, R) cosine sim (unit vectors)
        pair = torch.ones_like(cos, dtype=torch.bool)
        pair.fill_diagonal_(False)            # exclude self-pairs
        if self.cross_domain_only:
            pair &= dom_real.unsqueeze(0) != dom_real.unsqueeze(1)
        if pair.any():
            attract = (1.0 - cos)[pair].mean()
        else:
            attract = zero

        # ---- real-fake separation (hinge on Euclidean distance) ----
        if int(fake.sum()) > 0:
            e_fake = e[fake]
            dist = torch.cdist(e_real, e_fake)       # (R, F)
            separate = F.relu(self.margin - dist).mean()
        else:
            separate = zero

        return attract + separate


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, D, K = 16, 64, 4          # batch, embed dim, n domains

    # --- GRL / domain adversarial ---
    emb = torch.randn(B, D, requires_grad=True)
    dom = torch.randint(0, K, (B,))
    dloss = DomainAdversarialLoss(D, K)
    l = dloss(emb, dom, lambd=1.0)
    l.backward()
    # gradient on emb should be non-zero and (via reversal) point to *increase*
    # domain confusion; just assert it flows.
    assert emb.grad is not None and torch.isfinite(l), "GRL backward failed"
    print(f"DomainAdversarialLoss: {l.item():.4f}  (grad flows: {emb.grad.abs().sum()>0})")

    # --- Real-Anchored Contrastive Loss ---
    anchor = RealAnchoredContrastiveLoss(margin=1.0)
    labels = torch.randint(0, 2, (B,))
    domains = torch.randint(0, K, (B,))
    e = torch.randn(B, D, requires_grad=True)
    la = anchor(e, labels, domains)
    la.backward()
    assert torch.isfinite(la) and la.item() >= 0, "anchor loss invalid"
    print(f"RealAnchoredContrastiveLoss (random): {la.item():.4f}")

    # sanity: well-separated real manifold + distant fakes -> ~0 loss
    good = torch.zeros(B, D)
    good[labels == 0] = torch.randn(1, D) * 0.01          # reals collapse together
    good[labels == 1] = torch.randn(int((labels == 1).sum()), D) * 0.01 + 10.0  # fakes far away
    lg = anchor(F.normalize(good, dim=1) * 5, labels, domains)
    print(f"RealAnchoredContrastiveLoss (ideal geometry): {lg.item():.4f}  (should be small)")

    # edge cases: all-real and all-fake batches must not crash
    allreal = anchor(e, torch.zeros(B, dtype=torch.long), domains)
    allfake = anchor(e, torch.ones(B, dtype=torch.long), domains)
    print(f"all-real batch: {allreal.item():.4f}   all-fake batch: {allfake.item():.4f} (0.0 expected)")
    print("losses.py self-test passed.")
