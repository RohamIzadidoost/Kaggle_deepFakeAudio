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
# 3. ASDG-style aggregation-separation loss (prior-work baseline)
# --------------------------------------------------------------------------
class ASDGLoss(nn.Module):
    """Aggregation-separation domain generalization (Xie et al., IEEE TIFS 2024
    style): aggregate real embeddings across domains, AND separate fake
    embeddings across domains too.

    This exists specifically as the head-to-head baseline against
    RealAnchoredContrastiveLoss above. The two are structurally identical
    except for ONE choice, isolated here on purpose: what happens to fakes.
      * RealAnchoredContrastiveLoss (ours): fakes are pushed away from real by
        a margin, but never pulled toward each other -- unconstrained, because
        fake artifacts are generator-specific and forcing them together is the
        wrong inductive bias (this is the paper's stated differentiator).
      * ASDGLoss (prior work, this class): fakes across DIFFERENT domains are
        explicitly SEPARATED (cosine similarity pushed toward -1 / pairwise
        distance maximised), symmetric to how reals are aggregated. This is
        the standard "aggregation for real, separation for fake" DG recipe.

    Args:
        margin: real-vs-fake separation margin (Euclidean, same as the anchor
            loss) so both losses share this hyperparameter for a fair comparison.
        sep_margin: target MINIMUM cosine distance (1 - cos) for cross-domain
            fake-fake pairs; pairs already this far apart contribute 0 (hinge,
            not unbounded repulsion -- unbounded repulsion has no minimum and
            never converges).
    """

    def __init__(self, margin: float = 1.0, sep_margin: float = 1.0,
                cross_domain_only: bool = True):
        super().__init__()
        self.margin = margin
        self.sep_margin = sep_margin
        self.cross_domain_only = cross_domain_only

    def forward(self, embeddings, labels, domains):
        e = F.normalize(embeddings, dim=1)
        labels = labels.view(-1)
        domains = domains.view(-1)
        real = labels == 0
        fake = labels == 1
        device = e.device
        zero = torch.zeros((), device=device)

        def _cross_domain_pairs(mask):
            idx = mask.nonzero(as_tuple=True)[0]
            if len(idx) < 2:
                return None, None
            ee, dd = e[idx], domains[idx]
            cos = ee @ ee.t()
            pair = torch.ones_like(cos, dtype=torch.bool)
            pair.fill_diagonal_(False)
            if self.cross_domain_only:
                pair &= dd.unsqueeze(0) != dd.unsqueeze(1)
            return cos, pair

        # ---- aggregation: real-real attraction across domains (same as
        # RealAnchoredContrastiveLoss's real-real term, by design -- this is
        # the shared half of the two losses) ----
        cos_r, pair_r = _cross_domain_pairs(real)
        aggregate = (1.0 - cos_r)[pair_r].mean() if (pair_r is not None and pair_r.any()) else zero

        # ---- separation: fake-fake REPULSION across domains (the deliberate
        # contrast with the anchor loss, which leaves this term out entirely) ----
        cos_f, pair_f = _cross_domain_pairs(fake)
        if pair_f is not None and pair_f.any():
            # hinge: only penalise pairs closer (higher cosine) than sep_margin allows
            separate_fake = F.relu(cos_f[pair_f] - (1.0 - self.sep_margin)).mean()
        else:
            separate_fake = zero

        # ---- real-vs-fake margin (kept identical to the anchor loss so the
        # two losses differ in exactly the fake-fake term, nothing else) ----
        if int(real.sum()) > 0 and int(fake.sum()) > 0:
            dist = torch.cdist(e[real], e[fake])
            separate_rf = F.relu(self.margin - dist).mean()
        else:
            separate_rf = zero

        return aggregate + separate_fake + separate_rf


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

    # --- ASDGLoss ---
    asdg = ASDGLoss(margin=1.0, sep_margin=1.0)
    la2 = asdg(e, labels, domains)
    la2.backward()
    assert torch.isfinite(la2) and la2.item() >= 0, "ASDG loss invalid"
    print(f"ASDGLoss (random): {la2.item():.4f}")

    allreal2 = asdg(e, torch.zeros(B, dtype=torch.long), domains)
    allfake2 = asdg(e, torch.ones(B, dtype=torch.long), domains)
    # unlike the anchor loss, ASDG DOES have a fake-fake term, so an all-fake
    # batch of untrained/random embeddings is expected to be > 0 here (random
    # cosine similarities aren't perfectly separated); only all-real should be
    # numerically identical to the anchor loss's all-real case, since that term
    # is shared code between the two losses.
    assert abs(allreal2.item() - allreal.item()) < 1e-5, \
        "the shared real-real aggregation term must match the anchor loss exactly"
    print(f"ASDG all-real: {allreal2.item():.4f} (matches anchor loss exactly)   "
          f"all-fake: {allfake2.item():.4f} (>0 expected -- ASDG has a fake-fake term)")

    # The decisive test: the two losses must actually differ on fake geometry.
    # Construct embeddings where fakes from different domains are CLUSTERED
    # together (cos~1, i.e. the thing ASDG explicitly penalises and the anchor
    # loss explicitly ignores).
    torch.manual_seed(1)
    labels3 = torch.cat([torch.zeros(8, dtype=torch.long), torch.ones(8, dtype=torch.long)])
    domains3 = torch.cat([torch.randint(0, K, (8,)), torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])])
    clustered = torch.zeros(16, D)
    clustered[:8] = torch.randn(1, D) * 0.01                       # reals: tight cluster
    clustered[8:] = torch.randn(1, D) * 0.01 + 10.0                # fakes: ALSO a tight
                                                                    # cluster (same direction),
                                                                    # far from real -> anchor
                                                                    # loss is happy, ASDG is not
    clustered = F.normalize(clustered, dim=1) * 5
    l_anchor_val = anchor(clustered, labels3, domains3).item()
    l_asdg_val = asdg(clustered, labels3, domains3).item()
    print(f"clustered-fakes geometry: anchor={l_anchor_val:.4f} (should be small, "
          f"fakes-clustering is invisible to it)  asdg={l_asdg_val:.4f} (should be "
          f"large, ASDG explicitly penalises cross-domain fake clustering)")
    assert l_asdg_val > l_anchor_val + 0.1, \
        "ASDG and the anchor loss must disagree on clustered-fake geometry -- " \
        "if they don't, the fake-fake separation term isn't doing anything"

    print("losses.py self-test passed.")
