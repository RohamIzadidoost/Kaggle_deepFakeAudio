"""
Domain-balanced batch sampler for DA-RAD.

The source-adversarial (GRL) and Real-Anchored contrastive losses only work if a
batch contains (a) more than one source domain and (b) both classes, so there are
cross-domain real-real pairs and real-fake pairs to act on. A plain random sampler
gives neither guarantee. This sampler builds each batch from several domains, with
classes balanced within each domain where possible (single-class domains such as
MLAAD, which is fake-only, are handled gracefully).

Use as a `batch_sampler` for a DataLoader whose dataset returns (input, label, domain).
"""

import numpy as np
from torch.utils.data import Sampler


class DomainBalancedBatchSampler(Sampler):
    def __init__(self, domains, labels, batch_size=32, domains_per_batch=2,
                 num_batches=None, seed=0):
        self.domains = np.asarray(domains)
        self.labels = np.asarray(labels)
        self.uniq = np.unique(self.domains)
        self.domains_per_batch = min(domains_per_batch, len(self.uniq))
        self.per_domain = max(1, batch_size // self.domains_per_batch)
        self.batch_size = self.per_domain * self.domains_per_batch
        self.rng = np.random.default_rng(seed)

        # index pools per (domain, class) and per domain
        self.pool = {(d, c): np.where((self.domains == d) & (self.labels == c))[0]
                     for d in self.uniq for c in (0, 1)}
        self.dom_pool = {d: np.where(self.domains == d)[0] for d in self.uniq}

        n = len(self.domains)
        self.num_batches = num_batches or max(1, n // self.batch_size)

    def __len__(self):
        return self.num_batches

    def _draw(self, pool, k):
        return self.rng.choice(pool, size=k, replace=len(pool) < k)

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            doms = self.rng.choice(self.uniq, size=self.domains_per_batch, replace=False)
            for d in doms:
                n_real = self.per_domain - self.per_domain // 2
                n_fake = self.per_domain // 2
                picks = []
                for c, k in ((0, n_real), (1, n_fake)):
                    if len(self.pool[(d, c)]) and k:
                        picks.append(self._draw(self.pool[(d, c)], k))
                dpicks = np.concatenate(picks) if picks else np.empty(0, dtype=int)
                if len(dpicks) < self.per_domain:      # single-class domain: top up
                    dpicks = np.concatenate([dpicks, self._draw(self.dom_pool[d],
                                                                self.per_domain - len(dpicks))])
                batch.extend(dpicks[: self.per_domain].tolist())
            self.rng.shuffle(batch)
            yield batch


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # synthetic corpus: 4 domains; domain 3 is fake-only (like MLAAD)
    N = 4000
    domains = rng.integers(0, 4, N)
    labels = rng.integers(0, 2, N)
    labels[domains == 3] = 1                      # MLAAD-style: all fake

    s = DomainBalancedBatchSampler(domains, labels, batch_size=32, domains_per_batch=2, seed=1)
    print(f"batch_size={s.batch_size}  per_domain={s.per_domain}  num_batches={len(s)}")

    multi_dom = both_cls = 0
    for batch in s:
        assert len(batch) == s.batch_size
        b_dom = domains[batch]
        b_lab = labels[batch]
        multi_dom += len(np.unique(b_dom)) >= 2
        both_cls += (0 in b_lab) and (1 in b_lab)
    nb = len(s)
    print(f"batches with >=2 domains: {multi_dom}/{nb}")
    print(f"batches with both classes: {both_cls}/{nb}")
    assert multi_dom == nb, "every batch must be multi-domain"
    assert both_cls >= 0.95 * nb, "almost every batch should have both classes"
    print("sampler.py self-test passed.")
