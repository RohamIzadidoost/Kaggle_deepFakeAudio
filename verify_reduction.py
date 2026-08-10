"""Assert that adapt_adaptive reduces exactly to the published adapt().

Run: python verify_reduction.py     (uses the smoke config; needs a GPU)

Why this gate exists: the whole point of the adaptive arms is to be compared
against `ours_fixed`, and that comparison only means anything if the two share
a code path. If `adapt_adaptive(adapt_q=False, adapt_e=False)` and
`extended_pipeline`-style `adapt()` diverge for any reason -- a different tail
rule, a different RNG consumption order, an extra forward pass -- then every
difference in the results table is confounded by the refactor rather than by
the schedules.

RNG order is the subtle part. `adapt_adaptive` does an extra augmented scoring
pass to size the budget, and `augment()` draws from the CUDA RNG; that pass is
therefore gated on `adapt_q=True` so the fixed arm consumes exactly the draws
the published loop does. This script is what catches a regression there.
"""

import copy
import os

os.environ.setdefault("ADAPTIVE_SMOKE", "1")

import numpy as np
import torch

import adaptive_pipeline as P


def run(fn, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = fn(copy.deepcopy(SOURCE), IDX)
    _, s = P.score(model, IDX)
    del model
    torch.cuda.empty_cache()
    return s


if __name__ == "__main__":
    target, seed = P.TARGETS[0], P.SEEDS[0]
    ckpt = f"{P.CKPT_DIR}/source_{target}_seed{seed}.pt"
    assert os.path.exists(ckpt), f"missing {ckpt}"

    tgt_df = P.sample_target(P.pool[P.pool.corpus == target], P.TARGET_PER_CLASS, seed)
    IDX = P.idx_of(tgt_df)
    SOURCE = P.XLSRDetector(encoder_amp=P.ENCODER_AMP).to(P.DEVICE)
    SOURCE.load_state_dict(torch.load(ckpt, map_location=P.DEVICE), strict=False)
    print(f"{target} seed{seed}: {len(IDX)} clips")

    s_ref = run(lambda m, i: P.adapt(m, i))
    s_new = run(lambda m, i: P.adapt_adaptive(m, i, adapt_q=False, adapt_e=False,
                                              tag="reduction", seed=seed,
                                              target=target, method="_reduction_check"))

    same = np.array_equal(s_ref, s_new)
    dmax = float(np.abs(s_ref - s_new).max())
    print(f"\n  bitwise identical : {same}")
    print(f"  max |difference|  : {dmax:.3e}")

    # A non-zero but tiny delta means the two loops did the same arithmetic in a
    # different order, which is still a confound worth knowing about; a large one
    # means they are genuinely different computations.
    if same:
        print("\nPASS -- adapt_adaptive(False, False) is the published adapt()")
    elif dmax < 1e-6:
        print(f"\nPASS (within fp tolerance {dmax:.3e}) -- ordering differs, arithmetic does not")
    else:
        raise SystemExit(f"\nFAIL -- loops diverge by {dmax:.3e}; the ablation is confounded")
