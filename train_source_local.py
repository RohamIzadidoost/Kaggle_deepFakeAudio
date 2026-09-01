"""Train the paper's own source models on this box, so the main method is not
capped at the three seeds that survived the cloud.

Why this exists. `results_ext.csv` reports ten seeds per target, but only
`ckpt_ext/` seeds 0-2 were ever saved; seeds 3-9 were trained on a cloud
instance whose checkpoints are gone. Every local experiment on OUR model has
therefore been stuck at n=3 -- including the E curve, where three seeds cannot
support a per-target Wilcoxon (n=10 is the minimum that can reach p<0.05 at
all, 2/2^10).

Why it turns out to be affordable. Source training was assumed to be a full
300M-parameter XLS-R fine-tune that could not fit in 10 GB. It is not:
`XLSRDetector` freezes the whole encoder and unfreezes only the top-4
transformer layers, so roughly 50M parameters are trained. Optimiser state is
~800 MB rather than ~5 GB, which is why the saved checkpoints are 194 MB and
not 1.2 GB. With CACHE_ON_CPU=1 keeping the 4.3 GB waveform cache off the card,
the whole thing fits.

This imports `adaptive_pipeline` rather than copying it. That file guards its
grid behind `if __name__ == "__main__"`, so importing gives us its pool
construction, cache, model and `fit()` with no duplication and no drift -- the
repo already carries two verbatim copies of this pipeline and does not need a
third.

    python train_source_local.py --probe                  # VRAM/time feasibility only
    python train_source_local.py --fidelity               # retrain seed 0, compare to cloud
    python train_source_local.py --seeds 3 4 5 6 7 8 9    # the real thing
"""

import argparse
import os
import time

os.environ.setdefault("ADAPTIVE_SMOKE", "0")
os.environ.setdefault("CACHE_ON_CPU", "1")
# adaptive_pipeline omits MLAAD by design, because it does not train. This file
# does, and MLAAD is part of every source pool behind results_ext.csv. Training
# without it missed the cloud seed-0 checkpoint by +7.3 EER on Arabic and -8.1 on
# dataset2. Verified not to disturb the target pools: their path hashes are
# identical with and without it, since MLAAD rows are concatenated last.
os.environ.setdefault("INCLUDE_MLAAD", "1")

import pandas as pd
import torch

import adaptive_pipeline as AP


def source_pool_for(target, seed):
    """The source pool for a fold: every corpus except the target, plus MLAAD.

    Identical to extended_pipeline's rule, which is what produced the published
    checkpoints -- a different pool here would make locally trained seeds
    incomparable to the cloud-trained ones already in results_ext.csv.
    """
    sp = AP.pool[(AP.pool.corpus != target) & (AP.pool.corpus != "mlaad")]
    sp = pd.concat([sp, AP.mlaad_pool])
    return AP.sample_source(sp, AP.SOURCE_PER_CLASS, seed)


def train_one(target, seed, out_path, epochs=None, log=AP.log):
    epochs = epochs or AP.SOURCE_EPOCHS
    torch.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)

    src_df = source_pool_for(target, seed)
    src_idx = AP.idx_of(src_df)
    log(f"  source pool {len(src_df)} clips from {sorted(src_df.corpus.unique())}")

    torch.cuda.reset_peak_memory_stats()
    model = AP.XLSRDetector(encoder_amp=AP.ENCODER_AMP).to(AP.DEVICE)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log(f"  trainable {n_tr/1e6:.1f}M / {n_all/1e6:.1f}M params ({100*n_tr/n_all:.1f}%)")

    t0 = time.time()
    AP.fit(model, src_idx, epochs, tag=f"seed{seed}/{target}")
    mins = (time.time() - t0) / 60
    peak = torch.cuda.max_memory_allocated() / 2**30
    log(f"  trained in {mins:.1f} min, peak VRAM {peak:.2f} GiB")

    if out_path:
        torch.save({n: p.detach().cpu() for n, p in model.named_parameters()
                    if p.requires_grad}, out_path)
        log(f"  saved {out_path}")
    return model, dict(minutes=round(mins, 1), peak_gib=round(peak, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=AP.EER_TARGETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 4, 5, 6, 7, 8, 9])
    ap.add_argument("--ckpt_dir", default="ckpt_ext")
    ap.add_argument("--probe", action="store_true",
                    help="one fold, one epoch: does it fit, and how fast")
    ap.add_argument("--fidelity", action="store_true",
                    help="retrain a seed that already has a cloud checkpoint and "
                         "compare, before trusting any locally trained seed")
    args = ap.parse_args()

    if args.probe:
        AP.log("=== PROBE: one fold, one epoch ===")
        _, info = train_one("in_the_wild", 0, None, epochs=1)
        AP.log(f"PROBE RESULT: peak {info['peak_gib']} GiB, "
               f"{info['minutes']:.1f} min/epoch -> "
               f"{info['minutes']*AP.SOURCE_EPOCHS:.0f} min per source model, "
               f"{info['minutes']*AP.SOURCE_EPOCHS*28/60:.1f} h for 28 models")
        return

    if args.fidelity:
        # A locally trained seed is only usable if local training reproduces what
        # the cloud produced. Seed 0 has a surviving checkpoint, so retrain it here
        # and compare source EER on the same target pool. Trained to a scratch path
        # so the real checkpoint is never overwritten.
        rows = []
        for target in args.targets:
            cloud = f"{args.ckpt_dir}/source_{target}_seed0.pt"
            if not os.path.exists(cloud):
                AP.log(f"  no cloud checkpoint for {target}, skipping"); continue
            AP.log(f"=== fidelity: {target} seed 0 ===")
            tgt_df = AP.sample_target(AP.pool[AP.pool.corpus == target],
                                      AP.TARGET_PER_CLASS, 0)
            tgt_idx = AP.idx_of(tgt_df)

            ref = AP.XLSRDetector(encoder_amp=AP.ENCODER_AMP).to(AP.DEVICE)
            ref.load_state_dict(torch.load(cloud, map_location=AP.DEVICE), strict=False)
            m_ref, _, _ = AP.metrics(ref, tgt_idx)
            del ref; torch.cuda.empty_cache()

            local, info = train_one(target, 0, f"/tmp/local_{target}_seed0.pt")
            m_loc, _, _ = AP.metrics(local, tgt_idx)
            del local; torch.cuda.empty_cache()

            rows.append(dict(target=target, cloud_eer=round(m_ref["eer"], 2),
                             local_eer=round(m_loc["eer"], 2),
                             cloud_auc=round(m_ref["auc"], 4),
                             local_auc=round(m_loc["auc"], 4), **info))
            AP.log(f"  CLOUD EER {m_ref['eer']:.2f} / AUC {m_ref['auc']:.4f}  vs  "
                   f"LOCAL EER {m_loc['eer']:.2f} / AUC {m_loc['auc']:.4f}")
        df = pd.DataFrame(rows)
        df.to_csv("fidelity_local_vs_cloud.csv", index=False)
        AP.log("\n" + df.to_string(index=False))
        AP.log("wrote fidelity_local_vs_cloud.csv")
        return

    os.makedirs(args.ckpt_dir, exist_ok=True)
    for seed in args.seeds:
        for target in args.targets:
            out = f"{args.ckpt_dir}/source_{target}_seed{seed}.pt"
            if os.path.exists(out):
                AP.log(f"=== seed{seed}/{target}: checkpoint present, skipping ===")
                continue
            AP.log(f"=== training seed{seed}/{target} ===")
            try:
                _, info = train_one(target, seed, out)
            except Exception as e:
                AP.log(f"  !! failed: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
