# Training the main method locally (2026-08-31)

`results_ext.csv` reports ten seeds per target, but only `ckpt_ext/` seeds 0-2
survived the cloud instance. Every local experiment on our own model has
therefore been capped at n=3 -- including the E curve, where three seeds cannot
support a per-target Wilcoxon at all (n=10 is the minimum that can reach
p<0.05: 2/2^10 = 0.002).

## Source training was never infeasible

It was assumed to be a full 300M-parameter XLS-R fine-tune that would not fit on
a 10 GB card. It is not. `XLSRDetector` freezes the encoder and unfreezes only
the top-4 transformer layers: **50.6M trainable of 315.7M**, which is why the
saved checkpoints are 194 MB rather than 1.2 GB. Measured on this box, with
`CACHE_ON_CPU=1` keeping the 4.3 GB waveform cache off the GPU:

| | |
|---|---|
| peak VRAM | **3.64 GiB** |
| per source model (8 epochs) | **~6.6 min** |
| all 28 missing checkpoints | **~3 h** |

## The first attempt failed its own gate, and why

`train_source_local.py` imports `adaptive_pipeline` rather than making a third
verbatim copy of the pipeline. But that file states plainly that it omits MLAAD
*because it never trains* -- and MLAAD is part of every source pool behind
`results_ext.csv`. Training through it therefore dropped an entire corpus from
the source pool. Retraining seed 0 and comparing against the surviving cloud
checkpoints:

| target | cloud | local, no MLAAD | local, with MLAAD |
|---|---|---|---|
| arabic | 20.91 | **28.21** | 22.24 |
| dataset2 | 37.24 | **29.12** | 36.02 |
| in_the_wild | 12.78 | 14.60 | 13.28 |
| asvspoof2019 | 4.69 | 5.25 | 6.90 |

MLAAD also lives at a different root on this machine (`data/dataset_3`, not the
`data/mlaad` the pipeline globs), so the loader now searches both and logs which
it found.

## Gate criterion, and the result

Exact reproduction of a cloud run is not achievable -- different file ordering
into `sample_source`, different GPU non-determinism. The right criterion is
whether local training lands inside the **cloud's own seed-to-seed spread**,
which `results_ext.csv` measures over ten seeds:

| target | cloud seeds 0-9 | local seed 0 | z |
|---|---|---|---|
| arabic | 22.50 ± 1.86 | 22.24 | −0.1 |
| in_the_wild | 12.78 ± 1.24 | 13.28 | +0.4 |
| asvspoof2019 | 5.03 ± 1.31 | 6.90 | +1.4 |
| dataset2 | 33.54 ± 1.83 | 36.02 | +1.4 |

All four in range. **PASS.**

## What this does and does not license

It licenses treating locally trained seeds 3-9 as draws from the same
distribution as cloud seeds 0-9. The resulting ten-seed set has **mixed
provenance**: seeds 0-2 cloud-trained, seeds 3-9 trained here. That must be
stated wherever the ten-seed numbers are reported, and the two groups should be
compared for systematic offset before pooling.

It does *not* license claiming these are the same checkpoints the cloud
produced. Local seed 0 is not cloud seed 0 (6.90 vs 4.69 EER on ASVspoof2019);
it is a different draw from the same process.

## Verified, not assumed

Restoring MLAAD adds 8,000 rows to `pool`, which could have shifted the target
pools and invalidated every completed E-sweep result. It does not: SHA-256 of
the path lists for all four target corpora, and of their seed-0 samples, are
**identical** with and without MLAAD, because MLAAD rows are concatenated after
the per-corpus caps. The change is gated behind `INCLUDE_MLAAD=1`, off by
default, so results already recorded against `adaptive_pipeline` stay
reproducible.
