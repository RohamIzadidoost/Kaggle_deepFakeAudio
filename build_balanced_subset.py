"""
Carve a class-balanced, source-diverse subset out of the full manifest.

The merged manifest is ~97% fake (19k real vs 595k fake), and the fake class is
dominated by ASVspoof (74%, all one "unknown" generator). Training on it as-is
lets a model score ~97% by always predicting "fake" (the exact failure the paper's
LR/KNC hit: 0.00 recall on real). This script fixes both problems:

  * keeps every real file (the minority class), and
  * samples an equal-ish number of fake files, spread evenly across the three
    source datasets AND across MLAAD's ~90 generators, so the fake class is
    diverse rather than 74% ASVspoof.

Allocation uses "water-filling": the fake budget is split as evenly as possible
first across dataset_source, then across generator within each source, capped by
how many files each group actually has (leftover budget from small groups is
redistributed to groups that still have capacity).

Run:
  python build_balanced_subset.py --manifest manifest.csv --out manifest_balanced.csv

Caveat worth noting in your report: MLAAD has NO real audio, so every MLAAD fake
included teaches the model "MLAAD-domain audio => fake". That inflates in-domain
accuracy and is not the same as detecting deepfake artifacts. Tune --ratio / the
source mix if you want to lean less on MLAAD.
"""

import argparse

import pandas as pd


def water_fill(capacities: dict, budget: int) -> dict:
    """Distribute `budget` across keys as evenly as possible, capped per key.

    Small groups take all they have; their leftover share is redistributed to
    groups that still have room, until the budget is spent or every group is full.
    """
    alloc = {k: 0 for k in capacities}
    remaining = min(budget, sum(capacities.values()))
    active = {k for k, c in capacities.items() if c > 0}

    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        for k in list(active):
            room = capacities[k] - alloc[k]
            take = min(share, room, remaining)
            alloc[k] += take
            remaining -= take
            if alloc[k] >= capacities[k]:
                active.discard(k)
            if remaining <= 0:
                break
    return alloc


def sample_fake(fake: pd.DataFrame, n_target: int, seed: int) -> pd.DataFrame:
    """Sample ~n_target fake rows, balanced across dataset_source then generator."""
    source_caps = fake.groupby("dataset_source").size().to_dict()
    source_alloc = water_fill(source_caps, n_target)

    picks = []
    for source, n_source in source_alloc.items():
        if n_source <= 0:
            continue
        sub = fake[fake["dataset_source"] == source]
        gen_caps = sub.groupby("generator").size().to_dict()
        gen_alloc = water_fill(gen_caps, n_source)
        for generator, n_gen in gen_alloc.items():
            if n_gen <= 0:
                continue
            grp = sub[sub["generator"] == generator]
            picks.append(grp.sample(n=n_gen, random_state=seed))
    return pd.concat(picks, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="manifest_balanced.csv")
    ap.add_argument("--ratio", type=float, default=1.0,
                    help="fake-to-real ratio in the subset (1.0 = perfectly balanced)")
    ap.add_argument("--max_real", type=int, default=None,
                    help="cap on real files (default: keep all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, dtype={"extra": str}, low_memory=False)
    real = df[df["label"] == "real"]
    fake = df[df["label"] == "fake"]

    if args.max_real is not None and len(real) > args.max_real:
        real = real.sample(n=args.max_real, random_state=args.seed)

    n_fake = round(args.ratio * len(real))
    fake_subset = sample_fake(fake, n_fake, args.seed)

    out = pd.concat([real, fake_subset], ignore_index=True)
    out = out.sample(frac=1, random_state=args.seed).reset_index(drop=True)  # shuffle
    out.to_csv(args.out, index=False)

    print(f"Wrote {len(out)} rows to {args.out}")
    print(f"  real: {len(real)}  fake: {len(fake_subset)}")
    print("\nFake composition by source:")
    print(fake_subset.groupby("dataset_source").size().to_string())
    print(f"\nDistinct fake generators represented: {fake_subset['generator'].nunique()}")


if __name__ == "__main__":
    main()
