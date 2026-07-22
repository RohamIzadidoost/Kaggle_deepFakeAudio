"""
Split a manifest into SPEAKER-DISJOINT train/val/test CSVs.

Why not a plain random split: the paper (and an earlier version of this script)
split rows at random, so the same speaker's / TTS-reader's utterances land in both
train and test. The model then memorises voice identity instead of learning
real-vs-fake artifacts, and reported accuracy is inflated. Here we split on the
`speaker` column (built by build_manifest.py) so no speaker crosses splits, while
StratifiedGroupKFold keeps the real/fake ratio roughly constant in each split.

Run:
  python split_manifest.py --manifest manifest_balanced.csv --out_dir splits/
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def _grouped_holdout(df, frac, seed):
    """Split off ~frac of the rows as a speaker-disjoint, label-stratified holdout.

    Returns (rest_df, holdout_df). Implemented via one fold of a
    StratifiedGroupKFold with n_splits≈round(1/frac).
    """
    n_splits = max(2, round(1 / frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, hold_idx = next(sgkf.split(df, df["label"], groups=df["speaker"]))
    return df.iloc[rest_idx].copy(), df.iloc[hold_idx].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", default="splits")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, dtype={"extra": str}, low_memory=False)
    if "speaker" not in df.columns:
        raise SystemExit(
            "manifest has no 'speaker' column — rebuild it with the current "
            "build_manifest.py so speaker-disjoint splitting is possible."
        )
    print(f"Loaded {len(df)} rows / {df['speaker'].nunique()} speakers. "
          f"Label counts:\n{df['label'].value_counts()}")

    # Carve test first, then val out of what remains — both speaker-disjoint.
    train_val_df, test_df = _grouped_holdout(df, args.test_frac, args.seed)
    val_rel = args.val_frac / (1 - args.test_frac)
    train_df, val_df = _grouped_holdout(train_val_df, val_rel, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    train_df.to_csv(f"{args.out_dir}/train.csv", index=False)
    val_df.to_csv(f"{args.out_dir}/val.csv", index=False)
    test_df.to_csv(f"{args.out_dir}/test.csv", index=False)

    # Verify disjointness and report composition.
    tr, va, te = set(train_df["speaker"]), set(val_df["speaker"]), set(test_df["speaker"])
    assert not (tr & va) and not (tr & te) and not (va & te), "speaker leak between splits!"
    print("\nSpeaker-disjoint check: PASSED (no speaker shared across splits)")
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = split["label"].value_counts().to_dict()
        print(f"{name}: {len(split)} rows, {split['speaker'].nunique()} speakers -> {counts}")


if __name__ == "__main__":
    main()
