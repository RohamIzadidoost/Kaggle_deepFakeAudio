"""
Pool the re-foldered In-the-Wild mirror back into the canonical flat set.

The Kaggle mirror (`bhaveshkumars/release-in-the-wild`) ships ITW split into
train/val/test x real/fake. Published ITW numbers are quoted on the *whole*
release (19,963 real + 11,816 fake = 31,779 clips), so we pool all six folders
back together -- otherwise our reproduced source EER is not comparable to
anything in the literature and the correctness gate is meaningless.

Label convention here is the repo's own: LABEL_TO_IDX = {real: 0, fake: 1}.
Public checkpoints often use the opposite (bonafide=1); that flip is handled at
scoring time, not here.

    python build_itw_manifest.py --out manifest_itw.csv
"""

import argparse
import os
import pandas as pd

ROOT = "data/in_the_wild/release_in_the_wild"
SPLITS = ["train", "val", "test"]
CLASSES = {"real": 0, "fake": 1}

# Canonical In-the-Wild composition (Mueller et al., 2022).
EXPECT_REAL, EXPECT_FAKE = 19963, 11816


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default="manifest_itw.csv")
    args = ap.parse_args()

    rows = []
    for split in SPLITS:
        for cls, label in CLASSES.items():
            d = os.path.join(args.root, split, cls)
            if not os.path.isdir(d):
                raise SystemExit(f"missing directory: {d}")
            for fn in sorted(os.listdir(d)):
                if not fn.lower().endswith(".wav"):
                    continue
                rows.append({
                    "path": os.path.join(d, fn),
                    "label": label,
                    "label_name": cls,
                    "mirror_split": split,   # kept for provenance only; NOT used to subset
                    "clip_id": os.path.splitext(fn)[0],
                })

    df = pd.DataFrame(rows)

    # Filenames are unique across the canonical release; if the mirror duplicated
    # clips into several splits we must know before quoting a comparable EER.
    dupes = df["clip_id"].duplicated().sum()
    n_real = int((df.label == 0).sum())
    n_fake = int((df.label == 1).sum())

    print(f"real {n_real}  fake {n_fake}  total {len(df)}  duplicate clip_ids {dupes}")
    if (n_real, n_fake) != (EXPECT_REAL, EXPECT_FAKE):
        print(f"WARNING: composition differs from canonical ITW "
              f"({EXPECT_REAL} real / {EXPECT_FAKE} fake). Published-number "
              f"comparisons are only valid on the full release.")
    else:
        print("composition matches canonical In-the-Wild release")
    if dupes:
        print(f"WARNING: {dupes} duplicate clip ids -- mirror is not a clean partition")

    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
