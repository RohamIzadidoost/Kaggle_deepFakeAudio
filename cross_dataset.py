"""
Leave-one-source-out cross-dataset generalisation test.

The in-domain result (train/test drawn from the same three sources) can be
flattered by dataset-specific quirks — especially since MLAAD contributes only
fake audio. This probes the harder question: does the model detect deepfake
ARTIFACTS that transfer to an unseen source? For each held-out source we train a
fresh AttentiveSpecCNN on the other two and evaluate on the held-out one.

Caveats baked into the data:
  * MLAAD has no real audio -> when it is the test source we can only report the
    fake-detection RATE (recall on unseen generators), not EER.
  * When ASVspoof is held out, the only training real speech is dataset_2's ~2.3k
    LibriSpeech clips -> that fold is real-starved; read it as a lower bound.

Run:
  python cross_dataset.py --epochs 10
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from build_balanced_subset import sample_fake
from deepfake_dataset import LABEL_TO_IDX
from metrics import compute_eer
from model import AttentiveSpecCNN
from train import collect_scores, make_loader

SOURCES = {
    "ASVspoof": "dataset_1_asvspoof2021_DF",
    "dataset_2": "dataset_2_seedtts_style",
    "MLAAD": "dataset_3_mlaad_v5",
}


def build_fold(df, holdout, seed, out_dir, test_cap=4000):
    """Write train.csv (balanced, other two sources) and test.csv (held-out source)."""
    src = SOURCES[holdout]
    train_df = df[df["dataset_source"] != src]
    test_df = df[df["dataset_source"] == src]

    # balance train: keep all real, sample a matching, source/generator-diverse fake set
    real = train_df[train_df["label"] == "real"]
    fake = train_df[train_df["label"] == "fake"]
    fake_bal = sample_fake(fake, len(real), seed) if len(real) else fake
    train_bal = (pd.concat([real, fake_bal])
                 .sample(frac=1, random_state=seed).reset_index(drop=True))

    # cap the test set (balanced if the source has both classes) for fast eval
    if test_df["label"].nunique() > 1:
        per = test_cap // 2
        test_df = pd.concat([
            test_df[test_df.label == "real"].sample(min(per, (test_df.label == "real").sum()), random_state=seed),
            test_df[test_df.label == "fake"].sample(min(per, (test_df.label == "fake").sum()), random_state=seed),
        ])
    else:
        test_df = test_df.sample(min(test_cap, len(test_df)), random_state=seed)

    os.makedirs(out_dir, exist_ok=True)
    train_csv = f"{out_dir}/{holdout}_train.csv"
    test_csv = f"{out_dir}/{holdout}_test.csv"
    train_bal.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    return train_csv, test_csv, train_bal, test_df


def fit(train_csv, epochs, lr, workers, device):
    loader, ds = make_loader(train_csv, 64, workers, True)
    model = AttentiveSpecCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model.train()
        for specs, labels in tqdm(loader, desc=f"  epoch {epoch}/{epochs}", leave=False):
            optimizer.zero_grad()
            loss = criterion(model(specs.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="folds")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.manifest, dtype={"extra": str}, low_memory=False)

    rows = []
    for holdout in SOURCES:
        print(f"\n=== Hold out {holdout} (train on the other two) ===")
        train_csv, test_csv, train_bal, test_df = build_fold(df, holdout, args.seed, args.out_dir)
        print(f"  train {len(train_bal)} ({train_bal['label'].value_counts().to_dict()}), "
              f"test {len(test_df)} ({test_df['label'].value_counts().to_dict()})")

        model = fit(train_csv, args.epochs, args.lr, args.workers, device)
        test_loader, _ = make_loader(test_csv, 64, args.workers, False)
        y, s = collect_scores(model, test_loader, device)

        if len(np.unique(y)) > 1:
            eer, _ = compute_eer(y, s)
            from sklearn.metrics import balanced_accuracy_score, roc_auc_score
            preds = (s >= 0.5).astype(int)
            m = {"test_source": holdout, "metric": "EER",
                 "eer": eer * 100, "bal_acc": balanced_accuracy_score(y, preds) * 100,
                 "auc": roc_auc_score(y, s), "fake_recall": np.nan}
            print(f"  -> EER {eer*100:.2f}%  bal-acc {m['bal_acc']:.2f}%  AUC {m['auc']:.3f}")
        else:
            # fake-only source: report detection rate at 0.5
            fake_recall = float((s >= 0.5).mean()) * 100  # all y == fake here
            m = {"test_source": holdout, "metric": "fake-recall(unseen gen)",
                 "eer": np.nan, "bal_acc": np.nan, "auc": np.nan, "fake_recall": fake_recall}
            print(f"  -> fake-detection rate on unseen generators: {fake_recall:.2f}%")
        rows.append(m)

    res = pd.DataFrame(rows)
    print("\n" + "=" * 64)
    print("CROSS-DATASET (leave-one-source-out) GENERALISATION")
    print("=" * 64)
    with pd.option_context("display.float_format", lambda v: f"{v:.2f}"):
        print(res.to_string(index=False))
    res.to_csv("cross_dataset_results.csv", index=False)
    print("\nSaved cross_dataset_results.csv")


if __name__ == "__main__":
    main()
