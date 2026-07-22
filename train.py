"""
Train AttentiveSpecCNN on the speaker-disjoint splits and evaluate with EER.

Reports EER / balanced-accuracy / F1 on the test set (via metrics.py) and appends
the deep-learning row to baseline_results.csv so it sits next to the paper baseline.

Run:
  python train.py --epochs 15
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepfake_dataset import DeepfakeAudioDataset, LABEL_TO_IDX, compute_class_weights
from metrics import evaluation_report
from model import AttentiveSpecCNN


def make_loader(csv, batch_size, workers, train):
    ds = DeepfakeAudioDataset(csv, random_crop=train)
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=workers, pin_memory=True, drop_last=train), ds


@torch.no_grad()
def collect_scores(model, loader, device):
    """Return (labels, P(fake)) over a loader."""
    model.eval()
    ys, scores = [], []
    for specs, labels in loader:
        logits = model(specs.to(device))
        p_fake = torch.softmax(logits, dim=1)[:, LABEL_TO_IDX["fake"]]
        scores.append(p_fake.cpu().numpy())
        ys.append(labels.numpy())
    return np.concatenate(ys), np.concatenate(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="attentive_spec_cnn.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader, train_ds = make_loader("splits/train.csv", args.batch_size, args.workers, True)
    val_loader, _ = make_loader("splits/val.csv", args.batch_size, args.workers, False)
    test_loader, _ = make_loader("splits/test.csv", args.batch_size, args.workers, False)

    model = AttentiveSpecCNN().to(device)
    weights = compute_class_weights(train_ds.df).to(device)  # ~[1,1] on balanced data
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_eer = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for specs, labels in pbar:
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(specs), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{running / (pbar.n + 1):.4f}")

        yval, sval = collect_scores(model, val_loader, device)
        val = evaluation_report(yval, sval, verbose=False)
        scheduler.step(val["eer"])
        print(f"  val: EER {val['eer']*100:.2f}%  bal-acc {val['balanced_accuracy']*100:.2f}%  AUC {val['auc']:.4f}")

        if val["eer"] < best_val_eer:
            best_val_eer = val["eer"]
            torch.save(model.state_dict(), args.out)
            print(f"  -> saved best (val EER {best_val_eer*100:.2f}%)")

    # Final evaluation with the best checkpoint.
    print("\n" + "=" * 60)
    print(f"TEST-SET RESULTS (best val checkpoint, val EER {best_val_eer*100:.2f}%)")
    print("=" * 60)
    model.load_state_dict(torch.load(args.out))
    ytest, stest = collect_scores(model, test_loader, device)
    test = evaluation_report(ytest, stest)

    # Append to the comparison table alongside the paper baseline.
    row = {"features": "log-mel", "model": "AttentiveSpecCNN",
           "eer": test["eer"] * 100, "bal_acc": test["balanced_accuracy"] * 100,
           "f1_fake": test["f1_fake"] * 100, "auc": test["auc"]}
    try:
        res = pd.read_csv("baseline_results.csv")
        res = res[res["model"] != "AttentiveSpecCNN"]  # replace prior DL row
        res = pd.concat([res, pd.DataFrame([row])], ignore_index=True)
    except FileNotFoundError:
        res = pd.DataFrame([row])
    res.to_csv("results.csv", index=False)
    print("\nSaved combined comparison to results.csv")


if __name__ == "__main__":
    main()
