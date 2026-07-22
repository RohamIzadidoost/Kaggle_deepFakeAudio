"""
Reproduce the paper's "MFCC-GNB XtractNet" classical baseline on OUR merged data
and OUR speaker-disjoint splits, so we have an honest number to beat.

Faithful to the paper (Gujjar et al., 2024, Algorithm 1):
  MFCC features  ->  GNB probabilistic "transfer" features + NMF refinement
                 ->  {GNB, LogisticRegression, RandomForest, KNN} classifiers

Two feature sets are evaluated per classifier, mirroring the paper's Table 4:
  (a) raw MFCC features, and
  (b) the GNB+NMF "new transfer features".

Note on faithfulness: the paper's literal chain MFCC->GNB->NMF is degenerate (the
GNB step outputs only a 2-D probability vector, which NMF cannot meaningfully
refine). We implement the sensible interpretation of its stated intent — GNB
probability features concatenated with an NMF parts-based decomposition of the
MFCCs — which is what "enrich MFCC with GNB-probabilistic features refined by NMF"
actually amounts to. Unlike the paper we also report EER on a leakage-free split.

Run:
  python paper_baseline.py            # uses splits/{train,val,test}.csv
"""

import hashlib
import os

import numpy as np
import pandas as pd
import torch
import torchaudio
from joblib import Parallel, delayed
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from deepfake_dataset import LABEL_TO_IDX, load_audio
from metrics import compute_eer

SAMPLE_RATE = 16000
N_MFCC = 40
CACHE_DIR = "features_cache"

_mfcc = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 80},
)


def _file_features(filepath: str) -> np.ndarray:
    """MFCC mean+std over time -> a fixed 2*N_MFCC vector. Zero on unreadable files."""
    try:
        w, sr = load_audio(filepath)  # soundfile + ffmpeg fallback
        if w.shape[0] > 1:
            w = w.mean(0, keepdim=True)
        if sr != SAMPLE_RATE:
            w = torchaudio.functional.resample(w, sr, SAMPLE_RATE)
        m = _mfcc(w).squeeze(0)  # (N_MFCC, T)
        return torch.cat([m.mean(1), m.std(1)]).numpy()
    except Exception:
        return np.zeros(2 * N_MFCC, dtype="float32")


def extract_split(csv_path: str):
    """Return (X, y) for a split, caching features keyed by the file list hash."""
    df = pd.read_csv(csv_path, dtype={"extra": str}, low_memory=False)
    key = hashlib.md5(("|".join(df["filepath"])).encode()).hexdigest()[:12]
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{os.path.basename(csv_path)}.{key}.npz")

    if os.path.exists(cache):
        d = np.load(cache)
        return d["X"], d["y"]

    print(f"  extracting MFCC features for {len(df)} files ({os.path.basename(csv_path)})...")
    X = np.vstack(Parallel(n_jobs=-1, prefer="threads")(
        delayed(_file_features)(fp) for fp in df["filepath"]))
    y = df["label"].map(LABEL_TO_IDX).to_numpy()
    np.savez_compressed(cache, X=X, y=y)
    return X, y


class TransferFeatures:
    """Paper's GNB+NMF 'transfer' step. Fit on train, apply to any split."""

    def __init__(self, n_components: int = 16, seed: int = 0):
        self.n_components = n_components
        self.seed = seed

    def fit(self, X, y):
        self.scaler = MinMaxScaler().fit(X)          # NMF requires non-negative input
        self.nmf = NMF(n_components=self.n_components, init="nndsvda",
                       max_iter=500, random_state=self.seed).fit(self.scaler.transform(X))
        self.gnb = GaussianNB().fit(X, y)            # probabilistic transfer features
        return self

    def transform(self, X):
        p = self.gnb.predict_proba(X)                # (n, 2)
        # clip: test values can fall outside the train min/max and go slightly
        # negative after MinMaxScaler, which NMF.transform rejects.
        Xnn = np.clip(self.scaler.transform(X), 0.0, None)
        z = self.nmf.transform(Xnn)                  # (n, n_components)
        return np.hstack([p, z])


def _classifiers():
    # Hyperparameters follow the paper's Table 2 where they are meaningful.
    return {
        "GNB": GaussianNB(),
        "LR": LogisticRegression(max_iter=1000),
        "RF": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=0),
        "KNC": KNeighborsClassifier(n_neighbors=5),
    }


def _evaluate(clf, Xtr, ytr, Xte, yte):
    clf.fit(Xtr, ytr)
    fake_col = list(clf.classes_).index(LABEL_TO_IDX["fake"])
    scores = clf.predict_proba(Xte)[:, fake_col]
    preds = (scores >= 0.5).astype(int)
    eer, _ = compute_eer(yte, scores)
    return {
        "eer": eer * 100,
        "bal_acc": balanced_accuracy_score(yte, preds) * 100,
        "f1_fake": f1_score(yte, preds, pos_label=LABEL_TO_IDX["fake"]) * 100,
        "auc": roc_auc_score(yte, scores),
    }


def main():
    print("Loading features...")
    Xtr, ytr = extract_split("splits/train.csv")
    Xte, yte = extract_split("splits/test.csv")
    print(f"train {Xtr.shape}  test {Xte.shape}\n")

    # Feature set A: raw MFCC (standardized). B: GNB+NMF transfer features.
    scaler_mfcc = StandardScaler().fit(Xtr)
    feats = {"MFCC": (scaler_mfcc.transform(Xtr), scaler_mfcc.transform(Xte))}

    tf = TransferFeatures().fit(Xtr, ytr)
    Ttr, Tte = tf.transform(Xtr), tf.transform(Xte)
    scaler_tf = StandardScaler().fit(Ttr)
    feats["Transfer(GNB+NMF)"] = (scaler_tf.transform(Ttr), scaler_tf.transform(Tte))

    rows = []
    for feat_name, (Ftr, Fte) in feats.items():
        for clf_name, clf in _classifiers().items():
            m = _evaluate(clf, Ftr, ytr, Fte, yte)
            rows.append({"features": feat_name, "model": clf_name, **m})

    res = pd.DataFrame(rows)
    print("=" * 68)
    print("PAPER BASELINE on speaker-disjoint TEST set (lower EER = better)")
    print("=" * 68)
    with pd.option_context("display.float_format", lambda v: f"{v:.2f}"):
        print(res.to_string(index=False))
    best = res.loc[res["eer"].idxmin()]
    print(f"\nBest baseline: {best['model']} on {best['features']} "
          f"-> EER {best['eer']:.2f}%, F1(fake) {best['f1_fake']:.2f}%, "
          f"balanced acc {best['bal_acc']:.2f}%")
    print("This is the number your deep-learning model needs to beat.")
    res.to_csv("baseline_results.csv", index=False)
    print("\nSaved baseline_results.csv")


if __name__ == "__main__":
    main()
