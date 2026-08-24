"""
Quantify each protocol-inflation mechanism from the JASMP paper's audit.

Sect. "Auditing the reference evaluation" in main_jasmp.tex names four
mechanisms that inflate a reported score. Two of them (M1 pre-split
oversampling, M2 random k-fold over a speaker-recurrent corpus) can be measured
directly on our own benchmark by re-running the SAME classical pipeline under
progressively leakier protocols and watching the number move. That turns the
audit from a structural argument into a measurement, which is what a reviewer
will ask for.

Protocols compared (identical features, identical classifiers, only the split
and the resampling change):

  P0  speaker-disjoint, balanced       -- the honest protocol (the paper's
                                          in-domain results table)
  P1  random row split, balanced       -- M2 only: same speaker in train & test
  P2  random k-fold, balanced          -- M2 as the audited paper runs it
  P3  random k-fold + pre-split SMOTE  -- M1 + M2 together, on an IMBALANCED pool
                                          (SMOTE is a no-op on a 1:1 subset, so
                                           this arm deliberately uses a 4:1
                                           fake:real pool, mirroring the audited
                                           setup: imbalanced corpus, oversampled
                                           before splitting)

Reports accuracy AND EER for every arm, because the whole point is that accuracy
climbs under leakage while EER on the honest protocol does not.

CPU only, no GPU, no torch training. MFCC features are extracted once for the
whole balanced manifest and cached, so re-runs are fast.

Run:
  python leakage_ablation.py                       # all arms, seed 42
  python leakage_ablation.py --seeds 0 1 2 3 4     # multi-seed (recommended)

Writes leakage_ablation_results.csv.
"""

import argparse
import hashlib
import importlib
import os
import shutil
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from metrics import compute_eer  # numpy/sklearn only -- safe to import eagerly

# Mirrors deepfake_dataset.LABEL_TO_IDX, as a literal so this module does not
# import librosa just to learn the label convention. preflight() asserts the
# two agree before anything runs.
LABEL_TO_IDX = {"real": 0, "fake": 1}
FAKE = LABEL_TO_IDX["fake"]


# --------------------------------------------------------------------------- preflight
# paper_baseline imports torch/torchaudio at module level and deepfake_dataset
# imports librosa/soundfile, so a plain `from paper_baseline import ...` at the
# top of this file dies with an ImportError several frames deep before anything
# useful can be said. Defer it, and check every prerequisite up front instead.
def preflight(manifest_csv):
    """Report every missing prerequisite at once, then exit. No partial runs."""
    problems = []

    if not os.path.exists(manifest_csv):
        problems.append(
            f"manifest not found: {manifest_csv}\n"
            "      It is gitignored (it embeds absolute paths). Regenerate with:\n"
            "        python build_manifest.py --root data --out manifest.csv\n"
            "        python build_balanced_subset.py --manifest manifest.csv "
            "--out manifest_balanced.csv\n"
            "      That needs all three corpora decompressed under data/ "
            "(ASVspoof2021-DF, the LibriSpeech+TTS set, MLAAD v5).")

    missing = []
    for mod in ("torch", "torchaudio", "soundfile"):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        problems.append(
            f"missing packages: {', '.join(missing)}\n"
            "      Needed to decode audio and compute MFCCs. Install into the "
            "environment\n      you are running this from (see requirements.txt "
            "-- the pins matter).")

    # Not librosa: it is deliberately absent from requirements.txt (numba vs the
    # pinned numpy 1.26.4), and an empty leftover librosa/ directory satisfies
    # `import librosa` as a namespace package anyway, so importing it proves
    # nothing. The decoder fallback in deepfake_dataset shells out to ffmpeg,
    # and without ffmpeg roughly half the ASVspoof FLACs become silent zeros.
    if shutil.which("ffmpeg") is None:
        problems.append(
            "ffmpeg not found on PATH\n"
            "      libsndfile cannot decode a large fraction of the ASVspoof2021 "
            "FLACs;\n      deepfake_dataset.load_audio falls back to ffmpeg for "
            "those. Without it\n      they decode as silence and every arm of this "
            "ablation is measuring\n      a decoder bug instead of protocol leakage.")

    if not problems:
        from deepfake_dataset import LABEL_TO_IDX as _UPSTREAM
        assert _UPSTREAM == LABEL_TO_IDX, f"label convention changed upstream: {_UPSTREAM}"
        return

    print("Cannot run: prerequisites missing.\n", file=sys.stderr)
    for i, p in enumerate(problems, 1):
        print(f"  [{i}] {p}\n", file=sys.stderr)
    print("This experiment is CPU-only but not data-free -- it needs the merged\n"
          "corpora. Run it on the machine that holds them.", file=sys.stderr)
    sys.exit(1)


def load_backend():
    """Import the shared feature extractor and classifiers from paper_baseline.

    Deferred past preflight() so a missing torch reports as one clear line
    rather than an ImportError traceback.
    """
    global CACHE_DIR, TransferFeatures, _classifiers, _file_features
    from paper_baseline import (CACHE_DIR, TransferFeatures, _classifiers,
                                _file_features)


# --------------------------------------------------------------------------- features
def manifest_features(manifest_csv: str):
    """MFCC mean+std features for every row of a manifest, cached by file-list hash."""
    df = pd.read_csv(manifest_csv, dtype={"extra": str}, low_memory=False)
    key = hashlib.md5(("|".join(df["filepath"])).encode()).hexdigest()[:12]
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"leakage_ablation.{key}.npz")

    if os.path.exists(cache):
        d = np.load(cache)
        return df, d["X"], d["y"]

    print(f"Extracting MFCC features for {len(df)} files (one-off, then cached)...")
    X = np.vstack(Parallel(n_jobs=-1, prefer="threads")(
        delayed(_file_features)(fp) for fp in df["filepath"]))
    y = df["label"].map(LABEL_TO_IDX).to_numpy()

    # A file that fails to decode returns an all-zero vector (see the paper's
    # data-hygiene section). Count them loudly rather than letting them through.
    n_zero = int((np.abs(X).sum(axis=1) == 0).sum())
    if n_zero:
        print(f"  WARNING: {n_zero} files produced all-zero features "
              f"({n_zero / len(X):.1%}). Check the decoder before trusting any of this.")

    np.savez_compressed(cache, X=X, y=y)
    return df, X, y


# --------------------------------------------------------------------------- SMOTE
def smote(X, y, k=5, seed=0):
    """Minimal SMOTE: oversample the minority class to parity by k-NN interpolation.

    Implemented here rather than pulled from imbalanced-learn to avoid adding a
    dependency to the pinned environment. Faithful to Chawla et al. (2002) for
    the binary, full-parity case, which is all this ablation needs.
    """
    rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=2)
    minority = int(np.argmin(counts))
    n_needed = int(counts.max() - counts.min())
    if n_needed <= 0:
        return X, y

    Xm = X[y == minority]
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xm))).fit(Xm)
    _, idx = nn.kneighbors(Xm)                       # column 0 is the point itself

    base = rng.integers(0, len(Xm), size=n_needed)
    partner = idx[base, rng.integers(1, idx.shape[1], size=n_needed)]
    gap = rng.random((n_needed, 1))
    synth = Xm[base] + gap * (Xm[partner] - Xm[base])

    return np.vstack([X, synth]), np.concatenate([y, np.full(n_needed, minority)])


# --------------------------------------------------------------------------- evaluation
def evaluate(clf, Xtr, ytr, Xte, yte):
    clf.fit(Xtr, ytr)
    fake_col = list(clf.classes_).index(FAKE)
    scores = clf.predict_proba(Xte)[:, fake_col]
    preds = (scores >= 0.5).astype(int)
    eer, _ = compute_eer(yte, scores)
    return {
        "accuracy": accuracy_score(yte, preds) * 100,
        "bal_acc": balanced_accuracy_score(yte, preds) * 100,
        "eer": eer * 100,
        "auc": roc_auc_score(yte, scores),
    }


def run_fold(Xtr, ytr, Xte, yte):
    """Both feature sets x all four classifiers on one train/test fold."""
    scaler = StandardScaler().fit(Xtr)
    feats = {"MFCC": (scaler.transform(Xtr), scaler.transform(Xte))}

    tf = TransferFeatures().fit(Xtr, ytr)
    Ttr, Tte = tf.transform(Xtr), tf.transform(Xte)
    tscaler = StandardScaler().fit(Ttr)
    feats["Transfer(GNB+NMF)"] = (tscaler.transform(Ttr), tscaler.transform(Tte))

    rows = []
    for feat_name, (Ftr, Fte) in feats.items():
        for clf_name, clf in _classifiers().items():
            rows.append({"features": feat_name, "model": clf_name,
                         **evaluate(clf, Ftr, ytr, Fte, yte)})
    return rows


# --------------------------------------------------------------------------- protocols
def protocol_holdout(X, y, groups, seed, test_frac=0.1):
    """One speaker-disjoint (groups given) or random (groups None) holdout fold."""
    n_splits = max(2, round(1 / test_frac))
    if groups is not None:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        tr, te = next(sgkf.split(X, y, groups=groups))
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        tr, te = next(skf.split(X, y))
    return [tr], [te]


def protocol_kfold(X, y, seed, k=5):
    """Random stratified k-fold over rows -- the audited paper's protocol."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    trs, tes = zip(*skf.split(X, y))
    return list(trs), list(tes)


def imbalanced_pool(X, y, ratio, seed):
    """Subsample real down to 1:`ratio` real:fake, mirroring an imbalanced corpus."""
    rng = np.random.default_rng(seed)
    real_idx = np.flatnonzero(y != FAKE)
    fake_idx = np.flatnonzero(y == FAKE)
    n_real = max(1, int(round(len(fake_idx) / ratio)))
    keep = np.concatenate([rng.choice(real_idx, size=min(n_real, len(real_idx)),
                                      replace=False), fake_idx])
    rng.shuffle(keep)
    return X[keep], y[keep]


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest_balanced.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--imbalance_ratio", type=float, default=4.0,
                    help="fake:real ratio for the pre-split-SMOTE arm (P3)")
    ap.add_argument("--out", default="leakage_ablation_results.csv")
    args = ap.parse_args()

    preflight(args.manifest)
    load_backend()

    df, X, y = manifest_features(args.manifest)
    if "speaker" not in df.columns:
        raise SystemExit("manifest has no 'speaker' column -- rebuild with build_manifest.py")
    groups = df["speaker"].to_numpy()
    print(f"{len(df)} clips, {df['speaker'].nunique()} speakers, "
          f"class counts {np.bincount(y).tolist()} (0=real, 1=fake)\n")

    rows = []
    for seed in args.seeds:
        print(f"--- seed {seed} ---")

        # P0: the honest protocol -- speaker-disjoint holdout.
        print("  P0 speaker-disjoint holdout")
        for tr, te in zip(*protocol_holdout(X, y, groups, seed)):
            rows += [{"protocol": "P0_speaker_disjoint", "mechanisms": "none",
                      "seed": seed, **r} for r in run_fold(X[tr], y[tr], X[te], y[te])]

        # P1: random row holdout -- speakers now recur across the split (M2).
        print("  P1 random row holdout (M2)")
        for tr, te in zip(*protocol_holdout(X, y, None, seed)):
            rows += [{"protocol": "P1_random_holdout", "mechanisms": "M2",
                      "seed": seed, **r} for r in run_fold(X[tr], y[tr], X[te], y[te])]

        # P2: random k-fold -- the audited paper's protocol (M2, averaged over folds).
        print(f"  P2 random {args.kfolds}-fold (M2, as audited)")
        trs, tes = protocol_kfold(X, y, seed, k=args.kfolds)
        for fold, (tr, te) in enumerate(zip(trs, tes)):
            rows += [{"protocol": "P2_random_kfold", "mechanisms": "M2",
                      "seed": seed, "fold": fold,
                      **r} for r in run_fold(X[tr], y[tr], X[te], y[te])]

        # P3: pre-split SMOTE on an imbalanced pool, then random k-fold (M1 + M2).
        print(f"  P3 pre-split SMOTE @ {args.imbalance_ratio}:1 + random "
              f"{args.kfolds}-fold (M1+M2)")
        X_i, y_i = imbalanced_pool(X, y, args.imbalance_ratio, seed)
        X_s, y_s = smote(X_i, y_i, seed=seed)          # <-- BEFORE the split: the leak
        trs, tes = protocol_kfold(X_s, y_s, seed, k=args.kfolds)
        for fold, (tr, te) in enumerate(zip(trs, tes)):
            rows += [{"protocol": "P3_presplit_smote_kfold", "mechanisms": "M1+M2",
                      "seed": seed, "fold": fold,
                      **r} for r in run_fold(X_s[tr], y_s[tr], X_s[te], y_s[te])]

    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)

    print("\n" + "=" * 78)
    print("PROTOCOL-INFLATION ABLATION (mean over seeds/folds)")
    print("=" * 78)
    summary = (res.groupby(["protocol", "mechanisms", "features", "model"])
                  [["accuracy", "bal_acc", "eer", "auc"]].mean().reset_index())
    with pd.option_context("display.float_format", lambda v: f"{v:.2f}",
                           "display.max_rows", None):
        print(summary.to_string(index=False))

    print("\nHeadline (best accuracy per protocol -- the number a paper would report):")
    for proto in res["protocol"].unique():
        sub = summary[summary["protocol"] == proto]
        best = sub.loc[sub["accuracy"].idxmax()]
        print(f"  {proto:26s} acc {best['accuracy']:6.2f}%   "
              f"EER {best['eer']:5.2f}%   ({best['model']} / {best['features']})")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
