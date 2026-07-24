"""
Download & verify the extra corpora for the DA-RAD generalization study.

  - ASVspoof2019 LA (train/dev/eval flac + CM protocols)  -> data/asvspoof2019_LA/
  - In-the-Wild (real+fake wav)                            -> data/in_the_wild/

Uses the Kaggle CLI (auth at ~/.kaggle/kaggle.json). Idempotent: skips a corpus
whose expected files already exist. Verifies structure + audio counts afterwards.

Run:
  python download_data.py --root data
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

KAGGLE = os.path.join(os.path.dirname(sys.executable), "kaggle")

DATASETS = {
    "asvspoof2019_LA": {
        "slug": "azkurniwan/asvspoof-2019-la",
        "expect_files": [
            "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
            "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
        ],
        "expect_glob": ["ASVspoof2019_LA_train/flac/*.flac",
                        "ASVspoof2019_LA_dev/flac/*.flac"],
    },
    "in_the_wild": {
        "slug": "bhaveshkumars/release-in-the-wild",
        "expect_files": [],
        "expect_glob": ["release_in_the_wild/**/*.wav"],
    },
}


def _present(target: Path, cfg) -> bool:
    if not target.exists():
        return False
    if not all((target / f).exists() for f in cfg["expect_files"]):
        return False
    return all(next(target.glob(g), None) is not None for g in cfg["expect_glob"])


def _verify(target: Path, cfg):
    for f in cfg["expect_files"]:
        assert (target / f).exists(), f"missing expected file: {f}"
    counts = {}
    for g in cfg["expect_glob"]:
        counts[g] = sum(1 for _ in target.glob(g))
        assert counts[g] > 0, f"no files matched: {g}"
    return counts


def download(name, cfg, root: Path):
    target = root / name
    if _present(target, cfg):
        print(f"[{name}] already present — skipping download")
    else:
        target.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "KAGGLE_CONFIG_DIR": os.path.expanduser("~/.kaggle")}
        cmd = [KAGGLE, "datasets", "download", "-d", cfg["slug"], "-p", str(target), "--unzip"]
        print(f"[{name}] downloading {cfg['slug']} …")
        subprocess.run(cmd, check=True, env=env)
    counts = _verify(target, cfg)
    print(f"[{name}] verified OK: " + ", ".join(f"{g}={n}" for g, n in counts.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--only", choices=list(DATASETS), help="download just one corpus")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    names = [args.only] if args.only else list(DATASETS)
    for name in names:
        download(name, DATASETS[name], root)
    print("\nAll requested corpora present and verified.")


if __name__ == "__main__":
    main()
