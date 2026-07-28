"""
Download the Arabic Audio Deepfake (ArAD) dataset and extract it to wav files.

ArAD ships as HF parquet with an Audio column. We read the raw (already-wav) bytes
with decode=False (avoids the torchcodec dependency) and write them to:
    data/arabic_arad/<split>/<real|fake>/<idx>.wav
so the existing manifest/loader pipeline can pick them up. Idempotent.

Run:  python download_arabic.py
"""

import os

from datasets import Audio, load_dataset

OUT = "data/arabic_arad"
REPO = "DeepFake-Audio-Rangers/Arabic_Audio_Deepfake"


def main():
    ds = load_dataset(REPO).cast_column("audio", Audio(decode=False))
    names = ds["train"].features["label"].names   # ['fake', 'real']

    total = 0
    for split in ds:
        for i, ex in enumerate(ds[split]):
            label = names[ex["label"]]
            d = os.path.join(OUT, split, label)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{i}.wav")
            if not os.path.exists(path):
                with open(path, "wb") as f:
                    f.write(ex["audio"]["bytes"])
            total += 1
        n_real = sum(1 for _ in os.scandir(os.path.join(OUT, split, "real"))) if os.path.isdir(os.path.join(OUT, split, "real")) else 0
        n_fake = sum(1 for _ in os.scandir(os.path.join(OUT, split, "fake"))) if os.path.isdir(os.path.join(OUT, split, "fake")) else 0
        print(f"{split}: {n_real} real / {n_fake} fake")
    print(f"extracted {total} clips to {OUT}/")


if __name__ == "__main__":
    main()
