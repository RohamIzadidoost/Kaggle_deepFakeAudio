"""
Raw-waveform dataset for DA-RAD (SSL front-ends consume raw audio, not spectrograms).

Returns (waveform[samples], label_idx, domain_idx) where domain = source corpus
(dataset_source). Optional RawBoost augmentation is applied at train time. Reuses
deepfake_dataset.load_audio (soundfile + ffmpeg fallback).
"""

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from augment import RawBoost
from deepfake_dataset import LABEL_TO_IDX, load_audio


class RawAudioDataset(Dataset):
    def __init__(self, csv_path, sample_rate=16000, duration_sec=3.0, random_crop=True,
                 rawboost=False, rawboost_algo=4, domain_map=None):
        self.df = pd.read_csv(csv_path, dtype={"extra": str}, low_memory=False)
        self.sr = sample_rate
        self.target_len = int(sample_rate * duration_sec)
        self.random_crop = random_crop
        self.rb = RawBoost(sr=sample_rate, algo=rawboost_algo) if rawboost else None

        if domain_map is None:
            domain_map = {d: i for i, d in enumerate(sorted(self.df["dataset_source"].unique()))}
        self.domain_map = domain_map
        # sources not in the (train-derived) map -> -1; harmless because domain is
        # only used by the DG losses at train time, never at eval.
        self.domains = self.df["dataset_source"].map(domain_map).fillna(-1).astype(int).to_numpy()
        self.labels = self.df["label"].map(LABEL_TO_IDX).to_numpy()

    def __len__(self):
        return len(self.df)

    def _fix_len(self, wav):                       # wav: (1, n)
        n = wav.shape[1]
        if n < self.target_len:
            wav = torch.nn.functional.pad(wav, (0, self.target_len - n))
        elif n > self.target_len:
            start = (torch.randint(0, n - self.target_len + 1, (1,)).item()
                     if self.random_crop else (n - self.target_len) // 2)
            wav = wav[:, start:start + self.target_len]
        return wav

    def __getitem__(self, idx):
        fp = self.df.iloc[idx]["filepath"]
        try:
            wav, sr = load_audio(fp)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if sr != self.sr:
                wav = torchaudio.functional.resample(wav, sr, self.sr)
        except Exception as e:
            print(f"[WARN] failed to load {fp}: {e}")
            wav = torch.zeros(1, self.target_len)
        w = self._fix_len(wav).squeeze(0)          # (samples,)
        if self.rb is not None:
            w = self.rb(w)
        return w.float(), int(self.labels[idx]), int(self.domains[idx])


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "splits/train.csv"
    ds = RawAudioDataset(csv, rawboost=True)
    print(f"{len(ds)} items, domains={ds.domain_map}")
    w, y, d = ds[0]
    print(f"waveform {tuple(w.shape)} label={y} domain={d}")
