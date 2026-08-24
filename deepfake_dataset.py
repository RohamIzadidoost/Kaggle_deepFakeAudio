"""
PyTorch Dataset for the unified deepfake-audio manifest.

Loads audio -> resamples to a fixed sample rate -> pads/crops to a fixed
duration -> converts to a log-mel spectrogram. Also provides a helper to
compute class weights for a weighted loss, since the data is ~97% fake.

Usage:
    from deepfake_dataset import DeepfakeAudioDataset, compute_class_weights

    train_ds = DeepfakeAudioDataset("splits/train.csv")
    weights = compute_class_weights(train_ds.df)  # tensor([w_real, w_fake])

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)

    for specs, labels in loader:
        # specs: (B, 1, n_mels, T), labels: (B,) long, 0=real 1=fake
        ...
"""

import io
import shutil
import subprocess

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

LABEL_TO_IDX = {"real": 0, "fake": 1}


FFMPEG = shutil.which("ffmpeg")


def _ffmpeg_decode(filepath: str):
    """Decode via ffmpeg -> WAV on stdout -> soundfile. Returns (samples, ch), sr.

    Raises on any failure; callers must not silently swallow it (see load_audio).
    """
    if FFMPEG is None:
        raise RuntimeError(
            "ffmpeg not found on PATH, and libsndfile cannot decode this file. "
            "Install ffmpeg -- without it a large fraction of the ASVspoof2021 "
            "FLACs decode as silence.")
    proc = subprocess.run(
        [FFMPEG, "-v", "quiet", "-i", filepath, "-f", "wav", "-c:a", "pcm_f32le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(
            f"ffmpeg failed to decode {filepath}: "
            f"{proc.stderr.decode('utf-8', 'replace')[:200]}")
    return sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=True)


def load_audio(filepath: str):
    """Load audio as (waveform[channels, samples] float32 tensor, sample_rate).

    Tries soundfile first (fast, native for .wav/.flac). libsndfile silently
    fails to decode a large fraction of the ASVspoof2021 FLACs ("unknown error
    in flac decoder") even though the files are valid, so we fall back to
    ffmpeg, which reads them correctly. Without this fallback ~half of ASVspoof
    loaded as pure silence and got zero-filled, poisoning training with a
    "silence => real" shortcut.

    The fallback shells out to ffmpeg directly rather than going through
    librosa: librosa is deliberately absent from requirements.txt (it drags in
    numba, which conflicts with the pinned numpy 1.26.4). An empty leftover
    librosa/ directory in site-packages still satisfies `import librosa` as a
    namespace package, so the previous librosa-based fallback raised
    AttributeError, was swallowed by callers' bare excepts, and reinstated the
    exact zero-fill bug this function exists to prevent.
    """
    try:
        data, sr = sf.read(filepath, dtype="float32", always_2d=True)  # (samples, ch)
    except Exception:
        data, sr = _ffmpeg_decode(filepath)
    return torch.from_numpy(np.ascontiguousarray(data.T)).float(), sr


class DeepfakeAudioDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        sample_rate: int = 16000,
        duration_sec: float = 4.0,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        random_crop: bool = True,
    ):
        self.df = pd.read_csv(csv_path, dtype={"extra": str})
        self.sample_rate = sample_rate
        self.target_len = int(sample_rate * duration_sec)
        # random crop = light augmentation for training; set False for val/test
        # so evaluation is deterministic (center crop).
        self.random_crop = random_crop

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.df)

    def _load_and_fix_length(self, filepath: str) -> torch.Tensor:
        # load_audio() uses soundfile with an ffmpeg/librosa fallback (see its
        # docstring) so the many ASVspoof FLACs libsndfile can't decode still
        # load correctly instead of becoming silent zeros.
        waveform, sr = load_audio(filepath)  # (channels, samples)

        # mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        n = waveform.shape[1]
        if n < self.target_len:
            pad = self.target_len - n
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        elif n > self.target_len:
            if self.random_crop:
                # random crop acts as light augmentation at train time
                start = torch.randint(0, n - self.target_len + 1, (1,)).item()
            else:
                # deterministic center crop for reproducible evaluation
                start = (n - self.target_len) // 2
            waveform = waveform[:, start:start + self.target_len]

        return waveform

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        label = LABEL_TO_IDX[row["label"]]

        try:
            waveform = self._load_and_fix_length(filepath)
        except Exception as e:
            # A handful of corrupt/unreadable files are common at this scale.
            # Returning zeros keeps a DataLoader worker from crashing the run;
            # log so you can audit which files are bad afterward.
            print(f"[WARN] failed to load {filepath}: {e}")
            waveform = torch.zeros(1, self.target_len)

        mel = self.mel_transform(waveform)      # (1, n_mels, T)
        mel_db = self.db_transform(mel)
        return mel_db, label


def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Inverse-frequency class weights for nn.CrossEntropyLoss(weight=...).
    Index order matches LABEL_TO_IDX: [weight_real, weight_fake].
    """
    counts = df["label"].value_counts()
    n_real = counts.get("real", 0)
    n_fake = counts.get("fake", 0)
    total = n_real + n_fake

    # inverse frequency, normalized so weights are on a sane scale
    w_real = total / (2 * max(n_real, 1))
    w_fake = total / (2 * max(n_fake, 1))
    return torch.tensor([w_real, w_fake], dtype=torch.float32)
