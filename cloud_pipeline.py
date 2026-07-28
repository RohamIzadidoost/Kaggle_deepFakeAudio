# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Cross-Corpus Deepfake Voice Detection via Test-Time Adaptation
#
# Fine-tune an XLS-R detector on a **source** corpus, then adapt it *unsupervised*
# to each **unseen target** corpus (In-the-Wild, Arabic) using confident-pseudo-label
# self-training + channel consistency. Reports Equal Error Rate before vs. after adaptation.

# %%
# Bare environment: install everything. torch+torchaudio first (matched CUDA
# wheels from PyPI), then the rest. Takes a few minutes on the first run.
import sys, subprocess

def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

# The base image's numpy stack is inconsistent (some pkgs need numpy<2, pandas is
# built for another). Install ONE consistent numpy-1.26 stack that shadows it,
# plus the modern deps we need. librosa/numba removed (they re-pin numpy).
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "librosa", "numba"], check=False)
_pip("numpy==1.26.4", "pandas", "scikit-learn")     # matched to numpy 1.26
_pip("torch", "torchaudio", "soundfile", "datasets", "tqdm", "kaggle")
_pip("numpy==1.26.4")                                # re-pin last: some deps bump numpy
print("install complete --- now RESTART THE KERNEL (Kernel menu), then run from the imports cell")

# %%
import os, glob, copy, contextlib, multiprocessing as mp, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torchaudio, soundfile as sf
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, roc_auc_score
from datasets import load_dataset, Audio
from tqdm.auto import tqdm

try:
    mp.set_start_method("fork", force=True)   # DataLoader workers can pickle cell-defined classes
except (RuntimeError, ValueError):
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

def amp():                                    # bf16 where supported, else fp32 (T4/V100-safe)
    return torch.amp.autocast("cuda", dtype=torch.bfloat16) if USE_BF16 else contextlib.nullcontext()

SR, DUR = 16000, 3.0
TARGET_LEN = int(SR * DUR)
SOURCE_PER_CLASS = 2500      # balanced source clips per class
TARGET_PER_CLASS = 2000      # balanced target clips per class
SOURCE_EPOCHS, TTA_EPOCHS = 4, 4
torch.manual_seed(0); np.random.seed(0)

# %% [markdown]
# ## 0. Environment check — run this first; it fails fast if the env is wrong

# %%
print("python    ", sys.version.split()[0])
print("torch     ", torch.__version__)
print("torchaudio", torchaudio.__version__)
print("CUDA      ", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("bf16      ", USE_BF16, "(uses fp32 fallback if False)")
assert hasattr(torchaudio.pipelines, "WAV2VEC2_XLSR_300M"), \
    "XLS-R bundle missing: need torchaudio>=2.1 matching your torch."
print("XLS-R bundle: OK")

# %% [markdown]
# ## 1. Download corpora
# Upload `kaggle.json` to this folder first (Kaggle → Account → Create New Token);
# the cell below installs it. Arabic comes from the Hugging Face Hub.

# %%
import shutil
_kdir = os.path.expanduser("~/.kaggle")
if os.path.exists("kaggle.json"):
    os.makedirs(_kdir, exist_ok=True)
    shutil.copy("kaggle.json", f"{_kdir}/kaggle.json")
    os.chmod(f"{_kdir}/kaggle.json", 0o600)
    print("kaggle.json installed")
elif os.path.exists(f"{_kdir}/kaggle.json"):
    print("kaggle.json already in place")
else:
    print("!! upload kaggle.json to this folder, then re-run this cell")

# %%
# Kaggle via the Python API (the `kaggle` CLI is often not on PATH in notebooks)
import kaggle
kaggle.api.dataset_download_files("azkurniwan/asvspoof-2019-la",
                                  path="data/asvspoof2019_LA", unzip=True, quiet=False)
kaggle.api.dataset_download_files("bhaveshkumars/release-in-the-wild",
                                  path="data/in_the_wild", unzip=True, quiet=False)

def fetch_arabic(out="data/arabic_arad"):
    if glob.glob(f"{out}/**/*.wav", recursive=True):
        return
    ds = load_dataset("DeepFake-Audio-Rangers/Arabic_Audio_Deepfake").cast_column("audio", Audio(decode=False))
    names = ds["train"].features["label"].names
    for split in ds:
        for i, ex in enumerate(ds[split]):
            d = f"{out}/{split}/{names[ex['label']]}"
            os.makedirs(d, exist_ok=True)
            open(f"{d}/{i}.wav", "wb").write(ex["audio"]["bytes"])

fetch_arabic()

# %% [markdown]
# ## 2. Manifest: (path, label, corpus) for every clip

# %%
def build_manifest():
    rows = []
    proto = "data/asvspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    flac = "data/asvspoof2019_LA/ASVspoof2019_LA_train/flac"
    for line in open(proto):
        p = line.split()
        if len(p) >= 5:
            rows.append((f"{flac}/{p[1]}.flac", "fake" if p[-1] == "spoof" else "real", "asvspoof2019"))
    for corpus, root in [("in_the_wild", "data/in_the_wild"), ("arabic", "data/arabic_arad")]:
        for w in glob.glob(f"{root}/**/*.wav", recursive=True):
            label = "real" if f"{os.sep}real{os.sep}" in w else "fake" if f"{os.sep}fake{os.sep}" in w else None
            if label:
                rows.append((w, label, corpus))
    return pd.DataFrame(rows, columns=["path", "label", "corpus"])

def balanced(df, n, seed=0):
    real = df[df.label == "real"]; fake = df[df.label == "fake"]
    real = real.sample(min(n, len(real)), random_state=seed)
    fake = fake.sample(min(n, len(fake)), random_state=seed)
    return pd.concat([real, fake]).sample(frac=1, random_state=seed).reset_index(drop=True)

manifest = build_manifest()
manifest.groupby(["corpus", "label"]).size()

# %% [markdown]
# ## 3. Audio dataset

# %%
LBL = {"real": 0, "fake": 1}

def load_audio(path):
    d, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(np.ascontiguousarray(d.T)).float(), sr

class AudioDS(Dataset):
    def __init__(self, df): self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            w, sr = load_audio(r.path)
            if w.shape[0] > 1: w = w.mean(0, keepdim=True)
            if sr != SR: w = torchaudio.functional.resample(w, sr, SR)
            w = w.squeeze(0)
        except Exception:
            w = torch.zeros(TARGET_LEN)
        n = w.numel()
        if n < TARGET_LEN: w = F.pad(w, (0, TARGET_LEN - n))
        elif n > TARGET_LEN: s = (n - TARGET_LEN) // 2; w = w[s:s + TARGET_LEN]
        return w, LBL[r.label]

def loader(df, bs=16, shuffle=False):
    return DataLoader(AudioDS(df), batch_size=bs, shuffle=shuffle, num_workers=8, pin_memory=True)

# %% [markdown]
# ## 4. Model: frozen XLS-R (top-4 layers fine-tuned) + attentive pooling + classifier

# %%
class XLSRDetector(nn.Module):
    def __init__(self, n_finetune=4):
        super().__init__()
        self.ssl = torchaudio.pipelines.WAV2VEC2_XLSR_300M.get_model()
        for p in self.ssl.parameters(): p.requires_grad_(False)
        for p in self.ssl.model.encoder.transformer.layers[-n_finetune:].parameters(): p.requires_grad_(True)
        self.ssl.eval()
        self.attn = nn.Linear(1024, 1)
        self.proj = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3))
        self.cls = nn.Linear(256, 2)

    def forward(self, wav):
        with torch.amp.autocast("cuda", enabled=False):     # encoder stays fp32 under AMP
            feats, _ = self.ssl.extract_features(wav.float())
        x = feats[-1]
        a = torch.softmax(self.attn(x).squeeze(-1), 1)
        ctx = torch.bmm(a.unsqueeze(1), x).squeeze(1)
        emb = self.proj(ctx)
        return self.cls(emb), emb

# %% [markdown]
# ## 5. Metrics + train / score helpers

# %%
def eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1); fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2

@torch.no_grad()
def score(model, dl):
    model.eval(); ys, ss = [], []
    for wav, y in dl:
        logits, _ = model(wav.to(DEVICE))
        ss.append(torch.softmax(logits, 1)[:, 1].float().cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ss)

def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]

def fit(model, dl, epochs, lr=1e-4):
    opt = torch.optim.Adam(trainable(model), lr=lr)
    for _ in range(epochs):
        model.train(); model.ssl.eval()
        for wav, y in tqdm(dl, leave=False):
            opt.zero_grad()
            with amp():
                logits, _ = model(wav.to(DEVICE))
                loss = F.cross_entropy(logits, y.to(DEVICE))
            loss.backward(); opt.step()
    return model

# %% [markdown]
# ## 6. Train the source detector (ASVspoof2019-LA)

# %%
source_df = balanced(manifest[manifest.corpus == "asvspoof2019"], SOURCE_PER_CLASS)
source = XLSRDetector().to(DEVICE)
fit(source, loader(source_df, shuffle=True), SOURCE_EPOCHS)

# %% [markdown]
# ## 7. Test-time adaptation
# Adapt LayerNorm (top-4) + head to the unlabeled target: self-train on the model's own
# confident-ranked pseudo-labels, plus a channel-consistency term. Labels never used.

# %%
def set_tta_params(model):
    for p in model.parameters(): p.requires_grad_(False)
    for m in model.ssl.model.encoder.transformer.layers[-4:].modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters(): p.requires_grad_(True)
    for head in (model.attn, model.proj, model.cls):
        for p in head.parameters(): p.requires_grad_(True)

def tta(model, dl, epochs=TTA_EPOCHS, lr=1e-4, q=0.3):
    set_tta_params(model)
    opt = torch.optim.Adam(trainable(model), lr=lr)
    n = len(dl.dataset)
    for _ in range(epochs):
        _, s = score(model, dl)
        lo, hi = np.quantile(s, q), np.quantile(s, 1 - q)
        pl = np.full(n, -1); pl[s <= lo] = 0; pl[s >= hi] = 1; pl = torch.tensor(pl)
        model.train(); model.ssl.eval(); ptr = 0
        for wav, _ in dl:
            bs = wav.size(0); wav = wav.to(DEVICE); bpl = pl[ptr:ptr + bs].to(DEVICE); ptr += bs
            opt.zero_grad()
            with amp():
                logits, _ = model(wav); p = torch.softmax(logits, 1)
                conf = bpl >= 0
                loss = F.cross_entropy(logits[conf], bpl[conf].long()) if conf.any() else torch.zeros((), device=DEVICE)
                aug = wav * torch.empty(bs, 1, device=DEVICE).uniform_(0.7, 1.3) + 0.005 * torch.randn_like(wav)
                loss = loss + 0.3 * F.mse_loss(torch.softmax(model(aug)[0], 1), p.detach())
            if loss.requires_grad:
                loss.backward(); opt.step()
    return model

def tent(model, dl, epochs=TTA_EPOCHS, lr=1e-4):     # vanilla TTA baseline (entropy minimisation)
    set_tta_params(model)
    opt = torch.optim.Adam(trainable(model), lr=lr)
    for _ in range(epochs):
        model.train(); model.ssl.eval()
        for wav, _ in dl:
            opt.zero_grad()
            with amp():
                p = torch.softmax(model(wav.to(DEVICE))[0], 1)
                loss = -(p * torch.log(p + 1e-8)).sum(1).mean()
            loss.backward(); opt.step()
    return model

# %% [markdown]
# ## 8. Cross-corpus results: source-only vs. Tent (vanilla TTA) vs. ours

# %%
def eer_auc(model, dl):
    y, s = score(model, dl)
    return eer(y, s) * 100, roc_auc_score(y, s)

results = []
for corpus in ["in_the_wild", "arabic"]:
    dl = loader(balanced(manifest[manifest.corpus == corpus], TARGET_PER_CLASS))
    src_eer, src_auc = eer_auc(source, dl)
    tent_eer, tent_auc = eer_auc(tent(copy.deepcopy(source), dl), dl)
    tta_eer, tta_auc = eer_auc(tta(copy.deepcopy(source), dl), dl)
    results.append([corpus, src_eer, src_auc, tent_eer, tent_auc, tta_eer, tta_auc])

pd.DataFrame(results, columns=["target", "source EER%", "source AUC",
                               "Tent EER%", "Tent AUC", "ours EER%", "ours AUC"]).round(2)
