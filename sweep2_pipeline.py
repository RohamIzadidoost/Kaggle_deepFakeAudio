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
# # Extended Cross-Corpus Study: More Data, Languages, Methods
#
# Extends the validated 6-epoch pipeline to a full leave-one-corpus-out matrix
# over four EER-capable targets (ASVspoof2019, dataset2, In-the-Wild, Arabic),
# adds MLAAD (38 languages, fake-only) as a source-diversity corpus plus a
# held-out-language fake-recall diagnostic, and adds three more comparison
# points: BN/LayerNorm-stats-only adaptation, a supervised-target oracle
# (upper bound, not a fair TTA baseline), and a from-scratch RawNet2-lite
# baseline (no SSL) to separate "SSL helps" from "adaptation helps".
#
# **Run the PILOT first** (single fold, ~15 min) to confirm the batch/LR fix
# actually reproduces the locally-validated ~11-13% In-the-Wild source EER
# before committing the full matrix overnight. Flip `PILOT = False` only after
# the pilot's `source EER` line looks right.

# %%
import sys, subprocess

def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "librosa", "numba"], check=False)
_pip("numpy==1.26.4", "pandas", "scikit-learn")
_pip("torch", "torchaudio", "soundfile", "datasets", "tqdm", "kaggle", "matplotlib")
_pip("numpy==1.26.4")
print("install complete -- if first run, RESTART THE KERNEL, then continue")

# %%
import copy, glob, os, shutil, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torchaudio, soundfile as sf
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from tqdm.auto import tqdm

PILOT = False  # sweep2 always runs the full seed set; no pilot mode here.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SR = 16000
CROP_SEC, CACHE_SEC = 3.0, 4.0
CROP_LEN, CACHE_LEN = int(SR * CROP_SEC), int(SR * CACHE_SEC)

N_FINETUNE = 4
# --- the fix: batch=64/lr=1e-4 was undertrained (confirmed: 6-epoch run gave
# WORSE source EER than 12-epoch at the same batch/lr, i.e. not simply
# "overtraining" -- batch 32 + linearly-scaled lr is the safer middle ground). ---
BATCH = 32
LR, TTA_LR = 2e-4, 1e-4
Q, LAMBDA_CONS = 0.3, 0.3
N_HELDOUT_LANGS = 6         # MLAAD languages excluded from source, used only for the diagnostic

EER_TARGETS = ["asvspoof2019", "dataset2", "in_the_wild", "arabic"]

if PILOT:
    SEEDS, TARGETS = [0], ["in_the_wild"]
    SOURCE_EPOCHS, TTA_EPOCHS = 6, 4
    SOURCE_PER_CLASS, TARGET_PER_CLASS, MAX_PER_CORPUS_CLASS = 5000, 3000, 6000
    RUN_ORACLE, RUN_RAWNET, RUN_BN_ONLY = True, True, True
else:
    SEEDS, TARGETS = [0, 1, 2], EER_TARGETS
    SOURCE_EPOCHS, TTA_EPOCHS = 8, 4
    SOURCE_PER_CLASS, TARGET_PER_CLASS, MAX_PER_CORPUS_CLASS = 5000, 3000, 6000
    RUN_ORACLE, RUN_RAWNET, RUN_BN_ONLY = True, True, True

# ---------------------------------------------------------------------------
#  SWEEP2: adaptation-only grid. Reuses the *existing* source checkpoints from
#  the completed extended run -- no source training happens in this script.
#  Separate results/log filenames so nothing can collide with results_ext.csv.
# ---------------------------------------------------------------------------
SUFFIX = "_sweep2"
RESULTS_CSV, LOG_FILE = f"results{SUFFIX}.csv", f"run_log{SUFFIX}.txt"
CKPT_DIR = "ckpt_ext"                 # <-- READ-ONLY reuse, never written here
assert os.path.isdir(CKPT_DIR), f"{CKPT_DIR}/ not found -- unzip results_bundle.zip first"

# Part A: consistency-weight mechanism test. The paper hypothesises that on a
# LOW source-AUC target (dataset2, AUC .72) the consistency term amplifies bad
# pseudo-labels and should HURT as lambda grows, while on a HIGH source-AUC
# target (arabic, AUC .85) it should HELP. This tests that directly.
LAM_GRID, LAM_TARGETS = [0.0, 0.1, 0.3, 1.0], ["dataset2", "arabic"]

# Part B: adaptation-epoch compute/accuracy tradeoff (default E=4 -> E=8).
E_GRID, E_TARGETS = [8], EER_TARGETS

# never train anything in this script
RUN_ORACLE, RUN_RAWNET, RUN_BN_ONLY = False, False, False

def amp():
    return torch.amp.autocast("cuda", dtype=torch.bfloat16) if USE_BF16 else torch.amp.autocast("cuda", enabled=False)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

log(f"PILOT={PILOT} | device {DEVICE} {torch.cuda.get_device_name(0) if DEVICE=='cuda' else ''} | bf16 {USE_BF16}")
log(f"batch={BATCH} lr={LR} source_epochs={SOURCE_EPOCHS} targets={TARGETS}")
assert hasattr(torchaudio.pipelines, "WAV2VEC2_XLSR_300M"), "XLS-R bundle missing"

# %% [markdown]
# ## 1. Data
# Reuses the three corpora from the validated run (skips downloads if already
# present) and adds MLAAD (fake-only, 38 languages).

# %%
_kdir = os.path.expanduser("~/.kaggle")
if os.path.exists("kaggle.json"):
    os.makedirs(_kdir, exist_ok=True)
    shutil.copy("kaggle.json", f"{_kdir}/kaggle.json")
    os.chmod(f"{_kdir}/kaggle.json", 0o600)

KAGGLE_SETS = [
    ("azkurniwan/asvspoof-2019-la", "data/asvspoof2019_LA", "ASVspoof2019_LA_train/flac/*.flac"),
    ("bhaveshkumars/release-in-the-wild", "data/in_the_wild", "**/*.wav"),
    ("adarshsingh0903/audio-deepfake-detection-dataset", "data/dataset_2", "**/*.wav"),
    ("trapka/mlaadthe-multi-languagaudioanti-spoofing-dataset", "data/mlaad", "**/*.wav"),
]

import kaggle
for slug, path, probe in KAGGLE_SETS:
    if glob.glob(os.path.join(path, probe), recursive=True):
        log(f"already present: {path}")
        continue
    log(f"downloading {slug} (this may take a while for MLAAD, ~45GB)")
    kaggle.api.dataset_download_files(slug, path=path, unzip=True, quiet=False)

def fetch_arabic(out="data/arabic_arad"):
    if glob.glob(f"{out}/**/*.wav", recursive=True):
        log(f"already present: {out}")
        return
    from datasets import load_dataset, Audio
    ds = load_dataset("DeepFake-Audio-Rangers/Arabic_Audio_Deepfake").cast_column("audio", Audio(decode=False))
    names = ds["train"].features["label"].names
    for split in ds:
        for i, ex in enumerate(ds[split]):
            d = f"{out}/{split}/{names[ex['label']]}"
            os.makedirs(d, exist_ok=True)
            open(f"{d}/{i}.wav", "wb").write(ex["audio"]["bytes"])

fetch_arabic()

# %% [markdown]
# ## 2. Manifest
# `asvspoof2019` and `dataset2` are now EER-capable *targets* too (leave-one-out
# over all four). MLAAD is fake-only: it is a source-diversity corpus, and
# separately a held-out-language diagnostic (never an EER target).

# %%
LBL = {"real": 0, "fake": 1}

def build_manifest():
    rows = []
    proto = "data/asvspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    flac = "data/asvspoof2019_LA/ASVspoof2019_LA_train/flac"
    if os.path.exists(proto):
        for line in open(proto):
            p = line.split()
            if len(p) >= 5:
                lab = "fake" if p[-1] == "spoof" else "real"
                gen = p[-2] if lab == "fake" else "bonafide"
                rows.append((f"{flac}/{p[1]}.flac", lab, "asvspoof2019", gen, "en"))

    for d in sorted(glob.glob("data/dataset_2/*/")):
        gen = os.path.basename(d.rstrip("/"))
        lab = "real" if gen == "real_samples" else "fake"
        for w in glob.glob(f"{d}/**/*.wav", recursive=True):
            rows.append((w, lab, "dataset2", gen, "en"))

    for corpus, root, lang in [("in_the_wild", "data/in_the_wild", "en"), ("arabic", "data/arabic_arad", "ar")]:
        for w in glob.glob(f"{root}/**/*.wav", recursive=True):
            parts = w.split(os.sep)
            lab = "real" if "real" in parts else "fake" if "fake" in parts else None
            if lab:
                rows.append((w, lab, corpus, corpus, lang))

    # MLAAD: fake/<lang>/<model_dir>/*.wav, meta.csv columns are pipe-delimited
    for meta in glob.glob("data/mlaad/**/meta.csv", recursive=True):
        model_dir = os.path.dirname(meta)
        lang = os.path.basename(os.path.dirname(model_dir))
        gen = os.path.basename(model_dir)
        for w in glob.glob(f"{model_dir}/*.wav"):
            rows.append((w, "fake", "mlaad", gen, lang))

    return pd.DataFrame(rows, columns=["path", "label", "corpus", "generator", "language"])

manifest = build_manifest()
log("manifest:\n" + manifest.groupby(["corpus", "label"]).size().to_string())
log(f"MLAAD languages: {sorted(manifest[manifest.corpus=='mlaad'].language.unique())}")

# held out a fixed set of MLAAD languages entirely from source training
mlaad_langs = sorted(manifest[manifest.corpus == "mlaad"].language.unique())
rng = np.random.RandomState(0)
HELDOUT_LANGS = list(rng.choice(mlaad_langs, size=min(N_HELDOUT_LANGS, len(mlaad_langs)), replace=False))
log(f"MLAAD held-out languages (never seen in training): {HELDOUT_LANGS}")

# %% [markdown]
# ## 3. GPU-resident audio cache

# %%
def cap_per_class(df, n, seed=0):
    return pd.concat([g.sample(min(n, len(g)), random_state=seed) for _, g in df.groupby("label")])

# MLAAD source pool excludes held-out languages entirely
mlaad_pool = manifest[(manifest.corpus == "mlaad") & (~manifest.language.isin(HELDOUT_LANGS))]
mlaad_pool = mlaad_pool.sample(min(MAX_PER_CORPUS_CLASS, len(mlaad_pool)), random_state=0)
mlaad_heldout = manifest[(manifest.corpus == "mlaad") & (manifest.language.isin(HELDOUT_LANGS))]
mlaad_heldout = mlaad_heldout.sample(min(2000, len(mlaad_heldout)), random_state=0)

other = manifest[manifest.corpus != "mlaad"]
pool = pd.concat([cap_per_class(g, MAX_PER_CORPUS_CLASS) for _, g in other.groupby("corpus")]
                 + [mlaad_pool, mlaad_heldout]).reset_index(drop=True)
# re-derive mlaad subsets AFTER the reset -- the pre-reset frames' indices are
# stale once concatenated into `pool` and would silently misalign every
# idx_of() lookup that uses them (source pool AND the diagnostic).
mlaad_pool = pool[(pool.corpus == "mlaad") & (~pool.language.isin(HELDOUT_LANGS))]
mlaad_heldout = pool[(pool.corpus == "mlaad") & (pool.language.isin(HELDOUT_LANGS))]
log(f"cache pool: {len(pool)} clips\n" + pool.groupby(["corpus", "label"]).size().to_string())

def decode(path):
    try:
        d, sr = sf.read(path, dtype="float32", always_2d=True)
        w = d.mean(axis=1)
        if sr != SR:
            w = torchaudio.functional.resample(torch.from_numpy(w), sr, SR).numpy()
        return w
    except Exception:
        return np.zeros(1, dtype=np.float32)

def build_cache(df):
    n = len(df)
    buf = np.zeros((n, CACHE_LEN), dtype=np.float16)
    vlen = np.ones(n, dtype=np.int64)
    with ThreadPoolExecutor(max_workers=16) as ex:
        for i, w in enumerate(tqdm(ex.map(decode, df.path.tolist()), total=n, desc="decoding")):
            L = min(len(w), CACHE_LEN)
            buf[i, :L] = w[:L]
            vlen[i] = max(L, 1)
    return (torch.from_numpy(buf).to(DEVICE), torch.from_numpy(vlen).to(DEVICE),
            torch.tensor(df.label.map(LBL).values, dtype=torch.long, device=DEVICE))

t0 = time.time()
BUF, VLEN, Y = build_cache(pool)
log(f"cache ready: {tuple(BUF.shape)} fp16 = {BUF.numel()*2/1e9:.2f} GB on GPU ({time.time()-t0:.0f}s)")

_AR = torch.arange(CROP_LEN, device=DEVICE)

def get_batch(idx, train):
    span = (VLEN[idx] - CROP_LEN).clamp(min=0)
    start = (torch.rand(len(idx), device=DEVICE) * (span + 1).float()).long() if train else span // 2
    gidx = (start.unsqueeze(1) + _AR.unsqueeze(0)).clamp(max=CACHE_LEN - 1)
    return torch.gather(BUF[idx], 1, gidx).float(), Y[idx]

def augment(x):
    return x * torch.empty(x.size(0), 1, device=DEVICE).uniform_(0.7, 1.3) + 0.005 * torch.randn_like(x)

def idx_of(df):
    return torch.tensor(df.index.values, dtype=torch.long, device=DEVICE)

def sample_source(df, n_per_class, seed=0):
    real = df[df.label == "real"]
    real = real.sample(min(n_per_class, len(real)), random_state=seed)
    fakes = df[df.label == "fake"]
    groups = {k: g for k, g in fakes.groupby(["corpus", "generator"])}
    quota, left, keys = {}, min(n_per_class, len(fakes)), sorted(groups, key=lambda k: len(groups[k]))
    for i, k in enumerate(keys):
        take = min(left // (len(keys) - i), len(groups[k]))
        quota[k], left = take, left - take
    fake = pd.concat([groups[k].sample(quota[k], random_state=seed) for k in keys if quota[k]])
    return pd.concat([real, fake]).sample(frac=1, random_state=seed)

def sample_target(df, n_per_class, seed=0):
    return pd.concat([g.sample(min(n_per_class, len(g)), random_state=seed)
                      for _, g in df.groupby("label")]).sample(frac=1, random_state=seed)

# %% [markdown]
# ## 4. Models
# XLS-R detector (as before) plus a compact from-scratch RawNet2-lite (raw
# waveform CNN, no SSL) to separate "SSL helps" from "adaptation helps".

# %%
class XLSRDetector(nn.Module):
    def __init__(self, n_finetune=N_FINETUNE, encoder_amp=True):
        super().__init__()
        self.ssl = torchaudio.pipelines.WAV2VEC2_XLSR_300M.get_model()
        for p in self.ssl.parameters():
            p.requires_grad_(False)
        for p in self.ssl.model.encoder.transformer.layers[-n_finetune:].parameters():
            p.requires_grad_(True)
        self.ssl.eval()
        self.encoder_amp = encoder_amp
        self.attn = nn.Linear(1024, 1)
        self.proj = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3))
        self.cls = nn.Linear(256, 2)

    def forward(self, wav):
        ctx = amp() if self.encoder_amp else torch.amp.autocast("cuda", enabled=False)
        with ctx:
            feats, _ = self.ssl.extract_features(wav.float())
        x = feats[-1]
        a = torch.softmax(self.attn(x).squeeze(-1), 1)
        emb = self.proj(torch.bmm(a.unsqueeze(1), x).squeeze(1))
        return self.cls(emb), emb


class RawNet2Lite(nn.Module):
    """Compact raw-waveform CNN baseline: strided conv stem (SincNet-like
    receptive field) + residual conv blocks + GRU + linear head. No SSL
    pretraining -- trained from scratch per fold, to isolate how much of our
    result comes from XLS-R pretraining vs. the adaptation method itself."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, 251, stride=4, padding=125), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.MaxPool1d(4))

        def block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv1d(cin, cout, 5, stride=stride, padding=2), nn.BatchNorm1d(cout), nn.LeakyReLU(0.2),
                nn.Conv1d(cout, cout, 5, padding=2), nn.BatchNorm1d(cout))

        self.blocks = nn.ModuleList([block(64, 64, 2), block(64, 128, 2), block(128, 128, 1), block(128, 256, 2)])
        self.short = nn.ModuleList([nn.Conv1d(64, 64, 1, stride=2), nn.Conv1d(64, 128, 1, stride=2),
                                    nn.Identity(), nn.Conv1d(128, 256, 1, stride=2)])
        self.gru = nn.GRU(256, 128, batch_first=True, bidirectional=True)
        self.cls = nn.Linear(256, 2)
        self.encoder_amp = True  # dummy, matches XLSRDetector interface

    def forward(self, wav):
        x = self.stem(wav.unsqueeze(1))
        for blk, sc in zip(self.blocks, self.short):
            x = F.leaky_relu(blk(x) + sc(x), 0.2)
        x = x.transpose(1, 2)
        out, _ = self.gru(x)
        emb = out.mean(1)
        return self.cls(emb), emb


def trainable(m):
    return [p for p in m.parameters() if p.requires_grad]

_probe = XLSRDetector().to(DEVICE)
try:
    with amp():
        _probe(torch.randn(2, CROP_LEN, device=DEVICE))
    ENCODER_AMP = True
except RuntimeError:
    ENCODER_AMP = False
del _probe
torch.cuda.empty_cache()
log(f"encoder runs under autocast: {ENCODER_AMP}")

# %% [markdown]
# ## 5. Train / score / adapt

# %%
def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2

@torch.no_grad()
def score(model, idx):
    model.eval()
    out = []
    for i in range(0, len(idx), BATCH):
        x, _ = get_batch(idx[i:i + BATCH], train=False)
        with amp():
            out.append(torch.softmax(model(x)[0].float(), 1)[:, 1])
    return Y[idx].cpu().numpy(), torch.cat(out).cpu().numpy()

def metrics(model, idx):
    y, s = score(model, idx)
    return {"eer": eer(y, s) * 100, "auc": roc_auc_score(y, s),
            "acc": accuracy_score(y, (s >= 0.5).astype(int)) * 100}, y, s

def fit(model, idx, epochs, tag="", lr=LR):
    opt = torch.optim.Adam(trainable(model), lr=lr)
    for ep in range(epochs):
        model.train()
        if hasattr(model, "ssl"):
            model.ssl.eval()
        perm = idx[torch.randperm(len(idx), device=DEVICE)]
        tot = cnt = 0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            x, y = get_batch(b, train=True)
            opt.zero_grad(set_to_none=True)
            with amp():
                loss = F.cross_entropy(model(augment(x))[0], y)
            loss.backward(); opt.step()
            tot += loss.item() * len(b); cnt += len(b)
        log(f"  {tag} epoch {ep+1}/{epochs}  loss {tot/cnt:.4f}")
    return model

def set_tta_params(model):
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.ssl.model.encoder.transformer.layers[-N_FINETUNE:].modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad_(True)
    for head in (model.attn, model.proj, model.cls):
        for p in head.parameters():
            p.requires_grad_(True)

def adapt(model, idx, use_st=True, use_cons=True, epochs=TTA_EPOCHS, lam=LAMBDA_CONS):
    set_tta_params(model)
    opt = torch.optim.Adam(trainable(model), lr=TTA_LR)
    for _ in range(epochs):
        _, s = score(model, idx)
        lo, hi = np.quantile(s, Q), np.quantile(s, 1 - Q)
        pl = torch.full((len(idx),), -1, dtype=torch.long, device=DEVICE)
        pl[torch.from_numpy(s <= lo).to(DEVICE)] = 0
        pl[torch.from_numpy(s >= hi).to(DEVICE)] = 1
        model.train(); model.ssl.eval()
        order = torch.randperm(len(idx), device=DEVICE)
        for i in range(0, len(order), BATCH):
            sel = order[i:i + BATCH]
            x, _ = get_batch(idx[sel], train=False)
            bpl = pl[sel]
            opt.zero_grad(set_to_none=True)
            with amp():
                logits, _ = model(x)
                p = torch.softmax(logits, 1)
                loss = torch.zeros((), device=DEVICE)
                conf = bpl >= 0
                if use_st and conf.any():
                    loss = loss + F.cross_entropy(logits[conf], bpl[conf])
                if use_cons:
                    loss = loss + lam * F.mse_loss(torch.softmax(model(augment(x))[0], 1), p.detach())
            if loss.requires_grad:
                loss.backward(); opt.step()
    return model

def tent(model, idx, epochs=TTA_EPOCHS):
    set_tta_params(model)
    opt = torch.optim.Adam(trainable(model), lr=TTA_LR)
    for _ in range(epochs):
        model.train(); model.ssl.eval()
        order = torch.randperm(len(idx), device=DEVICE)
        for i in range(0, len(order), BATCH):
            x, _ = get_batch(idx[order[i:i + BATCH]], train=False)
            opt.zero_grad(set_to_none=True)
            with amp():
                p = torch.softmax(model(x)[0], 1)
                loss = -(p * torch.log(p + 1e-8)).sum(1).mean()
            loss.backward(); opt.step()
    return model

@torch.no_grad()
def bn_only(model, idx):
    """AdaBN-style baseline: recompute LayerNorm running behaviour via forward
    passes in train() mode only -- no gradient, no labels, no pseudo-labels."""
    set_tta_params(model)
    model.train(); model.ssl.eval()
    order = torch.randperm(len(idx), device=DEVICE)
    for i in range(0, len(order), BATCH):
        x, _ = get_batch(idx[order[i:i + BATCH]], train=False)
        with amp():
            model(x)
    return model

def oracle(model, idx_labeled, epochs=TTA_EPOCHS):
    """Supervised fine-tune directly on target labels. NOT a fair TTA baseline
    (uses labels) -- reported only as an upper-bound reference point."""
    set_tta_params(model)
    return fit(model, idx_labeled, epochs, tag="oracle", lr=TTA_LR)

# %% [markdown]
# ## 6. Experiment grid: leave-one-corpus-out x seeds x methods

# %%
def record(**row):
    pd.DataFrame([row]).to_csv(RESULTS_CSV, mode="a",
                               header=not os.path.exists(RESULTS_CSV), index=False)

# --- guard: the data pool must match the extended run or nothing is comparable ---
EXPECT_LANGS = ["mt", "sl", "hu", "hr", "fi", "lt"]
if sorted(HELDOUT_LANGS) != sorted(EXPECT_LANGS):
    log(f"!! FATAL: HELDOUT_LANGS={sorted(HELDOUT_LANGS)} != {sorted(EXPECT_LANGS)} "
        "-- source pool differs from the extended run, results would not be comparable")
    raise SystemExit(1)

# --- expected source EERs, used to verify each checkpoint loaded correctly ---
EXPECTED = {}
if os.path.exists("results_ext.csv"):
    _e = pd.read_csv("results_ext.csv")
    _e = _e[(_e.family == "xlsr") & (_e.setting == "transductive") & (_e.method == "source")]
    for _, r in _e.iterrows():
        EXPECTED[(int(r["seed"]), r["target"])] = float(r["eer"])
    log(f"ckpt verification: loaded {len(EXPECTED)} expected source EERs from results_ext.csv")
else:
    log("!! results_ext.csv not found -- cannot verify checkpoints, proceeding unverified")

JOBS = []
for _s in SEEDS:
    for _t in LAM_TARGETS:
        for _l in LAM_GRID:
            JOBS.append(dict(part="lambda", seed=_s, target=_t, lam=_l, epochs=TTA_EPOCHS))
for _s in SEEDS:
    for _t in E_TARGETS:
        for _E in E_GRID:
            JOBS.append(dict(part="epochs", seed=_s, target=_t, lam=LAMBDA_CONS, epochs=_E))

# --- resume support: skip anything already in the CSV ---
done = set()
if os.path.exists(RESULTS_CSV):
    _d = pd.read_csv(RESULTS_CSV)
    done = {(str(r["part"]), int(r["seed"]), str(r["target"]), float(r["lam"]), int(r["epochs"]))
            for _, r in _d.iterrows()}
    log(f"resume: {len(done)} jobs already recorded, will skip them")

def _key(j):
    return (j["part"], int(j["seed"]), j["target"], float(j["lam"]), int(j["epochs"]))

JOBS = [j for j in JOBS if _key(j) not in done]
log(f"{len(JOBS)} adaptation jobs to run "
    f"(part A lambda: {len(SEEDS)}x{len(LAM_TARGETS)}x{len(LAM_GRID)}, "
    f"part B epochs: {len(SEEDS)}x{len(E_TARGETS)}x{len(E_GRID)})")

by_pair = {}
for j in JOBS:
    by_pair.setdefault((j["seed"], j["target"]), []).append(j)

grid_t0 = time.time()
for (seed, target), jobs in by_pair.items():
    torch.manual_seed(seed); np.random.seed(seed)

    # reproduce the extended run's exact subsets (both samplers are seed-deterministic)
    src_pool = pool[(pool.corpus != target) & (pool.corpus != "mlaad")]
    src_pool = pd.concat([src_pool, mlaad_pool])
    _ = sample_source(src_pool, SOURCE_PER_CLASS, seed)          # parity with ext run
    tgt_df = sample_target(pool[pool.corpus == target], TARGET_PER_CLASS, seed)
    tgt_idx = idx_of(tgt_df)

    ckpt = f"{CKPT_DIR}/source_{target}_seed{seed}.pt"
    if not os.path.exists(ckpt):
        log(f"!! {ckpt} missing -- SKIPPING seed{seed}/{target} (refusing to retrain: "
            "a freshly trained source would not be comparable to the extended run)")
        continue

    source = XLSRDetector(encoder_amp=ENCODER_AMP).to(DEVICE)
    source.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)
    m0, _, _ = metrics(source, tgt_idx)
    exp = EXPECTED.get((seed, target))
    log(f"=== seed{seed}/{target} | n={len(tgt_idx)} | {len(jobs)} jobs | "
        f"ckpt source EER {m0['eer']:.2f}"
        + (f" (ext run: {exp:.2f})" if exp is not None else " (unverified)"))
    if exp is not None and abs(m0["eer"] - exp) > 0.5:
        log(f"  !! MISMATCH vs results_ext.csv ({m0['eer']:.2f} vs {exp:.2f}) -- skipping this "
            "pair; checkpoint or data pool differs, results would not be comparable")
        del source; torch.cuda.empty_cache(); continue

    for j in jobs:
        try:
            t0 = time.time()
            torch.manual_seed(seed); np.random.seed(seed)   # isolate the knob under test
            model = adapt(copy.deepcopy(source), tgt_idx, use_st=True,
                          use_cons=(j["lam"] > 0), lam=j["lam"], epochs=j["epochs"])
            m, _, _ = metrics(model, tgt_idx)
            record(part=j["part"], seed=seed, target=target, lam=j["lam"], epochs=j["epochs"],
                   method="ours", setting="transductive", family="xlsr",
                   eer=round(m["eer"], 3), auc=round(m["auc"], 4), acc=round(m["acc"], 2),
                   source_eer=round(m0["eer"], 3), n=len(tgt_idx),
                   minutes=round((time.time() - t0) / 60, 1))
            log(f"  [{j['part']:6s}] lam={j['lam']:<4} E={j['epochs']}: "
                f"EER {m['eer']:6.2f} (src {m0['eer']:5.2f}, delta {m['eer']-m0['eer']:+.2f})  "
                f"AUC {m['auc']:.3f}  ({(time.time()-t0)/60:.1f} min)")
            del model; torch.cuda.empty_cache()
        except Exception as e:
            log(f"  !! lam={j['lam']} E={j['epochs']} failed: {type(e).__name__}: {e}")

    del source; torch.cuda.empty_cache()

log(f"SWEEP2 COMPLETE in {(time.time()-grid_t0)/60:.1f} min | "
    f"peak GPU {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

# ---------------------------- summary ----------------------------
if os.path.exists(RESULTS_CSV):
    res = pd.read_csv(RESULTS_CSV)
    log("\n===== PART A: consistency weight (lambda), mean EER over seeds =====")
    a = res[res.part == "lambda"]
    if len(a):
        log("\n" + a.pivot_table(index="lam", columns="target", values="eer",
                                 aggfunc="mean").round(2).to_string())
        log("\n(source EER for reference)")
        log("\n" + a.groupby("target")["source_eer"].mean().round(2).to_string())
    log("\n===== PART B: adaptation epochs E=8 vs the E=4 default =====")
    b = res[res.part == "epochs"]
    if len(b):
        log("\n" + b.groupby("target")[["source_eer", "eer"]].mean().round(2).to_string())
