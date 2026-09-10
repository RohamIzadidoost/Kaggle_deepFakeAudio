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
# # Adaptive q and E for test-time adaptation
#
# Sections 1-5 below are copied verbatim from `extended_pipeline.py` (the
# pipeline that produced every published number) so that the manifest, the pool
# capping, the target sampling and the model are bit-identical and the
# `ours_fixed` control row is directly comparable to `results_ext.csv`. Only the
# adaptation loop and the experiment grid at the bottom are new.
#
# `extended_pipeline.py` is deliberately NOT edited: its resume guard keys on
# `(seed, target, setting)` only, so re-running it against the existing
# `results_ext.csv` would skip all 20 folds regardless of any methods added.
#
# ## What adapts, and what does not
#
# * **q** -- a curriculum ramp from a strict high-purity tail up to a
#   prevalence-split budget, replacing the fixed symmetric `q=0.3`. The ceiling
#   stays at the published 0.3; the adaptive parts are the ramp and the split.
# * **E** -- run to 8 (the sweep's better value) with label-free early stopping
#   on epoch-to-epoch score movement, replacing the fixed `E=4` that
#   `main.tex:693-696` calls "a conservative, compute-matched choice, not a
#   tuned optimum". The stop is a safety valve, not an efficiency measure:
#   every swept point says more epochs help, so the rule is biased toward
#   spending the full budget.
# * **lambda stays fixed at 0.3.** Making it adaptive was the original intent
#   and was dropped on evidence; see the note on `LAMBDA_CONS` below and the
#   recorded negative result in `test_adaptive_tta.py` check [2].
#
# Smoke run (default -- tiny subsets, separate output files):
#     python adaptive_pipeline.py
# Real run:
#     ADAPTIVE_SMOKE=0 python adaptive_pipeline.py

# %%
# Environment bootstrap, for a fresh cloud kernel only. Unlike the notebook this
# was copied from, this file gets *imported* (by verify_reduction.py), and this
# cell uninstalls librosa/numba and pins numpy -- running that on every import
# would churn the local venv that requirements.txt deliberately pins. Opt in
# with ADAPTIVE_BOOTSTRAP=1 on a machine that needs it.
import os, sys, subprocess

if os.environ.get("ADAPTIVE_BOOTSTRAP") == "1":
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

import adaptive_tta as AT

# Defaults to the smoke config; `ADAPTIVE_SMOKE=0 python adaptive_pipeline.py`
# runs the real thing. An env var rather than a literal toggle because
# verify_reduction.py imports this module and needs the smoke sizes without
# editing the file.
SMOKE = os.environ.get("ADAPTIVE_SMOKE", "1") == "1"

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
# LAMBDA_CONS stays fixed at the published 0.3 in every arm here. Gating it on
# unlabeled statistics was the original goal and was dropped on evidence:
#   * Pseudo-label churn under augmentation, the planned signal, is confounded
#     -- it moves lambda by 0.017 across the whole AUC range .60-.99 but by
#     0.165 across augmentation magnitude alone, so it reads augment()'s own
#     constants rather than the corpus (test_adaptive_tta.py check [2]).
#   * The tail-separation statistic from lambda_selector.py IS strongly related
#     to whether consistency helps (r=+0.74, p=0.006 vs delta_2 -- note the sign
#     is the OPPOSITE of the "high separation = reliable" reading the paper
#     assumed), but the whole gap between always-lambda=0.3 and a per-point
#     oracle is only 0.24 EER pooled, and leave-one-target-out selection lands
#     inside that gap. Twelve (target, seed) points cannot settle it.
# q and E are where the measured headroom is: E=4->8 is worth 1.2-1.6 EER in the
# existing sweep, and a fixed symmetric q costs 26.33 -> 42.45 EER on the skewed
# Protocol A pool.
N_HELDOUT_LANGS = 6         # MLAAD languages excluded from source, used only for the diagnostic

EER_TARGETS = ["asvspoof2019", "dataset2", "in_the_wild", "arabic"]

# Only seeds 0-2 have local source checkpoints (ckpt_ext); the cloud run's seeds
# 3-4 checkpoints were never saved, and retraining them is out of scope for the
# adaptation-only comparison.
if SMOKE:
    SEEDS, TARGETS = [0], ["in_the_wild"]
    TARGET_PER_CLASS, MAX_PER_CORPUS_CLASS = 64, 128
    ABLATION_SEEDS = [0]
else:
    SEEDS, TARGETS = [0, 1, 2], EER_TARGETS
    TARGET_PER_CLASS, MAX_PER_CORPUS_CLASS = 3000, 6000
    ABLATION_SEEDS = [0]          # single-knob arms only on seed 0 (compute)
    # OUR_SEEDS / OUR_TARGETS let the orchestrator slice the grid without editing
    # this file, so a long sweep can be run seed-by-seed and resumed.
    if os.environ.get("OUR_SEEDS"):
        SEEDS = [int(x) for x in os.environ["OUR_SEEDS"].split(",")]
    if os.environ.get("OUR_TARGETS"):
        TARGETS = os.environ["OUR_TARGETS"].split(",")

SOURCE_EPOCHS, TTA_EPOCHS = 8, 4  # TTA_EPOCHS = the fixed-E control; adaptive E uses AT.E_MAX
SOURCE_PER_CLASS = 5000
RUN_ORACLE, RUN_RAWNET, RUN_BN_ONLY = False, False, False   # adaptation-only run

SUFFIX = "_smoke" if SMOKE else ""
RESULTS_CSV = f"results_adaptive{SUFFIX}.csv"
TRACE_CSV = f"adaptive_trace{SUFFIX}.csv"      # per-epoch diagnostics; the paper figure
LOG_FILE = f"run_log_adaptive{SUFFIX}.txt"
# Source checkpoints are READ-ONLY inputs here -- this run never trains a source
# model. ckpt/ holds the single in_the_wild seed-0 model used for the smoke test;
# ckpt_ext/ holds the 12 (4 targets x seeds 0-2) models behind results_ext.csv.
CKPT_DIR = "ckpt" if SMOKE else "ckpt_ext"

def amp():
    return torch.amp.autocast("cuda", dtype=torch.bfloat16) if USE_BF16 else torch.amp.autocast("cuda", enabled=False)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

log(f"SMOKE={SMOKE} | device {DEVICE} {torch.cuda.get_device_name(0) if DEVICE=='cuda' else ''} | bf16 {USE_BF16}")
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

# This run adapts pre-trained checkpoints and never trains a source model, so it
# needs only the four EER target corpora. MLAAD is a *source*-diversity corpus
# and is deliberately absent from this list: the copied loop would otherwise
# start a 45 GB download on any machine where data/mlaad is missing.
REQUIRED_DATA = [
    ("data/asvspoof2019_LA", "ASVspoof2019_LA_train/flac/*.flac"),
    ("data/in_the_wild", "**/*.wav"),
    ("data/dataset_2", "**/*.wav"),
]

# Assert rather than download. Nothing here should ever touch the network: a
# missing corpus is a setup error to report, not multi-GB of traffic to start
# silently in the middle of an experiment.
for path, probe in REQUIRED_DATA:
    if not glob.glob(os.path.join(path, probe), recursive=True):
        raise SystemExit(f"missing corpus {path} (expected {probe}); "
                         f"fetch it with extended_pipeline.py, not from here")
    log(f"present: {path}")

def require_arabic(out="data/arabic_arad"):
    """Assert-only counterpart of extended_pipeline.py's fetch_arabic()."""
    if not glob.glob(f"{out}/**/*.wav", recursive=True):
        raise SystemExit(f"missing corpus {out}; fetch it with extended_pipeline.py, not from here")
    log(f"present: {out}")

require_arabic()

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

    # MLAAD is intentionally not read here -- it only ever served source
    # training, which this file does not do. Its absence does not change the
    # target pools: `pool[pool.corpus == target]` holds the same rows in the
    # same order either way, since the per-corpus caps are applied by groupby
    # and MLAAD rows are concatenated after them.
    #
    # INCLUDE_MLAAD=1 puts it back, for the one caller that DOES train
    # (train_source_local.py). Leaving it out while training silently drops a
    # whole corpus from the source pool: the first local seed-0 run without it
    # missed the cloud checkpoint by +7.3 EER on Arabic and -8.1 on dataset2,
    # far outside seed noise. Off by default so every result already recorded
    # against this file stays bit-reproducible.
    # NEW_TARGETS=1 adds the three ASVspoof2021 tracks as *targets*. They are
    # never added to any source pool -- see SOURCE_CORPORA in
    # train_source_local.py -- so the source model for each of the original four
    # targets is unchanged and their recorded ten-seed results stay comparable.
    #
    # All three share ASVspoof2019 lineage, which IS in the source pool. They are
    # therefore a condition-shift arm (codec/replay), not new corpora proper, and
    # must be reported as such. PA is replay: a different detection task again.
    if os.environ.get("NEW_TARGETS") == "1":
        import csv as _csv
        _specs = {
            "asvspoof2021la": ("data/dataset_1/LA-keys-full/keys/LA/CM/trial_metadata.txt",
                               ["data/dataset_1/ASVspoof2021_LA_eval/ASVspoof2021_LA_eval/flac"], 5),
            "asvspoof2021pa": ("data/dataset_1/PA-keys-full/keys/PA/CM/trial_metadata.txt",
                               [f"data/dataset_1/ASVspoof2021_PA_eval_part{p:02d}/ASVspoof2021_PA_eval/flac"
                                for p in range(4)], 9),
        }
        for _name, (_keys, _dirs, _lc) in _specs.items():
            _lab = {}
            for _line in open(_keys):
                _q = _line.split()
                if len(_q) > _lc and _q[_lc] in ("spoof", "bonafide"):
                    _lab[_q[1]] = "fake" if _q[_lc] == "spoof" else "real"
            for _d in _dirs:
                for _f in glob.glob(os.path.join(_d, "*.flac")):
                    _stem = os.path.splitext(os.path.basename(_f))[0]
                    if _stem in _lab:
                        rows.append((_f, _lab[_stem], _name, _name, "en"))
        # 2021-DF comes from the already-parsed manifest.csv, as in
        # make_target_manifests.py, rather than re-globbing 61 GB.
        _df = pd.read_csv("manifest.csv", low_memory=False)
        _df = _df[_df.dataset_source == "dataset_1_asvspoof2021_DF"]
        for _pth, _l in zip(_df.filepath.values, _df.label.values):
            rows.append((_pth, _l, "asvspoof2021df", "asvspoof2021df", "en"))
        # Two corpora with no ASVspoof lineage at all, fetched by
        # fetch_hf_corpus.py. These are what the "extending the target set erases
        # the EER benefit" finding actually needs: the three ASVspoof2021 tracks
        # are relatives of a corpus already in the source pool, which a reviewer
        # can fairly discount.
        for _hf in ("wavefake", "commercialtts"):
            for _lab in ("real", "fake"):
                _d = f"data/hf_{_hf}/{_lab}"
                for _w in sorted(glob.glob(f"{_d}/*.wav")):
                    rows.append((_w, _lab, f"hf_{_hf}", f"hf_{_hf}", "en"))
        log("NEW_TARGETS: added asvspoof2021 la/pa/df + hf_wavefake, hf_commercialtts "
            "as targets (never as source)")

    if os.environ.get("INCLUDE_MLAAD") == "1":
        # MLAAD ships to different roots on different machines -- the Kaggle
        # download lands in data/mlaad, this box has it under data/dataset_3.
        # Search both rather than assume, and say which one was found.
        metas = (glob.glob("data/mlaad/**/meta.csv", recursive=True)
                 or glob.glob("data/dataset_3/**/meta.csv", recursive=True))
        for meta in metas:
            model_dir = os.path.dirname(meta)
            lang = os.path.basename(os.path.dirname(model_dir))
            gen = os.path.basename(model_dir)
            for w in glob.glob(f"{model_dir}/*.wav"):
                rows.append((w, "fake", "mlaad", gen, lang))
        log(f"MLAAD: {len(metas)} model dirs found for source training")

    return pd.DataFrame(rows, columns=["path", "label", "corpus", "generator", "language"])

manifest = build_manifest()
log("manifest:\n" + manifest.groupby(["corpus", "label"]).size().to_string())
# No MLAAD in this run (see build_manifest), so no held-out-language split and
# no source-diversity pool. Kept as empty frames so the copied section-3 pool
# construction below stays character-identical to extended_pipeline.py.
if os.environ.get("INCLUDE_MLAAD") == "1" and (manifest.corpus == "mlaad").any():
    _langs = sorted(manifest[manifest.corpus == "mlaad"].language.unique())
    HELDOUT_LANGS = list(np.random.RandomState(0).choice(
        _langs, size=min(N_HELDOUT_LANGS, len(_langs)), replace=False))
    log(f"MLAAD held-out languages (never in source training): {HELDOUT_LANGS}")
else:
    HELDOUT_LANGS = []

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
    # CACHE_ON_CPU keeps the waveform cache in pinned host memory and moves only
    # the current batch to the GPU. Numerically identical -- same tensors, same
    # gather, same order -- it only changes where they live.
    #
    # Needed because this file was written for a 35 GB cloud slice. The whole pool
    # is 33,597 clips x 64,000 samples x fp16 = 4.30 GB resident, which on a 10 GB
    # card leaves no room for XLS-R plus the deepcopy that every adapt() arm makes:
    # the `source` (eval-only) arm ran fine and every adaptation arm died with
    # OOM. public_ckpt_tta.build_cache already solved this the same way, and says
    # so in its docstring.
    cache_dev = "cpu" if os.environ.get("CACHE_ON_CPU") == "1" else DEVICE
    buf_t = torch.from_numpy(buf)
    if cache_dev == "cpu":
        buf_t = buf_t.pin_memory()
    else:
        buf_t = buf_t.to(DEVICE)
    return (buf_t, torch.from_numpy(vlen).to(DEVICE),
            torch.tensor(df.label.map(LBL).values, dtype=torch.long, device=DEVICE))

t0 = time.time()
BUF, VLEN, Y = build_cache(pool)
log(f"cache ready: {tuple(BUF.shape)} fp16 = {BUF.numel()*2/1e9:.2f} GB on {BUF.device} ({time.time()-t0:.0f}s)")

_AR = torch.arange(CROP_LEN, device=DEVICE)

def get_batch(idx, train):
    span = (VLEN[idx] - CROP_LEN).clamp(min=0)
    start = (torch.rand(len(idx), device=DEVICE) * (span + 1).float()).long() if train else span // 2
    gidx = (start.unsqueeze(1) + _AR.unsqueeze(0)).clamp(max=CACHE_LEN - 1)
    rows = BUF[idx] if BUF.is_cuda else BUF[idx.cpu()].to(DEVICE, non_blocking=True)
    return torch.gather(rows, 1, gidx).float(), Y[idx]

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

def adapt(model, idx, use_st=True, use_cons=True, epochs=TTA_EPOCHS):
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
                    loss = loss + LAMBDA_CONS * F.mse_loss(torch.softmax(model(augment(x))[0], 1), p.detach())
            if loss.requires_grad:
                loss.backward(); opt.step()
    return model


def _bbse_pi_fake(model, tgt_idx, seed, target):
    """Label-free P(fake) on the target pool by black-box shift estimation.

    M[y, yhat] = P_source(yhat | y) from an in-domain labelled source-pool split;
    q = observed target prediction histogram; p = M^{-T} q, clipped to a simplex.
    Unlike the 2-component GMM (AT.estimate_prevalence_guarded), BBSE reads the
    *source* error structure and the *target prediction rate*, not the target
    score-distribution shape -- so it survives the calibration shift that this
    paper's thesis says is exactly what does transfer cross-corpus. On the skew
    sweep it estimates the prior to within ~1.4% at 50-95% fake where the GMM is
    ~28% off (r2_vuln2_prevalence_cpu.py).
    """
    src_df = sample_source(pool[pool.corpus != target], 1500, seed if seed is not None else 0)
    src_idx = idx_of(src_df)
    y_src, s_src = score(model, src_idx)
    y_src = np.asarray(y_src).astype(int)
    yh = (s_src >= 0.5).astype(int)
    M = np.array([[np.mean(yh[y_src == 0] == 0), np.mean(yh[y_src == 0] == 1)],
                  [np.mean(yh[y_src == 1] == 0), np.mean(yh[y_src == 1] == 1)]])
    _, s_t = score(model, tgt_idx)
    q = np.array([np.mean((s_t >= 0.5) == 0), np.mean((s_t >= 0.5) == 1)])
    try:
        p = np.linalg.solve(M.T, q)
    except np.linalg.LinAlgError:
        return 0.5
    p = np.clip(p, 1e-3, 1.0 - 1e-3)
    return float(p[1] / p.sum())


def adapt_dynq(model, idx, epochs=TTA_EPOCHS, seed=None, target=None, **_):
    """adapt(), but the confident-tail budget is split by a BBSE prevalence
    estimate instead of the fixed symmetric q=0.3.

    On a balanced pool BBSE returns ~0.5 and AT.tail_budget gives (Q, Q) exactly
    -- byte-identical to adapt(). On a 90%-fake pool it gives (~0.06, ~0.54): the
    pseudo-real bucket shrinks to the true real fraction instead of scooping up
    900 mislabelled fakes. Prevalence is estimated ONCE from the frozen
    pre-adaptation model (re-estimating per epoch lets a drifting model reinforce
    its own mistakes -- protocol_a.py:392).
    """
    set_tta_params(model)
    opt = torch.optim.Adam(trainable(model), lr=TTA_LR)
    pi_fake = _bbse_pi_fake(model, idx, seed, target)
    lo_frac, hi_frac = AT.tail_budget(Q, 1.0 - pi_fake)     # tail_budget takes pi_real
    log(f"  dynq: BBSE pi_fake={pi_fake:.3f}  budget=(lo {lo_frac:.3f}, hi {hi_frac:.3f})")
    for _ in range(epochs):
        _, s = score(model, idx)
        pl = torch.full((len(idx),), -1, dtype=torch.long, device=DEVICE)
        if lo_frac > 0:
            pl[torch.from_numpy(s <= np.quantile(s, lo_frac)).to(DEVICE)] = 0
        if hi_frac > 0:
            pl[torch.from_numpy(s >= np.quantile(s, 1.0 - hi_frac)).to(DEVICE)] = 1
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
                if conf.any():
                    loss = loss + F.cross_entropy(logits[conf], bpl[conf])
                loss = loss + LAMBDA_CONS * F.mse_loss(
                    torch.softmax(model(augment(x))[0], 1), p.detach())
            if loss.requires_grad:
                loss.backward(); opt.step()
    return model

# --- modern TTA baselines (SHOT / ETA / EATA / SAR) -------------------------
# The paper's only TTA comparison point was Tent (2021). Tent's collapse here is
# a headline observation, but the literature already answered it: reliable-sample
# filtering (ETA/EATA), sharpness-aware entropy (SAR), and information
# maximisation with pseudo-labels (SHOT). Comparing only against the method that
# is known to be unstable is the cheapest objection a reviewer can raise.
# Implementations live in tta_baselines.py against a callback protocol so the
# same code also runs on the third-party checkpoints.
import tta_baselines as TB


def _tb_forward(model, idx):
    """Callback: pool indices -> logits in canonical [real, fake] order."""
    def f(sel):
        x, _ = get_batch(idx[sel], train=False)
        return model(x)[0]
    return f


def shot_baseline(model, idx, epochs=TTA_EPOCHS, **_):
    """SHOT: information maximisation + centroid pseudo-labels, head FROZEN.

    SHOT's premise is that the source hypothesis (the classifier) is kept and
    only the feature extractor moves -- the exact opposite of our method, which
    adapts the head. Freezing the head here keeps that contrast real instead of
    turning SHOT into a re-parameterisation of ours.
    """
    set_tta_params(model)
    for head in (model.attn, model.proj, model.cls):
        for p_ in head.parameters():
            p_.requires_grad_(False)
    params = trainable(model)
    if not params:
        log("  shot: empty trainable set after freezing the head -- skipping")
        return model
    return TB.shot(model, len(idx), _tb_forward(model, idx), params, amp, DEVICE,
                   epochs=epochs, lr=TTA_LR, batch=BATCH, log=log)


def eata_baseline(model, idx, epochs=TTA_EPOCHS, use_fisher=True, seed=None,
                  target=None, **_):
    """ETA (use_fisher=False) / EATA (True). Same trainable set as ours."""
    set_tta_params(model)
    params = trainable(model)
    fisher = anchor_w = None
    if use_fisher:
        src_df = sample_source(pool[pool.corpus != target], 750,
                               seed if seed is not None else 0)
        src_idx = idx_of(src_df)
        fisher = TB.compute_fisher(model, params, len(src_idx),
                                   _tb_forward(model, src_idx), amp, DEVICE,
                                   batch=BATCH, log=log)
        anchor_w = [p_.detach().clone() for p_ in params]
    return TB.eata(model, len(idx), _tb_forward(model, idx), params, amp, DEVICE,
                   epochs=epochs, lr=TTA_LR, batch=BATCH, fisher=fisher,
                   anchor=anchor_w, log=log)


def sar_baseline(model, idx, epochs=TTA_EPOCHS, **_):
    """SAR: reliable-sample entropy + SAM + model recovery."""
    set_tta_params(model)
    params = trainable(model)
    return TB.sar(model, len(idx), _tb_forward(model, idx), params, amp, DEVICE,
                  epochs=epochs, lr=TTA_LR, batch=BATCH,
                  snapshot=lambda: _snapshot(model),
                  restore=lambda sn: _restore(model, sn), log=log)


# --- label-free budget selection -------------------------------------------
# The manuscript concedes that E is "not a hyperparameter but a mis-specified
# budget" and that its strongest numbers come from E=32, chosen post hoc. That
# concession costs more than the result is worth unless the budget can be picked
# without target labels. `adapt_traced` runs a long budget once and records, at
# every epoch, both the (label-using) EER/AUC and a set of candidate label-free
# monitors, so a selection rule can be fitted and validated leave-one-target-out
# afterwards from the trace alone. The labels in the trace are ANALYSIS ONLY --
# no monitor reads them.
STOP_TRACE_CSV = f"stop_trace{SUFFIX}.csv"


def _spearman(a, b):
    """Rank correlation without scipy (numpy-only, ties averaged)."""
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def _stop_monitors(s_now, s_prev, s_src, pi_hat):
    """Label-free statistics computable at any point during adaptation.

    * `rho_src`  -- Spearman correlation between the current scores and the
      FROZEN source model's scores on the same pool. The paper's thesis is that
      the source *ranking* is the asset that transfers; an adaptation that
      destroys it has thrown away the only thing it was standing on. This is the
      monitor that should catch a Tent-style collapse, because a collapse IS a
      ranking collapse (AUC .918 -> .502).
    * `prior_gap` -- |predicted fake rate - BBSE prior estimate|. BBSE gives a
      label-free estimate of the target prior; a model drifting away from it is
      re-labelling the pool, not re-calibrating on it.
    * `churn`    -- fraction of the pool whose hard prediction flipped since the
      previous epoch. The obvious convergence signal, included so that the
      principled monitors have to beat it rather than merely exist.
    * `conf`     -- mean max-probability. High confidence is what entropy
      minimisation maximises, so it should be the monitor that fails.
    """
    pred = (s_now >= 0.5)
    return dict(
        rho_src=_spearman(s_now, s_src),
        prior_gap=float(abs(pred.mean() - pi_hat)) if pi_hat is not None else np.nan,
        pred_rate=float(pred.mean()),
        churn=float(np.mean(pred != (s_prev >= 0.5))) if s_prev is not None else np.nan,
        conf=float(np.mean(np.maximum(s_now, 1 - s_now))),
        tail_gap=float(np.quantile(s_now, 1 - Q) - np.quantile(s_now, Q)),
    )


def _trace_row(**row):
    pd.DataFrame([row]).to_csv(STOP_TRACE_CSV, mode="a",
                               header=not os.path.exists(STOP_TRACE_CSV), index=False)


def adapt_traced(model, idx, epochs=32, variant="ours", seed=None, target=None,
                 use_bbse_budget=False, **_):
    """One long adaptation run, fully instrumented per epoch.

    `variant`: "ours" (confident-tail self-training + consistency) or "tent"
    (entropy minimisation) -- the collapse case the monitor has to catch.
    Returns the model at the FINAL epoch; every intermediate epoch's metrics are
    in STOP_TRACE_CSV, which is what the selection analysis reads.
    """
    set_tta_params(model)
    opt = torch.optim.Adam(trainable(model), lr=TTA_LR)
    y_true, s_src = score(model, idx)          # frozen-source reference ranking
    pi_hat = None
    lo_frac = hi_frac = Q
    if use_bbse_budget:
        pi_hat = _bbse_pi_fake(model, idx, seed, target)
        lo_frac, hi_frac = AT.tail_budget(Q, 1.0 - pi_hat)
    else:
        # a prior estimate is still recorded as a monitor even when the budget
        # does not use it, so `prior_gap` is available on every arm
        pi_hat = _bbse_pi_fake(model, idx, seed, target)
    m0 = _stop_monitors(s_src, None, s_src, pi_hat)
    _trace_row(seed=seed, target=target, variant=variant, epoch=0,
               eer=eer(y_true, s_src) * 100,
               auc=roc_auc_score(y_true, s_src),
               acc=accuracy_score(y_true, (s_src >= 0.5).astype(int)) * 100,
               pi_hat=pi_hat, n_conf=0, **m0)
    # One scoring pass per epoch, not two: the scores measured at the END of
    # epoch t are exactly the scores adapt() would compute at the START of
    # epoch t+1 to build its pseudo-labels, so reusing them is semantically
    # identical and halves the instrumentation cost (0.36 min per pass x 32
    # epochs x 8 folds is 1.5 GPU-h of pure duplication).
    s_prev, s = s_src, s_src
    for ep in range(epochs):
        pl = torch.full((len(idx),), -1, dtype=torch.long, device=DEVICE)
        n_conf = 0
        if variant == "ours":
            if lo_frac > 0:
                pl[torch.from_numpy(s <= np.quantile(s, lo_frac)).to(DEVICE)] = 0
            if hi_frac > 0:
                pl[torch.from_numpy(s >= np.quantile(s, 1 - hi_frac)).to(DEVICE)] = 1
            n_conf = int((pl >= 0).sum())
        model.train(); model.ssl.eval()
        order = torch.randperm(len(idx), device=DEVICE)
        for i in range(0, len(order), BATCH):
            sel = order[i:i + BATCH]
            x, _ = get_batch(idx[sel], train=False)
            opt.zero_grad(set_to_none=True)
            with amp():
                logits, _ = model(x)
                if variant == "tent":
                    p = torch.softmax(logits, 1)
                    loss = -(p * torch.log(p + 1e-8)).sum(1).mean()
                else:
                    bpl = pl[sel]
                    p = torch.softmax(logits, 1)
                    loss = torch.zeros((), device=DEVICE)
                    conf = bpl >= 0
                    if conf.any():
                        loss = loss + F.cross_entropy(logits[conf], bpl[conf])
                    loss = loss + LAMBDA_CONS * F.mse_loss(
                        torch.softmax(model(augment(x))[0], 1), p.detach())
            if loss.requires_grad:
                loss.backward(); opt.step()
        _, s_now = score(model, idx)
        mon = _stop_monitors(s_now, s_prev, s_src, pi_hat)
        _trace_row(seed=seed, target=target, variant=variant, epoch=ep + 1,
                   eer=eer(y_true, s_now) * 100,
                   auc=roc_auc_score(y_true, s_now),
                   acc=accuracy_score(y_true, (s_now >= 0.5).astype(int)) * 100,
                   pi_hat=pi_hat, n_conf=n_conf, **mon)
        log(f"    trace {variant} ep{ep+1}: EER {eer(y_true, s_now)*100:.2f} "
            f"AUC {roc_auc_score(y_true, s_now):.4f} rho_src {mon['rho_src']:.3f} "
            f"prior_gap {mon['prior_gap']:.3f} churn {mon['churn']:.3f}")
        s_prev, s = s, s_now
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
# ## 6. Adaptive adaptation
#
# `adapt_adaptive` is one loop with two independent switches so that every arm
# of the ablation runs the same code path. With both switches off it must be
# numerically identical to `adapt` above (the published method) -- that
# reduction is asserted by `verify_reduction.py`, and nothing downstream is
# comparable if it fails. To keep it exact, the extra augmented scoring pass
# that `select_q_max` needs is done ONCE before the loop and only when
# `adapt_q=True`, so no arm consumes RNG that the fixed arm would not.

# %%
@torch.no_grad()
def score_aug(model, idx):
    """Score the augmented view. Used once, pre-adaptation, to size the budget."""
    model.eval()
    out = []
    for i in range(0, len(idx), BATCH):
        x, _ = get_batch(idx[i:i + BATCH], train=False)
        with amp():
            out.append(torch.softmax(model(augment(x))[0].float(), 1)[:, 1])
    return torch.cat(out).cpu().numpy()


def _snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def _restore(model, snap):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in snap:
                p.copy_(snap[n])


_trace_rows = []


def record_trace(**row):
    _trace_rows.append(row)
    pd.DataFrame([row]).to_csv(TRACE_CSV, mode="a",
                               header=not os.path.exists(TRACE_CSV), index=False)


def adapt_adaptive(model, idx, adapt_q=True, adapt_e=True, tag="", seed=None, target=None,
                   method="", epochs=TTA_EPOCHS):
    """Confident-tail self-training + channel consistency, with adaptive q and E.

    lambda is fixed at LAMBDA_CONS in every arm -- see the note at the top of
    this file for why gating it was dropped.
    """
    set_tta_params(model)
    opt = torch.optim.Adam(trainable(model), lr=TTA_LR)
    pre = _snapshot(model)

    # --- budget sizing, once, from the frozen pre-adaptation model -------------
    # Re-estimating per epoch would let a drifting self-trained model reinforce
    # its own mistakes (the rationale recorded at protocol_a.py:392-394).
    # q_max is the published Q, not a data-derived value: AT.select_q_max is
    # confounded by augmentation magnitude and returns the grid minimum on real
    # audio regardless of corpus (see its docstring). The adaptive parts of q are
    # the curriculum ramp and the prevalence-weighted split.
    pi_real, bic_delta, shrunk, q_max = 0.5, float("nan"), False, Q
    if adapt_q:
        _, s0 = score(model, idx)
        pi_real, bic_delta, shrunk = AT.estimate_prevalence_guarded(s0)
        log(f"  {tag} pi_real={pi_real:.4f} (bic_delta={bic_delta:.1f}, shrunk={shrunk})  q_max={q_max:.3f}")

    n_epochs = AT.E_MAX if adapt_e else epochs
    pl_prev, s_prev, shift_hist, stopped = None, None, [], ""

    for ep in range(n_epochs):
        _, s = score(model, idx)

        bad, why = AT.collapsed(s)
        if bad:
            # Hand back the pre-adaptation model: a no-op run beats the silent
            # catastrophic one that main.tex:460-479 describes for Tent.
            log(f"  {tag} COLLAPSE at epoch {ep} ({why}) -- reverting to pre-adaptation model")
            _restore(model, pre)
            record_trace(seed=seed, target=target, method=method, epoch=ep, q_t=float("nan"),
                         lo_frac=float("nan"), hi_frac=float("nan"), lambda_t=LAMBDA_CONS,
                         c_epoch=float("nan"), shift=float("nan"), pi_real=pi_real, shrunk=shrunk, q_max=q_max,
                         n_conf=0, stopped=why)
            return model

        if adapt_q:
            q_t = AT.q_schedule(ep, AT.E_MAX, q_max=q_max)
            lo_frac, hi_frac = AT.tail_budget(q_t, pi_real)
        else:
            q_t, lo_frac, hi_frac = Q, Q, Q

        pl_np = AT.pseudo_labels(s, lo_frac, hi_frac)
        c_epoch = AT.epoch_churn(pl_prev, pl_np) if pl_prev is not None else float("nan")
        shift = AT.score_shift(s_prev, s) if s_prev is not None else float("nan")
        if s_prev is not None:
            shift_hist.append(shift)
        pl_prev, s_prev = pl_np, s

        pl = torch.from_numpy(pl_np).to(DEVICE)
        n_conf = int((pl >= 0).sum())

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
                if conf.any():
                    loss = loss + F.cross_entropy(logits[conf], bpl[conf])
                loss = loss + LAMBDA_CONS * F.mse_loss(torch.softmax(model(augment(x))[0], 1), p.detach())
            if loss.requires_grad:
                loss.backward(); opt.step()

        stop, why = AT.should_stop(shift_hist, ep, max_epochs=n_epochs) if adapt_e else (ep + 1 >= n_epochs, "fixed_E")
        record_trace(seed=seed, target=target, method=method, epoch=ep, q_t=round(q_t, 4),
                     lo_frac=round(lo_frac, 4), hi_frac=round(hi_frac, 4), lambda_t=LAMBDA_CONS,
                     c_epoch=round(c_epoch, 4) if c_epoch == c_epoch else float("nan"),
                     shift=round(shift, 5) if shift == shift else float("nan"),
                     pi_real=round(pi_real, 4), shrunk=shrunk, q_max=q_max,
                     n_conf=n_conf, stopped=why if stop else "")
        log(f"  {tag} ep {ep+1}/{n_epochs}  q={q_t:.3f} budget=({lo_frac:.3f},{hi_frac:.3f}) "
            f"conf={n_conf}/{len(idx)} shift={shift:.5f} churn={c_epoch:.4f}"
            + (f"  STOP({why})" if stop else ""))
        if stop:
            stopped = why
            break

    return model


# %% [markdown]
# ## 7. Experiment grid
#
# Adaptation only -- source checkpoints are read, never trained. Rows are
# appended immediately so an interrupted run still leaves usable data, and the
# resume guard keys on (seed, target, method, setting), a strictly finer key
# than `extended_pipeline.py`'s, so a restart resumes per method.

# %%
def record(**row):
    pd.DataFrame([row]).to_csv(RESULTS_CSV, mode="a",
                               header=not os.path.exists(RESULTS_CSV), index=False)


_KEY = ["seed", "target", "method", "setting"]
_prior = (pd.read_csv(RESULTS_CSV)[_KEY] if os.path.exists(RESULTS_CSV)
          else pd.DataFrame(columns=_KEY))


def done(seed, target, method, setting):
    if not len(_prior):
        return False
    return ((_prior.seed == seed) & (_prior.target == target)
            & (_prior.method == method) & (_prior.setting == setting)).any()


def run_grid():
    grid_t0 = time.time()
    for seed in SEEDS:
        for target in TARGETS:
            tag = f"seed{seed}/{target}"
            ckpt = f"{CKPT_DIR}/source_{target}_seed{seed}.pt"
            if not os.path.exists(ckpt):
                log(f"=== {tag}: no checkpoint at {ckpt}, skipping (this run never trains) ===")
                continue

            log(f"=== {tag} ===")
            # TARGET_SKEW=0.9 makes the target pool 90% fake instead of balanced.
            # Both the published q=0.3 symmetric pseudo-labelling AND the trivial
            # median-threshold baseline implicitly assume a balanced pool: q takes
            # the top and bottom 30% as fake/real regardless of the true prior, and
            # the median predicts exactly 50% positive. If they degrade together
            # under skew they share that assumption, which would also explain the
            # documented Protocol A collapse (26.33 -> 42.46 EER on a 97%-spoof
            # pool) that the paper reports but does not account for.
            _skew = os.environ.get("TARGET_SKEW")
            if _skew:
                _sk = float(_skew)
                _sub = pool[pool.corpus == target]
                _n = 2 * TARGET_PER_CLASS
                _want = {"fake": int(round(_n * _sk)), "real": int(round(_n * (1 - _sk)))}
                _parts = []
                for _lab, _g in _sub.groupby("label"):
                    _parts.append(_g.sample(min(_want.get(_lab, 0), len(_g)), random_state=seed))
                tgt_df = pd.concat(_parts).sample(frac=1, random_state=seed)
                log(f"  SKEW {_sk}: pool {tgt_df.label.value_counts().to_dict()}")
            else:
                tgt_df = sample_target(pool[pool.corpus == target], TARGET_PER_CLASS, seed)
            tgt_idx = idx_of(tgt_df)
            log(f"  target {len(tgt_df)} clips")

            # A corrupt checkpoint must cost one fold, not the whole run. An
            # incomplete 2.3 GB upload of ckpt_ext killed a session here: torch
            # raised "failed finding central directory" on one file and the
            # exception propagated out of the fold loop, abandoning the ten
            # folds that came after it.
            source = XLSRDetector(encoder_amp=ENCODER_AMP).to(DEVICE)
            try:
                source.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)
            except Exception as e:
                log(f"  !! {ckpt} failed to load ({type(e).__name__}: {e}) -- "
                    f"skipping this fold; re-upload the file and re-run to resume")
                del source
                torch.cuda.empty_cache()
                continue

            methods = {
                "source":        None,
                "ours_fixed":    lambda m, i, **k: adapt(m, i),
                "ours_adaptive": lambda m, i, **k: adapt_adaptive(m, i, True, True, **k),
            }
            # TTA_BASELINES=1 replaces the arms with the modern TTA
            # baselines the paper is missing. Ours and Tent are re-run in the
            # same fold so the comparison is on identical pools and seeds
            # rather than joined across runs.
            # STOP_TRACE=1 runs one long instrumented budget per fold, for
            # the label-free budget-selection analysis. The EER at every epoch
            # is in the trace, so "what would the rule have picked" is answered
            # without re-running anything.
            if os.environ.get("STOP_TRACE"):
                _e = int(os.environ.get("STOP_EPOCHS", "32"))
                methods = {
                    "source":       None,
                    "traced_ours":  (lambda ee: (lambda m, i, **k: adapt_traced(
                        m, i, epochs=ee, variant="ours", **k)))(_e),
                    "traced_tent":  (lambda ee: (lambda m, i, **k: adapt_traced(
                        m, i, epochs=ee, variant="tent", **k)))(_e),
                }
            elif os.environ.get("TTA_BASELINES"):
                _e = int(os.environ.get("BASELINE_EPOCHS", TTA_EPOCHS))
                methods = {
                    "source":      None,
                    "ours_fixed":  (lambda ee: (lambda m, i, **k: adapt(m, i, epochs=ee)))(_e),
                    "tent":        (lambda ee: (lambda m, i, **k: tent(m, i, epochs=ee)))(_e),
                    "shot":        (lambda ee: (lambda m, i, **k: shot_baseline(m, i, epochs=ee)))(_e),
                    "eta":         (lambda ee: (lambda m, i, **k: eata_baseline(m, i, epochs=ee, use_fisher=False, **k)))(_e),
                    "eata":        (lambda ee: (lambda m, i, **k: eata_baseline(m, i, epochs=ee, use_fisher=True, **k)))(_e),
                    "sar":         (lambda ee: (lambda m, i, **k: sar_baseline(m, i, epochs=ee)))(_e),
                }
            # DYNQ=1: the minimal dynamic-q arm -- published adapt() with the
            # confident-tail budget split by a BBSE prevalence estimate instead
            # of the fixed symmetric q=0.3, and nothing else changed (no q ramp,
            # no adaptive E). Point: the prevalence-weighted budget was reported
            # to fail on Protocol A, but it was fed by the 2-component GMM, which
            # r2_vuln2_prevalence_cpu.py shows is ~28% wrong at 90% skew. BBSE is
            # ~1.4% off there. Run with TARGET_SKEW to test whether a correct
            # prior makes dynamic q actually work on unbalanced pools.
            elif os.environ.get("DYNQ"):
                methods = {
                    "source":     None,
                    "ours_fixed": lambda m, i, **k: adapt(m, i),
                    "ours_dynq":  lambda m, i, **k: adapt_dynq(m, i, seed=k.get("seed"),
                                                               target=k.get("target")),
                    "ours_gmmq":  lambda m, i, **k: adapt_adaptive(m, i, True, False, **k),
                }
            # OUR_E_SWEEP="8,16,32" replaces the arms above with published-adapt()
            # runs at those epoch counts, to map the shape of the E curve on OUR
            # source model. Motivation: E is an *epoch* count, so the update budget
            # is a function of pool size; four epochs over In-the-Wild's full
            # 31,779 clips is 5.3x the steps of four over a 6,000-clip pool, and
            # matching the steps on third-party checkpoints beat the full-pool
            # result. The note at the top of this file already records
            # "E=4->8 is worth 1.2-1.6 EER in the existing sweep" -- this maps
            # the rest of the curve, and specifically whether it plateaus
            # (safe: pick a big budget, no labels needed) or peaks and declines
            # (unusable: locating the peak would need target labels).
            # OUR_BASELINE_E="4,32" runs tent and self-training-only at those
            # epoch counts. This exists to close a fairness hole rather than to
            # add a result: we claim E=32 beats the published E=4 for our
            # method, which gives it 8x the updates. A reviewer will ask whether
            # the baselines improve as much when handed the same budget, and
            # until they are run at matched E the comparison is confounded.
            if os.environ.get("OUR_BASELINE_E"):
                _bs = [int(x) for x in os.environ["OUR_BASELINE_E"].split(",")]
                methods = {"source": None}
                for _e in _bs:
                    methods[f"tent_E{_e}"] = (
                        lambda ee: (lambda m, i, **k: tent(m, i, epochs=ee)))(_e)
                    methods[f"st_only_E{_e}"] = (
                        lambda ee: (lambda m, i, **k: adapt(m, i, use_st=True,
                                                            use_cons=False, epochs=ee)))(_e)
            elif os.environ.get("OUR_E_SWEEP"):
                _es = [int(x) for x in os.environ["OUR_E_SWEEP"].split(",")]
                methods = {"source": None}
                for _e in _es:
                    # bind e per-iteration; a bare closure would capture the last value
                    methods[f"ours_E{_e}"] = (lambda ee: (lambda m, i, **k: adapt(m, i, epochs=ee)))(_e)
            if seed in ABLATION_SEEDS and not (os.environ.get("OUR_E_SWEEP")
                                               or os.environ.get("OUR_BASELINE_E")
                                               or os.environ.get("TTA_BASELINES")
                                               or os.environ.get("STOP_TRACE")
                                               or os.environ.get("DYNQ")):
                methods["ours_aq"] = lambda m, i, **k: adapt_adaptive(m, i, True, False, **k)
                methods["ours_ae"] = lambda m, i, **k: adapt_adaptive(m, i, False, True, **k)

            # The setting label must be computed BEFORE the resume guard, not
            # only at record time. Keying the guard on a hardcoded "transductive"
            # while writing rows under "skew0.9" makes every skewed fold look
            # already-done: the first attempt at this experiment "completed" ten
            # jobs in 0.6 min each and wrote nothing. This is the resume-guard
            # hazard the repo already documents; it silently skips folds rather
            # than failing.
            _setting = ("transductive" if not os.environ.get("TARGET_SKEW")
                        else f"skew{os.environ['TARGET_SKEW']}")
            for name, fn in methods.items():
                if done(seed, target, name, _setting):
                    log(f"  {name}: already recorded, skipping")
                    continue
                try:
                    torch.manual_seed(seed); np.random.seed(seed)
                    t0 = time.time()
                    model = source if fn is None else fn(copy.deepcopy(source), tgt_idx,
                                                         tag=f"{tag}/{name}", seed=seed,
                                                         target=target, method=name)
                    m, y, s = metrics(model, tgt_idx)
                    record(seed=seed, target=target, method=name, setting=_setting, family="xlsr",
                           eer=round(m["eer"], 3), auc=round(m["auc"], 4), acc=round(m["acc"], 2),
                           n=len(tgt_idx), minutes=round((time.time() - t0) / 60, 1))
                    log(f"  {name:14s} EER {m['eer']:6.2f}  AUC {m['auc']:.3f}  ({(time.time()-t0)/60:.1f} min)")
                    if fn is not None:
                        del model; torch.cuda.empty_cache()
                except Exception as e:
                    log(f"  !! {name} failed: {type(e).__name__}: {e}")

            # inductive check: adapt on one half, evaluate on the disjoint half.
            # Skipped during an E sweep -- it exercises adapt_adaptive(), which is
            # a different arm from the fixed-E adapt() the sweep is measuring.
            if not (os.environ.get("OUR_E_SWEEP") or os.environ.get("OUR_BASELINE_E")
                    or os.environ.get("TTA_BASELINES") or os.environ.get("STOP_TRACE")
                    or os.environ.get("DYNQ")) \
                    and not done(seed, target, "ours_adaptive", "inductive"):
                try:
                    torch.manual_seed(seed); np.random.seed(seed)
                    half = len(tgt_idx) // 2
                    perm = torch.randperm(len(tgt_idx), device=DEVICE)
                    a_idx, b_idx = tgt_idx[perm[:half]], tgt_idx[perm[half:]]
                    m_src, _, _ = metrics(source, b_idx)
                    model = adapt_adaptive(copy.deepcopy(source), a_idx, True, True,
                                           tag=f"{tag}/inductive", seed=seed, target=target,
                                           method="ours_adaptive_inductive")
                    m_ada, _, _ = metrics(model, b_idx)
                    for nm, mm in (("source", m_src), ("ours_adaptive", m_ada)):
                        record(seed=seed, target=target, method=nm, setting="inductive", family="xlsr",
                               eer=round(mm["eer"], 3), auc=round(mm["auc"], 4), acc=round(mm["acc"], 2),
                               n=len(b_idx), minutes=0)
                    log(f"  inductive: source EER {m_src['eer']:.2f} -> adaptive EER {m_ada['eer']:.2f}")
                    del model; torch.cuda.empty_cache()
                except Exception as e:
                    log(f"  !! inductive failed: {type(e).__name__}: {e}")

            del source; torch.cuda.empty_cache()

    log(f"grid done in {(time.time() - grid_t0)/60:.1f} min -> {RESULTS_CSV}, {TRACE_CSV}")


if __name__ == "__main__":
    run_grid()
