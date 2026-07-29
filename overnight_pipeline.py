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
# # Cross-Corpus Audio Deepfake Detection via Test-Time Adaptation
#
# Leave-one-corpus-out study. For each held-out target corpus we train an XLS-R
# detector on the remaining corpora, then adapt it to the *unlabeled* target with
# confident-pseudo-label self-training + channel consistency, and compare against
# source-only and Tent (entropy-minimisation TTA).
#
# Designed to run unattended: all audio is decoded once into GPU memory, results
# are appended to `results.csv` as they complete, and progress is mirrored to
# `run_log.txt` (`tail -f run_log.txt` from a terminal to watch).
#
# **Set `SMOKE = True` for a ~3 min end-to-end validation, then `SMOKE = False`
# and Run All before leaving.**

# %%
import sys, subprocess

def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "librosa", "numba"], check=False)
_pip("numpy==1.26.4", "pandas", "scikit-learn")
_pip("torch", "torchaudio", "soundfile", "datasets", "tqdm", "kaggle", "matplotlib")
_pip("numpy==1.26.4")
print("install complete -- if this is the first run, RESTART THE KERNEL now, then continue")

# %%
import copy, glob, json, os, shutil, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torchaudio, soundfile as sf
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from tqdm.auto import tqdm

SMOKE = True                      # <-- flip to False for the real run

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SR = 16000
CROP_SEC, CACHE_SEC = 3.0, 4.0            # cache 4 s, train on random 3 s crops
CROP_LEN, CACHE_LEN = int(SR * CROP_SEC), int(SR * CACHE_SEC)

N_FINETUNE = 4                            # top XLS-R layers to fine-tune
LR, TTA_LR = 1e-4, 1e-4
Q = 0.3                                   # confident-tail quantile for pseudo-labels
LAMBDA_CONS = 0.3

if SMOKE:
    SEEDS, TARGETS = [0], ["in_the_wild"]
    SOURCE_EPOCHS, TTA_EPOCHS, BATCH = 1, 1, 8
    SOURCE_PER_CLASS, TARGET_PER_CLASS, MAX_PER_CORPUS_CLASS = 32, 32, 64
else:
    SEEDS, TARGETS = [0, 1, 2], ["in_the_wild", "arabic"]
    SOURCE_EPOCHS, TTA_EPOCHS, BATCH = 12, 4, 64
    SOURCE_PER_CLASS, TARGET_PER_CLASS, MAX_PER_CORPUS_CLASS = 5000, 3000, 6000

# smoke artefacts are kept separate so a throwaway smoke model can never be
# resumed by the real run
SUFFIX = "_smoke" if SMOKE else ""
RESULTS_CSV, LOG_FILE = f"results{SUFFIX}.csv", f"run_log{SUFFIX}.txt"
CKPT_DIR = f"ckpt{SUFFIX}"
os.makedirs(CKPT_DIR, exist_ok=True)

def amp():
    return torch.amp.autocast("cuda", dtype=torch.bfloat16) if USE_BF16 else torch.amp.autocast("cuda", enabled=False)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

log(f"python {sys.version.split()[0]} | torch {torch.__version__} | torchaudio {torchaudio.__version__}")
log(f"device {DEVICE} {torch.cuda.get_device_name(0) if DEVICE=='cuda' else ''} | bf16 {USE_BF16} | SMOKE {SMOKE}")
assert hasattr(torchaudio.pipelines, "WAV2VEC2_XLSR_300M"), "XLS-R bundle missing"

# %% [markdown]
# ## 1. Data
# Upload `kaggle.json` next to this notebook. Arabic comes from the Hugging Face Hub.

# %%
_kdir = os.path.expanduser("~/.kaggle")
if os.path.exists("kaggle.json"):
    os.makedirs(_kdir, exist_ok=True)
    shutil.copy("kaggle.json", f"{_kdir}/kaggle.json")
    os.chmod(f"{_kdir}/kaggle.json", 0o600)
    log("kaggle.json installed")

KAGGLE_SETS = [
    ("azkurniwan/asvspoof-2019-la", "data/asvspoof2019_LA", "ASVspoof2019_LA_train/flac/*.flac"),
    ("bhaveshkumars/release-in-the-wild", "data/in_the_wild", "**/*.wav"),
    ("adarshsingh0903/audio-deepfake-detection-dataset", "data/dataset_2", "**/*.wav"),
]

import kaggle
for slug, path, probe in KAGGLE_SETS:
    if glob.glob(os.path.join(path, probe), recursive=True):
        log(f"already present: {path}")
        continue
    log(f"downloading {slug}")
    kaggle.api.dataset_download_files(slug, path=path, unzip=True, quiet=False)

def fetch_arabic(out="data/arabic_arad"):
    if glob.glob(f"{out}/**/*.wav", recursive=True):
        log(f"already present: {out}")
        return
    from datasets import load_dataset, Audio
    log("downloading Arabic ArAD from HF")
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
# `dataset_2` contributes eight distinct TTS generators; keeping them all (not just
# two) is what gives the source pool its generator diversity.

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
                rows.append((f"{flac}/{p[1]}.flac", lab, "asvspoof2019", gen))

    for d in sorted(glob.glob("data/dataset_2/*/")):
        gen = os.path.basename(d.rstrip("/"))
        lab = "real" if gen == "real_samples" else "fake"
        for w in glob.glob(f"{d}/**/*.wav", recursive=True):
            rows.append((w, lab, "dataset2", gen))

    for corpus, root in [("in_the_wild", "data/in_the_wild"), ("arabic", "data/arabic_arad")]:
        for w in glob.glob(f"{root}/**/*.wav", recursive=True):
            parts = w.split(os.sep)
            lab = "real" if "real" in parts else "fake" if "fake" in parts else None
            if lab:
                rows.append((w, lab, corpus, corpus))

    return pd.DataFrame(rows, columns=["path", "label", "corpus", "generator"])

manifest = build_manifest()
log("manifest:\n" + manifest.groupby(["corpus", "label"]).size().to_string())
log(f"dataset2 generators: {sorted(manifest[manifest.corpus=='dataset2'].generator.unique())}")

# %% [markdown]
# ## 3. GPU-resident audio cache
# Every clip we may touch is decoded once into a single fp16 tensor on the GPU.
# Training then never touches the filesystem — no DataLoader workers, no shared
# memory, and the GPU stays saturated instead of waiting on I/O.

# %%
def cap_per_class(df, n, seed=0):
    out = []
    for lab, g in df.groupby("label"):
        out.append(g.sample(min(n, len(g)), random_state=seed))
    return pd.concat(out)

pool = pd.concat([cap_per_class(g, MAX_PER_CORPUS_CLASS)
                  for _, g in manifest.groupby("corpus")]).reset_index(drop=True)
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
    return (torch.from_numpy(buf).to(DEVICE),
            torch.from_numpy(vlen).to(DEVICE),
            torch.tensor(df.label.map(LBL).values, dtype=torch.long, device=DEVICE))

t0 = time.time()
BUF, VLEN, Y = build_cache(pool)
log(f"cache ready: {tuple(BUF.shape)} fp16 = {BUF.numel()*2/1e9:.2f} GB on GPU ({time.time()-t0:.0f}s)")

_AR = torch.arange(CROP_LEN, device=DEVICE)

def get_batch(idx, train):
    """Slice CROP_LEN samples out of the cache: random offset when training."""
    span = (VLEN[idx] - CROP_LEN).clamp(min=0)
    start = (torch.rand(len(idx), device=DEVICE) * (span + 1).float()).long() if train else span // 2
    gidx = (start.unsqueeze(1) + _AR.unsqueeze(0)).clamp(max=CACHE_LEN - 1)
    return torch.gather(BUF[idx], 1, gidx).float(), Y[idx]

def augment(x):
    """Channel perturbation: random gain + light additive noise."""
    return x * torch.empty(x.size(0), 1, device=DEVICE).uniform_(0.7, 1.3) + 0.005 * torch.randn_like(x)

# %% [markdown]
# ## 4. Sampling
# Source fakes are water-filled across corpora and generators so no single
# generator dominates; target sets are simply class-balanced.

# %%
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
# ## 5. Model

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
        if self.encoder_amp:
            feats, _ = self.ssl.extract_features(wav.float())
        else:
            with torch.amp.autocast("cuda", enabled=False):
                feats, _ = self.ssl.extract_features(wav.float())
        x = feats[-1]
        a = torch.softmax(self.attn(x).squeeze(-1), 1)
        emb = self.proj(torch.bmm(a.unsqueeze(1), x).squeeze(1))
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
# ## 6. Train / score / adapt

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

def fit(model, idx, epochs, tag=""):
    opt = torch.optim.Adam(trainable(model), lr=LR)
    for ep in range(epochs):
        model.train(); model.ssl.eval()
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
    """Unsupervised TTA. Labels are never read; only the model's own scores."""
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

def tent(model, idx, epochs=TTA_EPOCHS):
    """Vanilla TTA baseline: entropy minimisation."""
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

# %% [markdown]
# ## 7. Experiment grid
# Leave-one-corpus-out × seeds × methods. Each row is appended to `results.csv`
# immediately, so a crash or a stopped kernel never loses completed work.

# %%
def idx_of(df):
    return torch.tensor(df.index.values, dtype=torch.long, device=DEVICE)

def record(**row):
    pd.DataFrame([row]).to_csv(RESULTS_CSV, mode="a", header=not os.path.exists(RESULTS_CSV), index=False)

METHODS = {
    "source":  None,
    "tent":    lambda m, i: tent(m, i),
    "st_only": lambda m, i: adapt(m, i, use_st=True, use_cons=False),
    "ours":    lambda m, i: adapt(m, i, use_st=True, use_cons=True),
}

grid_t0 = time.time()
for seed in SEEDS:
    for target in TARGETS:
        torch.manual_seed(seed); np.random.seed(seed)
        tag = f"seed{seed}/{target}"
        log(f"=== {tag} ===")

        src_df = sample_source(pool[pool.corpus != target], SOURCE_PER_CLASS, seed)
        tgt_df = sample_target(pool[pool.corpus == target], TARGET_PER_CLASS, seed)
        src_idx, tgt_idx = idx_of(src_df), idx_of(tgt_df)
        log(f"  source {len(src_df)} clips from {sorted(src_df.corpus.unique())} | target {len(tgt_df)}")

        ckpt = f"{CKPT_DIR}/source_{target}_seed{seed}.pt"
        source = XLSRDetector(encoder_amp=ENCODER_AMP).to(DEVICE)
        if os.path.exists(ckpt):
            source.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)
            log(f"  loaded {ckpt}")
        else:
            fit(source, src_idx, SOURCE_EPOCHS, tag=tag)
            torch.save({n: p.detach().cpu() for n, p in source.named_parameters() if p.requires_grad}, ckpt)

        # transductive: adapt on the target pool and score it
        for name, fn in METHODS.items():
            try:
                t0 = time.time()
                model = source if fn is None else fn(copy.deepcopy(source), tgt_idx)
                m, y, s = metrics(model, tgt_idx)
                record(seed=seed, target=target, method=name, setting="transductive",
                       eer=round(m["eer"], 3), auc=round(m["auc"], 4), acc=round(m["acc"], 2),
                       n=len(tgt_idx), minutes=round((time.time() - t0) / 60, 1))
                log(f"  {name:8s} EER {m['eer']:6.2f}  AUC {m['auc']:.3f}  acc@0.5 {m['acc']:5.1f}  ({(time.time()-t0)/60:.1f} min)")
                if name in ("source", "ours") and seed == SEEDS[0] and target == TARGETS[0]:
                    np.savez(f"scores{SUFFIX}_{target}_{name}.npz", y=y, s=s)
                if fn is not None:
                    del model; torch.cuda.empty_cache()
            except Exception as e:
                log(f"  !! {name} failed: {type(e).__name__}: {e}")

        # inductive: adapt on half the target, evaluate on the disjoint half
        try:
            half = len(tgt_idx) // 2
            perm = torch.randperm(len(tgt_idx), device=DEVICE)
            a_idx, b_idx = tgt_idx[perm[:half]], tgt_idx[perm[half:]]
            m_src, _, _ = metrics(source, b_idx)
            model = adapt(copy.deepcopy(source), a_idx)
            m_ada, _, _ = metrics(model, b_idx)
            record(seed=seed, target=target, method="source", setting="inductive",
                   eer=round(m_src["eer"], 3), auc=round(m_src["auc"], 4), acc=round(m_src["acc"], 2), n=len(b_idx), minutes=0)
            record(seed=seed, target=target, method="ours", setting="inductive",
                   eer=round(m_ada["eer"], 3), auc=round(m_ada["auc"], 4), acc=round(m_ada["acc"], 2), n=len(b_idx), minutes=0)
            log(f"  inductive: source EER {m_src['eer']:.2f} -> ours EER {m_ada['eer']:.2f} (held-out half)")
            del model; torch.cuda.empty_cache()
        except Exception as e:
            log(f"  !! inductive failed: {type(e).__name__}: {e}")

        del source; torch.cuda.empty_cache()

log(f"GRID COMPLETE in {(time.time()-grid_t0)/60:.1f} min | peak GPU {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

# %% [markdown]
# ## 8. TTA hyperparameter sensitivity
# The main grid finishes well before morning, so we spend the remaining GPU time
# mapping how the adaptation behaves as its three knobs move. Reuses the cached
# source checkpoints — no retraining. Appended to `sweep.csv`.

# %%
SWEEP_CSV = f"sweep{SUFFIX}.csv"
SWEEP = {"q": [0.1, 0.2, 0.3, 0.4], "epochs": [2, 4, 8], "lam": [0.0, 0.3, 1.0]}
SWEEP_SEED, BASE = SEEDS[0], {"q": Q, "epochs": TTA_EPOCHS, "lam": LAMBDA_CONS}

def run_sweep():
    global Q, LAMBDA_CONS
    for target in TARGETS:
        ckpt = f"{CKPT_DIR}/source_{target}_seed{SWEEP_SEED}.pt"
        if not os.path.exists(ckpt):
            log(f"sweep: no checkpoint for {target}, skipping")
            continue
        torch.manual_seed(SWEEP_SEED); np.random.seed(SWEEP_SEED)
        tgt_idx = idx_of(sample_target(pool[pool.corpus == target], TARGET_PER_CLASS, SWEEP_SEED))
        source = XLSRDetector(encoder_amp=ENCODER_AMP).to(DEVICE)
        source.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)

        for knob, values in SWEEP.items():
            for v in values:
                cfg = dict(BASE, **{knob: v})
                if cfg == BASE and knob != "q":
                    continue
                try:
                    t0 = time.time()
                    Q, LAMBDA_CONS = cfg["q"], cfg["lam"]
                    model = adapt(copy.deepcopy(source), tgt_idx,
                                  use_cons=cfg["lam"] > 0, epochs=cfg["epochs"])
                    m, _, _ = metrics(model, tgt_idx)
                    pd.DataFrame([dict(target=target, knob=knob, **cfg,
                                       eer=round(m["eer"], 3), auc=round(m["auc"], 4))]
                                 ).to_csv(SWEEP_CSV, mode="a",
                                          header=not os.path.exists(SWEEP_CSV), index=False)
                    log(f"  sweep {target} {knob}={v} -> EER {m['eer']:.2f} AUC {m['auc']:.3f} ({(time.time()-t0)/60:.1f} min)")
                    del model; torch.cuda.empty_cache()
                except Exception as e:
                    log(f"  !! sweep {target} {knob}={v} failed: {type(e).__name__}: {e}")
        del source; torch.cuda.empty_cache()
    Q, LAMBDA_CONS = BASE["q"], BASE["lam"]

if not SMOKE:
    log("=== hyperparameter sweep ===")
    run_sweep()
    log("SWEEP COMPLETE")

# %% [markdown]
# ## 9. Summary

# %%
res = pd.read_csv(RESULTS_CSV)
summary = (res[res.setting == "transductive"]
           .groupby(["target", "method"])
           .agg(eer_mean=("eer", "mean"), eer_std=("eer", "std"),
                auc_mean=("auc", "mean"), n_seeds=("eer", "size"))
           .round(3).reset_index())
print(summary.to_string(index=False))
print()
print(res[res.setting == "inductive"].groupby(["target", "method"])
      .agg(eer_mean=("eer", "mean"), auc_mean=("auc", "mean")).round(3).to_string())

if os.path.exists(SWEEP_CSV):
    print("\n--- TTA hyperparameter sensitivity ---")
    print(pd.read_csv(SWEEP_CSV).sort_values(["target", "knob", "eer"]).to_string(index=False))

# %%
order = ["source", "tent", "st_only", "ours"]
pretty = {"source": "Source-only (no adaptation)", "tent": "Tent (entropy-min TTA)",
          "st_only": "Self-training only", "ours": "\\textbf{Ours (ST + consistency)}"}
tgts = sorted(summary.target.unique())
print("\\begin{tabular}{@{}l" + "c" * len(tgts) + "@{}}\n\\toprule")
print("Method & " + " & ".join(t.replace("_", "-") for t in tgts) + " \\\\\n\\midrule")
for mth in order:
    cells = []
    for t in tgts:
        r = summary[(summary.target == t) & (summary.method == mth)]
        cells.append(f"{r.eer_mean.iloc[0]:.2f} $\\pm$ {r.eer_std.iloc[0]:.2f}" if len(r) and pd.notna(r.eer_std.iloc[0])
                     else (f"{r.eer_mean.iloc[0]:.2f}" if len(r) else "--"))
    print(f"{pretty[mth]} & " + " & ".join(cells) + " \\\\")
print("\\bottomrule\n\\end{tabular}")

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tgt = TARGETS[0]
if os.path.exists(f"scores{SUFFIX}_{tgt}_source.npz") and os.path.exists(f"scores{SUFFIX}_{tgt}_ours.npz"):
    a, b = np.load(f"scores{SUFFIX}_{tgt}_source.npz"), np.load(f"scores{SUFFIX}_{tgt}_ours.npz")
    logit = lambda s: np.log(np.clip(s, 1e-4, 1 - 1e-4) / (1 - np.clip(s, 1e-4, 1 - 1e-4)))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True, sharey=True)
    for ax, (name, d) in zip(axes, [("Source model", a), ("After test-time adaptation", b)]):
        y, s = d["y"], d["s"]
        ax.hist(logit(s[y == 0]), bins=np.linspace(-10, 10, 45), alpha=.6, color="#2ca02c", label="real", density=True)
        ax.hist(logit(s[y == 1]), bins=np.linspace(-10, 10, 45), alpha=.6, color="#d62728", label="fake", density=True)
        ax.axvline(0, ls="--", c="k", lw=1.2)
        ax.set_title(f"{name}\nAUC={roc_auc_score(y,s):.3f}  acc@0.5={accuracy_score(y,(s>=.5))*100:.0f}%  EER={eer(y,s)*100:.1f}%", fontsize=8.5)
        ax.set_xlabel("logit P(fake)"); ax.legend(fontsize=7.5, loc="upper left")
    axes[0].set_yticks([])
    fig.tight_layout(); fig.savefig("fig_score_dist.png", dpi=150)
    log("wrote fig_score_dist.png")
