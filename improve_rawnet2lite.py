"""
Properly train RawNet2Lite (the non-SSL baseline) to resolve the ambiguity in
Limitations item 5: was its collapse in the main paper evidence that SSL
pretraining is necessary, or just an artefact of it being undertrained
(8 epochs, no LR schedule)?

Same leave-one-corpus-out protocol, same manifest/data as extended_pipeline,
but: more epochs (40 vs 8), cosine LR decay, and run on the local RTX 3080
(unlimited time) instead of burning cloud H200 budget on a small model.
"""

import glob, os, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torchaudio, soundfile as sf
from sklearn.metrics import roc_curve, roc_auc_score

DEVICE = "cuda"
SR, CROP_SEC, CACHE_SEC = 16000, 3.0, 4.0
CROP_LEN, CACHE_LEN = int(SR * CROP_SEC), int(SR * CACHE_SEC)
BATCH = 32
SOURCE_PER_CLASS, TARGET_PER_CLASS, MAX_PER_CORPUS_CLASS = 5000, 3000, 6000
EPOCHS = 40                 # vs 8 in the original cloud ablation
SEEDS = [0, 1, 2]
EER_TARGETS = ["asvspoof2019", "dataset2", "in_the_wild", "arabic"]
RESULTS_CSV, LOG_FILE = "results_rawnet2lite_v2.csv", "run_log_rawnet2lite_v2.txt"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def record(**row):
    pd.DataFrame([row]).to_csv(RESULTS_CSV, mode="a", header=not os.path.exists(RESULTS_CSV), index=False)

LBL = {"real": 0, "fake": 1}

def build_manifest():
    rows = []
    proto = "data/asvspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    flac = "data/asvspoof2019_LA/ASVspoof2019_LA_train/flac"
    for line in open(proto):
        p = line.split()
        if len(p) >= 5:
            lab = "fake" if p[-1] == "spoof" else "real"
            rows.append((f"{flac}/{p[1]}.flac", lab, "asvspoof2019", p[-2] if lab == "fake" else "bonafide"))
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
    for meta in glob.glob("data/dataset_3/Mlaad_v5/mlaad_v5/**/meta.csv", recursive=True):
        model_dir = os.path.dirname(meta)
        gen = os.path.basename(model_dir)
        for w in glob.glob(f"{model_dir}/*.wav"):
            rows.append((w, "fake", "mlaad", gen))
    return pd.DataFrame(rows, columns=["path", "label", "corpus", "generator"])

manifest = build_manifest()
log("manifest:\n" + manifest.groupby(["corpus", "label"]).size().to_string())

def cap_per_class(df, n, seed=0):
    return pd.concat([g.sample(min(n, len(g)), random_state=seed) for _, g in df.groupby("label")])

mlaad_pool = manifest[manifest.corpus == "mlaad"].sample(
    min(MAX_PER_CORPUS_CLASS, (manifest.corpus == "mlaad").sum()), random_state=0)
other = manifest[manifest.corpus != "mlaad"]
pool = pd.concat([cap_per_class(g, MAX_PER_CORPUS_CLASS) for _, g in other.groupby("corpus")]
                 + [mlaad_pool]).reset_index(drop=True)
# re-derive AFTER reset_index -- the pre-reset frame's indices are stale once
# concatenated into `pool` and misalign every idx_of() lookup that uses them
# (this exact bug wasted cloud GPU time once before; caught here in smoke test).
mlaad_pool = pool[pool.corpus == "mlaad"]
log(f"cache pool: {len(pool)} clips")

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
    buf, vlen = np.zeros((n, CACHE_LEN), dtype=np.float16), np.ones(n, dtype=np.int64)
    with ThreadPoolExecutor(max_workers=16) as ex:
        for i, w in enumerate(ex.map(decode, df.path.tolist())):
            L = min(len(w), CACHE_LEN)
            buf[i, :L] = w[:L]
            vlen[i] = max(L, 1)
    return (torch.from_numpy(buf).to(DEVICE), torch.from_numpy(vlen).to(DEVICE),
            torch.tensor(df.label.map(LBL).values, dtype=torch.long, device=DEVICE))

t0 = time.time()
BUF, VLEN, Y = build_cache(pool)
log(f"cache ready: {tuple(BUF.shape)} = {BUF.numel()*2/1e9:.2f} GB ({time.time()-t0:.0f}s)")
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
    real = df[df.label == "real"].sample(min(n_per_class, (df.label == "real").sum()), random_state=seed)
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

class RawNet2Lite(nn.Module):
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
        self.dropout = nn.Dropout(0.3)
        self.cls = nn.Linear(256, 2)

    def forward(self, wav):
        x = self.stem(wav.unsqueeze(1))
        for blk, sc in zip(self.blocks, self.short):
            x = F.leaky_relu(blk(x) + sc(x), 0.2)
        x = x.transpose(1, 2)
        out, _ = self.gru(x)
        emb = self.dropout(out.mean(1))
        return self.cls(emb), emb

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
        out.append(torch.softmax(model(x)[0], 1)[:, 1])
    return Y[idx].cpu().numpy(), torch.cat(out).cpu().numpy()

def fit(model, idx, epochs, tag=""):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        model.train()
        perm = idx[torch.randperm(len(idx), device=DEVICE)]
        tot = cnt = 0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            x, y = get_batch(b, train=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(augment(x))[0], y)
            loss.backward(); opt.step()
            tot += loss.item() * len(b); cnt += len(b)
        sched.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            log(f"  {tag} epoch {ep+1}/{epochs} loss {tot/cnt:.4f} lr {sched.get_last_lr()[0]:.2e}")
    return model

grid_t0 = time.time()
for seed in SEEDS:
    for target in EER_TARGETS:
        torch.manual_seed(seed); np.random.seed(seed)
        tag = f"seed{seed}/{target}"
        log(f"=== {tag} ===")
        src_pool = pool[(pool.corpus != target) & (pool.corpus != "mlaad")]
        src_pool = pd.concat([src_pool, mlaad_pool])
        src_idx = idx_of(sample_source(src_pool, SOURCE_PER_CLASS, seed))
        tgt_idx = idx_of(sample_target(pool[pool.corpus == target], TARGET_PER_CLASS, seed))

        model = RawNet2Lite().to(DEVICE)
        t0 = time.time()
        fit(model, src_idx, EPOCHS, tag=tag)
        y, s = score(model, tgt_idx)
        e, a = eer(y, s) * 100, roc_auc_score(y, s)
        record(seed=seed, target=target, eer=round(e, 3), auc=round(a, 4),
              minutes=round((time.time() - t0) / 60, 1))
        log(f"  RawNet2Lite-v2 EER {e:.2f}  AUC {a:.3f}  ({(time.time()-t0)/60:.1f} min)")
        del model; torch.cuda.empty_cache()

log(f"DONE in {(time.time()-grid_t0)/60:.1f} min")
res = pd.read_csv(RESULTS_CSV)
print(res.groupby("target").agg(eer_mean=("eer", "mean"), eer_std=("eer", "std"),
                                auc_mean=("auc", "mean")).round(3).to_string())
