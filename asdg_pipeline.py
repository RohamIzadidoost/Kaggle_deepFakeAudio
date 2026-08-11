"""ASDG baseline (Xie et al., IEEE TIFS 2024 style aggregation-separation DG),
run through the SAME leave-one-corpus-out grid, seeds, and hyperparameters as
the paper's DANN baseline (results_dann.csv) and the main TTA grid
(results_ext.csv), for a direct three-way comparison.

Why this exists: `main.tex`'s Related Work names ASDG as the canonical prior
domain-generalization method for this task and states our method's asymmetric
real-anchoring is "differentiated from ASDG and shown to beat it" -- but that
head-to-head never actually ran (Task #8 in the project tracker). This is it.

Loss (ASDGLoss, losses.py): CE + lambda_asdg * (aggregate real cross-domain +
separate fake cross-domain + real-vs-fake margin). Structurally identical to
this project's own RealAnchoredContrastiveLoss except for exactly one term:
ASDG additionally repels fake embeddings from DIFFERENT source domains apart
from each other, which the anchor loss deliberately omits (fakes are
generator-specific; forcing them apart -- or together -- is not obviously
correct, whereas leaving them unconstrained matches the paper's stated
inductive bias). losses.py's self-test proves the two losses are byte-identical
on their shared (real-real) term and disagree only on fake geometry.

lambda_asdg=0.3, margin=1.0 match trainer.py's defaults for lambda_anchor
--the project's own prior DA-RAD config -- so the two DG losses are compared
at matched weight, not an arbitrarily-tuned-in-ASDG's-favor or -disfavor value.

"Domain" for the aggregation/separation terms = source CORPUS of origin
(asvspoof2019/dataset2/in_the_wild/arabic/mlaad, whichever three + mlaad are in
the source pool for a given held-out target) -- matching how the original
DA-RAD ablation and RealAnchoredContrastiveLoss both defined domains.

Runs locally (per-fold GPU-resident cache, ~2GB, small enough that the 33GB
cloud MIG slice extended_pipeline.py was built for is not needed here).
Set SMOKE=True first.
"""

import copy
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import roc_auc_score

from losses import ASDGLoss

SMOKE = False   # <-- run this first; only then set False for the real run

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SR = 16000
CROP_SEC, CACHE_SEC = 3.0, 4.0
CROP_LEN, CACHE_LEN = int(SR * CROP_SEC), int(SR * CACHE_SEC)

N_FINETUNE, BATCH = 4, 32
LR = 2e-4                      # matches the main grid's source-training LR
LAMBDA_ASDG, ASDG_MARGIN = 0.3, 1.0   # == trainer.py's lambda_anchor/anchor_margin defaults
EER_TARGETS = ["asvspoof2019", "dataset2", "in_the_wild", "arabic"]
DECODE_WORKERS = 16

if SMOKE:
    SEEDS = [0]
    TARGETS = ["in_the_wild"]
    SOURCE_EPOCHS = 2
    SOURCE_PER_CLASS, TARGET_PER_CLASS, MAX_PER_CORPUS_CLASS = 300, 300, 600
else:
    # ASDG_SEEDS override: P1.2 takes this baseline from 3 to 5 seeds. Default
    # unchanged; the (seed, target) resume guard still skips completed folds.
    SEEDS = [int(s) for s in os.environ["ASDG_SEEDS"].split(",")] \
        if os.environ.get("ASDG_SEEDS") else [0, 1, 2]
    TARGETS = EER_TARGETS
    SOURCE_EPOCHS = 8           # matches the main grid + DANN baseline
    SOURCE_PER_CLASS, TARGET_PER_CLASS, MAX_PER_CORPUS_CLASS = 5000, 3000, 6000

SUFFIX = "_smoke" if SMOKE else ""
RESULTS_CSV = f"results_asdg{SUFFIX}.csv"
LOG_FILE = f"run_log_asdg{SUFFIX}.txt"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def record(**row):
    pd.DataFrame([row]).to_csv(RESULTS_CSV, mode="a",
                              header=not os.path.exists(RESULTS_CSV), index=False)


def amp():
    return torch.amp.autocast("cuda", dtype=torch.bfloat16 if USE_BF16 else torch.float16,
                              enabled=torch.cuda.is_available())


# ---------------------------------------------------------------- manifest
# Local MLAAD path -- extended_pipeline.py/analysis_pipeline.py use
# "data/mlaad/**" which is the CLOUD layout and does not exist on this
# machine; the verified local path (already used successfully this session by
# improve_rawnet2lite.py / lambda_selector.py) is data/dataset_3/Mlaad_v5/...
LBL = {"real": 0, "fake": 1}


def build_manifest():
    rows = []
    proto = "data/asvspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    flac = "data/asvspoof2019_LA/ASVspoof2019_LA_train/flac"
    for line in open(proto):
        p = line.split()
        if len(p) >= 5:
            lab = "fake" if p[-1] == "spoof" else "real"
            rows.append((f"{flac}/{p[1]}.flac", lab, "asvspoof2019"))
    for d in sorted(glob.glob("data/dataset_2/*/")):
        gen = os.path.basename(d.rstrip("/"))
        lab = "real" if gen == "real_samples" else "fake"
        for w in glob.glob(f"{d}/**/*.wav", recursive=True):
            rows.append((w, lab, "dataset2"))
    for corpus, root in [("in_the_wild", "data/in_the_wild"), ("arabic", "data/arabic_arad")]:
        for w in glob.glob(f"{root}/**/*.wav", recursive=True):
            parts = w.split(os.sep)
            lab = "real" if "real" in parts else "fake" if "fake" in parts else None
            if lab:
                rows.append((w, lab, corpus))
    for meta in glob.glob("data/dataset_3/Mlaad_v5/mlaad_v5/**/meta.csv", recursive=True):
        model_dir = os.path.dirname(meta)
        for w in glob.glob(f"{model_dir}/*.wav"):
            rows.append((w, "fake", "mlaad"))
    return pd.DataFrame(rows, columns=["path", "label", "corpus"])


manifest = build_manifest()
log("manifest:\n" + manifest.groupby(["corpus", "label"]).size().to_string())

DOMAIN_ID = {c: i for i, c in enumerate(sorted(manifest.corpus.unique()))}
log(f"domain ids: {DOMAIN_ID}")


def cap_per_class(df, n, seed=0):
    return pd.concat([g.sample(min(n, len(g)), random_state=seed) for _, g in df.groupby("label")])


mlaad_pool_full = manifest[manifest.corpus == "mlaad"]
mlaad_pool_full = mlaad_pool_full.sample(min(MAX_PER_CORPUS_CLASS, len(mlaad_pool_full)), random_state=0)
other = manifest[manifest.corpus != "mlaad"]


def sample_source(target, n_per_class, seed):
    """Source pool for a held-out target: the other 3 EER corpora + MLAAD,
    same composition as the main grid and DANN baseline."""
    src = pd.concat([cap_per_class(g, MAX_PER_CORPUS_CLASS, seed)
                     for c, g in other.groupby("corpus") if c != target] + [mlaad_pool_full])
    real = src[src.label == "real"].sample(min(n_per_class, (src.label == "real").sum()), random_state=seed)
    fake = src[src.label == "fake"].sample(min(n_per_class, (src.label == "fake").sum()), random_state=seed)
    return pd.concat([real, fake]).sample(frac=1, random_state=seed).reset_index(drop=True)


def sample_target(target, n_per_class, seed):
    df = manifest[manifest.corpus == target]
    return pd.concat([g.sample(min(n_per_class, len(g)), random_state=seed)
                      for _, g in df.groupby("label")]).sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------- audio / cache
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
    with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as ex:
        for i, w in enumerate(ex.map(decode, df.path.tolist())):
            L = min(len(w), CACHE_LEN)
            buf[i, :L] = w[:L]
            vlen[i] = max(L, 1)
    y = torch.tensor(df.label.map(LBL).values, dtype=torch.long, device=DEVICE)
    dom = torch.tensor(df.corpus.map(DOMAIN_ID).values, dtype=torch.long, device=DEVICE)
    return torch.from_numpy(buf).to(DEVICE), torch.from_numpy(vlen).to(DEVICE), y, dom


_AR = torch.arange(CROP_LEN, device=DEVICE)


def get_batch(buf, vlen, y, dom, idx, train):
    span = (vlen[idx] - CROP_LEN).clamp(min=0)
    start = (torch.rand(len(idx), device=DEVICE) * (span + 1).float()).long() if train else span // 2
    gidx = (start.unsqueeze(1) + _AR.unsqueeze(0)).clamp(max=CACHE_LEN - 1)
    x = torch.gather(buf[idx], 1, gidx).float()
    return x, y[idx], dom[idx]


def augment(x):
    return x * torch.empty(x.size(0), 1, device=DEVICE).uniform_(0.7, 1.3) \
        + 0.005 * torch.randn_like(x)


# ---------------------------------------------------------------- model
class XLSRDetector(nn.Module):
    def __init__(self, n_finetune=N_FINETUNE):
        super().__init__()
        self.ssl = torchaudio.pipelines.WAV2VEC2_XLSR_300M.get_model()
        for p in self.ssl.parameters():
            p.requires_grad_(False)
        for p in self.ssl.model.encoder.transformer.layers[-n_finetune:].parameters():
            p.requires_grad_(True)
        self.ssl.eval()
        self.attn = nn.Linear(1024, 1)
        self.proj = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3))
        self.cls = nn.Linear(256, 2)

    def forward(self, wav):
        with amp():
            feats, _ = self.ssl.extract_features(wav.float())
            x = feats[-1]
            a = torch.softmax(self.attn(x).squeeze(-1), 1)
            emb = self.proj(torch.bmm(a.unsqueeze(1), x).squeeze(1))
            return self.cls(emb), emb


def trainable(m):
    return [p for p in m.parameters() if p.requires_grad]


def eer_of(y, s):
    from metrics import compute_eer
    e, _ = compute_eer(np.asarray(y), np.asarray(s))
    return e * 100


@torch.no_grad()
def score(model, buf, vlen, y, dom, n):
    model.eval()
    idx = torch.arange(n, device=DEVICE)
    out = []
    for i in range(0, n, BATCH):
        x, _, _ = get_batch(buf, vlen, y, dom, idx[i:i + BATCH], train=False)
        out.append(torch.softmax(model(x)[0].float(), 1)[:, 1])
    return y.cpu().numpy(), torch.cat(out).cpu().numpy()


# ---------------------------------------------------------------- ASDG training
def train_asdg(target, seed, epochs=SOURCE_EPOCHS):
    torch.manual_seed(seed)
    np.random.seed(seed)

    src_df = sample_source(target, SOURCE_PER_CLASS, seed)
    tgt_df = sample_target(target, TARGET_PER_CLASS, seed)
    log(f"  seed{seed}/{target}: source {len(src_df)} clips "
        f"({src_df.corpus.value_counts().to_dict()}), target {len(tgt_df)} clips")

    buf, vlen, y, dom = build_cache(src_df)
    tbuf, tvlen, ty, tdom = build_cache(tgt_df)

    model = XLSRDetector().to(DEVICE)
    asdg_loss = ASDGLoss(margin=ASDG_MARGIN, sep_margin=ASDG_MARGIN)
    opt = torch.optim.Adam(trainable(model), lr=LR)

    n = len(src_df)
    for ep in range(epochs):
        model.train()
        model.ssl.eval()
        perm = torch.randperm(n, device=DEVICE)
        tot_ce = tot_asdg = cnt = 0
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            x, yb, domb = get_batch(buf, vlen, y, dom, b, train=True)
            opt.zero_grad(set_to_none=True)
            logits, emb = model(augment(x))
            logits, emb = logits.float(), emb.float()
            loss_ce = F.cross_entropy(logits, yb)
            loss_asdg = asdg_loss(emb, yb, domb)
            loss = loss_ce + LAMBDA_ASDG * loss_asdg
            loss.backward()
            opt.step()
            tot_ce += loss_ce.item() * len(b)
            tot_asdg += loss_asdg.item() * len(b)
            cnt += len(b)
        log(f"  asdg seed{seed}/{target} epoch {ep + 1}/{epochs} "
            f"ce {tot_ce / cnt:.4f}  asdg {tot_asdg / cnt:.4f}")

    del buf, vlen, y, dom
    torch.cuda.empty_cache()
    return model, tbuf, tvlen, ty, tdom


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    log(f"=== ASDG baseline | SMOKE={SMOKE} | targets={TARGETS} | seeds={SEEDS} ===")

    done = set()
    if os.path.exists(RESULTS_CSV):
        prev = pd.read_csv(RESULTS_CSV)
        done = set(zip(prev.seed, prev.target))
    log(f"already recorded (seed, target): {done or '(none)'}")

    for seed in SEEDS:
        for target in TARGETS:
            if (seed, target) in done:
                log(f"seed{seed}/{target}: already recorded, skipping")
                continue
            tf = time.time()
            model, tbuf, tvlen, ty, tdom = train_asdg(target, seed)
            y, s = score(model, tbuf, tvlen, ty, tdom, len(ty))
            e = eer_of(y, s)
            auc = roc_auc_score(y, s)
            acc = float(((s >= 0.5).astype(int) == y).mean()) * 100
            minutes = (time.time() - tf) / 60
            record(seed=seed, target=target, method="asdg", eer=round(e, 3),
                   auc=round(auc, 4), acc=round(acc, 2), minutes=round(minutes, 1),
                   smoke=SMOKE)
            log(f"  ASDG {target} seed{seed}: EER {e:.2f}  AUC {auc:.3f}  "
                f"acc {acc:.1f}  ({minutes:.1f} min)")
            del model, tbuf, tvlen, ty, tdom
            torch.cuda.empty_cache()

    log(f"DONE in {(time.time() - t0) / 60:.1f} min | "
        f"peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
