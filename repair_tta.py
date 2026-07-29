"""
RePAIR-TTA pilot: reliability-gated pairwise test-time adaptation.

Two questions, both answered against the existing st_cons method:
  A. On a balanced target pool, does pairwise adaptation match or beat st_cons?
  B. Under class-prior skew (10-90% fake), is it more robust than symmetric tails?

Per the proposal's decision rule, RePAIR-TTA is only worth pursuing if it matches
on (A) and clearly wins on (B). Labels are used for evaluation and diagnostics
only, never for adaptation.

Reuses the source checkpoint trained by overnight_pipeline.
"""

import copy, glob, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torchaudio, soundfile as sf
from sklearn.metrics import roc_curve, roc_auc_score

DEVICE = "cuda"
SR, CROP_SEC, CACHE_SEC = 16000, 3.0, 4.0
CROP_LEN, CACHE_LEN = int(SR * CROP_SEC), int(SR * CACHE_SEC)
N_FINETUNE, TTA_LR, BATCH = 4, 1e-4, 16
Q, LAMBDA_CONS, TTA_EPOCHS = 0.3, 0.3, 3
MAX_PER_CORPUS_CLASS = 2200
TARGET = "in_the_wild"
CKPT = f"ckpt/source_{TARGET}_seed0.pt"
K_VIEWS, DELTA, MARGIN, ETA_Q = 4, 0.5, 1.0, 0.5
LBL = {"real": 0, "fake": 1}

torch.backends.cuda.matmul.allow_tf32 = True
USE_BF16 = torch.cuda.is_bf16_supported()

def amp():
    return torch.amp.autocast("cuda", dtype=torch.bfloat16) if USE_BF16 else torch.amp.autocast("cuda", enabled=False)

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- data
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
    return pd.DataFrame(rows, columns=["path", "label", "corpus", "generator"])


def decode(path):
    try:
        d, sr = sf.read(path, dtype="float32", always_2d=True)
        w = d.mean(axis=1)
        if sr != SR:
            w = torchaudio.functional.resample(torch.from_numpy(w), sr, SR).numpy()
        return w
    except Exception:
        return np.zeros(1, dtype=np.float32)


def cap_per_class(df, n, seed=0):
    return pd.concat([g.sample(min(n, len(g)), random_state=seed) for _, g in df.groupby("label")])


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
        feats, _ = self.ssl.extract_features(wav.float())
        x = feats[-1]
        a = torch.softmax(self.attn(x).squeeze(-1), 1)
        emb = self.proj(torch.bmm(a.unsqueeze(1), x).squeeze(1))
        return self.cls(emb), emb


def trainable(m):
    return [p for p in m.parameters() if p.requires_grad]


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


# ---------------------------------------------------------------- helpers
def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2


def get_batch(idx):
    span = (VLEN[idx] - CROP_LEN).clamp(min=0)
    start = span // 2
    gidx = (start.unsqueeze(1) + _AR.unsqueeze(0)).clamp(max=CACHE_LEN - 1)
    return torch.gather(BUF[idx], 1, gidx).float(), Y[idx]


def augment(x):
    return x * torch.empty(x.size(0), 1, device=DEVICE).uniform_(0.7, 1.3) + 0.005 * torch.randn_like(x)


@torch.no_grad()
def score(model, idx):
    model.eval()
    out = []
    for i in range(0, len(idx), BATCH):
        x, _ = get_batch(idx[i:i + BATCH])
        with amp():
            out.append(torch.softmax(model(x)[0].float(), 1)[:, 1])
    return Y[idx].cpu().numpy(), torch.cat(out).cpu().numpy()


def evaluate(model, idx):
    y, s = score(model, idx)
    return eer(y, s) * 100, roc_auc_score(y, s)


# ---------------------------------------------------------------- methods
def st_cons(model, idx, epochs=TTA_EPOCHS, q=Q):
    """Current method: symmetric confident tails + channel consistency."""
    set_tta_params(model)
    opt = torch.optim.Adam(trainable(model), lr=TTA_LR)
    for _ in range(epochs):
        _, s = score(model, idx)
        lo, hi = np.quantile(s, q), np.quantile(s, 1 - q)
        pl = torch.full((len(idx),), -1, dtype=torch.long, device=DEVICE)
        pl[torch.from_numpy(s <= lo).to(DEVICE)] = 0
        pl[torch.from_numpy(s >= hi).to(DEVICE)] = 1
        model.train(); model.ssl.eval()
        order = torch.randperm(len(idx), device=DEVICE)
        for i in range(0, len(order), BATCH):
            sel = order[i:i + BATCH]
            x, _ = get_batch(idx[sel])
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
    return model


@torch.no_grad()
def score_views(model, idx, K=K_VIEWS):
    """Mean and variance of the fake-vs-real logit margin across K channel views."""
    model.eval()
    zs = []
    for k in range(K):
        out = []
        for i in range(0, len(idx), BATCH):
            x, _ = get_batch(idx[i:i + BATCH])
            with amp():
                lg = model(x if k == 0 else augment(x))[0].float()
            out.append(lg[:, 1] - lg[:, 0])
        zs.append(torch.cat(out))
    Z = torch.stack(zs)
    return Z.mean(0), Z.var(0, unbiased=False)


def reliability(zbar, v):
    n = len(zbar)
    r = torch.argsort(torch.argsort(zbar)).float() / max(n - 1, 1)
    w = (2 * r - 1).abs() * torch.exp(-v / v.median().clamp(min=1e-6))
    return r, w


def repair(model, idx, epochs=TTA_EPOCHS, use_cons=True, delta=DELTA, eta_q=ETA_Q, margin=MARGIN):
    """RePAIR-TTA: reliability-gated pairwise ranking adaptation."""
    set_tta_params(model)
    opt = torch.optim.Adam(trainable(model), lr=TTA_LR)
    for _ in range(epochs):
        zbar, v = score_views(model, idx)
        r, w = reliability(zbar, v)
        eta = torch.quantile(w, eta_q) ** 2
        model.train(); model.ssl.eval()
        order = torch.randperm(len(idx), device=DEVICE)
        for i in range(0, len(order), BATCH):
            sel = order[i:i + BATCH]
            x, _ = get_batch(idx[sel])
            rb, wb = r[sel], w[sel]
            opt.zero_grad(set_to_none=True)
            with amp():
                logits, _ = model(x)
                z = logits[:, 1] - logits[:, 0]
                dr = rb.unsqueeze(0) - rb.unsqueeze(1)          # [i,j] = r_j - r_i
                ww = wb.unsqueeze(1) * wb.unsqueeze(0)
                mask = (dr > delta) & (ww > eta)
                loss = torch.zeros((), device=DEVICE)
                if mask.any():
                    dz = z.unsqueeze(0) - z.unsqueeze(1)        # [i,j] = z_j - z_i
                    pair = F.softplus(margin - dz)
                    loss = loss + (ww * mask * pair).sum() / (ww * mask).sum().clamp(min=1e-8)
                if use_cons:
                    p = torch.softmax(logits, 1)
                    loss = loss + LAMBDA_CONS * F.mse_loss(torch.softmax(model(augment(x))[0], 1), p.detach())
            if loss.requires_grad:
                loss.backward(); opt.step()
    return model


# ------------------------------------------------- diagnostics (labels: analysis only)
def diagnostics(model, idx):
    zbar, v = score_views(model, idx)
    r, w = reliability(zbar, v)
    y = Y[idx]
    implied = (r > 0.5).long()                       # what the rank would pseudo-label
    acc_by_decile = []
    for d in range(10):
        m = (w >= torch.quantile(w, d / 10)) & (w <= torch.quantile(w, (d + 1) / 10))
        acc_by_decile.append(float((implied[m] == y[m]).float().mean()) if m.any() else float("nan"))

    eta = torch.quantile(w, ETA_Q) ** 2
    dr = r.unsqueeze(0) - r.unsqueeze(1)
    ww = w.unsqueeze(1) * w.unsqueeze(0)
    mask = (dr > DELTA) & (ww > eta)
    yi = y.unsqueeze(1).expand_as(mask)              # [i,j] -> y_i
    yj = y.unsqueeze(0).expand_as(mask)              # [i,j] -> y_j
    sel = mask & (yi != yj)                          # pairs spanning both classes
    correct = (mask & (yi == 0) & (yj == 1)).sum()
    purity_span = float(correct) / max(int(sel.sum()), 1)
    purity_all = float(correct) / max(int(mask.sum()), 1)
    same = float((mask & (yi == yj)).sum()) / max(int(mask.sum()), 1)
    return acc_by_decile, purity_span, purity_all, same, int(mask.sum())


# ---------------------------------------------------------------- run
if __name__ == "__main__":
    manifest = build_manifest()
    pool = pd.concat([cap_per_class(g, MAX_PER_CORPUS_CLASS)
                      for _, g in manifest.groupby("corpus")]).reset_index(drop=True)
    log(f"pool {len(pool)} clips")
    BUF, VLEN, Y = build_cache(pool)
    _AR = torch.arange(CROP_LEN, device=DEVICE)
    log(f"cache {tuple(BUF.shape)} = {BUF.numel()*2/1e9:.2f} GB")

    source = XLSRDetector().to(DEVICE)
    source.load_state_dict(torch.load(CKPT, map_location=DEVICE), strict=False)
    log(f"loaded {CKPT}")

    tgt = pool[pool.corpus == TARGET]
    rng = np.random.RandomState(0)
    rows = []

    def make_pool(prev, n_total=1400, seed=0):
        n_fake = int(round(prev * n_total))
        f = tgt[tgt.label == "fake"].sample(min(n_fake, (tgt.label == "fake").sum()), random_state=seed)
        r = tgt[tgt.label == "real"].sample(min(n_total - n_fake, (tgt.label == "real").sum()), random_state=seed)
        return torch.tensor(pd.concat([f, r]).index.values, dtype=torch.long, device=DEVICE)

    log("=== A. balanced target (prevalence 0.5) ===")
    idx = make_pool(0.5)
    e, a = evaluate(source, idx); log(f"  source   EER {e:5.2f}  AUC {a:.3f}"); rows.append(("0.5", "source", e, a))
    for name, fn in [("st_cons", st_cons), ("repair", repair), ("repair_nocons", lambda m, i: repair(m, i, use_cons=False))]:
        t0 = time.time()
        e, a = evaluate(fn(copy.deepcopy(source), idx), idx)
        log(f"  {name:14s} EER {e:5.2f}  AUC {a:.3f}   ({(time.time()-t0)/60:.1f} min)")
        rows.append(("0.5", name, e, a))
        torch.cuda.empty_cache()

    dec, ps, pa, same, npairs = diagnostics(source, idx)
    log(f"  diagnostics: pair-purity(span-class) {ps:.3f} | purity(all pairs) {pa:.3f} | same-class pairs {same:.3f} | n={npairs}")
    log(f"  pseudo-label acc by reliability decile: {[round(x,3) for x in dec]}")

    log("=== B. class-prior stress test ===")
    for prev in [0.1, 0.3, 0.7, 0.9]:
        idx = make_pool(prev)
        e, a = evaluate(source, idx); log(f"  prev={prev}  source   EER {e:5.2f}  AUC {a:.3f}"); rows.append((str(prev), "source", e, a))
        for name, fn in [("st_cons", st_cons), ("repair", repair)]:
            e, a = evaluate(fn(copy.deepcopy(source), idx), idx)
            log(f"  prev={prev}  {name:8s} EER {e:5.2f}  AUC {a:.3f}")
            rows.append((str(prev), name, e, a))
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows, columns=["prevalence", "method", "eer", "auc"])
    df.to_csv("repair_results.csv", index=False)
    print("\n" + df.pivot(index="prevalence", columns="method", values="eer").round(2).to_string())
    log("saved repair_results.csv")
