"""
Embedding geometry + domain-invariance probe, using the exact checkpoints and
target pools from the cloud leave-one-corpus-out grid (loaded from cloud_bundle/).

Two pieces of new evidence for the paper:
  1. t-SNE of target embeddings, source vs. adapted, colored by (a) real/fake and
     (b) source-corpus-mixture-of-origin proxy (here: real vs fake per corpus,
     since the target pool is single-corpus by construction -- so we instead
     compare SOURCE-POOL vs TARGET-POOL embeddings to visualize the domain gap
     closing).
  2. Domain-invariance probe: a linear classifier trained to predict
     source-vs-target origin from frozen embeddings. If adaptation makes the
     representation domain-invariant, probe accuracy should drop toward chance.
"""

import glob, os
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torchaudio, soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR, CROP_SEC = 16000, 3.0
CROP_LEN = int(SR * CROP_SEC)
N_FINETUNE = 4
TARGET = "in_the_wild"
CKPT = f"cloud_bundle/ckpt/source_{TARGET}_seed0.pt"
N_PER_POOL = 500          # subsample for a fast, readable t-SNE

torch.backends.cuda.matmul.allow_tf32 = True
USE_BF16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False


def amp():
    return torch.amp.autocast("cuda", dtype=torch.bfloat16) if USE_BF16 else torch.amp.autocast("cuda", enabled=False)


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


def load_audio(path):
    try:
        d, sr = sf.read(path, dtype="float32", always_2d=True)
        w = d.mean(axis=1)
        if sr != SR:
            w = torchaudio.functional.resample(torch.from_numpy(w), sr, SR).numpy()
        n = len(w)
        if n < CROP_LEN:
            w = np.pad(w, (0, CROP_LEN - n))
        else:
            s = (n - CROP_LEN) // 2
            w = w[s:s + CROP_LEN]
        return w.astype(np.float32)
    except Exception:
        return np.zeros(CROP_LEN, dtype=np.float32)


@torch.no_grad()
def embed(model, paths, bs=16):
    model.eval()
    out = []
    for i in range(0, len(paths), bs):
        batch = paths[i:i + bs]
        with ThreadPoolExecutor(max_workers=8) as ex:
            wavs = list(ex.map(load_audio, batch))
        x = torch.tensor(np.stack(wavs), device=DEVICE)
        with amp():
            _, e = model(x)
        out.append(F.normalize(e.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def augment(x):
    return x * torch.empty(x.size(0), 1, device=DEVICE).uniform_(0.7, 1.3) + 0.005 * torch.randn_like(x)


def adapt(model, paths, q=0.3, lam=0.3, epochs=4, lr=1e-4, bs=16):
    set_tta_params(model)
    opt = torch.optim.Adam(trainable(model), lr=lr)
    for _ in range(epochs):
        s = []
        with torch.no_grad():
            for i in range(0, len(paths), bs):
                batch = paths[i:i + bs]
                with ThreadPoolExecutor(max_workers=8) as ex:
                    wavs = list(ex.map(load_audio, batch))
                x = torch.tensor(np.stack(wavs), device=DEVICE)
                with amp():
                    logits, _ = model(x)
                s.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
        s = np.concatenate(s)
        lo, hi = np.quantile(s, q), np.quantile(s, 1 - q)
        pl = np.full(len(paths), -1)
        pl[s <= lo] = 0
        pl[s >= hi] = 1
        pl_t = torch.tensor(pl, device=DEVICE)

        model.train(); model.ssl.eval()
        order = np.random.permutation(len(paths))
        for i in range(0, len(order), bs):
            sel = order[i:i + bs]
            with ThreadPoolExecutor(max_workers=8) as ex:
                wavs = list(ex.map(load_audio, [paths[j] for j in sel]))
            x = torch.tensor(np.stack(wavs), device=DEVICE)
            bpl = pl_t[sel]
            opt.zero_grad(set_to_none=True)
            with amp():
                logits, _ = model(x)
                p = torch.softmax(logits, 1)
                loss = torch.zeros((), device=DEVICE)
                conf = bpl >= 0
                if conf.any():
                    loss = loss + F.cross_entropy(logits[conf], bpl[conf].long())
                loss = loss + lam * F.mse_loss(torch.softmax(model(augment(x))[0], 1), p.detach())
            if loss.requires_grad:
                loss.backward(); opt.step()
    return model


def domain_probe(emb_a, emb_b, n_splits=5):
    X = np.concatenate([emb_a, emb_b])
    y = np.concatenate([np.zeros(len(emb_a)), np.ones(len(emb_b))])
    clf = LogisticRegression(max_iter=2000)
    scores = cross_val_score(clf, X, y, cv=n_splits)
    return scores.mean(), scores.std()


if __name__ == "__main__":
    np.random.seed(0)
    src_pool = pd.read_csv(f"cloud_bundle/source_pool_{TARGET}_seed0.csv").sample(N_PER_POOL, random_state=0)
    tgt_pool = pd.read_csv(f"cloud_bundle/target_pool_{TARGET}_seed0.csv").sample(N_PER_POOL, random_state=0)
    print(f"source pool: {len(src_pool)} clips from {sorted(src_pool.corpus.unique())}")
    print(f"target pool: {len(tgt_pool)} clips from {sorted(tgt_pool.corpus.unique())}")

    model = XLSRDetector().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE), strict=False)

    print("embedding source pool (pre-adaptation model)...")
    e_src = embed(model, src_pool.path.tolist())
    print("embedding target pool (pre-adaptation)...")
    e_tgt_before = embed(model, tgt_pool.path.tolist())

    acc_before, std_before = domain_probe(e_src, e_tgt_before)
    print(f"domain probe BEFORE adaptation: {acc_before*100:.1f}% +- {std_before*100:.1f}%  (chance=50%)")

    print("adapting on target pool (unsupervised TTA)...")
    model = adapt(model, tgt_pool.path.tolist())

    print("re-embedding source pool (adapted model)...")
    e_src2 = embed(model, src_pool.path.tolist())
    print("re-embedding target pool (adapted model)...")
    e_tgt_after = embed(model, tgt_pool.path.tolist())

    acc_after, std_after = domain_probe(e_src2, e_tgt_after)
    print(f"domain probe AFTER adaptation:  {acc_after*100:.1f}% +- {std_after*100:.1f}%  (chance=50%)")

    y_src = src_pool.label.map({"real": 0, "fake": 1}).values
    y_tgt = tgt_pool.label.map({"real": 0, "fake": 1}).values

    print("running t-SNE (before)...")
    X_before = np.concatenate([e_src, e_tgt_before])
    Z_before = TSNE(n_components=2, init="pca", random_state=0, perplexity=30).fit_transform(X_before)
    print("running t-SNE (after)...")
    X_after = np.concatenate([e_src2, e_tgt_after])
    Z_after = TSNE(n_components=2, init="pca", random_state=0, perplexity=30).fit_transform(X_after)

    n = len(src_pool)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, Z, title, acc in [(axes[0], Z_before, "Before adaptation", acc_before),
                              (axes[1], Z_after, "After adaptation", acc_after)]:
        ax.scatter(Z[:n, 0], Z[:n, 1], c=["#2ca02c" if v == 0 else "#d62728" for v in y_src],
                  marker="o", s=10, alpha=0.5, label="source (o)")
        ax.scatter(Z[n:, 0], Z[n:, 1], c=["#2ca02c" if v == 0 else "#d62728" for v in y_tgt],
                  marker="^", s=14, alpha=0.7, label="target ($\\triangle$)")
        ax.set_title(f"{title}\ndomain-probe acc={acc*100:.0f}%")
        ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", label="source", markersize=7),
              plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", label="target", markersize=8),
              plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#2ca02c", label="real", markersize=8),
              plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#d62728", label="fake", markersize=8)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig("fig_embedding_geometry.png", dpi=150)
    print("wrote fig_embedding_geometry.png")

    with open("embedding_results.txt", "w") as f:
        f.write(f"domain_probe_before: {acc_before:.4f} +- {std_before:.4f}\n")
        f.write(f"domain_probe_after: {acc_after:.4f} +- {std_after:.4f}\n")
    print("wrote embedding_results.txt")
