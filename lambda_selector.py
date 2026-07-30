"""
Label-free lambda-selection heuristic: a practitioner deploying TTA has no
target labels, so cannot compute AUC to decide whether the consistency term
will help or hurt (Limitations item 1). This tests whether a statistic
computable from RAW UNLABELED TARGET SCORES ALONE predicts it instead.

Proxy: confident-tail separation on the source model's own scores over the
unlabeled target pool -- gap = mean(score | score >= Q0.7) - mean(score | score <= Q0.3).
Large gap => the model already separates target clips into two confident
groups (good ranking, consistency should help); small gap => scores are smeared
in the middle (poor ranking, consistency has nothing reliable to reinforce and
may hurt, per the dataset2 case).

Validated by correlating this label-free proxy against the REAL Delta_2
(consistency's marginal EER contribution: self-training-only -> ours), computed
using labels only for validation here, never for the proxy itself. Uses the 12
actually-trained checkpoints in ckpt_ext/ (4 targets x 3 seeds) -- no new
training, just inference.
"""

import glob, os
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torchaudio, soundfile as sf
from concurrent.futures import ThreadPoolExecutor

DEVICE = "cuda"
SR, CROP_SEC, CACHE_SEC = 16000, 3.0, 4.0
CROP_LEN, CACHE_LEN = int(SR * CROP_SEC), int(SR * CACHE_SEC)
BATCH = 4
TARGET_PER_CLASS, MAX_PER_CORPUS_CLASS = 800, 1600
N_FINETUNE = 4
SEEDS = [0, 1, 2]
EER_TARGETS = ["asvspoof2019", "dataset2", "in_the_wild", "arabic"]
CKPT_DIR = "ckpt_ext"
Q = 0.3

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
    return pd.DataFrame(rows, columns=["path", "label", "corpus"])

manifest = build_manifest()

def cap_per_class(df, n, seed=0):
    return pd.concat([g.sample(min(n, len(g)), random_state=seed) for _, g in df.groupby("label")])

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

_AR = torch.arange(CROP_LEN, device=DEVICE)

def get_batch(buf, vlen, idx):
    span = (vlen[idx] - CROP_LEN).clamp(min=0)
    gidx = ((span // 2).unsqueeze(1) + _AR.unsqueeze(0)).clamp(max=CACHE_LEN - 1)
    return torch.gather(buf[idx], 1, gidx).float()

def sample_target(df, n_per_class, seed=0):
    return pd.concat([g.sample(min(n_per_class, len(g)), random_state=seed)
                      for _, g in df.groupby("label")]).sample(frac=1, random_state=seed).reset_index(drop=True)

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

@torch.no_grad()
def score(model, buf, vlen, n):
    model.eval()
    out = []
    idx = torch.arange(n, device=DEVICE)
    for i in range(0, n, BATCH):
        x = get_batch(buf, vlen, idx[i:i + BATCH])
        out.append(torch.softmax(model(x)[0].float(), 1)[:, 1])
    return torch.cat(out).cpu().numpy()

def tail_separation(s, q=Q):
    lo, hi = np.quantile(s, q), np.quantile(s, 1 - q)
    return float(s[s >= hi].mean() - s[s <= lo].mean())   # in [0, 1], label-free

def tail_purity_proxy(s, q=Q):
    """Fraction of scores NOT in the ambiguous middle -- another label-free stat."""
    lo, hi = np.quantile(s, q), np.quantile(s, 1 - q)
    mid_spread = hi - lo
    return float(1.0 - mid_spread)  # smaller quantile gap along the raw scale => less separation... use with care

rows = []
for target in EER_TARGETS:
    tgt_df = manifest[manifest.corpus == target]
    for seed in SEEDS:
        ckpt = f"{CKPT_DIR}/source_{target}_seed{seed}.pt"
        if not os.path.exists(ckpt):
            print(f"  !! missing {ckpt}, skipping")
            continue
        # small, per-target/seed cache -- freed right after, so this never
        # competes with the concurrent RawNet2Lite job for more than ~1 GB
        sampled = sample_target(tgt_df, TARGET_PER_CLASS, seed)
        buf, vlen, _ = build_cache(sampled)
        model = XLSRDetector().to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)
        s = score(model, buf, vlen, len(sampled))
        gap = tail_separation(s)
        score_std = float(s.std())
        frac_middle = float(((s > 0.4) & (s < 0.6)).mean())
        rows.append(dict(target=target, seed=seed, tail_gap=gap, score_std=score_std,
                         frac_middle=frac_middle))
        print(f"  {target} seed{seed}: tail_gap={gap:.3f}  std={score_std:.3f}  frac_middle={frac_middle:.3f}")
        del model, buf, vlen; torch.cuda.empty_cache()

proxy = pd.DataFrame(rows)
proxy.to_csv("lambda_selector_proxy.csv", index=False)

# validate against the REAL Delta_2 (consistency's marginal contribution) from the main grid
ext = pd.read_csv("results_ext.csv")
t = ext[(ext.setting == "transductive") & (ext.family == "xlsr")]
piv = t.pivot_table(index=["target", "seed"], columns="method", values="eer")
piv["delta2"] = piv["ours"] - piv["st_only"]   # negative = consistency helped
piv = piv.reset_index()[["target", "seed", "delta2"]]

merged = proxy.merge(piv, on=["target", "seed"])
merged.to_csv("lambda_selector_validation.csv", index=False)
print("\n=== validation: label-free proxy vs. real Delta_2 (consistency benefit) ===")
print(merged.to_string(index=False))
for col in ["tail_gap", "score_std", "frac_middle"]:
    r = merged[col].corr(merged["delta2"])
    print(f"corr({col}, delta2) = {r:+.3f}  (negative = proxy correctly predicts consistency helps)")
