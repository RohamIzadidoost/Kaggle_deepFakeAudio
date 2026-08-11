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
# # Divergence Analysis, DANN Baseline, and Per-Language Study
#
# Run this **after** `extended_pipeline.ipynb` has finished (it reuses those
# checkpoints; if a checkpoint is missing it trains its own).
#
# Four independent parts, each separately toggleable so one failure cannot take
# down the rest:
#
# * **A. Domain divergence.** Estimate the proxy $\mathcal{A}$-distance
#   $d_\mathcal{A}=2(1-2\varepsilon)$ between source and target in the detector's
#   embedding space, before and after adaptation. Ben-David's bound says
#   $\varepsilon_T \le \varepsilon_S + \tfrac12 d_{\mathcal{H}\Delta\mathcal{H}} + \lambda^\*$;
#   the interesting question is whether our TTA lowers the divergence term (i.e.
#   aligns domains) or leaves it alone and simply relocates the decision boundary.
# * **B. DANN baseline.** Classic domain-adversarial training (gradient reversal
#   on a source-vs-target discriminator), same protocol and seeds as the main
#   grid, so the paper can answer "why not DANN?" with numbers.
# * **C. Per-language MLAAD.** Fake-recall for each of the ~38 languages, split
#   into languages seen and unseen during training.
# * **D. Embedding geometry.** t-SNE of source vs target embeddings, before and
#   after adaptation, plus a linear domain probe.

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from tqdm.auto import tqdm

RUN_DIVERGENCE, RUN_DANN, RUN_PERLANG, RUN_TSNE = True, True, True, True
# ANALYSIS_PARTS lets a supervisor run just one part (P1.2 needs DANN only; the
# divergence / per-language / t-SNE analyses are already done and would be
# recomputed for nothing). Default keeps all four on.
if os.environ.get("ANALYSIS_PARTS"):
    _p = {s.strip() for s in os.environ["ANALYSIS_PARTS"].split(",")}
    RUN_DIVERGENCE, RUN_DANN = "div" in _p, "dann" in _p
    RUN_PERLANG, RUN_TSNE = "lang" in _p, "tsne" in _p

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
torch.backends.cuda.matmul.allow_tf32 = True

SR = 16000
CROP_SEC, CACHE_SEC = 3.0, 4.0
CROP_LEN, CACHE_LEN = int(SR * CROP_SEC), int(SR * CACHE_SEC)

N_FINETUNE, BATCH = 4, 32
LR, TTA_LR = 2e-4, 1e-4
Q, LAMBDA_CONS, TTA_EPOCHS = 0.3, 0.3, 4
SOURCE_EPOCHS = 8
SOURCE_PER_CLASS, TARGET_PER_CLASS, MAX_PER_CORPUS_CLASS = 5000, 3000, 6000
N_HELDOUT_LANGS = 6
EER_TARGETS = ["asvspoof2019", "dataset2", "in_the_wild", "arabic"]

SEEDS_DIV = [0]                     # divergence analysis: seed 0 is enough
# DANN_SEEDS override for P1.2 (3 -> 5 seeds). NOTE: unlike the other pipelines
# this part has NO resume guard -- it appends whatever seeds it is given, so
# pass only the seeds that are actually missing from results_dann.csv.
SEEDS_DANN = [int(s) for s in os.environ["DANN_SEEDS"].split(",")] \
    if os.environ.get("DANN_SEEDS") else [0, 1, 2]
TARGETS = EER_TARGETS

SRC_CKPT_DIR = "ckpt_ext"           # reuse the extended grid's source models (read-only, external)
OUT_DIR = "analysis_res"            # everything THIS notebook writes lives here
CKPT_DIR = f"{OUT_DIR}/ckpt_analysis"
LOG_FILE = f"{OUT_DIR}/run_log_analysis.txt"
os.makedirs(CKPT_DIR, exist_ok=True)

def amp():
    return torch.amp.autocast("cuda", dtype=torch.bfloat16) if USE_BF16 else torch.amp.autocast("cuda", enabled=False)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def record(part, **row):
    """One CSV per part -- the parts have different column sets, and appending
    them to a single file would silently write later rows under the first
    header, misaligning every column."""
    f = f"{OUT_DIR}/results_{part}.csv"
    pd.DataFrame([dict(part=part, **row)]).to_csv(f, mode="a", header=not os.path.exists(f), index=False)

log(f"device {DEVICE} | bf16 {USE_BF16} | parts: div={RUN_DIVERGENCE} dann={RUN_DANN} lang={RUN_PERLANG} tsne={RUN_TSNE}")

# %% [markdown]
# ## Data, cache, model (same definitions as the extended pipeline)

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
                rows.append((f"{flac}/{p[1]}.flac", lab, "asvspoof2019",
                             p[-2] if lab == "fake" else "bonafide", "en"))
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
    for meta in glob.glob("data/mlaad/**/meta.csv", recursive=True):
        model_dir = os.path.dirname(meta)
        lang = os.path.basename(os.path.dirname(model_dir))
        gen = os.path.basename(model_dir)
        for w in glob.glob(f"{model_dir}/*.wav"):
            rows.append((w, "fake", "mlaad", gen, lang))
    return pd.DataFrame(rows, columns=["path", "label", "corpus", "generator", "language"])

manifest = build_manifest()
log("manifest:\n" + manifest.groupby(["corpus", "label"]).size().to_string())

mlaad_langs = sorted(manifest[manifest.corpus == "mlaad"].language.unique())
HELDOUT_LANGS = list(np.random.RandomState(0).choice(
    mlaad_langs, size=min(N_HELDOUT_LANGS, len(mlaad_langs)), replace=False))
log(f"MLAAD held-out languages: {HELDOUT_LANGS}")

def cap_per_class(df, n, seed=0):
    return pd.concat([g.sample(min(n, len(g)), random_state=seed) for _, g in df.groupby("label")])

# for the per-language study we keep a wider, language-stratified MLAAD sample.
# NB: an explicit loop, not groupby().apply() -- pandas 3 drops the grouping
# column from the result, which silently empties every per-language lookup.
mlaad_all = manifest[manifest.corpus == "mlaad"]
per_lang = pd.concat([g.sample(min(300, len(g)), random_state=0)
                      for _, g in mlaad_all.groupby("language")])
other = manifest[manifest.corpus != "mlaad"]
pool = pd.concat([cap_per_class(g, MAX_PER_CORPUS_CLASS) for _, g in other.groupby("corpus")]
                 + [per_lang]).reset_index(drop=True)
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
    buf, vlen = np.zeros((n, CACHE_LEN), dtype=np.float16), np.ones(n, dtype=np.int64)
    with ThreadPoolExecutor(max_workers=16) as ex:
        for i, w in enumerate(tqdm(ex.map(decode, df.path.tolist()), total=n, desc="decoding")):
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

def trainable(m):
    return [p for p in m.parameters() if p.requires_grad]

_probe = XLSRDetector().to(DEVICE)
try:
    with amp():
        _probe(torch.randn(2, CROP_LEN, device=DEVICE))
    ENCODER_AMP = True
except RuntimeError:
    ENCODER_AMP = False
del _probe; torch.cuda.empty_cache()
log(f"encoder autocast: {ENCODER_AMP}")

def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2

@torch.no_grad()
def embed_and_score(model, idx):
    """Return (labels, fake-probabilities, 256-d embeddings) for idx."""
    model.eval()
    S, E = [], []
    for i in range(0, len(idx), BATCH):
        x, _ = get_batch(idx[i:i + BATCH], train=False)
        with amp():
            logits, emb = model(x)
        S.append(torch.softmax(logits.float(), 1)[:, 1])
        E.append(emb.float())
    return Y[idx].cpu().numpy(), torch.cat(S).cpu().numpy(), torch.cat(E).cpu().numpy()

def metrics_from(y, s):
    return {"eer": eer(y, s) * 100, "auc": roc_auc_score(y, s),
            "acc": accuracy_score(y, (s >= 0.5).astype(int)) * 100}

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
        log(f"  {tag} epoch {ep+1}/{epochs} loss {tot/cnt:.4f}")
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

def adapt(model, idx, epochs=TTA_EPOCHS):
    set_tta_params(model)
    opt = torch.optim.Adam(trainable(model), lr=TTA_LR)
    for _ in range(epochs):
        _, s, _ = embed_and_score(model, idx)
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
                if conf.any():
                    loss = loss + F.cross_entropy(logits[conf], bpl[conf])
                loss = loss + LAMBDA_CONS * F.mse_loss(torch.softmax(model(augment(x))[0], 1), p.detach())
            if loss.requires_grad:
                loss.backward(); opt.step()
    return model

def get_source_model(target, seed):
    """Reuse the extended grid's checkpoint when available, else train one."""
    m = XLSRDetector(encoder_amp=ENCODER_AMP).to(DEVICE)
    for d in (SRC_CKPT_DIR, CKPT_DIR):
        ck = f"{d}/source_{target}_seed{seed}.pt"
        if os.path.exists(ck):
            m.load_state_dict(torch.load(ck, map_location=DEVICE), strict=False)
            log(f"  loaded {ck}")
            return m
    src_pool = pd.concat([pool[(pool.corpus != target) & (pool.corpus != "mlaad")], mlaad_pool])
    src_idx = idx_of(sample_source(src_pool, SOURCE_PER_CLASS, seed))
    fit(m, src_idx, SOURCE_EPOCHS, tag=f"seed{seed}/{target}")
    torch.save({n: p.detach().cpu() for n, p in m.named_parameters() if p.requires_grad},
               f"{CKPT_DIR}/source_{target}_seed{seed}.pt")
    return m

# %% [markdown]
# ## A. Domain divergence (proxy $\mathcal{A}$-distance)
#
# A logistic-regression discriminator is cross-validated to separate source from
# target embeddings; its error $\varepsilon$ gives
# $d_\mathcal{A}=2(1-2\varepsilon)\in[0,2]$. $d_\mathcal{A}=0$ means the domains
# are indistinguishable, $2$ means perfectly separable. We measure it on the
# source model and again after test-time adaptation, and pair it with the EER
# change so the two can be correlated.

# %%
def proxy_a_distance(Es, Et, seed=0, cv=5):
    n = min(len(Es), len(Et))
    rs = np.random.RandomState(seed)
    Es = Es[rs.choice(len(Es), n, replace=False)]
    Et = Et[rs.choice(len(Et), n, replace=False)]
    X = np.vstack([Es, Et])
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    y = np.concatenate([np.zeros(n), np.ones(n)])
    acc = cross_val_score(LogisticRegression(max_iter=3000), X, y, cv=cv, scoring="accuracy").mean()
    return 2 * (2 * acc - 1), acc          # d_A, discriminator accuracy

if RUN_DIVERGENCE:
    log("=== A. domain divergence ===")
    for seed in SEEDS_DIV:
        for target in TARGETS:
            try:
                torch.manual_seed(seed); np.random.seed(seed)
                src_pool = pd.concat([pool[(pool.corpus != target) & (pool.corpus != "mlaad")], mlaad_pool])
                src_idx = idx_of(sample_source(src_pool, 1500, seed))
                tgt_idx = idx_of(sample_target(pool[pool.corpus == target], TARGET_PER_CLASS, seed))

                source = get_source_model(target, seed)
                _, _, Es0 = embed_and_score(source, src_idx)
                yt, st0, Et0 = embed_and_score(source, tgt_idx)
                dA0, dacc0 = proxy_a_distance(Es0, Et0, seed)
                m0 = metrics_from(yt, st0)

                adapted = adapt(copy.deepcopy(source), tgt_idx)
                _, _, Es1 = embed_and_score(adapted, src_idx)
                yt, st1, Et1 = embed_and_score(adapted, tgt_idx)
                dA1, dacc1 = proxy_a_distance(Es1, Et1, seed)
                m1 = metrics_from(yt, st1)

                record("divergence", seed=seed, target=target,
                       dA_source=round(dA0, 4), dA_adapted=round(dA1, 4),
                       domacc_source=round(dacc0, 4), domacc_adapted=round(dacc1, 4),
                       eer_source=round(m0["eer"], 3), eer_adapted=round(m1["eer"], 3),
                       auc_source=round(m0["auc"], 4), auc_adapted=round(m1["auc"], 4))
                log(f"  {target}: d_A {dA0:.3f} -> {dA1:.3f} (domain-clf acc {dacc0:.3f} -> {dacc1:.3f}) | "
                    f"EER {m0['eer']:.2f} -> {m1['eer']:.2f}")

                if RUN_TSNE and seed == SEEDS_DIV[0]:
                    np.savez(f"{OUT_DIR}/emb_{target}_seed{seed}.npz",
                             Es0=Es0[:1500], Et0=Et0[:1500], Es1=Es1[:1500], Et1=Et1[:1500],
                             yt=yt[:1500])
                del adapted, source; torch.cuda.empty_cache()
            except Exception as e:
                log(f"  !! divergence {target} failed: {type(e).__name__}: {e}")

# %% [markdown]
# ## B. DANN baseline
#
# Classic domain-adversarial training: a discriminator on the shared embedding
# tries to tell source from (unlabeled) target while a gradient-reversal layer
# makes the encoder fool it. This uses unlabeled target data at **training**
# time, whereas our method uses it at **test** time; both are unsupervised, so
# the comparison is fair. $\lambda$ is ramped $0\!\to\!\lambda_{\max}$ on the
# usual DANN schedule, since a full-strength reversal from step 0 is unstable.

# %%
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g.neg() * ctx.lambd, None

def train_dann(target, seed, epochs=SOURCE_EPOCHS, lambda_max=0.5):
    torch.manual_seed(seed); np.random.seed(seed)
    src_pool = pd.concat([pool[(pool.corpus != target) & (pool.corpus != "mlaad")], mlaad_pool])
    src_idx = idx_of(sample_source(src_pool, SOURCE_PER_CLASS, seed))
    tgt_idx = idx_of(sample_target(pool[pool.corpus == target], TARGET_PER_CLASS, seed))

    model = XLSRDetector(encoder_amp=ENCODER_AMP).to(DEVICE)
    disc = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2)).to(DEVICE)
    opt = torch.optim.Adam(trainable(model) + list(disc.parameters()), lr=LR)

    total_steps = epochs * (len(src_idx) // BATCH + 1)
    step = 0
    for ep in range(epochs):
        model.train(); model.ssl.eval()
        perm = src_idx[torch.randperm(len(src_idx), device=DEVICE)]
        tot = cnt = 0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            tb = tgt_idx[torch.randint(0, len(tgt_idx), (len(b),), device=DEVICE)]
            xs, ys = get_batch(b, train=True)
            xt, _ = get_batch(tb, train=True)
            p = step / max(1, total_steps)
            lambd = lambda_max * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
            opt.zero_grad(set_to_none=True)
            with amp():
                logits_s, emb_s = model(augment(xs))
                _, emb_t = model(augment(xt))
                loss_cls = F.cross_entropy(logits_s, ys)
                emb = torch.cat([emb_s, emb_t])
                dom = torch.cat([torch.zeros(len(emb_s), dtype=torch.long, device=DEVICE),
                                 torch.ones(len(emb_t), dtype=torch.long, device=DEVICE)])
                loss_dom = F.cross_entropy(disc(GradReverse.apply(emb, lambd)), dom)
                loss = loss_cls + loss_dom
            loss.backward(); opt.step()
            step += 1
            tot += loss_cls.item() * len(b); cnt += len(b)
        log(f"  dann seed{seed}/{target} epoch {ep+1}/{epochs} cls-loss {tot/cnt:.4f} lambda {lambd:.2f}")
    return model, tgt_idx

if RUN_DANN:
    log("=== B. DANN baseline ===")
    for seed in SEEDS_DANN:
        for target in TARGETS:
            try:
                t0 = time.time()
                model, tgt_idx = train_dann(target, seed)
                y, s, _ = embed_and_score(model, tgt_idx)
                m = metrics_from(y, s)
                record("dann", seed=seed, target=target, method="dann",
                       eer=round(m["eer"], 3), auc=round(m["auc"], 4), acc=round(m["acc"], 2),
                       minutes=round((time.time() - t0) / 60, 1))
                log(f"  DANN {target} seed{seed}: EER {m['eer']:.2f} AUC {m['auc']:.3f} ({(time.time()-t0)/60:.1f} min)")
                del model; torch.cuda.empty_cache()
            except Exception as e:
                log(f"  !! dann {target} seed{seed} failed: {type(e).__name__}: {e}")

# %% [markdown]
# ## C. Per-language MLAAD
# MLAAD is fake-only, so the meaningful quantity is fake-recall at the model's
# own threshold. We report it per language and split by whether the language was
# present in source training.

# %%
if RUN_PERLANG:
    log("=== C. per-language MLAAD ===")
    for target in TARGETS:
        try:
            source = get_source_model(target, 0)
            n_scored, n_skipped = 0, 0
            for lang in mlaad_langs:
                sub = pool[(pool.corpus == "mlaad") & (pool.language == lang)]
                if len(sub) < 20:
                    n_skipped += 1
                    continue
                _, s, _ = embed_and_score(source, idx_of(sub))
                record("perlang", target=target, language=lang,
                       seen_in_training=lang not in HELDOUT_LANGS,
                       n=len(sub), recall=round(float((s >= 0.5).mean()) * 100, 2),
                       mean_score=round(float(s.mean()), 4))
                n_scored += 1
            log(f"  {target}: scored {n_scored}/{len(mlaad_langs)} languages"
                + (f" ({n_skipped} skipped, <20 clips)" if n_skipped else ""))
            del source; torch.cuda.empty_cache()
        except Exception as e:
            log(f"  !! perlang {target} failed: {type(e).__name__}: {e}")

# %% [markdown]
# ## D. Embedding geometry (t-SNE + linear domain probe)

# %%
if RUN_TSNE:
    log("=== D. embedding geometry ===")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        files = sorted(glob.glob(f"{OUT_DIR}/emb_*_seed*.npz"))
        if files:
            fig, axes = plt.subplots(2, len(files), figsize=(4 * len(files), 7.5), squeeze=False)
            for col, f in enumerate(files):
                d = np.load(f)
                name = os.path.basename(f)[len("emb_"):].rsplit("_seed", 1)[0]
                for row, (tagE, Es, Et) in enumerate([("source", d["Es0"], d["Et0"]),
                                                      ("adapted", d["Es1"], d["Et1"])]):
                    n = min(600, len(Es), len(Et))
                    X = np.vstack([Es[:n], Et[:n]])
                    Z = TSNE(n_components=2, init="pca", perplexity=30,
                             random_state=0).fit_transform(X)
                    ax = axes[row][col]
                    ax.scatter(Z[:n, 0], Z[:n, 1], s=4, alpha=.5, label="source", color="#1f77b4")
                    ax.scatter(Z[n:, 0], Z[n:, 1], s=4, alpha=.5, label="target", color="#ff7f0e")
                    ax.set_title(f"{name} ({tagE})", fontsize=10)
                    ax.set_xticks([]); ax.set_yticks([])
                    if row == 0 and col == 0:
                        ax.legend(fontsize=9, markerscale=3)
            fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_tsne.png", dpi=150)
            log(f"  wrote {OUT_DIR}/fig_tsne.png")
        else:
            log(f"  no {OUT_DIR}/emb_*.npz found (run part A first)")
    except Exception as e:
        log(f"  !! tsne failed: {type(e).__name__}: {e}")

# %% [markdown]
# ## Summary

# %%
def load_part(part):
    # each part has its own file with a consistent schema, so pandas infers
    # dtypes correctly and no coercion is needed
    f = f"{OUT_DIR}/results_{part}.csv"
    return pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()

d = load_part("divergence")
if len(d):
    print("=== domain divergence (proxy A-distance) ===")
    print(d[["target", "dA_source", "dA_adapted", "domacc_source", "domacc_adapted",
             "eer_source", "eer_adapted"]].to_string(index=False))
    d = d.copy()
    d["eer_gain"] = d.eer_source - d.eer_adapted
    d["dA_change"] = d.dA_adapted - d.dA_source
    print("\nper-target: does adaptation reduce divergence, and does it track the gain?")
    print(d[["target", "dA_change", "eer_gain"]].round(3).to_string(index=False))
    if len(d) > 2:
        print(f"\ncorr(d_A source, EER gain)   = {d.dA_source.corr(d.eer_gain):+.3f}")
        print(f"corr(d_A source, source EER) = {d.dA_source.corr(d.eer_source):+.3f}")

b = load_part("dann")
if len(b):
    print("\n=== DANN baseline ===")
    print(b.groupby("target").agg(eer_mean=("eer", "mean"), eer_std=("eer", "std"),
                                  auc_mean=("auc", "mean"), n=("eer", "size")).round(3).to_string())

p = load_part("perlang")
if len(p):
    print("\n=== MLAAD per-language fake-recall ===")
    print(p.groupby("seen_in_training").agg(recall_mean=("recall", "mean"),
                                            recall_std=("recall", "std"),
                                            n_langs=("language", "nunique")).round(2).to_string())
    print("\nworst 8 languages (averaged over targets):")
    print(p.groupby(["language", "seen_in_training"]).recall.mean()
          .sort_values().head(8).round(2).to_string())
