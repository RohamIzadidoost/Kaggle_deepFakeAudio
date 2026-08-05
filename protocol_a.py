"""Protocol A: train on ASVspoof2019-LA train only, evaluate on the official
ASVspoof2021-DF eval partition.

Why this exists: every number in the paper so far comes from our own
leave-one-corpus-out protocol, where the only comparison points are ablations we
built ourselves. This is the one protocol where *published* baselines exist
(Wav2DF-TSL 1.95%, AASIST-classifier 2.87%, Wav2Vec2-AASIST 8.54%,
challenge-era DF top-1 ~15.6%), so it is what makes a SOTA comparison possible
at all.

Two structural differences from `extended_pipeline.py`, both forced by scale:

  * The DF eval set is ~459k clips = ~59 GB at fp16/4s. It CANNOT live in a
    GPU-resident cache the way the leave-one-corpus-out target pools do. Scoring
    therefore streams: CPU threads decode a chunk, the GPU scores it, the chunk
    is dropped. Only the small 2019-LA train pool (25,380 clips, 3.25 GB) is
    cached on the GPU.
  * Adaptation cannot run 4 epochs of backprop over 459k clips. We adapt on an
    unlabeled subsample (ADAPT_N) and then score the full official trial list.
    No labels are used at any point in adaptation.

Reported rows, mirroring the paper's existing transductive/inductive convention:
  source              -- no adaptation
  ours (full eval)    -- adapt on subsample, score the full official list
  ours (disjoint)     -- same model, scored only on trials NOT in the adapt
                         subsample; the clean inductive number

Set SMOKE = True first (a few minutes, tiny subsets) before the real run.
"""

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

from eval_protocol import DF_KEYS_DEFAULT, load_df_keys, score_official_df
from rawboost_gpu import RawBoostGPU

SMOKE = False   # <-- run this first; only then set False for the real run
RECIPE = "rawboost"   # "baseline" (gain+noise, 8ep) or "rawboost" (Phase 1b)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SR = 16000
CROP_SEC, CACHE_SEC = 3.0, 4.0
CROP_LEN, CACHE_LEN = int(SR * CROP_SEC), int(SR * CACHE_SEC)

N_FINETUNE = 4
BATCH = 32
LR, TTA_LR = 2e-4, 1e-4
Q, LAMBDA_CONS = 0.3, 0.3          # same defaults as the main paper
DECODE_WORKERS = 16

LA19_PROTO = "data/asvspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
LA19_FLAC = "data/asvspoof2019_LA/ASVspoof2019_LA_train/flac"
DF_PARTS = "data/dataset_1/ASVspoof2021_DF_eval_part0*/**/*.flac"

if SMOKE:
    SOURCE_EPOCHS, TTA_EPOCHS = 1, 1
    MAX_TRAIN, MAX_EVAL, ADAPT_N = 400, 800, 200
elif RECIPE == "rawboost":
    # Phase 1b: source loss bottomed at 0.0001 under the baseline recipe (gain
    # +/-30% + 0.005*randn) -- textbook memorization of the 25,380-clip 2019-LA
    # train set rather than learned synthesis cues. More epochs alone would make
    # this worse; RawBoost is the actual fix (channel/device perturbation strong
    # enough that memorizing exact per-clip waveforms stops being a viable
    # shortcut). 16 epochs: double the baseline's, since a stronger augmentation
    # needs more passes to converge but should NOT re-reach loss~0 if it's
    # working -- watch the loss curve, not just the final EER.
    SOURCE_EPOCHS, TTA_EPOCHS = 16, 4
    MAX_TRAIN, MAX_EVAL, ADAPT_N = None, None, 20000
else:
    SOURCE_EPOCHS, TTA_EPOCHS = 8, 4
    MAX_TRAIN, MAX_EVAL, ADAPT_N = None, None, 20000

RECIPE_TAG = "" if RECIPE == "baseline" else f"_{RECIPE}"
SUFFIX = "_smoke" if SMOKE else RECIPE_TAG
RESULTS_CSV = "results_protocol_a.csv" if not SMOKE else "results_protocol_a_smoke.csv"
LOG_FILE = f"run_log_protocol_a{SUFFIX}.txt"
CKPT = f"ckpt_protocol_a{SUFFIX}.pt"
METHOD_SUFFIX = "" if (SMOKE or RECIPE == "baseline") else f"_{RECIPE}"


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


# ---------------------------------------------------------------- manifests
def build_la19_train():
    """ASVspoof2019-LA train split. Kept at full size and native class balance
    (2580 bonafide / 22800 spoof) because that is what published Protocol-A
    numbers train on -- do not rebalance here or comparability is lost."""
    rows = []
    for line in open(LA19_PROTO):
        p = line.split()
        if len(p) >= 5:
            rows.append((f"{LA19_FLAC}/{p[1]}.flac", 1 if p[-1] == "spoof" else 0))
    df = pd.DataFrame(rows, columns=["path", "y"])
    missing = ~df.path.map(os.path.exists)
    if missing.any():
        raise FileNotFoundError(f"{missing.sum()} of {len(df)} 2019-LA train flacs missing")
    return df


def build_df_eval(phase="eval"):
    """Official DF eval trials that are actually on disk, restricted to `phase`.

    part03 is absent locally (458,871 of 611,829 trials present). That omission
    is uniform across codec / label / phase (verified 75-25 in every stratum),
    so pooled EER over what remains is an unbiased estimate of the full-set
    number -- but n_missing is recorded so the shortfall is always visible.
    """
    keys = load_df_keys(DF_KEYS_DEFAULT)
    rows = []
    for p in glob.glob(DF_PARTS, recursive=True):
        utt = os.path.basename(p)[:-5]
        k = keys.get(utt)
        if k is not None and (phase is None or k[1] == phase):
            rows.append((p, k[0], utt))
    return pd.DataFrame(rows, columns=["path", "y", "utt"])


# ---------------------------------------------------------------- audio io
def decode(path):
    try:
        d, sr = sf.read(path, dtype="float32", always_2d=True)
        w = d.mean(axis=1)
        if sr != SR:
            w = torchaudio.functional.resample(torch.from_numpy(w), sr, SR).numpy()
        return w
    except Exception:
        return np.zeros(1, dtype=np.float32)


def _pack(paths):
    """Decode a list of paths into a (n, CACHE_LEN) fp16 array + valid lengths."""
    n = len(paths)
    buf = np.zeros((n, CACHE_LEN), dtype=np.float16)
    vlen = np.ones(n, dtype=np.int64)
    with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as ex:
        for i, w in enumerate(ex.map(decode, paths)):
            L = min(len(w), CACHE_LEN)
            buf[i, :L] = w[:L]
            vlen[i] = max(L, 1)
    return buf, vlen


class CpuPool:
    """Decoded-audio pool held in pinned CPU memory, yielding GPU batches.

    The 10 GB local card cannot host a GPU-resident cache the way the 33 GB
    cloud MIG slice did: the full 2019-LA train pool alone is 3.25 GB at
    fp16/4s, which on top of model+optimizer+activations (~5.5 GB measured)
    plus the desktop's ~1 GB leaves no headroom. Per-batch transfer is 4 MB --
    negligible over PCIe -- and decouples pool size from VRAM entirely.
    """

    def __init__(self, paths, labels=None):
        buf, vlen = _pack(list(paths))
        self.buf = torch.from_numpy(buf).pin_memory()
        self.vlen = torch.from_numpy(vlen)
        self.y = (torch.tensor(np.asarray(labels), dtype=torch.long)
                  if labels is not None else None)
        self.n = len(self.buf)

    def gb(self):
        return self.buf.numel() * 2 / 1e9

    def batch(self, idx_cpu, train):
        """idx_cpu: CPU LongTensor of pool indices -> (x_gpu, y_gpu_or_None)."""
        xb = self.buf[idx_cpu].to(DEVICE, non_blocking=True)
        vb = self.vlen[idx_cpu].to(DEVICE, non_blocking=True)
        x = _crop(xb, vb, train=train)
        y = self.y[idx_cpu].to(DEVICE, non_blocking=True) if self.y is not None else None
        return x, y


_AR = torch.arange(CROP_LEN, device=DEVICE)


def _crop(buf_gpu, vlen_gpu, train):
    """Center crop (eval) or random crop (train) to CROP_LEN, batched on GPU."""
    span = (vlen_gpu - CROP_LEN).clamp(min=0)
    if train:
        start = (torch.rand(len(span), device=DEVICE) * (span + 1).float()).long()
    else:
        start = span // 2
    gidx = (start.unsqueeze(1) + _AR.unsqueeze(0)).clamp(max=CACHE_LEN - 1)
    return torch.gather(buf_gpu, 1, gidx).float()


def augment(x):
    """Weak gain+noise perturbation -- the paper's original recipe. Kept as the
    TTA-consistency perturbation even under RECIPE='rawboost' so that changing
    the SOURCE training augmentation and changing the CONSISTENCY perturbation
    are isolated, testable-separately variables (Phase 2.3 of the plan is the
    latter change; do not conflate the two in one run)."""
    return x * torch.empty(x.size(0), 1, device=DEVICE).uniform_(0.7, 1.3) \
        + 0.005 * torch.randn_like(x)


# Source-training augmentation only. bank_size=4096 trades exact-reference
# equivalence (verified in rawboost_gpu.py's self-test) for ~13x throughput --
# necessary here since fit_source runs ~794 batches/epoch x 16 epochs.
_source_rawboost = RawBoostGPU(sr=SR, algo=4, bank_size=4096) if RECIPE == "rawboost" else None


def source_augment(x):
    return _source_rawboost(x) if _source_rawboost is not None else augment(x)


# ---------------------------------------------------------------- model
class XLSRDetector(nn.Module):
    """Same architecture as extended_pipeline.py:285, so Protocol-A numbers are
    directly comparable to the leave-one-corpus-out numbers in the paper."""

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
        # Whole forward under autocast, not just the encoder: extended_pipeline.py
        # relies on every call site wrapping model() in `with amp():`, and a
        # missed one silently gives "mat1 and mat2 must have the same dtype"
        # (bf16 features hitting an fp32 head). Self-contained is safer.
        with amp():
            feats, _ = self.ssl.extract_features(wav.float())
            x = feats[-1]
            a = torch.softmax(self.attn(x).squeeze(-1), 1)
            emb = self.proj(torch.bmm(a.unsqueeze(1), x).squeeze(1))
            return self.cls(emb), emb


def set_tta_params(model):
    """Trainable set for adaptation: top-layer LayerNorm affine + head only
    (extended_pipeline.py:401)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.ssl.model.encoder.transformer.layers[-N_FINETUNE:].modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad_(True)
    for head in (model.attn, model.proj, model.cls):
        for p in head.parameters():
            p.requires_grad_(True)


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def eer_of(y, s):
    from metrics import compute_eer
    e, _ = compute_eer(np.asarray(y), np.asarray(s))
    return e * 100


def estimate_prevalence(scores, min_prior=0.02, max_prior=0.98):
    """Label-free estimate of P(real) from a score distribution alone.

    Why this exists: naive TTA on the real DF eval pool collapsed AUC .870 ->
    .686. Cause, confirmed against ground truth: `adapt()`'s Q=0.3 pseudo-labels
    the bottom 30% of the pool as pseudo-real regardless of the pool's actual
    class balance. DF eval is 97.5% spoof / 2.5% bonafide, so that bottom-30%
    bucket is mechanically dominated by low-scoring spoof clips mislabeled real
    -- self-training then teaches the model that a lot of spoof IS real.

    Not `mean(scores)`: that assumes the classifier is calibrated, which is
    exactly the property this paper's own thesis says does NOT transfer
    cross-corpus (ranking transfers, calibration doesn't). This instead fits a
    2-component GMM to the score histogram and reads off the low-scoring
    component's mixture weight -- it exploits rank/separation, not the absolute
    score value, so it survives the same calibration drift the paper's method
    already relies on surviving.

    Validated against real withheld labels on the actual DF adapt pool: true
    bonafide prevalence 2.47%, estimated 2.63% (both directions of a synthetic
    sweep also recovered the true prior to within a few points down to 10%;
    accuracy degrades at the very extreme, hence the `min_prior` floor).
    """
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=2, random_state=0, n_init=3).fit(
        scores.reshape(-1, 1))
    means, weights = gm.means_.flatten(), gm.weights_
    pi_real = float(weights[np.argmin(means)])
    return float(np.clip(pi_real, min_prior, max_prior))


# ---------------------------------------------------------------- train / score
def fit_source(model, df, epochs):
    """Train on the 2019-LA train pool. Cosine schedule + weight decay (the
    recipe already validated locally in improve_rawnet2lite.py:179)."""
    pool = CpuPool(df.path.tolist(), df.y.values)
    log(f"  train pool {pool.n} clips = {pool.gb():.2f} GB (pinned CPU)")

    opt = torch.optim.Adam(trainable(model), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    for ep in range(epochs):
        model.train()
        model.ssl.eval()
        perm = torch.randperm(pool.n)
        tot = cnt = 0
        for i in range(0, pool.n, BATCH):
            b = perm[i:i + BATCH]
            x, y = pool.batch(b, train=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(source_augment(x))[0].float(), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            cnt += len(b)
        sched.step()
        log(f"  source epoch {ep + 1}/{epochs} loss {tot / cnt:.4f} "
            f"lr {sched.get_last_lr()[0]:.2e}")

    del pool
    torch.cuda.empty_cache()
    return model


@torch.no_grad()
def stream_score(model, df, chunk=4096, tag=""):
    """Score a large manifest without ever caching it whole: decode a chunk on
    CPU threads, score it on GPU, discard it."""
    model.eval()
    out = np.zeros(len(df), dtype=np.float32)
    paths = df.path.tolist()
    t0 = time.time()
    for c0 in range(0, len(paths), chunk):
        sl = slice(c0, min(c0 + chunk, len(paths)))
        pool = CpuPool(paths[sl])
        got = []
        for i in range(0, pool.n, BATCH):
            x, _ = pool.batch(torch.arange(i, min(i + BATCH, pool.n)), train=False)
            got.append(torch.softmax(model(x)[0].float(), 1)[:, 1])
        out[sl] = torch.cat(got).cpu().numpy()
        del pool
        torch.cuda.empty_cache()
        done = min(c0 + chunk, len(paths))
        if c0 == 0 or done % (chunk * 10) == 0 or done == len(paths):
            rate = done / max(time.time() - t0, 1e-9)
            log(f"    {tag}scored {done}/{len(paths)}  {rate:.0f} clips/s  "
                f"eta {(len(paths) - done) / max(rate, 1e-9) / 60:.1f} min")
    return out


def adapt(model, df, epochs=TTA_EPOCHS, prior_aware=False):
    """Confident-tail pseudo-label self-training + channel consistency, on an
    unlabeled subsample. Same objective as extended_pipeline.py:412; the pool is
    small enough (ADAPT_N) to cache on GPU.

    `prior_aware`: split the total confident-label budget (2*Q of the pool) by
    estimated class prevalence instead of splitting it evenly between the two
    tails. At prevalence=0.5 this is IDENTICAL to the original symmetric
    behaviour (lo_frac = hi_frac = Q), so the balanced leave-one-corpus-out grid
    in the main paper is unaffected -- this only changes behaviour on skewed
    pools like the real DF eval set (97.5% spoof), where naive Q=0.3 collapsed
    AUC .870 -> .686 by pseudo-labeling mostly-spoof clips as "confident real".
    Prevalence is estimated ONCE from the frozen pre-adaptation model and held
    fixed for all epochs -- re-estimating from a self-training model each epoch
    would let a drifting estimate reinforce its own mistakes.
    """
    set_tta_params(model)
    opt = torch.optim.Adam(trainable(model), lr=TTA_LR)

    pool = CpuPool(df.path.tolist())
    n = pool.n
    log(f"  adapt pool {n} clips = {pool.gb():.2f} GB (pinned CPU)  "
        f"prior_aware={prior_aware}")

    lo_frac, hi_frac = Q, Q
    if prior_aware:
        model.eval()
        with torch.no_grad():
            s0 = []
            for i in range(0, n, BATCH):
                x, _ = pool.batch(torch.arange(i, min(i + BATCH, n)), train=False)
                s0.append(torch.softmax(model(x)[0].float(), 1)[:, 1])
            s0 = torch.cat(s0).cpu().numpy()
        pi_real = estimate_prevalence(s0)
        lo_frac, hi_frac = 2 * Q * pi_real, 2 * Q * (1 - pi_real)
        log(f"  estimated P(real) = {pi_real:.4f}  ->  "
            f"pseudo-real budget {lo_frac:.4f}  pseudo-fake budget {hi_frac:.4f}")

    for ep in range(epochs):
        # re-score the pool and re-assign confident-tail pseudo-labels each epoch
        model.eval()
        with torch.no_grad():
            s = []
            for i in range(0, n, BATCH):
                x, _ = pool.batch(torch.arange(i, min(i + BATCH, n)), train=False)
                s.append(torch.softmax(model(x)[0].float(), 1)[:, 1])
            s = torch.cat(s).cpu().numpy()
        lo, hi = np.quantile(s, lo_frac), np.quantile(s, 1 - hi_frac)
        pl = torch.full((n,), -1, dtype=torch.long, device=DEVICE)
        pl[torch.from_numpy(s <= lo).to(DEVICE)] = 0
        pl[torch.from_numpy(s >= hi).to(DEVICE)] = 1

        model.train()
        model.ssl.eval()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            sel = perm[i:i + BATCH]
            x, _ = pool.batch(sel, train=False)
            bpl = pl[sel.to(DEVICE)]
            opt.zero_grad(set_to_none=True)
            logits = model(x)[0].float()
            p = torch.softmax(logits, 1)
            loss = torch.zeros((), device=DEVICE)
            conf = bpl >= 0
            if conf.any():
                loss = loss + F.cross_entropy(logits[conf], bpl[conf])
            loss = loss + LAMBDA_CONS * F.mse_loss(
                torch.softmax(model(augment(x))[0].float(), 1), p.detach())
            if loss.requires_grad:
                loss.backward()
                opt.step()
        log(f"  adapt epoch {ep + 1}/{epochs}  confident {int((pl >= 0).sum())}/{n}")

    del pool
    torch.cuda.empty_cache()
    return model


# ---------------------------------------------------------------- main
def report(name, df, scores, adapt_utts=None):
    """Official pooled DF EER (+ AUC), and the disjoint-subset number when the
    model was adapted on part of the pool."""
    smap = dict(zip(df.utt, scores))
    off = score_official_df(smap, phase="eval")
    auc = roc_auc_score(df.y.values, scores)
    log(f"  {name:22s} official DF EER {off['eer']:6.2f}%  AUC {auc:.4f}  "
        f"n_scored {off['n_scored']}  n_missing {off['n_missing']}")
    record(method=name, setting="official_eval", eer=round(off["eer"], 3),
           auc=round(auc, 4), n_scored=off["n_scored"], n_missing=off["n_missing"],
           threshold=round(off["threshold"], 6), smoke=SMOKE)

    if adapt_utts:
        m = ~df.utt.isin(adapt_utts)
        if m.sum() > 100 and df.y[m].nunique() > 1:
            e = eer_of(df.y[m].values, scores[m.values])
            a = roc_auc_score(df.y[m].values, scores[m.values])
            log(f"  {name + ' (disjoint)':22s} EER {e:6.2f}%  AUC {a:.4f}  n {int(m.sum())}")
            record(method=name, setting="disjoint_eval", eer=round(e, 3),
                   auc=round(a, 4), n_scored=int(m.sum()), n_missing=-1,
                   threshold=-1, smoke=SMOKE)


def main():
    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    log(f"=== Protocol A | SMOKE={SMOKE} | bf16={USE_BF16} ===")

    train = build_la19_train()
    ev = build_df_eval(phase="eval")
    log(f"2019-LA train: {len(train)} clips "
        f"({int((train.y == 0).sum())} bonafide / {int((train.y == 1).sum())} spoof)")
    log(f"DF eval (phase=eval, on disk): {len(ev)} clips "
        f"({int((ev.y == 0).sum())} bonafide / {int((ev.y == 1).sum())} spoof)")

    # NB: use concat-over-groupby, not groupby().apply() -- pandas 3.x silently
    # drops the grouping column from apply() results, which would kill `.y` here.
    def cap_per_class(df, n):
        return pd.concat([g.sample(min(len(g), n), random_state=0)
                          for _, g in df.groupby("y")])

    if MAX_TRAIN:
        train = cap_per_class(train, MAX_TRAIN // 2)
        log(f"  SMOKE: train subsampled to {len(train)}")
    if MAX_EVAL:
        ev = cap_per_class(ev, MAX_EVAL // 2)
        log(f"  SMOKE: eval subsampled to {len(ev)}")
    ev = ev.reset_index(drop=True)

    model = XLSRDetector().to(DEVICE)
    if os.path.exists(CKPT):
        model.load_state_dict(torch.load(CKPT, map_location=DEVICE), strict=False)
        log(f"loaded existing source checkpoint {CKPT} (delete it to retrain)")
    else:
        log("training source model on 2019-LA train only")
        fit_source(model, train, SOURCE_EPOCHS)
        torch.save({n: p.detach().cpu() for n, p in model.named_parameters()
                    if p.requires_grad}, CKPT)
        log(f"saved {CKPT}")

    # resume: skip a method entirely if its official_eval row is already recorded
    # -- a 22-min re-score of the full 400k pool is not something to redo by
    # accident when only adding one new method to an existing results file.
    done = set()
    if os.path.exists(RESULTS_CSV):
        prev = pd.read_csv(RESULTS_CSV)
        done = set(prev[prev.setting == "official_eval"].method)
    log(f"already recorded: {done or '(none)'}")

    m_source = "source" + METHOD_SUFFIX
    m_ours = "ours" + METHOD_SUFFIX
    m_ours_prior = "ours_prior" + METHOD_SUFFIX

    if m_source not in done:
        log(f"scoring DF eval with the {m_source} model")
        report(m_source, ev, stream_score(model, ev, tag=f"{m_source} "))
    else:
        log(f"{m_source} already recorded, skipping re-score")

    adapt_df = ev.sample(min(ADAPT_N, len(ev)), random_state=0)
    import copy

    if m_ours not in done:
        log(f"[naive] adapting on {len(adapt_df)} unlabeled DF eval clips "
            f"(symmetric Q, no labels used)")
        adapted = adapt(copy.deepcopy(model), adapt_df, prior_aware=False)
        log(f"scoring DF eval with the naive-adapted ({m_ours}) model")
        report(m_ours, ev, stream_score(adapted, ev, tag=f"{m_ours} "), set(adapt_df.utt))
    else:
        log(f"{m_ours} already recorded, skipping")

    if m_ours_prior not in done:
        log(f"[prior-aware] adapting on {len(adapt_df)} unlabeled DF eval clips "
            f"(prevalence-weighted budget, no labels used)")
        adapted_p = adapt(copy.deepcopy(model), adapt_df, prior_aware=True)
        log(f"scoring DF eval with the prior-aware-adapted ({m_ours_prior}) model")
        report(m_ours_prior, ev, stream_score(adapted_p, ev, tag=f"{m_ours_prior} "),
               set(adapt_df.utt))
    else:
        log(f"{m_ours_prior} already recorded, skipping")

    log(f"DONE in {(time.time() - t0) / 60:.1f} min | "
        f"peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
