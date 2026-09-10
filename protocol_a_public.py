"""Protocol A on PUBLISHED checkpoints: the official ASVspoof2021-DF eval.

Why this exists (ICASSP_REVIEW_AND_PLAN.md, W2): every EER gain in the paper is
measured on our own source model, which scores 26.33% on the one protocol with
published baselines (Wav2Vec2-AASIST 8.54%, AASIST 2.87%, challenge top-1
~15.6%). A reviewer reads "relative gain on a weak model" and discounts the
whole diagnosis as an artefact of under-training.

This script removes that objection by running the method on somebody else's
SOTA-grade weights, on the official protocol, at the protocol's real 97%-spoof
class balance:

  * checkpoints: DeepFense ASV19_Wav2Vec2_AASIST_NoAug_Seed{2,42,240} -- XLS-R
    300M + AASIST trained on ASVspoof2019-LA train, i.e. exactly the Protocol-A
    training condition the published baselines use. Optionally ash56/ssl-aasist
    (WaveFake-trained), which is a different training condition and is reported
    separately.
  * arms: source / naive fixed-q TTA / prior-corrected (BBSE) TTA /
    label-free median-threshold control.
  * eval: official DF CM keys, phase='eval', pooled EER (eval_protocol.py).

Labels are used ONLY for scoring, never in adaptation. The BBSE prior estimate
reads the ASVspoof2019-LA *train* split, which is the checkpoint's own labelled
training data and therefore legitimately available at deployment.

    PUBA_SMOKE=1 python protocol_a_public.py          # ~10 min sanity run
    python protocol_a_public.py                       # full
"""

import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

import public_ckpt_tta as P
from eval_protocol import DF_KEYS_DEFAULT, load_df_keys, score_official_df
from metrics import compute_eer

SMOKE = os.environ.get("PUBA_SMOKE", "0") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SR = 16000
BATCH = int(os.environ.get("PUBA_BATCH", "32"))
DECODE_WORKERS = 16

Q, LAMBDA_CONS, TTA_LR = 0.3, 0.3, 1e-4
TTA_EPOCHS = int(os.environ.get("PUBA_EPOCHS", "4"))

DF_PARTS = "data/dataset_1/ASVspoof2021_DF_eval_part0*/**/*.flac"
LA19_PROTO = "data/asvspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
LA19_FLAC = "data/asvspoof2019_LA/ASVspoof2019_LA_train/flac"

# PUBA_CORPUS selects the evaluation corpus:
#   df2021 -- the official ASVspoof2021-DF eval partition (97% spoof, the
#             protocol with published baselines)
#   itw    -- the complete In-the-Wild benchmark, all 31,779 clips (62.8% spoof)
# Both are run whole, not subsampled: the paper's existing third-party arm caps
# every pool at 6,000 clips, and a reviewer is entitled to the standard
# benchmark at its published size.
CORPUS = os.environ.get("PUBA_CORPUS", "df2021")

# Where each checkpoint's own labelled training data lives. BBSE needs the
# source confusion matrix M, which is measured on data the deployer legitimately
# has: the checkpoint's training corpus. A checkpoint whose training data is
# undocumented cannot get an honest M and is therefore excluded from the BBSE
# arm rather than fed a proxy.
BBSE_SOURCE = {
    "deepfense_w2v2_aasist_s2": "la19",
    "deepfense_w2v2_aasist_s42": "la19",
    "deepfense_w2v2_aasist_s240": "la19",
    "ssl_aasist_wavefake": "wavefake",
}

CKPTS = os.environ.get("PUBA_CKPTS",
                       "deepfense_w2v2_aasist_s2,deepfense_w2v2_aasist_s42,"
                       "deepfense_w2v2_aasist_s240").split(",")
ARMS = os.environ.get("PUBA_ARMS", "source,ours_fixed,ours_bbse").split(",")

if SMOKE:
    ADAPT_N, MAX_EVAL, BBSE_N = 400, 2000, 200
    TTA_EPOCHS = 1
    CKPTS = CKPTS[:1]
else:
    ADAPT_N = int(os.environ.get("PUBA_ADAPT_N", "20000"))
    MAX_EVAL, BBSE_N = None, 4000

SUFFIX = ("_smoke" if SMOKE else "") + ("" if CORPUS == "df2021" else f"_{CORPUS}")
RESULTS_CSV = f"results_protocol_a_public{SUFFIX}.csv"
SCORES_DIR = f"scores_protocol_a_public{SUFFIX}"
LOG_FILE = f"run_log_protocol_a_public{SUFFIX}.txt"
os.makedirs(SCORES_DIR, exist_ok=True)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def record(**row):
    pd.DataFrame([row]).to_csv(RESULTS_CSV, mode="a",
                               header=not os.path.exists(RESULTS_CSV), index=False)


def done_rows():
    if not os.path.exists(RESULTS_CSV):
        return set()
    d = pd.read_csv(RESULTS_CSV)
    return set(zip(d.ckpt, d.method, d.setting))


# ------------------------------------------------------------------ manifests
def build_df_eval():
    import glob
    keys = load_df_keys(DF_KEYS_DEFAULT)
    rows = []
    for p in glob.glob(DF_PARTS, recursive=True):
        utt = os.path.basename(p)[:-5]
        k = keys.get(utt)
        if k is not None and k[1] == "eval":
            rows.append((p, k[0], utt))
    df = pd.DataFrame(rows, columns=["path", "label", "utt"])
    return df.sort_values("utt").reset_index(drop=True)


def build_la19_train():
    rows = []
    for line in open(LA19_PROTO):
        p = line.split()
        if len(p) >= 5:
            rows.append((f"{LA19_FLAC}/{p[1]}.flac", 1 if p[-1] == "spoof" else 0))
    return pd.DataFrame(rows, columns=["path", "label"])


def build_wavefake():
    import glob
    rows = [(w, 0 if "/real/" in w else 1)
            for w in glob.glob("data/hf_wavefake/*/*.wav")]
    return pd.DataFrame(rows, columns=["path", "label"])


def build_itw():
    """The complete In-the-Wild benchmark: 31,779 clips, label from the path."""
    import glob
    rows = []
    for w in glob.glob("data/in_the_wild/**/*.wav", recursive=True):
        parts = w.split(os.sep)
        lab = 0 if "real" in parts else 1 if "fake" in parts else None
        if lab is not None:
            rows.append((w, lab, os.path.relpath(w, "data/in_the_wild")))
    df = pd.DataFrame(rows, columns=["path", "label", "utt"])
    return df.sort_values("utt").reset_index(drop=True)


def build_eval():
    if CORPUS == "df2021":
        return build_df_eval()
    if CORPUS == "itw":
        return build_itw()
    raise ValueError(f"unknown PUBA_CORPUS={CORPUS!r}")


def build_bbse_source(kind):
    if kind == "la19":
        df = build_la19_train()
    elif kind == "wavefake":
        df = build_wavefake()
    else:
        return None
    df = df[df.path.map(os.path.exists)].reset_index(drop=True)
    return df if len(df) else None


# ------------------------------------------------------------------ audio
def decode_batch(paths, crop):
    buf = torch.empty((len(paths), crop), dtype=torch.float16)
    with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as ex:
        for i, w in enumerate(ex.map(lambda p: P.load_clip(p, crop), paths)):
            buf[i] = torch.from_numpy(np.ascontiguousarray(w))
    return buf


@torch.no_grad()
def stream_score(model, paths, crop, fake_col, chunk=2048, tag=""):
    """Score a large manifest without caching it whole.

    Decode and GPU work overlap: measured in isolation this box decodes 533
    clips/s and scores 204 clips/s, so running them in series caps throughput at
    147 clips/s. One prefetch thread hides the decode entirely and the full
    400k-clip official eval drops from ~45 min to ~33 min per pass -- times the
    nine passes this study needs, that is four GPU-hours saved.
    """
    model.eval()
    out = np.zeros(len(paths), dtype=np.float32)
    t0 = time.time()
    starts = list(range(0, len(paths), chunk))
    pre = ThreadPoolExecutor(max_workers=1)
    nxt = pre.submit(decode_batch, paths[starts[0]:starts[0] + chunk], crop)
    for k, c0 in enumerate(starts):
        sl = slice(c0, min(c0 + chunk, len(paths)))
        buf = nxt.result()
        if k + 1 < len(starts):
            n0 = starts[k + 1]
            nxt = pre.submit(decode_batch, paths[n0:n0 + chunk], crop)
        got = []
        for i in range(0, len(buf), BATCH):
            x = buf[i:i + BATCH].to(DEVICE, non_blocking=True).float()
            with P.amp_ctx(DEVICE):
                logits = model(x)[0]
            got.append(torch.softmax(logits.float(), 1)[:, fake_col].cpu())
        out[sl] = torch.cat(got).numpy()
        del buf
        done = min(c0 + chunk, len(paths))
        if c0 == 0 or done % (chunk * 20) == 0 or done == len(paths):
            rate = done / max(time.time() - t0, 1e-9)
            log(f"    {tag}{done}/{len(paths)}  {rate:.0f} clips/s  "
                f"eta {(len(paths) - done) / max(rate, 1e-9) / 60:.1f} min")
    pre.shutdown(wait=True)
    return out


# ------------------------------------------------------------------ BBSE
def bbse_pi_fake(model, crop, fake_col, la19, target_scores, seed=0):
    """P(fake) on the target pool by black-box shift estimation.

    M[y, yhat] = P_source(yhat | y) measured on a class-balanced slice of the
    checkpoint's OWN labelled training corpus (ASVspoof2019-LA train); q is the
    target prediction histogram. p = M^{-T} q. Reads source error structure and
    target prediction rate only -- never the shape of the target score
    distribution, which is what miscalibration corrupts.
    """
    bal = pd.concat([g.sample(min(len(g), BBSE_N // 2), random_state=seed)
                     for _, g in la19.groupby("label")]).reset_index(drop=True)
    s_src = stream_score(model, bal.path.tolist(), crop, fake_col, tag="bbse-M ")
    yh = (s_src >= 0.5).astype(int)
    y = bal.label.values.astype(int)
    M = np.array([[np.mean(yh[y == 0] == 0), np.mean(yh[y == 0] == 1)],
                  [np.mean(yh[y == 1] == 0), np.mean(yh[y == 1] == 1)]])
    q = np.array([np.mean((target_scores >= 0.5) == 0),
                  np.mean((target_scores >= 0.5) == 1)])
    try:
        p = np.linalg.solve(M.T, q)
    except np.linalg.LinAlgError:
        return 0.5, M, q
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(p[1] / p.sum()), M, q


def tail_budget(q_base, pi_real):
    """Split a total confident budget 2*q_base by the estimated prior.

    Same rule as adaptive_tta.tail_budget: the pseudo-real bucket may not be
    larger than the estimated real fraction, so a 97%-spoof pool stops scooping
    up 30% of the pool as 'real'.
    """
    import adaptive_tta as AT
    return AT.tail_budget(q_base, pi_real)


# ------------------------------------------------------------------ adaptation
def adapt(model, buf, crop, fake_col, lo_frac, hi_frac, epochs=TTA_EPOCHS, tag=""):
    """Confident-tail pseudo-label self-training + channel consistency.

    lo_frac/hi_frac are the pseudo-real / pseudo-fake budgets. (Q, Q) reproduces
    the published fixed-q method exactly.
    """
    trainable, info = P.set_tta_params(model)
    n_bn = P.freeze_batchnorm(model)
    log(f"  {tag}trainable {info['n_trainable_params']:,} params; BN frozen {n_bn}; "
        f"budget=(lo {lo_frac:.4f}, hi {hi_frac:.4f})")
    opt = torch.optim.Adam(trainable, lr=TTA_LR)
    n = len(buf)
    for ep in range(epochs):
        # rescore the pool with the current model
        model.eval()
        s = []
        with torch.no_grad():
            for i in range(0, n, BATCH):
                x = buf[i:i + BATCH].to(DEVICE, non_blocking=True).float()
                with P.amp_ctx(DEVICE):
                    s.append(torch.softmax(model(x)[0].float(), 1)[:, fake_col].cpu())
        s = torch.cat(s).numpy()
        pl = torch.full((n,), -1, dtype=torch.long)
        if lo_frac > 0:
            pl[torch.from_numpy(s <= np.quantile(s, lo_frac))] = 0
        if hi_frac > 0:
            pl[torch.from_numpy(s >= np.quantile(s, 1 - hi_frac))] = 1
        model.train()
        P.freeze_batchnorm(model)
        order = torch.randperm(n)
        for i in range(0, n, BATCH):
            sel = order[i:i + BATCH]
            x = buf[sel].to(DEVICE, non_blocking=True).float()
            bpl = pl[sel].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with P.amp_ctx(DEVICE):
                logits = model(x)[0]
                # fake_col may be 0; map to a canonical [real, fake] ordering so
                # cross-entropy against pseudo-labels means the same thing on
                # every checkpoint.
                lg = logits if fake_col == 1 else logits.flip(1)
                p = torch.softmax(lg, 1)
                loss = torch.zeros((), device=DEVICE)
                conf = bpl >= 0
                if conf.any():
                    loss = loss + F.cross_entropy(lg[conf], bpl[conf])
                lg_a = P.augment(x)
                lga = model(lg_a)[0]
                lga = lga if fake_col == 1 else lga.flip(1)
                loss = loss + LAMBDA_CONS * F.mse_loss(torch.softmax(lga, 1), p.detach())
            if loss.requires_grad:
                loss.backward()
                opt.step()
        log(f"  {tag}epoch {ep+1}/{epochs} done "
            f"(pseudo-labelled {int((pl >= 0).sum())}/{n})")
    return model


# ------------------------------------------------------------------ reporting
def report(ckpt_name, method, setting, ev, scores, adapt_utts=None, extra=None):
    if setting == "disjoint_eval" and adapt_utts is not None:
        mask = ~ev.utt.isin(adapt_utts).values
        sub, sc = ev[mask], scores[mask]
    else:
        sub, sc = ev, scores
    y = sub.label.values.astype(int)
    eer, _ = compute_eer(y, sc)
    auc = float(roc_auc_score(y, sc))
    acc = float(accuracy_score(y, (sc >= 0.5).astype(int))) * 100
    # label-free median-threshold control: what a one-line rule would deliver
    acc_med = float(accuracy_score(y, (sc >= np.median(sc)).astype(int))) * 100
    row = dict(ckpt=ckpt_name, method=method, setting=setting,
               eer=round(eer * 100, 3), auc=round(auc, 4), acc=round(acc, 3),
               acc_median_rule=round(acc_med, 3), n=len(sub),
               attainable=round(100 - eer * 100, 3),
               deficit=round((100 - eer * 100) - acc, 3), smoke=SMOKE)
    if extra:
        row.update(extra)
    record(**row)
    log(f"  >>> {ckpt_name} {method} [{setting}] EER {row['eer']:.3f} "
        f"AUC {row['auc']:.4f} acc {row['acc']:.2f} "
        f"(median-rule {row['acc_median_rule']:.2f}, deficit {row['deficit']:.2f})")
    return row


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    log(f"=== published checkpoints on {CORPUS} | smoke={SMOKE} "
        f"| ckpts={CKPTS} | arms={ARMS} ===")
    ev = build_eval()
    log(f"[{CORPUS}] eval clips on disk: {len(ev)}  "
        f"spoof {ev.label.mean()*100:.2f}%")
    if MAX_EVAL:
        ev = ev.sample(MAX_EVAL, random_state=0).reset_index(drop=True)
        log(f"SMOKE: subsampled to {len(ev)}")
    bbse_pools = {}
    for kind in set(BBSE_SOURCE.values()):
        pool_df = build_bbse_source(kind)
        if pool_df is not None:
            bbse_pools[kind] = pool_df
            log(f"BBSE source pool '{kind}': {len(pool_df)} clips "
                f"({pool_df.label.value_counts().to_dict()})")

    rng = np.random.RandomState(0)
    adapt_idx = rng.choice(len(ev), min(ADAPT_N, len(ev)), replace=False)
    adapt_df = ev.iloc[np.sort(adapt_idx)].reset_index(drop=True)
    log(f"adapt pool (unlabeled) {len(adapt_df)} clips; "
        f"true spoof rate {adapt_df.label.mean()*100:.2f}% (NOT used)")

    already = done_rows()
    for name in CKPTS:
        cfg = P.CHECKPOINTS[name]
        crop, fake_col = cfg["crop"], cfg["fake_col"]
        log(f"--- {name} | {cfg['arch']} | crop {crop} | fake_col {fake_col} ---")
        base, nf, nb = P.build_model(name, DEVICE, log=log)
        log(f"  loaded {nf} front-end + {nb} back-end tensors")

        # source scores over the full official eval (cached to disk)
        spath = f"{SCORES_DIR}/{name}__source.npy"
        if os.path.exists(spath):
            s_src = np.load(spath)
            log(f"  reusing cached source scores {spath}")
        else:
            s_src = stream_score(base, ev.path.tolist(), crop, fake_col,
                                 tag=f"{name}/source ")
            np.save(spath, s_src)
        if (name, "source", "official_eval") not in already and "source" in ARMS:
            report(name, "source", "official_eval", ev, s_src)
            report(name, "source", "disjoint_eval", ev, s_src, set(adapt_df.utt))

        # decode the adapt pool once, reuse for every adaptation arm
        need_adapt = [a for a in ARMS if a != "source"
                      and (name, a, "official_eval") not in already]
        if not need_adapt:
            del base
            torch.cuda.empty_cache()
            continue
        log(f"  decoding adapt pool ({len(adapt_df)} clips) ...")
        abuf = decode_batch(adapt_df.path.tolist(), crop)
        s_adapt_src = s_src[adapt_df.index.values] if False else None

        for arm in need_adapt:
            model = copy.deepcopy(base)
            extra = {}
            if arm == "ours_fixed":
                lo, hi = Q, Q
            elif arm == "ours_bbse":
                src_kind = BBSE_SOURCE.get(name)
                if src_kind not in bbse_pools:
                    log(f"  [{arm}] no labelled training corpus on disk for "
                        f"{name} -- skipping (BBSE cannot be run honestly "
                        f"without the checkpoint's own source data)")
                    del model
                    torch.cuda.empty_cache()
                    continue
                # target prediction histogram from the FROZEN source model on
                # the adapt pool only (no labels, no eval-set leakage beyond the
                # unlabeled pool the method already sees).
                with torch.no_grad():
                    s_pool = []
                    model.eval()
                    for i in range(0, len(abuf), BATCH):
                        x = abuf[i:i + BATCH].to(DEVICE).float()
                        with P.amp_ctx(DEVICE):
                            s_pool.append(torch.softmax(
                                model(x)[0].float(), 1)[:, fake_col].cpu())
                    s_pool = torch.cat(s_pool).numpy()
                pi_fake, M, q = bbse_pi_fake(model, crop, fake_col,
                                             bbse_pools[src_kind], s_pool)
                lo, hi = tail_budget(Q, 1.0 - pi_fake)
                extra = dict(pi_fake_hat=round(pi_fake, 4),
                             pi_fake_true=round(float(adapt_df.label.mean()), 4),
                             lo_frac=round(lo, 4), hi_frac=round(hi, 4))
                log(f"  BBSE pi_fake_hat={pi_fake:.4f} (true "
                    f"{adapt_df.label.mean():.4f})  M={M.round(3).tolist()} "
                    f"q={q.round(3).tolist()}")
            else:
                raise ValueError(arm)
            log(f"  [{arm}] adapting ...")
            ta = time.time()
            model = adapt(model, abuf, crop, fake_col, lo, hi, tag=f"{arm} ")
            extra["adapt_min"] = round((time.time() - ta) / 60, 1)
            s = stream_score(model, ev.path.tolist(), crop, fake_col,
                             tag=f"{name}/{arm} ")
            np.save(f"{SCORES_DIR}/{name}__{arm}.npy", s)
            report(name, arm, "official_eval", ev, s, extra=extra)
            report(name, arm, "disjoint_eval", ev, s, set(adapt_df.utt), extra=extra)
            del model
            torch.cuda.empty_cache()
        del abuf, base
        torch.cuda.empty_cache()

    log(f"DONE in {(time.time()-t0)/60:.1f} min | peak GPU "
        f"{torch.cuda.max_memory_allocated()/1e9:.1f} GB")


if __name__ == "__main__":
    main()
