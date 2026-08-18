"""
Apply our published test-time adaptation to a PUBLIC, third-party detector.

Why this exists (see HANDOFF_PUBLIC_CKPT.md): every number in the paper so far
compares our source model against our baselines on our protocol. This script
runs the *unmodified published TTA config* on somebody else's released weights,
on a standard benchmark (In-the-Wild), so the contribution stops depending on
our own baseline being the only comparator.

Labels are used ONLY for scoring. Never for adaptation.

    python public_ckpt_tta.py --mode signcheck --limit 200 --device cpu
    python public_ckpt_tta.py --mode source
    python public_ckpt_tta.py --mode ours --seed 0

Design notes that are easy to get wrong, and are asserted rather than assumed:

* **Score polarity.** This repo is `[real=0, fake=1]`, so P(fake) is column 1.
  Public anti-spoofing repos overwhelmingly use `[spoof=0, bonafide=1]`, making
  column 1 *real*. Getting this backwards yields `100 - x` EER with no error.
  Each checkpoint declares `fake_col`, and `--mode signcheck` verifies it
  empirically against known labels before any result is believed.
* **Crop length.** SSL-AASIST is trained at 64,600 samples (~4.04 s) and its
  own inference snippet pads by tile-repeat. We match that exactly; using our
  3 s convention would handicap the source model and invalidate the comparison.
* **BatchNorm.** The AASIST back-end contains BatchNorm2d. Calling `model.train()`
  during adaptation would silently update BN running statistics on target audio
  -- that is AdaBN, a *different* adaptation mechanism, and it would confound the
  measurement of ours. We force all BN modules to eval() so the only thing that
  adapts is what our method says adapts. (AdaBN remains available separately as
  the `bn_only` baseline.)
* **Parameter selection.** `tta.py`'s `select_tta_params` hardcodes our own
  module path and would select nothing on a foreign tree -- a silent no-op that
  looks like "adaptation didn't help". Here the transformer stack is discovered
  generically and a non-zero tensor count is asserted.
"""

import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

from metrics import compute_eer

# ---------------------------------------------------------------- published config
# Exactly the config behind every `ours` row in results_ext.csv. Do not improvise.
Q = 0.3              # confident top/bottom quantile -> pseudo-labels
LAMBDA_CONS = 0.3    # consistency weight
TTA_LR = 1e-4
TTA_EPOCHS = 4
N_FINETUNE = 4       # top-4 transformer blocks' LayerNorms
USE_ANCHOR = False   # PROJECT_LOG.md S4: anchor collapses the model to chance

SR = 16000

CHECKPOINTS = {
    "ssl_aasist_wavefake": dict(
        kind="ssl_aasist",
        path="public_ckpt/ssl_aasist/pytorch_model.bin",
        # ash56/ssl-aasist scores batch_out[:, 1] as the BONAFIDE score
        # (its calculate_EER treats bonafide as the target class), so P(fake)
        # is column 0 under our convention.
        fake_col=0,
        crop=64600,
        source="ash56/ssl-aasist (Garg et al. 2025, arXiv:2502.05674)",
        arch="SSL-AASIST (XLS-R 300M + AASIST)",
        train_data="WaveFake / LJSpeech, HiFiGAN vocoder",
    ),
}

# DeepFense released w2v2+AASIST trained on ASVspoof2019 at three seeds. Same
# fairseq XLS-R front-end format as above, so the same port applies; only the
# state-dict prefixes and the crop length differ (their config.yaml pads to
# 64,000). Their label_map is `bonafide: 1, spoof: 0`, i.e. the same flip --
# still verified empirically per checkpoint by `--mode signcheck`.
for _seed in (2, 42, 240):
    CHECKPOINTS[f"deepfense_w2v2_aasist_s{_seed}"] = dict(
        kind="deepfense",
        path=f"public_ckpt/deepfense_w2v2_aasist_s{_seed}/best_model.pth",
        fake_col=0,
        crop=64000,
        expect_seed=_seed,
        source=f"DeepFense/ASV19_Wav2Vec2_AASIST_NoAug_Seed{_seed}",
        arch="w2v2(XLS-R 300M) + AASIST",
        train_data="ASVspoof2019 LA train",
    )


# ------------------------------------------------------------------------ audio
def load_clip(path, crop):
    """Match upstream's pad(): first `crop` samples, tile-repeat if too short."""
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {sr}")
    if len(x) == 0:
        return np.zeros(crop, dtype=np.float32)
    if len(x) >= crop:
        return x[:crop]
    reps = int(crop / len(x)) + 1
    return np.tile(x, reps)[:crop].astype(np.float32)


def build_cache(df, crop, log=print):
    """Decode the whole pool once into a pinned fp16 CPU tensor.

    31,779 clips x 64,600 samples x 2 bytes = ~4.1 GB, which does not fit on a
    10 GB GPU alongside a 316 M-param model and a doubled forward pass, so the
    cache lives on the host and batches are moved per step.
    """
    n = len(df)
    buf = torch.empty((n, crop), dtype=torch.float16)
    t0 = time.time()
    for i, p in enumerate(df.path.values):
        buf[i] = torch.from_numpy(load_clip(p, crop))
        if (i + 1) % 5000 == 0:
            log(f"  decoded {i+1}/{n} ({time.time()-t0:.0f}s)")
    log(f"  decoded {n}/{n} in {time.time()-t0:.0f}s")
    return buf, torch.from_numpy(df.label.values.astype(np.int64))


# ------------------------------------------------------------------------ model
def _split_state_dict(cfg):
    """Return (frontend_sd, backend_sd) in this repo's module naming.

    Both supported checkpoint families wrap the *same* fairseq XLS-R front-end;
    they differ only in prefixes and in where the 2-way classifier lives.
    """
    if cfg["kind"] == "ssl_aasist":
        sd = torch.load(cfg["path"], map_location="cpu", weights_only=True)
        front = {k[len("ssl_model.model."):]: v for k, v in sd.items()
                 if k.startswith("ssl_model.model.")}
        back = {k: v for k, v in sd.items() if not k.startswith("ssl_model.")}
        return front, back

    if cfg["kind"] == "deepfense":
        obj = torch.load(cfg["path"], map_location="cpu", weights_only=False)
        sd = obj["model_state"]
        front = {k[len("frontend.model."):]: v for k, v in sd.items()
                 if k.startswith("frontend.model.")}
        back = {}
        for k, v in sd.items():
            if k.startswith("backend."):
                back[k[len("backend."):]] = v
            elif k.startswith("losses.0.fc."):
                # their classifier head is carried on the loss module
                back["out_layer." + k[len("losses.0.fc."):]] = v
        return front, back

    raise ValueError(f"unsupported kind {cfg['kind']}")


def verify_checkpoint_identity(name, cfg, log=print):
    """Confirm the file on disk is the checkpoint the directory name claims.

    A download-script quoting bug once fetched Seed240 into both the s2 and s240
    directories; the runs succeeded and produced plausible numbers under the
    wrong label. The upstream `config.yaml` records the training seed, so check
    it rather than trusting a path.
    """
    want = cfg.get("expect_seed")
    if want is None:
        return
    side = os.path.join(os.path.dirname(cfg["path"]), "config.yaml")
    if not os.path.exists(side):
        log(f"  WARNING: no config.yaml beside {cfg['path']}; identity unverified")
        return
    m = re.search(r"^seed:\s*(\d+)\s*$", open(side).read(), re.M)
    if not m:
        log(f"  WARNING: no seed in {side}; identity unverified")
        return
    got = int(m.group(1))
    if got != want:
        raise RuntimeError(
            f"checkpoint identity mismatch for {name}: {side} says seed {got}, "
            f"expected {want}. The file is not what its directory claims.")
    log(f"  identity ok: {name} is training seed {got}")


def build_model(name, device, log=print):
    cfg = CHECKPOINTS[name]
    verify_checkpoint_identity(name, cfg, log)
    import aasist_backend as A

    model = A.Model()
    ssl, back = _split_state_dict(cfg)

    n_front = model.ssl_model.load_fairseq_weights(ssl)      # strict
    res = model.load_state_dict(back, strict=False)
    missing = [k for k in res.missing_keys if not k.startswith("ssl_model.")]
    if missing or res.unexpected_keys:
        raise RuntimeError(f"backend load failed: missing={missing[:5]} "
                           f"unexpected={res.unexpected_keys[:5]}")
    return model.to(device).eval(), n_front, len(back)


def find_transformer_layers(model):
    """Locate the transformer block stack generically.

    Returns the largest nn.ModuleList whose children each contain at least one
    nn.LayerNorm -- i.e. the encoder block stack -- without depending on any
    particular attribute path (trap 2 in the handoff).
    """
    best = None
    for mod in model.modules():
        if not isinstance(mod, nn.ModuleList) or len(mod) == 0:
            continue
        if all(any(isinstance(m, nn.LayerNorm) for m in blk.modules()) for blk in mod):
            if best is None or len(mod) > len(best):
                best = mod
    if best is None:
        raise RuntimeError("no transformer block stack found")
    return best


def set_tta_params(model, n_finetune=N_FINETUNE, head_names=("out_layer",)):
    """Freeze everything, then unfreeze top-N blocks' LayerNorms + classifier."""
    for p in model.parameters():
        p.requires_grad_(False)

    layers = find_transformer_layers(model)
    n_ln = 0
    for blk in layers[-n_finetune:]:
        for m in blk.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad_(True)
                    n_ln += 1

    n_head = 0
    for hn in head_names:
        head = getattr(model, hn, None)
        if head is not None:
            for p in head.parameters():
                p.requires_grad_(True)
                n_head += 1

    trainable = [p for p in model.parameters() if p.requires_grad]
    # A silent empty selection makes TTA a no-op that reads as "it didn't help".
    assert n_ln > 0, "no LayerNorm parameters selected"
    assert len(trainable) > 0, "empty trainable set"
    return trainable, dict(n_blocks=len(layers), n_ln_tensors=n_ln,
                           n_head_tensors=n_head,
                           n_trainable_tensors=len(trainable),
                           n_trainable_params=sum(p.numel() for p in trainable))


def freeze_batchnorm(model):
    """Keep BN in eval so running stats never adapt (see module docstring)."""
    n = 0
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            n += 1
    return n


# ----------------------------------------------------------------- score / adapt
def augment(x):
    """Channel perturbation, identical to the published pipeline."""
    return x * torch.empty(x.size(0), 1, device=x.device).uniform_(0.7, 1.3) \
        + 0.005 * torch.randn_like(x)


def amp_ctx(device):
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return torch.amp.autocast("cuda", enabled=False)


@torch.no_grad()
def score(model, buf, idx, fake_col, batch, device):
    model.eval()
    out = []
    for i in range(0, len(idx), batch):
        sel = idx[i:i + batch]
        x = buf[sel].to(device, non_blocking=True).float()
        with amp_ctx(device):
            logits = model(x)[0]
        out.append(torch.softmax(logits.float(), 1)[:, fake_col].cpu())
    return torch.cat(out).numpy()


def evaluate(y, s):
    eer, _ = compute_eer(y, s)
    return dict(eer=eer * 100, auc=float(roc_auc_score(y, s)),
                acc=float(accuracy_score(y, (s >= 0.5).astype(int))) * 100)


def adapt(model, buf, idx, fake_col, batch, device, use_st=True, use_cons=True,
          epochs=TTA_EPOCHS, log=print):
    """Published TTA: self-training on confident tails + channel consistency.

    Unlabeled. `idx` indexes the adaptation pool; labels are never read here.
    """
    trainable, info = set_tta_params(model)
    log(f"  trainable: {info['n_trainable_tensors']} tensors / "
        f"{info['n_trainable_params']:,} params "
        f"(LN {info['n_ln_tensors']}, head {info['n_head_tensors']}, "
        f"{info['n_blocks']} blocks)")
    opt = torch.optim.Adam(trainable, lr=TTA_LR)

    for ep in range(epochs):
        s = score(model, buf, idx, fake_col, batch, device)
        lo, hi = np.quantile(s, Q), np.quantile(s, 1 - Q)
        pl = np.full(len(idx), -1, dtype=np.int64)
        pl[s <= lo] = 0                      # confident real
        pl[s >= hi] = 1                      # confident fake
        pl_t = torch.from_numpy(pl)

        model.train()
        freeze_batchnorm(model)              # no AdaBN confound
        order = torch.randperm(len(idx))
        tot = cnt = 0.0
        for i in range(0, len(order), batch):
            sel = order[i:i + batch]
            x = buf[idx[sel]].to(device, non_blocking=True).float()
            bpl = pl_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            with amp_ctx(device):
                logits = model(x)[0]
                p = torch.softmax(logits, 1)
                loss = torch.zeros((), device=device)
                conf = bpl >= 0
                if use_st and conf.any():
                    # pseudo-labels are in OUR convention (0=real,1=fake); map to
                    # the checkpoint's column order before the CE.
                    tgt = bpl[conf] if fake_col == 1 else (1 - bpl[conf])
                    loss = loss + F.cross_entropy(logits[conf], tgt)
                if use_cons:
                    loss = loss + LAMBDA_CONS * F.mse_loss(
                        torch.softmax(model(augment(x))[0], 1), p.detach())
            if loss.requires_grad:
                lv = float(loss.detach())
                loss.backward()
                opt.step()
                tot += lv * len(sel)
                cnt += len(sel)
        log(f"  epoch {ep+1}/{epochs} loss {tot/max(cnt,1):.4f} "
            f"(pseudo-labelled {int((pl>=0).sum())}/{len(idx)})")
    return model, info


def tent(model, buf, idx, batch, device, epochs=TTA_EPOCHS, log=print):
    """Tent (Wang et al. 2021): entropy minimisation on the same parameter set.

    Reported because PROJECT_LOG.md S5 / main.tex record that naive
    entropy-minimisation collapses to ~chance on this task, and it is the
    self-training + consistency structure that stabilises adaptation. Running it
    on a third-party checkpoint tests whether that failure mode is a property of
    our source model or of the problem.
    """
    trainable, info = set_tta_params(model)
    log(f"  trainable: {info['n_trainable_tensors']} tensors / "
        f"{info['n_trainable_params']:,} params")
    opt = torch.optim.Adam(trainable, lr=TTA_LR)
    for ep in range(epochs):
        model.train()
        freeze_batchnorm(model)
        order = torch.randperm(len(idx))
        tot = cnt = 0.0
        for i in range(0, len(order), batch):
            sel = order[i:i + batch]
            x = buf[idx[sel]].to(device, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            with amp_ctx(device):
                p = torch.softmax(model(x)[0], 1)
                loss = -(p * torch.log(p + 1e-8)).sum(1).mean()
            lv = float(loss.detach())
            loss.backward()
            opt.step()
            tot += lv * len(sel)
            cnt += len(sel)
        log(f"  tent epoch {ep+1}/{epochs} entropy {tot/max(cnt,1):.4f}")
    return model, info


# ------------------------------------------------------------------------ modes
def mode_signcheck(args, df, cfg, device, log):
    """Verify score polarity empirically before any number is believed."""
    n = args.limit // 2
    sub = pd.concat([df[df.label == 0].head(n), df[df.label == 1].head(n)])
    sub = sub.reset_index(drop=True)
    log(f"sign check on {len(sub)} clips ({(sub.label==0).sum()} real, "
        f"{(sub.label==1).sum()} fake) on {device}")

    model, n_front, n_back = build_model(args.ckpt, device)
    log(f"loaded {args.ckpt}: {n_front} frontend + {n_back} backend tensors")
    buf, y = build_cache(sub, cfg["crop"], log)
    idx = torch.arange(len(sub))

    for col in (0, 1):
        s = score(model, buf, idx, col, args.batch, device)
        m = evaluate(y.numpy(), s)
        mean_real, mean_fake = s[y.numpy() == 0].mean(), s[y.numpy() == 1].mean()
        flag = "  <-- declared fake_col" if col == cfg["fake_col"] else ""
        log(f"  col{col}: EER {m['eer']:5.2f}%  AUC {m['auc']:.4f}  "
            f"mean(real) {mean_real:.3f}  mean(fake) {mean_fake:.3f}{flag}")

    s = score(model, buf, idx, cfg["fake_col"], args.batch, device)
    yn = y.numpy()
    ok = s[yn == 1].mean() > s[yn == 0].mean()
    m = evaluate(yn, s)
    log("")
    if ok and m["eer"] < 50:
        log(f"PASS: with fake_col={cfg['fake_col']}, fakes score higher than "
            f"reals and EER {m['eer']:.2f}% < 50%")
    else:
        log(f"FAIL: polarity is wrong (EER {m['eer']:.2f}%). "
            f"Flip fake_col before trusting anything downstream.")
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["signcheck", "source", "ours", "st_only", "cons_only", "tent"])
    ap.add_argument("--ckpt", default="ssl_aasist_wavefake")
    ap.add_argument("--manifest", default="manifest_itw.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="subsample the pool (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inductive", action="store_true",
                    help="adapt on half the pool, evaluate on the disjoint half")
    ap.add_argument("--out", default="results_public_ckpt.csv")
    ap.add_argument("--log", default="run_log_public_ckpt.txt")
    args = ap.parse_args()

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(args.log, "a") as f:
            f.write(line + "\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = CHECKPOINTS[args.ckpt]
    df = pd.read_csv(args.manifest)
    log(f"=== {args.mode} | {args.ckpt} | seed {args.seed} | {len(df)} clips ===")
    log(f"    {cfg['arch']}, trained on {cfg['train_data']}")

    if args.mode == "signcheck":
        args.limit = args.limit or 200
        return mode_signcheck(args, df, cfg, args.device, log)

    if args.limit:
        # class-balanced subsample; avoid groupby.apply, whose handling of the
        # grouping column changes across pandas versions
        parts = [g.sample(min(len(g), args.limit // 2), random_state=args.seed)
                 for _, g in df.groupby("label")]
        df = pd.concat(parts).reset_index(drop=True)
        log(f"subsampled to {len(df)} clips")

    t0 = time.time()
    model, n_front, n_back = build_model(args.ckpt, args.device)
    buf, y = build_cache(df, cfg["crop"], log)
    yn = y.numpy()
    all_idx = torch.arange(len(df))

    if args.inductive:
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(df), generator=g)
        adapt_idx, eval_idx = perm[:len(df) // 2], perm[len(df) // 2:]
    else:
        adapt_idx = eval_idx = all_idx

    setting = "inductive" if args.inductive else "transductive"

    s = score(model, buf, eval_idx, cfg["fake_col"], args.batch, args.device)
    base = evaluate(yn[eval_idx.numpy()], s)
    log(f"source: EER {base['eer']:.2f}%  AUC {base['auc']:.4f}  acc {base['acc']:.2f}%")

    # the source row is measured on the same eval set as the adapted row, so it
    # carries the same setting label -- in inductive mode that is the held-out half
    rows = [dict(seed=args.seed, target="in_the_wild", method="source",
                 setting=setting, family=args.ckpt, **base,
                 n=len(eval_idx), minutes=(time.time() - t0) / 60,
                 trainable_params=0, crop=cfg["crop"], batch=args.batch)]

    if args.mode != "source":
        if args.mode == "tent":
            log(f"adapting with Tent (entropy minimisation), lr={TTA_LR}, E={TTA_EPOCHS}")
            model, info = tent(model, buf, adapt_idx, args.batch, args.device, log=log)
        else:
            use_st = args.mode in ("ours", "st_only")
            use_cons = args.mode in ("ours", "cons_only")
            log(f"adapting: self-training={use_st} consistency={use_cons} "
                f"anchor={USE_ANCHOR} (Q={Q}, lambda={LAMBDA_CONS}, lr={TTA_LR}, E={TTA_EPOCHS})")
            model, info = adapt(model, buf, adapt_idx, cfg["fake_col"], args.batch,
                                args.device, use_st=use_st, use_cons=use_cons, log=log)
        s = score(model, buf, eval_idx, cfg["fake_col"], args.batch, args.device)
        adp = evaluate(yn[eval_idx.numpy()], s)
        log(f"{args.mode} ({setting}): EER {adp['eer']:.2f}%  AUC {adp['auc']:.4f}  "
            f"acc {adp['acc']:.2f}%   [delta EER {adp['eer']-base['eer']:+.2f}]")
        rows.append(dict(seed=args.seed, target="in_the_wild", method=args.mode,
                         setting=setting, family=args.ckpt, **adp,
                         n=len(eval_idx), minutes=(time.time() - t0) / 60,
                         trainable_params=info["n_trainable_params"],
                         crop=cfg["crop"], batch=args.batch))

    # Fixed column order on every append. Runs differ in which fields they
    # produce, and a header written once by the first run would otherwise
    # silently misalign every later row.
    cols = ["seed", "target", "method", "setting", "family", "eer", "auc", "acc",
            "n", "minutes", "trainable_params", "crop", "batch"]
    pd.DataFrame(rows).reindex(columns=cols).to_csv(
        args.out, mode="a", index=False, header=not os.path.exists(args.out))
    log(f"appended {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
