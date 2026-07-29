"""
Properly re-seeded TTA hyperparameter sweep on the real cloud checkpoints.

The cloud sweep.csv shares one continuous RNG stream across all sweep points, so
adjacent values differ partly by batch-order/augmentation noise, not just the
knob. Here every configuration gets `torch.manual_seed(RUN_SEED)` right before
adaptation starts, so differences are attributable to the knob alone (still one
seed per point -- for a real CI you'd repeat each point over seeds too, but this
already removes the confound that made the cloud sweep's absolute values
untrustworthy).
"""

import glob
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torchaudio, soundfile as sf
from sklearn.metrics import roc_curve, roc_auc_score

DEVICE = "cuda"
SR, CROP_SEC = 16000, 3.0
CROP_LEN = int(SR * CROP_SEC)
N_FINETUNE = 4
RUN_SEED = 0
BS = 16
TARGETS = ["in_the_wild", "arabic"]
GRID = {"q": [0.1, 0.2, 0.3, 0.4], "epochs": [2, 4, 8], "lam": [0.0, 0.3, 1.0]}
BASE = {"q": 0.3, "epochs": 4, "lam": 0.3}
OUT_CSV = "sweep_reseeded.csv"

torch.backends.cuda.matmul.allow_tf32 = True
USE_BF16 = torch.cuda.is_bf16_supported()


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


def batched(paths, bs=BS):
    for i in range(0, len(paths), bs):
        chunk = paths[i:i + bs]
        with ThreadPoolExecutor(max_workers=8) as ex:
            wavs = list(ex.map(load_audio, chunk))
        yield torch.tensor(np.stack(wavs), device=DEVICE)


@torch.no_grad()
def score(model, paths):
    model.eval()
    out = []
    for x in batched(paths):
        with amp():
            out.append(torch.softmax(model(x)[0].float(), 1)[:, 1])
    return torch.cat(out).cpu().numpy()


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2


def augment(x):
    return x * torch.empty(x.size(0), 1, device=DEVICE).uniform_(0.7, 1.3) + 0.005 * torch.randn_like(x)


def adapt(model, paths, q, epochs, lam, lr=1e-4):
    set_tta_params(model)
    opt = torch.optim.Adam(trainable(model), lr=lr)
    for _ in range(epochs):
        s = score(model, paths)
        lo, hi = np.quantile(s, q), np.quantile(s, 1 - q)
        pl = np.full(len(paths), -1)
        pl[s <= lo] = 0
        pl[s >= hi] = 1
        pl_t = torch.tensor(pl, device=DEVICE)

        model.train(); model.ssl.eval()
        order = np.random.permutation(len(paths))
        ptr = 0
        for x in batched([paths[i] for i in order]):
            bpl = pl_t[order[ptr:ptr + x.size(0)]]; ptr += x.size(0)
            opt.zero_grad(set_to_none=True)
            with amp():
                logits, _ = model(x)
                p = torch.softmax(logits, 1)
                loss = torch.zeros((), device=DEVICE)
                conf = bpl >= 0
                if conf.any():
                    loss = loss + F.cross_entropy(logits[conf], bpl[conf].long())
                if lam > 0:
                    loss = loss + lam * F.mse_loss(torch.softmax(model(augment(x))[0], 1), p.detach())
            if loss.requires_grad:
                loss.backward(); opt.step()
    return model


def load_fresh(target):
    ckpt = f"cloud_bundle/ckpt/source_{target}_seed0.pt"
    m = XLSRDetector().to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)
    return m


if __name__ == "__main__":
    rows = []
    for target in TARGETS:
        df = pd.read_csv(f"cloud_bundle/target_pool_{target}_seed0.csv")
        y = df.label.map({"real": 0, "fake": 1}).values
        paths = df.path.tolist()

        base_model = load_fresh(target)
        s0 = score(base_model, paths)
        base_eer, base_auc = eer(y, s0) * 100, roc_auc_score(y, s0)
        print(f"{target} source: EER {base_eer:.2f} AUC {base_auc:.3f}", flush=True)
        rows.append(dict(target=target, knob="source", q=None, epochs=None, lam=None, eer=base_eer, auc=base_auc))

        seen = set()
        for knob, values in GRID.items():
            for v in values:
                cfg = dict(BASE, **{knob: v})
                key = tuple(sorted(cfg.items()))
                if key in seen:
                    continue
                seen.add(key)
                torch.manual_seed(RUN_SEED); np.random.seed(RUN_SEED)
                model = load_fresh(target)
                model = adapt(model, paths, q=cfg["q"], epochs=cfg["epochs"], lam=cfg["lam"])
                s = score(model, paths)
                e, a = eer(y, s) * 100, roc_auc_score(y, s)
                print(f"  {target} {knob}={v} (q={cfg['q']},ep={cfg['epochs']},lam={cfg['lam']}) "
                      f"-> EER {e:.2f} AUC {a:.3f}", flush=True)
                rows.append(dict(target=target, knob=knob, q=cfg["q"], epochs=cfg["epochs"], lam=cfg["lam"],
                                 eer=round(e, 3), auc=round(a, 4)))
                pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
                del model; torch.cuda.empty_cache()

    print("done ->", OUT_CSV)
