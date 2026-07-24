"""
Test-Time Adaptation for cross-corpus deepfake detection + validation harness.

Source model = fine-tuned XLS-R (protocols/ft_xlsr.pt), adapted UNSUPERVISED to a
target corpus (labels used only for EER/AUC). Adapts LayerNorm affine params of the
top-4 layers + head.

Real-Manifold TTA = self-train on confident-RANKED pseudo-labels + pull the
pseudo-real cluster toward a real prototype + channel-consistency, with anti-collapse
(confident tails only, LN-only params, pseudo-fake push). Each component is
toggleable for ablation, and adapt/eval CSVs can differ (inductive check).

  python tta.py --mode source
  python tta.py --mode tent
  python tta.py --mode realmanifold [--no-anchor] [--no-consistency] [--no-selftrain]
                 [--seed S] [--adapt_csv A.csv] [--eval_csv B.csv] [--tag NAME]
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from darad_dataset import RawAudioDataset
from metrics import compute_eer
from models.dg_model import DARADModel
from models.ssl_frontend import SSLFrontend

DEV = "cuda"
CKPT = "protocols/ft_xlsr.pt"
TARGET = "protocols/protoB_itw_test.csv"


def build_source():
    m = DARADModel(SSLFrontend("WAV2VEC2_XLSR_300M", n_trainable_layers=4), embed_dim=256).to(DEV)
    m(torch.zeros(1, 48000, device=DEV))
    m.load_state_dict(torch.load(CKPT, map_location=DEV), strict=False)
    return m


def select_tta_params(model):
    for p in model.parameters():
        p.requires_grad_(False)
    top = model.encoder.model.model.encoder.transformer.layers[-4:]
    for mod in top.modules():
        if isinstance(mod, nn.LayerNorm):
            for p in mod.parameters():
                p.requires_grad_(True)
    for head in (model.proj, model.classifier, model.pool):
        for p in head.parameters():
            p.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def score_all(model, loader):
    model.eval()
    ys, ss, embs = [], [], []
    for wav, y, _ in loader:
        logits, emb = model(wav.to(DEV))
        ss.append(torch.softmax(logits, 1)[:, 1].float().cpu().numpy())
        embs.append(F.normalize(emb, dim=1).float().cpu())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ss), torch.cat(embs)


def report(tag, y, s):
    eer, _ = compute_eer(y, s)
    auc = roc_auc_score(y, s)
    print(f"[{tag}] EER {eer*100:.2f}%   AUC {auc:.4f}")
    return eer * 100, auc


def _augment(wav):
    gain = torch.empty(wav.size(0), 1, device=wav.device).uniform_(0.7, 1.3)
    return wav * gain + 0.005 * torch.randn_like(wav)


def adapt(model, loader, mode, use_st, use_anchor, use_cons, epochs=4, lr=1e-4, q=0.3, margin=1.0):
    opt = torch.optim.Adam(select_tta_params(model), lr=lr)
    n = len(loader.dataset)
    for _ in range(epochs):
        _, scores, embs = score_all(model, loader)
        pl = np.full(n, -1)
        mu = None
        if mode == "realmanifold":
            lo, hi = np.quantile(scores, q), np.quantile(scores, 1 - q)
            pl[scores <= lo] = 0
            pl[scores >= hi] = 1
            mu = F.normalize(embs[pl == 0].mean(0), dim=0).to(DEV)
        pl_t = torch.tensor(pl)

        model.train()
        ptr = 0
        for wav, _, _ in loader:
            bs = wav.size(0); wav = wav.to(DEV)
            bpl = pl_t[ptr:ptr + bs].to(DEV); ptr += bs
            opt.zero_grad()
            logits, emb = model(wav)
            p = torch.softmax(logits, 1)
            if mode == "tent":
                loss = -(p * torch.log(p + 1e-8)).sum(1).mean()
            else:
                emn = F.normalize(emb, dim=1)
                real, fake, conf = bpl == 0, bpl == 1, bpl >= 0
                loss = torch.zeros((), device=DEV)
                if use_st and conf.any():
                    loss = loss + F.cross_entropy(logits[conf], bpl[conf].long())
                if use_anchor and real.any():
                    loss = loss + 0.3 * (1 - (emn[real] * mu).sum(1)).mean()
                if use_anchor and fake.any():
                    loss = loss + 0.3 * F.relu(margin - (emn[fake] - mu).norm(dim=1)).mean()
                if use_cons:
                    p_aug = torch.softmax(model(_augment(wav))[0], 1)
                    loss = loss + 0.3 * F.mse_loss(p_aug, p.detach())
            if loss.requires_grad:
                loss.backward(); opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["source", "tent", "realmanifold"])
    ap.add_argument("--no-selftrain", dest="st", action="store_false", default=True)
    ap.add_argument("--no-anchor", dest="anchor", action="store_false", default=True)
    ap.add_argument("--no-consistency", dest="cons", action="store_false", default=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adapt_csv", default=TARGET)
    ap.add_argument("--eval_csv", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    eval_csv = args.eval_csv or args.adapt_csv
    tag = args.tag or args.mode

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    adapt_loader = DataLoader(RawAudioDataset(args.adapt_csv, duration_sec=3.0, random_crop=False),
                              batch_size=16, shuffle=False, num_workers=args.workers, pin_memory=True)
    eval_loader = (adapt_loader if eval_csv == args.adapt_csv else
                   DataLoader(RawAudioDataset(eval_csv, duration_sec=3.0, random_crop=False),
                              batch_size=16, shuffle=False, num_workers=args.workers, pin_memory=True))

    model = build_source()
    y, s, _ = score_all(model, eval_loader)
    b_eer, b_auc = report(f"{tag} source", y, s)

    if args.mode != "source":
        adapt(model, adapt_loader, args.mode, args.st, args.anchor, args.cons, epochs=args.epochs, lr=args.lr)
        y, s, _ = score_all(model, eval_loader)
        a_eer, a_auc = report(f"{tag} adapted (seed {args.seed})", y, s)
        with open("tta_results.txt", "a") as f:
            f.write(f"{tag}\tseed{args.seed}\tEER {a_eer:.2f}\tAUC {a_auc:.4f}\t"
                    f"(source EER {b_eer:.2f}/AUC {b_auc:.4f})\n")


if __name__ == "__main__":
    main()
