"""
DA-RAD trainer — config-driven, supports the full objective and its ablations.

Loss:  L = CE(real/fake) + lambda_grl(t) * L_domain_adv + lambda_anchor * L_anchor
  * lambda_grl ramps 0 -> lambda_grl over training (standard DANN schedule; GRL is
    unstable at full strength from step 0).
  * Set --lambda_grl 0 and/or --lambda_anchor 0 and --no-rawboost to run the
    ablation rows (XLS-R, +RawBoost, +GRL, +Anchor=DA-RAD).

Encoder: --encoder xlsr (frozen XLS-R-300M, downloads weights) or stub (tests).
Frozen SSL by default so only the head/losses train on a 10 GB GPU; AMP + grad-accum.

Run (real):   python trainer.py --train_csv splits/train.csv --val_csv splits/val.csv --encoder xlsr
Run (smoke):  python trainer.py --train_csv <tiny> --val_csv <tiny> --encoder stub --epochs 1
"""

import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from darad_dataset import RawAudioDataset
from losses import DomainAdversarialLoss, RealAnchoredContrastiveLoss
from metrics import evaluation_report
from models.dg_model import DARADModel
from models.ssl_frontend import SSLFrontend, StubEncoder
from sampler import DomainBalancedBatchSampler

LABEL_FAKE = 1


def grl_lambda(step, total, max_lambda):
    """DANN schedule: 0 -> max_lambda following 2/(1+e^{-10p}) - 1."""
    p = step / max(1, total)
    return max_lambda * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0)


def build_encoder(name, n_trainable_layers=0, out_dim=256):
    if name == "stub":
        return StubEncoder(out_dim=out_dim)
    if name == "xlsr":
        return SSLFrontend("WAV2VEC2_XLSR_300M", n_trainable_layers=n_trainable_layers)
    raise ValueError(name)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, scores = [], []
    for wav, y, _ in loader:
        logits, _ = model(wav.to(device))
        p_fake = torch.softmax(logits, dim=1)[:, LABEL_FAKE]
        scores.append(p_fake.float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--encoder", default="xlsr", choices=["xlsr", "stub"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--domains_per_batch", type=int, default=2)
    ap.add_argument("--n_trainable_layers", type=int, default=0,
                    help="fine-tune the top K XLS-R transformer layers (0 = frozen)")
    ap.add_argument("--lambda_grl", type=float, default=0.5)
    ap.add_argument("--lambda_anchor", type=float, default=0.3)
    ap.add_argument("--anchor_margin", type=float, default=1.0)
    ap.add_argument("--rawboost", dest="rawboost", action="store_true", default=True)
    ap.add_argument("--no-rawboost", dest="rawboost", action="store_false")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="darad.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | encoder={args.encoder} | rawboost={args.rawboost} "
          f"| lambda_grl={args.lambda_grl} lambda_anchor={args.lambda_anchor}")

    # datasets (shared domain map so domain ids are consistent train/val)
    train_ds = RawAudioDataset(args.train_csv, duration_sec=args.duration,
                               random_crop=True, rawboost=args.rawboost)
    val_ds = RawAudioDataset(args.val_csv, duration_sec=args.duration,
                             random_crop=False, rawboost=False,
                             domain_map=train_ds.domain_map)
    n_domains = len(train_ds.domain_map)

    train_sampler = DomainBalancedBatchSampler(
        train_ds.domains, train_ds.labels, batch_size=args.batch_size,
        domains_per_batch=min(args.domains_per_batch, n_domains), seed=args.seed)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # model + losses
    model = DARADModel(build_encoder(args.encoder, args.n_trainable_layers), embed_dim=256).to(device)
    # materialise the lazily-loaded encoder now so its (fine-tuned) params exist
    # before the optimizer is built.
    with torch.no_grad():
        model(torch.zeros(1, int(args.duration * 16000), device=device))
    dom_loss = DomainAdversarialLoss(embed_dim=256, n_domains=n_domains).to(device)
    anchor_loss = RealAnchoredContrastiveLoss(margin=args.anchor_margin)

    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train_params:,} (n_trainable_layers={args.n_trainable_layers})")
    params = [p for p in model.parameters() if p.requires_grad] + list(dom_loss.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    # bf16 autocast (no GradScaler): fp16 scaling overflows and skips every step
    # once the encoder is fine-tuned (large grads); bf16 has fp32 range.
    use_amp = args.amp and device == "cuda"
    amp_dtype = torch.bfloat16

    total_steps = args.epochs * len(train_loader)
    best_eer, step = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for i, (wav, y, dom) in enumerate(pbar):
            wav, y, dom = wav.to(device), y.to(device), dom.to(device)
            lam = grl_lambda(step, total_steps, args.lambda_grl)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits, emb = model(wav)
                loss = F.cross_entropy(logits, y)
                if args.lambda_grl > 0 and n_domains > 1:
                    loss = loss + dom_loss(emb, dom, lambd=lam)
                if args.lambda_anchor > 0:
                    loss = loss + args.lambda_anchor * anchor_loss(emb, y, dom)
                loss = loss / args.grad_accum
            loss.backward()
            if (i + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
            step += 1
            pbar.set_postfix(loss=f"{loss.item()*args.grad_accum:.3f}", grl_lam=f"{lam:.2f}")

        y_val, s_val = evaluate(model, val_loader, device)
        rep = evaluation_report(y_val, s_val, verbose=False)
        print(f"  val: EER {rep['eer']*100:.2f}%  bal-acc {rep['balanced_accuracy']*100:.2f}%  AUC {rep['auc']:.4f}")
        if rep["eer"] < best_eer:
            best_eer = rep["eer"]
            # save only trainable params (skip the ~1.2GB frozen XLS-R encoder)
            trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
            torch.save(trainable, args.out)
            print(f"  -> saved best (val EER {best_eer*100:.2f}%)")

    print(f"\nBest val EER: {best_eer*100:.2f}%  (checkpoint: {args.out})")


if __name__ == "__main__":
    main()
