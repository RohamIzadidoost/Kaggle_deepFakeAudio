"""
Explainability for AttentiveSpecCNN: show WHY a clip was called real or fake.

Two complementary saliency views, saved as a figure per clip:
  * Temporal attention — the model's own attention-pooling weights, i.e. which
    moments of audio it focused on (a direct read-out of the architecture).
  * Grad-CAM — gradient-weighted activation of the last conv layer, highlighting
    the time-FREQUENCY regions that pushed the decision toward the predicted class.

Run:
  python explain.py --n 6                       # 6 test clips (mix of real/fake)
  python explain.py --file path/to/clip.wav     # one specific file
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from deepfake_dataset import DeepfakeAudioDataset, LABEL_TO_IDX
from model import AttentiveSpecCNN

IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


def grad_cam_and_attention(model, mel, device):
    """Return (pred_idx, p_fake, attn[T], cam[n_mels, T]) for one spectrogram."""
    model.eval()
    x = mel.unsqueeze(0).to(device)          # (1, 1, n_mels, T)

    captured = {}
    def hook(_m, _i, out):
        captured["fmap"] = out
        out.retain_grad()
    handle = model.features.register_forward_hook(hook)

    logits, attn = model(x, return_attn=True)
    p_fake = torch.softmax(logits, dim=1)[0, LABEL_TO_IDX["fake"]].item()
    pred_idx = int(logits.argmax(1).item())

    model.zero_grad()
    logits[0, pred_idx].backward()           # explain the predicted class
    handle.remove()

    fmap = captured["fmap"]                   # (1, C, F', T')
    grads = fmap.grad
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * fmap).sum(1)).squeeze(0)          # (F', T')
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(cam[None, None], size=mel.shape[-2:], # -> (n_mels, T)
                        mode="bilinear", align_corners=False)[0, 0]

    return pred_idx, p_fake, attn.squeeze(0).detach().cpu().numpy(), cam.detach().cpu().numpy()


def plot_explanation(mel, attn, cam, pred_idx, p_fake, true_label, out_path):
    mel_np = mel.squeeze(0).numpy()          # (n_mels, T)
    T = mel_np.shape[1]
    attn_t = np.interp(np.linspace(0, len(attn) - 1, T), np.arange(len(attn)), attn)

    correct = (pred_idx == LABEL_TO_IDX.get(true_label)) if true_label else None
    mark = "" if correct is None else ("  ✓" if correct else "  ✗ (wrong)")
    title = (f"pred: {IDX_TO_LABEL[pred_idx]}  (P_fake={p_fake:.2f})"
             + (f"   |   true: {true_label}{mark}" if true_label else ""))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 1]})

    axes[0].imshow(mel_np, origin="lower", aspect="auto", cmap="magma")
    axes[0].set_ylabel("mel bin")
    axes[0].set_title(title, fontsize=12)

    axes[1].imshow(mel_np, origin="lower", aspect="auto", cmap="gray")
    axes[1].imshow(cam, origin="lower", aspect="auto", cmap="jet", alpha=0.5)
    axes[1].set_ylabel("mel bin")
    axes[1].set_title("Grad-CAM: time-frequency regions driving the decision", fontsize=10)

    axes[2].plot(np.arange(T), attn_t, color="tab:blue")
    axes[2].fill_between(np.arange(T), attn_t, color="tab:blue", alpha=0.3)
    axes[2].set_ylabel("attention")
    axes[2].set_xlabel("time frame")
    axes[2].set_title("Temporal attention: which moments the model focused on", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="attentive_spec_cnn.pt")
    ap.add_argument("--csv", default="splits/test.csv")
    ap.add_argument("--file", default=None, help="explain one specific audio file")
    ap.add_argument("--n", type=int, default=6, help="number of clips to explain from --csv")
    ap.add_argument("--out_dir", default="explanations")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AttentiveSpecCNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    os.makedirs(args.out_dir, exist_ok=True)

    # Build the list of (mel, true_label, name) to explain.
    if args.file:
        tmp = pd.DataFrame([{"filepath": args.file, "label": "real"}])  # label unused
        tmp.to_csv(f"{args.out_dir}/_one.csv", index=False)
        ds = DeepfakeAudioDataset(f"{args.out_dir}/_one.csv", random_crop=False)
        samples = [(ds[0][0], None, os.path.splitext(os.path.basename(args.file))[0])]
    else:
        df = pd.read_csv(args.csv, dtype={"extra": str}, low_memory=False)
        # balanced mix of real and fake
        half = max(1, args.n // 2)
        picks = pd.concat([
            df[df["label"] == "real"].sample(half, random_state=args.seed),
            df[df["label"] == "fake"].sample(args.n - half, random_state=args.seed),
        ])
        picks.to_csv(f"{args.out_dir}/_picks.csv", index=False)
        ds = DeepfakeAudioDataset(f"{args.out_dir}/_picks.csv", random_crop=False)
        samples = [(ds[i][0], picks.iloc[i]["label"],
                    f"{picks.iloc[i]['label']}_{os.path.basename(picks.iloc[i]['filepath'])[:20]}")
                   for i in range(len(picks))]

    for mel, true_label, name in samples:
        pred_idx, p_fake, attn, cam = grad_cam_and_attention(model, mel, device)
        out_path = os.path.join(args.out_dir, f"{name}.png")
        plot_explanation(mel, attn, cam, pred_idx, p_fake, true_label, out_path)
        tag = IDX_TO_LABEL[pred_idx]
        print(f"  {name}: pred={tag} P_fake={p_fake:.2f} -> {out_path}")

    print(f"\nSaved {len(samples)} explanation figures to {args.out_dir}/")


if __name__ == "__main__":
    main()
