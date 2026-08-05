"""
Figure for the paper: real-vs-fake detector scores on the unseen In-the-Wild
corpus, BEFORE and AFTER test-time adaptation, on a logit axis (scores pile near
1.0, so logit spreads them out). Shows the central insight: the source model puts
almost everything past the 0.5 threshold (mis-calibration) yet still ranks real
below fake (high AUC); TTA shifts the real mass back across the threshold.

Re-plots from the saved scores in fig_score_dist.npz (no GPU needed). Regenerate
the scores with the model by running the earlier scoring path if the .npz is absent.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from metrics import compute_eer

d = np.load("fig_score_dist.npz")
y, S = d["y"], {"Source model (unseen In-the-Wild)": d["s_src"],
                "After test-time adaptation": d["s_adp"]}


def logit(s):
    s = np.clip(s, 1e-4, 1 - 1e-4)
    return np.log(s / (1 - s))


fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True, sharey=True)
bins = np.linspace(-10, 10, 45)
for ax, (title, s) in zip(axes, S.items()):
    auc = roc_auc_score(y, s)
    acc = accuracy_score(y, (s >= 0.5).astype(int)) * 100
    eer, _ = compute_eer(y, s)
    ax.hist(logit(s[y == 0]), bins=bins, alpha=0.6, color="#2ca02c", label="real", density=True)
    ax.hist(logit(s[y == 1]), bins=bins, alpha=0.6, color="#d62728", label="fake", density=True)
    ax.axvline(0, ls="--", c="black", lw=1.2)
    ax.text(0, ax.get_ylim()[1] * 0.6, "  thr=0.5", fontsize=9, rotation=90, va="center")
    ax.set_title(f"{title}\nAUC={auc:.3f}   acc@0.5={acc:.0f}%   EER={eer*100:.1f}%", fontsize=9.5)
    ax.set_xlabel("logit  P(fake)", fontsize=9)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, loc="upper left")
axes[0].set_yticks([])
fig.tight_layout()
fig.savefig("fig_score_dist.png", dpi=150)
print("wrote fig_score_dist.png")
