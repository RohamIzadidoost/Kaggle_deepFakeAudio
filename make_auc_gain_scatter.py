"""
Scatter of source AUC vs. EER gain (source - ours), one point per (target, seed),
from the real leave-one-corpus-out grid (results_ext.csv). Supports the
Limitations claim that TTA's benefit tracks source ranking quality.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

d = pd.read_csv("results_ext.csv")
t = d[(d.setting == "transductive") & (d.family == "xlsr")]
piv = t.pivot_table(index=["target", "seed"], columns="method", values=["eer", "auc"])
piv["eer_gain"] = piv[("eer", "source")] - piv[("eer", "ours")]
piv["auc_source"] = piv[("auc", "source")]
piv = piv.reset_index()

TARGET_LABEL = {"asvspoof2019": "ASVspoof2019", "dataset2": "LibriSpeech-TTS",
                "in_the_wild": "In-the-Wild", "arabic": "Arabic (ArAD)"}
COLORS = {"asvspoof2019": "#1f77b4", "dataset2": "#d62728",
         "in_the_wild": "#2ca02c", "arabic": "#9467bd"}

fig, ax = plt.subplots(figsize=(4.2, 3.3))
for target, g in piv.groupby("target"):
    ax.scatter(g["auc_source"], g["eer_gain"], s=46, alpha=0.85,
              color=COLORS[target], label=TARGET_LABEL[target],
              edgecolor="white", linewidth=0.6, zorder=3)

# least-squares trend line
x, y = piv["auc_source"].values, piv["eer_gain"].values
m, b = np.polyfit(x, y, 1)
xs = np.linspace(x.min() - 0.01, x.max() + 0.01, 50)
ax.plot(xs, m * xs + b, "--", color="gray", linewidth=1.2, zorder=1)

r = np.corrcoef(x, y)[0, 1]
ax.axhline(0, color="black", linewidth=0.8, zorder=1)
ax.set_xlabel("Source AUC (before adaptation)")
ax.set_ylabel("EER gain: source $-$ ours (points)")
ax.set_title(f"$r = {r:+.2f}$  ($n=12$)", fontsize=10)
ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
ax.grid(alpha=0.25, zorder=0)
fig.tight_layout()
fig.savefig("fig_auc_gain.png", dpi=150)
print(f"wrote fig_auc_gain.png  (r={r:+.3f}, slope={m:+.3f}, intercept={b:+.3f})")
