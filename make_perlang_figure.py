"""
Figure for the paper: MLAAD fake-clip recall by language, averaged over the
four leave-one-corpus-out source models, sorted ascending. Reconstructed from
results_perlang.csv (no GPU/model access needed -- the per-(target,language)
recall numbers are already computed and cached there).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results_perlang.csv")
agg = (df.groupby("language")
         .agg(recall=("recall", "mean"), seen=("seen_in_training", "first"))
         .reset_index()
         .sort_values("recall"))

overall_mean = df.recall.mean()

fig, ax = plt.subplots(figsize=(10, 4.2))
colors = ["#ff7f0e" if not s else "#2ca02c" for s in agg.seen]
ax.bar(agg.language, agg.recall, color=colors, width=0.7)
ax.axhline(overall_mean, ls="--", c="black", lw=1.2)
ax.text(len(agg) - 1, overall_mean + 1.2, f"mean = {overall_mean:.1f}%",
        ha="right", fontsize=9)

mt_row = agg[agg.language == "mt"]
if len(mt_row):
    mt_idx = agg.index.get_loc(mt_row.index[0])
    mt_worst = df[(df.language == "mt")].recall.min()
    ax.annotate(f"outlier: In-the-Wild model\nscores mt at {mt_worst:.1f}%",
                xy=(mt_idx, agg.recall.iloc[mt_idx]),
                xytext=(mt_idx + 2, agg.recall.iloc[mt_idx] - 12),
                fontsize=9, ha="left",
                arrowprops=dict(arrowstyle="->", lw=1))

ax.set_ylabel("Fake-clip recall (%)", fontsize=10)
ax.set_xlabel("MLAAD language", fontsize=10)
ax.set_title("Per-language fake-clip recall, averaged over 4 source models "
              "(orange = held out of every source pool)", fontsize=10)
ax.set_xticks(range(len(agg)))
ax.set_xticklabels(agg.language, rotation=90, fontsize=9)
ax.tick_params(axis="y", labelsize=9)
ax.set_ylim(50, 100)
fig.tight_layout()
fig.savefig("fig_perlang.png", dpi=150)
print("wrote fig_perlang.png")
