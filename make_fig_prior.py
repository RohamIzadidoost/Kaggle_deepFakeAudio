"""Figure: what label-free operating-point rules do as the target prior moves.

Reads prevalence_rules_*.csv (produced by analyze_prevalence_rules.py from the
cached official-DF score dumps) and draws accuracy vs target prior for the
shipped threshold, the median rule, the BBSE-quantile threshold and the
labelled oracle, with the benchmark's own prior marked.

The point of the figure is the crossing: the median rule is the BEST rule at
50/50 -- which is the only prevalence at which the manuscript's earlier
threshold-control experiments ever measured it -- and the worst by ~40 points at
the prevalence the benchmark actually has.

    python make_fig_prior.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("prevalence_rules_df2021.csv")
g = df.groupby("prior").mean(numeric_only=True).reset_index()
true_prior = float(g.prior.max())

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))

# --- left: what each label-free threshold rule delivers as the prior moves ---
ax = axes[0]
for col, lab, c, mk, ls in [
        ("shipped_acc", "shipped $\\tau=0.5$", "#444444", "o", "-"),
        ("median_acc", "median rule", "#c0392b", "s", "-"),
        ("bbse_acc", "BBSE $\\tau$ (ours)", "#1f77b4", "^", "--"),
        ("oracle_acc", "oracle $\\tau$ (labelled)", "#7f7f7f", "", ":")]:
    if col in g:
        ax.plot(g.prior, g[col], ls, color=c, marker=mk, ms=4, lw=1.6, label=lab)
ax.axvline(true_prior, color="#999999", lw=0.9, ls=":")
ax.annotate("benchmark's\nown prior", xy=(true_prior, 66), xytext=(0.855, 66),
            fontsize=6, color="#666666", ha="right", va="center",
            arrowprops=dict(arrowstyle="->", color="#999999", lw=0.7))
ax.set_xlabel("target prior $P(\\mathrm{fake})$", fontsize=8)
ax.set_ylabel("accuracy %", fontsize=8)
ax.set_title("label-free operating-point rules", fontsize=9)
ax.legend(fontsize=6.2, loc="lower left", framealpha=0.9)

# --- right: the prior estimate itself ---
ax = axes[1]
ax.plot([0.45, 1.0], [0.45, 1.0], ":", color="#999999", lw=1.0, label="perfect")
ax.plot(g.prior, [0.5] * len(g), "-", color="#c0392b", lw=1.6, marker="s", ms=4,
        label="median rule's implicit prior")
ax.plot(g.prior, g.pi_hat, "-", color="#1f77b4", lw=1.6, marker="^", ms=4,
        label="BBSE $\\hat\\pi$ (label-free)")
ax.axvline(true_prior, color="#999999", lw=0.9, ls=":")
ax.set_xlabel("true target prior $P(\\mathrm{fake})$", fontsize=8)
ax.set_ylabel("estimated prior", fontsize=8)
ax.set_title("the prior, estimated without labels", fontsize=9)
ax.legend(fontsize=6.2, loc="upper left", framealpha=0.9)

for ax in axes:
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig("fig_prior_rules.png", dpi=300, bbox_inches="tight")
print("wrote fig_prior_rules.png")
print(g.round(2).to_string(index=False))
