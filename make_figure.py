"""The prevalence figure: how tight the counting bound is, and what accuracy
does against a moving trivial baseline.

Panel (a) plots measured initial-pass pseudo-label purity against the counting
bound of Eq. (1) for every constrained tail in the audited grid. If the bound
were loose the points would sit well above the diagonal; the interesting claim
is that they sit on it.

Panel (b) plots accuracy at threshold 0.5 against spoof prevalence for each
budget, with the always-majority trivial baseline drawn as a dashed line. The
same accuracy number means something different at every prevalence, which is
why the tables lead with EER and balanced accuracy.

Reads only the audit outputs, so it needs no GPU and no score decoding:

    env/bin/python make_figure.py        # writes fig_prevalence.pdf
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.linewidth": .6, "lines.linewidth": .9,
})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

scores = pd.read_csv("manuscript_audit/scores.csv")

fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.35, 3.3))

# (a) outcome against the contamination the counting argument implies
con = pd.read_csv("manuscript_audit/contamination.csv")
con = con[con.setting == "available_pool"]
groups = [("itw", "ssl_aasist_wavefake", "o", "#1b1b1b", "WF / In-the-Wild"),
          ("itw", None, "^", "#7a7a7a", "DeepFense / In-the-Wild"),
          ("df2021", None, "s", "#b0b0b0", "DeepFense / DF")]
for corpus, ckpt, marker, colour, label in groups:
    g = con[(con.corpus == corpus)
            & ((con.ckpt == ckpt) if ckpt else (con.ckpt != "ssl_aasist_wavefake"))]
    ax.scatter(g.excess, g.d_ba, s=15, marker=marker, facecolors="none",
               edgecolors=colour, linewidths=.8, label=label, zorder=3)
ax.axvspan(715, 1427, color="#ececec", zorder=0)          # no cell lands here
ax.axhline(0, ls=":", c="#b0b0b0", lw=.8, zorder=1)
ax.set_xscale("symlog", linthresh=100)
ax.set_xlabel(r"implied excess contamination $N\sum_c (b_c-\pi_c)^+$ (clips)")
ax.set_ylabel(r"$\Delta$ balanced accuracy (pts)")
ax.set_xlim(-20, 12000)
ax.legend(loc="upper right", frameon=False, handletextpad=.4, borderpad=.2)
ax.text(.03, .90, "(a)", transform=ax.transAxes, fontweight="bold")

# (b) accuracy against a moving trivial baseline
itw = scores[(scores.corpus == "itw") & (scores.ckpt == "ssl_aasist_wavefake")
             & (scores.setting == "available_pool")]
arms = [("source", .3, "s", "#1b1b1b", "source"),
        ("ours_fixed", .02, "o", "#1b1b1b", "$q=.02$"),
        ("ours_fixed", .1, "v", "#6e6e6e", "$q=.10$"),
        ("ours_fixed", .3, "x", "#6e6e6e", "$q=.30$"),
        ("ours_bbse", .3, "^", "#a8a8a8", "BBSE")]
for arm, q, marker, colour, label in arms:
    r = itw[(itw.arm == arm) & np.isclose(itw.q, q)].sort_values("prior")
    if not len(r):
        continue
    bx.plot(r.prior, r.accuracy, marker=marker, ms=3.2, c=colour, label=label,
            mfc="none", mew=.8)
grid = np.logspace(np.log10(.008), np.log10(.99), 200)
bx.plot(grid, 100*np.maximum(grid, 1-grid), ls="--", c="#b0b0b0", lw=.8,
        label="trivial (always majority)")
bx.set_xscale("log")
bx.set_xlabel(r"spoof prevalence $\pi$ of the pool")
bx.set_ylabel("accuracy at 0.5 (%)")
bx.set_ylim(44, 103)
bx.legend(loc="lower center", frameon=False, ncol=3, handletextpad=.4,
          columnspacing=1.0, borderpad=.2)
bx.text(.03, .93, "(b)", transform=bx.transAxes, fontweight="bold")

for a in (ax, bx):
    a.tick_params(length=2.5, width=.6)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
fig.tight_layout(pad=.25, h_pad=.9)
fig.savefig("fig_prevalence.pdf")
print("wrote fig_prevalence.pdf")
lo, hi = con[con.excess <= 714], con[con.excess >= 1428]
print(f"panel (a): {len(lo)} cells below the gap ({(lo.d_ba > 5).sum()} improving "
      f"by >5 pts, max {lo.d_ba.max():.1f}), {len(hi)} above it (max {hi.d_ba.max():.1f})")
