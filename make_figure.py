"""The two panels the paper's claim rests on.

(a) Confident-tail adaptation is an operating-point procedure: the accuracy
    ceiling its scores can reach barely moves, while the gap between that
    ceiling and the accuracy actually obtained at threshold 0.5 moves by orders
    of magnitude more. Every run sits far below the diagonal.

(b) What sets the new operating point is the contamination the counting bound
    implies. Within a checkpoint and a prevalence, the post-adaptation gap rises
    monotonically with Delta; the curves are offset by checkpoint and prior but
    never cross zero slope.

Reads the audit outputs only; no GPU, no decoding.
    env/bin/python make_figure.py        # writes fig_prevalence.pdf
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "legend.fontsize": 5.6, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.linewidth": .6, "lines.linewidth": .9,
})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

scores = pd.read_csv("manuscript_audit/scores.csv")
con = pd.read_csv("manuscript_audit/contamination.csv")
con = con[con.setting == "available_pool"]

rows = []
for _, r in con.iterrows():
    sel = scores[(scores.corpus == r.corpus) & (scores.ckpt == r.ckpt)
                 & np.isclose(scores.q, r.q) & np.isclose(scores["skew"], r["skew"])
                 & (scores.setting == "available_pool") & (scores.seed == 0)
                 & (scores.eval_sub == 0)]
    a, b = sel[sel.arm == "ours_fixed"], sel[sel.arm == "source"]
    if len(a) != 1 or len(b) != 1:
        continue
    a, b = a.iloc[0], b.iloc[0]
    rows.append(dict(ckpt=r.ckpt, corpus=r.corpus, prior=r.prior, q=r.q, delta=r.delta,
                     d_ceiling=abs(a.oracle_accuracy - b.oracle_accuracy),
                     d_point=abs(a.threshold_gap - b.threshold_gap),
                     gap_ad=a.threshold_gap))
d = pd.DataFrame(rows)

fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.35, 3.15))

# (a) the ceiling does not move; the operating point does
ax.scatter(d.d_ceiling.clip(lower=3e-3), d.d_point.clip(lower=3e-3), s=13, marker="o",
           facecolors="none", edgecolors="#1b1b1b", linewidths=.8, zorder=3)
lim = [3e-3, 60]
ax.plot(lim, lim, ls=":", c="#b0b0b0", lw=.8, zorder=1)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xlabel("shift in the achievable ceiling (pts)")
ax.set_ylabel("shift in the\noperating point (pts)")
ax.text(.03, .88, "(a)", transform=ax.transAxes, fontweight="bold")

# (b) contamination sets where the new operating point lands
series = [("ssl_aasist_wavefake", .010, "o", "#1b1b1b", r"WF, $\pi{=}.01$"),
          ("ssl_aasist_wavefake", .970, "s", "#1b1b1b", r"WF, $\pi{=}.97$"),
          ("ssl_aasist_wavefake", .372, "^", "#8a8a8a", r"WF, $\pi{=}.37$"),
          ("deepfense_w2v2_aasist_s2", .972, "v", "#8a8a8a", r"DF-2, $\pi{=}.97$"),
          ("deepfense_w2v2_aasist_s42", .970, "x", "#b0b0b0", r"DF-42, $\pi{=}.97$")]
for ckpt, prior, marker, colour, label in series:
    g = d[(d.ckpt == ckpt) & (np.isclose(d.prior, prior, atol=.006))].sort_values("delta")
    if len(g) < 2:
        continue
    bx.plot(g.delta, g.gap_ad, marker=marker, ms=3.2, mfc="none", mew=.8,
            c=colour, label=label)
bx.set_xlabel(r"implied contamination $\Delta=\sum_c(b_c-\pi_c)^+$")
bx.set_ylabel("threshold gap after\nadaptation (pts)")
bx.legend(loc="upper left", frameon=False, ncol=2, handletextpad=.4,
          columnspacing=.9, borderpad=.2)
bx.set_ylim(-3, 52)
bx.text(.90, .88, "(b)", transform=bx.transAxes, fontweight="bold")

for a_ in (ax, bx):
    a_.tick_params(length=2.5, width=.6)
    for side in ("top", "right"):
        a_.spines[side].set_visible(False)
fig.tight_layout(pad=.25, h_pad=.9)
fig.savefig("fig_prevalence.pdf")
print(f"wrote fig_prevalence.pdf  ({len(d)} runs; ceiling median "
      f"{d.d_ceiling.median():.3f} pts vs operating point {d.d_point.median():.2f} pts; "
      f"{(d.d_ceiling > d.d_point).sum()} runs above the diagonal)")
