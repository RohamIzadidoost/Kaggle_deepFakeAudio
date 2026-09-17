"""The two panels the paper's claims rest on, without the oracle decomposition.

(a) Adaptation moves the decision far more than the ordering the decision is
    applied to. Both axes are threshold-free on one side (EER) and
    threshold-bound on the other (accuracy at 0.5); no decomposition of
    accuracy into a ceiling and a gap is involved, so the comparison is not
    exposed to the class prior pinning a ceiling term.

(b) Where the decision lands is ordered by the contamination the counting bound
    implies. Pooled over every cell, not within a prevalence: within a
    checkpoint and prevalence Delta is a monotone transform of q and the two
    cannot be separated, so the panel plots all strata together.

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
                 & (scores.setting == "available_pool") & (scores.seed == r.seed)
                 & (scores.eval_sub == 0)]
    a, b = sel[sel.arm == "ours_fixed"], sel[sel.arm == "source"]
    if len(a) != 1 or len(b) != 1:
        continue
    a, b = a.iloc[0], b.iloc[0]
    rows.append(dict(ckpt=r.ckpt, corpus=r.corpus, prior=r.prior, q=r.q, seed=r.seed,
                     delta=r.delta, gap_ad=a.threshold_gap,
                     d_acc=abs(a.accuracy - b.accuracy), d_eer=abs(a.eer - b.eer)))
d = pd.DataFrame(rows)

fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.35, 3.05))

# (a) the decision moves; the ordering does not
ax.scatter(d.d_eer.clip(lower=3e-3), d.d_acc.clip(lower=3e-3), s=13, marker="o",
           facecolors="none", edgecolors="#1b1b1b", linewidths=.8, zorder=3)
lim = [3e-3, 70]
ax.plot(lim, lim, ls=":", c="#b0b0b0", lw=.8, zorder=1)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xlabel("shift in EER (pts)")
ax.set_ylabel("shift in accuracy\nat 0.5 (pts)")
ax.text(.03, .88, "(a)", transform=ax.transAxes, fontweight="bold")

# (b) contamination orders the resulting operating point, pooled over all strata
groups = [("ssl_aasist_wavefake", "itw", "o", "#1b1b1b", "WF / ITW"),
          ("deepfense_w2v2_aasist_s2", "df2021", "s", "#6e6e6e", "DF-2 / DF"),
          ("deepfense_w2v2_aasist_s2", "itw", "v", "#9a9a9a", "DF-2 / ITW"),
          ("deepfense_w2v2_aasist_s42", "itw", "^", "#9a9a9a", "DF-42 / ITW"),
          ("deepfense_w2v2_aasist_s240", "itw", "x", "#b8b8b8", "DF-240 / ITW")]
for ckpt, corpus, marker, colour, label in groups:
    g = d[(d.ckpt == ckpt) & (d.corpus == corpus)]
    if not len(g):
        continue
    kw = (dict(c=colour) if marker == "x"
          else dict(facecolors="none", edgecolors=colour))
    bx.scatter(g.delta, g.gap_ad, s=14, marker=marker, linewidths=.8,
               label=label, zorder=3, **kw)
bx.set_xlabel(r"implied contamination $\Delta=\sum_c(b_c-\pi_c)^+$")
bx.set_ylabel("threshold gap after\nadaptation (pts)")
bx.legend(loc="lower right", frameon=False, ncol=2, handletextpad=.3,
          columnspacing=.8, borderpad=.2)
bx.set_ylim(-4, 62)
bx.text(.03, .88, "(b)", transform=bx.transAxes, fontweight="bold")

for a_ in (ax, bx):
    a_.tick_params(length=2.5, width=.6)
    for side in ("top", "right"):
        a_.spines[side].set_visible(False)
fig.tight_layout(pad=.25, h_pad=.9)
fig.savefig("fig_prevalence.pdf")
s0 = d[d.seed == 0]
print(f"panel (a): {(d.d_acc > d.d_eer).sum()}/{len(d)} runs above the diagonal")
print(f"wrote fig_prevalence.pdf ({len(d)} runs). accuracy/EER median ratio "
      f"{d.d_acc.median()/d.d_eer.median():.0f}x; pooled Spearman(delta,gap)="
      f"{s0[['delta','gap_ad']].corr('spearman').iloc[0,1]:+.3f} vs (q,gap)="
      f"{s0[['q','gap_ad']].corr('spearman').iloc[0,1]:+.3f}")
