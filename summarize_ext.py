"""Regenerate every paper table from results_ext.csv.

Point of this script: the leave-one-corpus-out grid gets re-run whenever the
recipe changes (seed count, augmentation, schedule), and each re-run invalidates
all six tables in `main.tex`. Hand-transcribing numbers out of a log is how wrong
values end up in a paper, so keep this the single source of truth: re-run the
grid, run this, copy the emitted rows.

Every cell reports its own n, because seed coverage is not uniform -- the cloud
budget ran out mid-grid, so `arabic` has seeds 0-3 while the other three targets
have 0-4. Do NOT silently average over different n without saying so.

    python summarize_ext.py            # human-readable
    python summarize_ext.py --latex    # LaTeX table bodies
"""

import sys

import numpy as np
import pandas as pd

CSV = "results_ext.csv"
TARGETS = ["asvspoof2019", "dataset2", "in_the_wild", "arabic"]
NAME = {"asvspoof2019": "ASVspoof2019", "dataset2": "LibriSpeech-TTS",
        "in_the_wild": "In-the-Wild", "arabic": "Arabic"}
METHODS = ["source", "bn_only", "tent", "st_only", "ours", "oracle"]
MLABEL = {"source": "Source-only", "bn_only": "BN-only recalib.", "tent": "Tent",
          "st_only": "Self-training only", "ours": "Ours (TTA)",
          "oracle": "Oracle (supervised on target)"}


def load():
    d = pd.read_csv(CSV)
    return d, d[(d.setting == "transductive") & (d.family == "xlsr")]


def auc_fmt(v):
    """0.991 -> '.991' (IEEE house style).

    Must strip only a LEADING zero. A blanket .replace("0.", ".") also eats the
    "0." inside other numbers -- it silently turned a Tent std of 20.52 into 2.52
    and an Arabic EER of 20.61 into 2.61. Wrong-by-an-order-of-magnitude numbers
    that still look plausible are the worst possible failure here.
    """
    s = f"{v:.3f}"
    return s[1:] if s.startswith("0.") else s


def main_table(t, latex=False):
    hdr = "TABLE: cross-corpus EER% mean+-std / AUC (transductive)"
    print(f"\n{'=' * 78}\n{hdr}\n{'=' * 78}")
    for m in METHODS:
        cells = []
        for tg in TARGETS:
            g = t[(t.target == tg) & (t.method == m)]
            if latex:
                cells.append(f"{g.eer.mean():.2f}$\\pm${g.eer.std():.2f} / "
                             f"{auc_fmt(g.auc.mean())}")
            else:
                cells.append(f"{g.eer.mean():5.2f}+-{g.eer.std():4.2f}/"
                             f"{g.auc.mean():.3f}(n={len(g)})")
        if latex:
            print(f"{MLABEL[m]} & " + " & ".join(cells) + r" \\")
        else:
            print(f"{m:10s} " + "  ".join(cells))


def seed_detail(t):
    """Per-seed EER for every method. This is where catastrophic single-seed
    failures show up -- Tent looks merely noisy in the mean but collapses
    outright on individual seeds."""
    print(f"\n{'=' * 78}\nPER-SEED EER (watch for single-seed collapse)\n{'=' * 78}")
    for m in METHODS:
        print(f"\n{m}:")
        for tg in TARGETS:
            g = t[(t.target == tg) & (t.method == m)].sort_values("seed")
            s = " ".join(f"s{int(r.seed)}={r.eer:6.2f}" for r in g.itertuples())
            flag = ""
            if len(g) > 1 and g.eer.max() > 40 and g.eer.min() < 20:
                flag = "   <-- COLLAPSE on some seed(s)"
            print(f"  {NAME[tg]:16s} {s}  mean {g.eer.mean():6.2f} "
                  f"std {g.eer.std():5.2f}{flag}")


def gain_decomp(t, latex=False):
    print(f"\n{'=' * 78}\nTABLE: gain decomposition "
          f"(D1 = source->ST-only, D2 = ST-only->ours)\n{'=' * 78}")
    piv = t.pivot_table(index=["target", "seed"], columns="method",
                        values="eer").reset_index()
    for tg in TARGETS:
        g = piv[piv.target == tg]
        d1, d2 = (g.st_only - g.source).mean(), (g.ours - g.st_only).mean()
        tot = g.ours - g.source
        if latex:
            print(f"{NAME[tg]} & {g.source.mean():.2f} & ${d1:+.2f}$ & "
                  f"${d2:+.2f}$" + r" \\")
        else:
            print(f"{NAME[tg]:16s} src {g.source.mean():6.2f}  D1 {d1:+6.2f}  "
                  f"D2 {d2:+6.2f}  total {tot.mean():+6.2f}  "
                  f"better in {int((tot < 0).sum())}/{len(g)} seeds")


def inductive(d, latex=False):
    print(f"\n{'=' * 78}\nTABLE: inductive check (adapt on half, eval on disjoint half)"
          f"\n{'=' * 78}")
    ind = d[(d.setting == "inductive") & (d.family == "xlsr")]
    ip = ind.pivot_table(index=["target", "seed"], columns="method",
                         values="eer").reset_index()
    for tg in TARGETS:
        g = ip[ip.target == tg]
        if not {"source", "ours"} <= set(g.columns) or g.empty:
            continue
        if latex:
            print(f"{NAME[tg]} & {g.source.mean():.2f} & {g.ours.mean():.2f}" + r" \\")
        else:
            print(f"{NAME[tg]:16s} src {g.source.mean():6.2f} -> ours "
                  f"{g.ours.mean():6.2f}  ({g.ours.mean() - g.source.mean():+.2f}, "
                  f"better in {int((g.ours < g.source).sum())}/{len(g)})")


def backbone(d, t, latex=False):
    print(f"\n{'=' * 78}\nTABLE: backbone comparison (source-only, no adaptation)"
          f"\n{'=' * 78}")
    r = d[d.family == "rawnet2lite"]
    for tg in TARGETS:
        g, x = r[r.target == tg], t[(t.target == tg) & (t.method == "source")]
        if latex:
            print(f"{NAME[tg]} & {x.eer.mean():.2f} / {auc_fmt(x.auc.mean())}"
                  f" & {g.eer.mean():.2f} / {auc_fmt(g.auc.mean())}" + r" \\")
        else:
            print(f"{NAME[tg]:16s} XLS-R {x.eer.mean():6.2f}/{x.auc.mean():.3f}   "
                  f"RawNet2Lite {g.eer.mean():6.2f}/{g.auc.mean():.3f} (n={len(g)})")


def auc_gain_corr(t):
    print(f"\n{'=' * 78}\nAUC-vs-gain correlation (the ranking-precondition claim)"
          f"\n{'=' * 78}")
    pa = t.pivot_table(index=["target", "seed"], columns="method",
                       values=["eer", "auc"])
    gain = pa[("eer", "source")] - pa[("eer", "ours")]
    auc = pa[("auc", "source")]
    ok = gain.notna() & auc.notna()
    print(f"n={int(ok.sum())}   r={np.corrcoef(auc[ok], gain[ok])[0, 1]:+.3f}")


def _check_formatting():
    """Guard the LaTeX formatter against the leading-zero-strip bug."""
    assert auc_fmt(0.991) == ".991"
    assert auc_fmt(0.718) == ".718"
    assert auc_fmt(1.0) == "1.000"          # must not become '.000'
    # the actual regressions this caught, as literal cases
    assert f"{20.52:.2f}" == "20.52" and auc_fmt(0.891) == ".891"
    assert auc_fmt(0.0) == ".000"


if __name__ == "__main__":
    _check_formatting()
    latex = "--latex" in sys.argv
    d, t = load()
    cov = t[t.method == "ours"].groupby("target").seed.nunique().to_dict()
    print(f"{CSV}: {len(d)} rows | seed coverage per target: "
          f"{ {NAME[k]: v for k, v in cov.items()} }")
    main_table(t, latex)
    seed_detail(t)
    gain_decomp(t, latex)
    inductive(d, latex)
    backbone(d, t, latex)
    auc_gain_corr(t)
    print("\nAll numbers above are computed from the CSV -- do not hand-edit "
          "them into main.tex from a log.")
