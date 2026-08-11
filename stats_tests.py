"""Pooled significance tests for the ICASSP paper (P0.2).

Reads results_ext.csv (source/ours/tent/st_only, transductive setting),
results_dann.csv, results_asdg.csv. Pairs runs by (target, seed) and reports:
  (a) Wilcoxon signed-rank, ours vs. {source, tent, st_only, dann, asdg}, pooled
      across all targets/seeds available for that comparison
  (b) 10k-resample bootstrap 95% CI on per-target mean EER, source vs. ours

No GPU, no training — pure post-hoc stats on committed CSVs.
"""
import csv
import numpy as np
from scipy.stats import wilcoxon

RESULTS_EXT = "results_ext.csv"
RESULTS_DANN = "results_dann.csv"
RESULTS_ASDG = "results_asdg.csv"


def load_ext(path=RESULTS_EXT):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if not row.get("eer"):
                continue
            row["eer"] = float(row["eer"])
            row["seed"] = int(row["seed"])
            rows.append(row)
    return rows


def load_simple(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if not row.get("eer"):
                continue
            row["eer"] = float(row["eer"])
            row["seed"] = int(row["seed"])
            rows.append(row)
    return rows


def paired_eer(rows_a, method_a, rows_b, method_b, setting=None):
    """Pair EER by (target, seed) between two method slices. Returns (a_vals, b_vals, keys)."""
    def index(rows, method):
        idx = {}
        for r in rows:
            if r["method"] != method:
                continue
            if r.get("family", "xlsr") != "xlsr":
                continue
            if setting is not None and r.get("setting", setting) != setting:
                continue
            idx[(r["target"], r["seed"])] = r["eer"]
        return idx

    idx_a = index(rows_a, method_a)
    idx_b = index(rows_b, method_b)
    keys = sorted(set(idx_a) & set(idx_b))
    a = np.array([idx_a[k] for k in keys])
    b = np.array([idx_b[k] for k in keys])
    return a, b, keys


def report_wilcoxon(name, a, b, n):
    if n < 2:
        print(f"{name}: n={n}, skipped (too few pairs)")
        return
    stat, p = wilcoxon(a, b)
    direction = "lower" if np.median(b - a) < 0 else "higher"
    print(f"{name}: n={n}, ours {direction} EER, median diff={np.median(b - a):+.3f}, "
          f"W={stat:.1f}, p={p:.4g}")


def bootstrap_ci(vals, n_resamples=10000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals)
    means = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                       for _ in range(n_resamples)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return vals.mean(), lo, hi


def main():
    ext = load_ext()

    print("=== Pooled Wilcoxon signed-rank (transductive, all targets/seeds) ===")
    a, b, keys = paired_eer(ext, "source", ext, "ours", setting="transductive")
    report_wilcoxon("ours vs. source", a, b, len(keys))

    a, b, keys = paired_eer(ext, "tent", ext, "ours", setting="transductive")
    report_wilcoxon("ours vs. tent", a, b, len(keys))

    a, b, keys = paired_eer(ext, "st_only", ext, "ours", setting="transductive")
    report_wilcoxon("ours vs. st_only", a, b, len(keys))

    dann = load_simple(RESULTS_DANN)
    a, b, keys = paired_eer(dann, "dann", ext, "ours", setting="transductive")
    report_wilcoxon("ours vs. dann", a, b, len(keys))

    asdg = load_simple(RESULTS_ASDG)
    a, b, keys = paired_eer(asdg, "asdg", ext, "ours", setting="transductive")
    report_wilcoxon("ours vs. asdg", a, b, len(keys))

    print("\n=== Bootstrap 95% CI on per-target mean EER (source vs. ours, transductive) ===")
    targets = sorted({r["target"] for r in ext if r["method"] == "ours"})
    out_rows = []
    for t in targets:
        for method in ("source", "ours"):
            vals = [r["eer"] for r in ext
                     if r["target"] == t and r["method"] == method
                     and r.get("family", "xlsr") == "xlsr"
                     and r.get("setting") == "transductive"]
            if not vals:
                continue
            mean, lo, hi = bootstrap_ci(vals)
            print(f"{t:14s} {method:8s} n_clips_runs={len(vals)}  mean EER={mean:.2f}  "
                  f"95% CI=[{lo:.2f}, {hi:.2f}]")
            out_rows.append({"target": t, "method": method, "mean_eer": mean,
                              "ci_lo": lo, "ci_hi": hi, "n_seeds": len(vals)})

    with open("stats_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "method", "mean_eer", "ci_lo", "ci_hi", "n_seeds"])
        w.writeheader()
        w.writerows(out_rows)
    print("\nWrote stats_results.csv")


if __name__ == "__main__":
    main()
