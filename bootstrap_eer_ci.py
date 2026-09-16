"""Paired stratified bootstrap confidence intervals for the reported EERs.

Reviewer item 5: put uncertainty on the EER numbers, and report the disjoint
win counts next to the available-pool ones.

Resampling is stratified by class, so every replicate keeps the pool's
prevalence fixed: an unstratified bootstrap would also jitter the class prior,
which is the quantity this study intervenes on, and would confound sampling
noise with the effect under test. Replicates are paired -- the same resampled
trials are scored for every arm -- so a delta interval is an interval on the
difference between two arms on one pool, not on two independent samples.

EER matches metrics.compute_eer exactly (ROC crossing, interpolated, ties
grouped); the binned form is checked against it on the observed data for every
arm before any replicate is drawn. Manifest reconstruction and the adaptation
draw are imported from audit_icassp, so the trials are the audited ones.

    python bootstrap_eer_ci.py            # BOOT_B=2000 replicates, seed 12345
"""
import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
_spec = importlib.util.spec_from_file_location("audit_icassp", "audit_icassp.py")
audit = importlib.util.module_from_spec(_spec)
sys.modules["audit_icassp"] = audit
_spec.loader.exec_module(audit)
from metrics import compute_eer

B = int(os.environ.get("BOOT_B", "2000"))
SEED = int(os.environ.get("BOOT_SEED", "12345"))
OUT = Path("manuscript_audit")
FOLDER = {"df2021": "scores_protocol_a_public", "itw": "scores_protocol_a_public_itw"}


# Score filenames are not written to one convention across runs (some seed
# variants carry no q or k suffix at all), so the files are indexed with the
# same regex audit_icassp parses them with rather than reconstructed by rule.
_INDEX = {}


def _index(corpus):
    if corpus not in _INDEX:
        table = {}
        for path in sorted(Path(FOLDER[corpus]).glob("*.npy")):
            ckpt, arm, *tag = path.stem.split("__")
            if arm == "M":
                continue
            suffix = tag[0] if tag else ""
            m = re.fullmatch(r"s(\d+)_e(\d+)(?:_q([\d.]+))?(?:_k([\d.]+))?", suffix) if suffix else None
            if suffix and not m:
                raise ValueError(f"Unrecognized score suffix: {path}")
            seed, sub, q, skew = (int(m[1]), int(m[2]), float(m[3] or .3),
                                  float(m[4] or 0)) if m else (0, 0, .3, 0)
            table[(ckpt, arm, seed, sub, round(q, 6), round(skew, 6))] = path
        _INDEX[corpus] = table
    return _INDEX[corpus]


def score_path(corpus, ckpt, arm, seed, sub, q, skew):
    return _index(corpus).get(
        (ckpt, arm, seed, sub or 0, round(q, 6), round(skew or 0, 6)),
        Path("/nonexistent"))


def binned(y, s):
    """Descending-threshold bin counts, the sufficient statistic for the ROC."""
    uniq, inv = np.unique(s, return_inverse=True)
    desc = (len(uniq) - 1) - inv
    return desc, len(uniq)


def eer_from_counts(pos_desc, neg_desc, n_pos, n_neg):
    fpr = np.concatenate(([0.], np.cumsum(neg_desc) / n_neg))
    tpr = np.concatenate(([0.], np.cumsum(pos_desc) / n_pos))
    delta = fpr - (1 - tpr)
    idx = int(np.flatnonzero(delta >= 0)[0])
    if delta[idx] == 0:
        return float(fpr[idx])
    w = -delta[idx-1] / (delta[idx] - delta[idx-1])
    return float(fpr[idx-1] + w * (fpr[idx] - fpr[idx-1]))


def cell(corpus, ckpt, seed, sub, skew, arms, pools):
    """arms: list of (label, arm, q); the first is the reference."""
    ev, adapt = audit.select_pool(pools[corpus], seed, sub, skew)
    y = ev.label.to_numpy()
    loaded = []
    for label, arm, q in arms:
        p = score_path(corpus, ckpt, arm, seed, sub, q, skew)
        if not p.exists():
            return []
        s = np.load(p, allow_pickle=False)
        if s.shape != y.shape:
            raise ValueError(f"{p}: {s.shape} vs {y.shape}")
        loaded.append((label, s))
    mask = np.ones(len(y), dtype=bool)
    mask[adapt] = False
    rows = []
    for setting, take in [("available_pool", np.ones(len(y), dtype=bool)),
                          ("disjoint_eval", mask)]:
        yy = y[take]
        if len(np.unique(yy)) != 2:
            continue
        pos, neg = np.flatnonzero(yy == 1), np.flatnonzero(yy == 0)
        prepared, point = [], []
        for label, s in loaded:
            ss = s[take]
            desc, U = binned(yy, ss)
            prepared.append((label, desc, U))
            observed = eer_from_counts(
                np.bincount(desc[pos], minlength=U), np.bincount(desc[neg], minlength=U),
                len(pos), len(neg))
            reference, _ = compute_eer(yy, ss)
            if abs(observed - reference) > 1e-12:
                raise ValueError(f"binned EER disagrees with metrics.compute_eer: {label}")
            point.append(observed)
        rng = np.random.default_rng(SEED)
        draws = np.empty((B, len(prepared)))
        for b in range(B):
            pi = pos[rng.integers(0, len(pos), len(pos))]
            ni = neg[rng.integers(0, len(neg), len(neg))]
            for a, (_, desc, U) in enumerate(prepared):
                draws[b, a] = eer_from_counts(np.bincount(desc[pi], minlength=U),
                                              np.bincount(desc[ni], minlength=U),
                                              len(pos), len(neg))
        sym = next((i for i, (l, _, _) in enumerate(prepared)
                    if l in ("Sym q=.3", "q=.30")), None)
        for a, (label, _, _) in enumerate(prepared):
            d = draws[:, a] - draws[:, 0]
            ds = draws[:, a] - draws[:, sym] if sym is not None else None
            rows.append(dict(
                corpus=corpus, ckpt=ckpt, seed=seed, eval_sub=sub or 0, skew=skew or 0,
                setting=setting, arm=label, n=int(take.sum()), n_pos=len(pos),
                eer=100*point[a],
                eer_lo=100*np.percentile(draws[:, a], 2.5),
                eer_hi=100*np.percentile(draws[:, a], 97.5),
                d_vs_ref=100*(point[a]-point[0]),
                d_lo=100*np.percentile(d, 2.5), d_hi=100*np.percentile(d, 97.5),
                d_vs_sym=100*(point[a]-point[sym]) if sym is not None else np.nan,
                dsym_lo=100*np.percentile(ds, 2.5) if sym is not None else np.nan,
                dsym_hi=100*np.percentile(ds, 97.5) if sym is not None else np.nan,
                B=B))
    return rows


def main():
    pools, _ = audit.manifests()
    df_arms = [("Source", "source", .3), ("Sym q=.3", "ours_fixed", .3),
               ("BBSE", "ours_bbse", .3)]
    rows = []
    for ckpt in ["deepfense_w2v2_aasist_s2", "deepfense_w2v2_aasist_s42",
                 "deepfense_w2v2_aasist_s240"]:
        for seed, sub in [(0, 0), (1, 50000), (2, 50000)]:
            rows += cell("df2021", ckpt, seed, sub, 0, df_arms, pools)
            print(f"  df2021 {ckpt} seed{seed} done", flush=True)
    for ckpt in ["deepfense_w2v2_aasist_s2", "deepfense_w2v2_aasist_s42",
                 "deepfense_w2v2_aasist_s240", "ssl_aasist_wavefake"]:
        rows += cell("itw", ckpt, 0, 0, 0, df_arms, pools)
        print(f"  itw {ckpt} native done", flush=True)
    skew_arms = [("Source", "source", .3), ("q=.02", "ours_fixed", .02),
                 ("q=.10", "ours_fixed", .1), ("q=.30", "ours_fixed", .3),
                 ("BBSE", "ours_bbse", .3)]
    for skew in [.01, .02, .05, .1, .9, .95, .97]:
        got = cell("itw", "ssl_aasist_wavefake", 0, 0, skew, skew_arms, pools)
        rows += got
        print(f"  itw skew {skew} {'done' if got else 'SKIPPED (arms missing)'}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "bootstrap_eer.csv", index=False)
    print(f"\nwrote {OUT/'bootstrap_eer.csv'} ({len(out)} rows, B={B})\n")
    df = out[(out.corpus == "df2021")]
    for setting in ["available_pool", "disjoint_eval"]:
        d = df[df.setting == setting]
        b = d[d.arm == "BBSE"].set_index(["ckpt", "seed"])
        s = d[d.arm == "Source"].set_index(["ckpt", "seed"])
        y = d[d.arm == "Sym q=.3"].set_index(["ckpt", "seed"])
        wins_src = (b.eer < s.eer.reindex(b.index)).sum()
        wins_sym = (b.eer < y.eer.reindex(b.index)).sum()
        excl = ((b.d_lo > 0) | (b.d_hi < 0)).sum()
        excl_sym = ((b.dsym_lo > 0) | (b.dsym_hi < 0)).sum()
        print(f"{setting}: BBSE beats source {wins_src}/{len(b)} "
              f"({excl}/{len(b)} with a CI excluding zero), "
              f"beats symmetric {wins_sym}/{len(b)} ({excl_sym}/{len(b)} excluding zero)")
    print()
    print(out[out.corpus == "df2021"][
        ["ckpt", "seed", "setting", "arm", "eer", "eer_lo", "eer_hi",
         "d_vs_ref", "d_lo", "d_hi"]].to_string(index=False))


if __name__ == "__main__":
    main()
