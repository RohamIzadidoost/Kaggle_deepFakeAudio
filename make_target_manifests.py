"""Build per-target evaluation pools for the public-checkpoint TTA arm.

Reproduces `extended_pipeline.py` sections 2-3 for the *target* side only, so a
public checkpoint is scored on exactly the clips our own source model was
scored on in `results_ext.csv`. Nothing here trains, downloads, or touches the
source pool.

The reproduction is verbatim on purpose -- same `build_manifest()` row order
(unsorted glob included), same `cap_per_class(..., 6000, random_state=0)`, then
same `sample_target(..., 3000, random_state=seed)`. Pool *size* is asserted
against the `n` column recorded in `results_ext.csv`, which is the only
independently recorded property of those pools; identical size plus an
identical construction rule is as close to proof of pool identity as the
artifacts allow. Membership could still differ if the corpus directories were
modified since the grid ran, which is why the size gate is an assert, not a
warning.

Also audits decodability, because `extended_pipeline.decode()` swallows every
decoder failure into a zero-filled clip -- a silent failure mode this repo has
been bitten by before. If a target pool contains clips libsndfile cannot read,
both this arm *and* the corresponding rows of `results_ext.csv` were computed on
silence, and we need to know that before adding points to a plot.

Usage:
    python make_target_manifests.py                    # all targets, seed 0
    python make_target_manifests.py --seeds 0 1 2
    python make_target_manifests.py --targets arabic --no-audit
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import soundfile as sf

SR = 16000
MAX_PER_CORPUS_CLASS = 6000     # extended_pipeline.py:96
TARGET_PER_CLASS = 3000         # extended_pipeline.py:96
LBL = {"real": 0, "fake": 1}

# The pool size each target had in the recorded 10-seed grid. Read off
# results_ext.csv (`n` for method=source, setting=transductive) and hardcoded
# here so this script fails loudly if a corpus directory has changed under it.
EXPECTED_N = {
    "asvspoof2019": 5580,
    "dataset2": 4447,
    "in_the_wild": 6000,
    "arabic": 5570,
    # asvspoof2021df is NOT in results_ext.csv -- it was never an
    # extended_pipeline target (it appears only as Protocol A, under its natural
    # 96.3%-fake skew). 6000 is what the same 3000-per-class rule yields here, so
    # the gate is by construction rather than against a prior run. Stated
    # explicitly because "the gate passed" means something weaker for this one.
    "asvspoof2021df": 6000,
    # Neither of these appears in results_ext.csv either -- same caveat as
    # asvspoof2021df: the gate is by construction, not against a prior run.
    "asvspoof2021la": 6000,
    "asvspoof2021pa": 6000,
}

# Targets that come from the pre-parsed manifest.csv rather than from a fresh
# directory walk. build_manifest.py already resolved ASVspoof2021-DF's labels and
# speakers out of trial_metadata.txt across three eval parts; re-globbing 61 GB
# to rediscover that would be slower and would risk disagreeing with it.
FROM_MANIFEST = {
    "asvspoof2021df": dict(csv="manifest.csv", source="dataset_1_asvspoof2021_DF"),
}

# Corpora read straight from an ASVspoof2021 key file. Both were already on disk
# and unused. Note the label column MOVES between tracks -- LA puts it at field 6,
# PA at field 10, because PA carries extra replay-configuration fields first.
# Reading column 6 for PA would silently label everything by microphone id.
FROM_KEYS = {
    "asvspoof2021la": dict(
        keys="data/dataset_1/LA-keys-full/keys/LA/CM/trial_metadata.txt",
        flac_globs=["data/dataset_1/ASVspoof2021_LA_eval/ASVspoof2021_LA_eval/flac"],
        label_col=5, lang="en",
    ),
    # Physical Access = REPLAY attacks: a genuine human voice played through a
    # speaker and re-recorded. That is a different detection problem from
    # synthetic-speech detection, and every model in this study was trained for
    # the latter. Included deliberately as an extreme out-of-domain arm -- the
    # place to find out whether adaptation is inert or harmful when the source
    # model has no usable ranking at all. Never to be pooled with the
    # synthetic-speech targets.
    "asvspoof2021pa": dict(
        keys="data/dataset_1/PA-keys-full/keys/PA/CM/trial_metadata.txt",
        flac_globs=[f"data/dataset_1/ASVspoof2021_PA_eval_part{p:02d}/ASVspoof2021_PA_eval/flac"
                    for p in range(4)],
        label_col=9, lang="en", different_task=True,
    ),
}


def build_manifest():
    """Verbatim from extended_pipeline.py, minus MLAAD (never an EER target)."""
    rows = []
    proto = "data/asvspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    flac = "data/asvspoof2019_LA/ASVspoof2019_LA_train/flac"
    if os.path.exists(proto):
        for line in open(proto):
            p = line.split()
            if len(p) >= 5:
                lab = "fake" if p[-1] == "spoof" else "real"
                gen = p[-2] if lab == "fake" else "bonafide"
                rows.append((f"{flac}/{p[1]}.flac", lab, "asvspoof2019", gen, "en"))

    for d in sorted(glob.glob("data/dataset_2/*/")):
        gen = os.path.basename(d.rstrip("/"))
        lab = "real" if gen == "real_samples" else "fake"
        for w in glob.glob(f"{d}/**/*.wav", recursive=True):
            rows.append((w, lab, "dataset2", gen, "en"))

    for corpus, root, lang in [("in_the_wild", "data/in_the_wild", "en"),
                               ("arabic", "data/arabic_arad", "ar")]:
        for w in glob.glob(f"{root}/**/*.wav", recursive=True):
            parts = w.split(os.sep)
            lab = "real" if "real" in parts else "fake" if "fake" in parts else None
            if lab:
                rows.append((w, lab, corpus, corpus, lang))

    return pd.DataFrame(rows, columns=["path", "label", "corpus", "generator", "language"])


def load_from_manifest(target, spec):
    """Read a target out of the repo's already-parsed manifest.csv.

    Returns the same five columns build_manifest() produces, so everything
    downstream (capping, sampling, the size gate) is shared.
    """
    df = pd.read_csv(spec["csv"], low_memory=False)
    df = df[df.dataset_source == spec["source"]]
    if df.empty:
        raise SystemExit(f"{spec['csv']} has no rows for {spec['source']!r}")
    # manifest.csv stores absolute paths; keep them, they are what will be read.
    out = pd.DataFrame({
        "path": df.filepath.values,
        "label": df.label.values,
        "corpus": target,
        "generator": df.generator.fillna("unknown").values,
        "language": "en",
    })
    print(f"  {target}: {len(out)} clips from {spec['csv']} "
          f"({out.label.value_counts().to_dict()}, "
          f"{df.speaker.nunique()} speakers)")
    return out


def load_from_keys(target, spec):
    """Build a corpus from an ASVspoof2021 CM key file plus its flac directories.

    Only trials whose audio is actually present are kept, and the count of
    missing ones is printed: a partially-downloaded eval set would otherwise
    silently shrink one class and shift every metric.
    """
    lab_col = spec["label_col"]
    labels, speakers = {}, {}
    for line in open(spec["keys"]):
        p = line.split()
        if len(p) <= lab_col:
            continue
        raw = p[lab_col]
        if raw not in ("spoof", "bonafide"):
            continue
        labels[p[1]] = "fake" if raw == "spoof" else "real"
        speakers[p[1]] = p[0]

    present, missing = [], 0
    for d in spec["flac_globs"]:
        for f in glob.glob(os.path.join(d, "*.flac")):
            stem = os.path.splitext(os.path.basename(f))[0]
            if stem in labels:
                present.append((f, labels[stem], speakers[stem]))
            else:
                missing += 1
    if not present:
        raise SystemExit(f"{target}: no audio matched {spec['keys']}")

    out = pd.DataFrame({
        "path": [p[0] for p in present],
        "label": [p[1] for p in present],
        "corpus": target,
        "generator": [p[2] for p in present],
        "language": spec.get("lang", "en"),
    })
    counts = out.label.value_counts().to_dict()
    print(f"  {target}: {len(out)} clips with labels {counts}, "
          f"{out.generator.nunique()} speakers, {missing} audio files not in the key list")
    return out


def cap_per_class(df, n, seed=0):
    return pd.concat([g.sample(min(n, len(g)), random_state=seed) for _, g in df.groupby("label")])


def sample_target(df, n_per_class, seed=0):
    return pd.concat([g.sample(min(n_per_class, len(g)), random_state=seed)
                      for _, g in df.groupby("label")]).sample(frac=1, random_state=seed)


def audit(df, name, limit=0):
    """Try to decode every clip the way the TTA arm will. Reports, never fixes.

    Mirrors `public_ckpt_tta.load_clip`: libsndfile first, FFmpeg on failure.
    The two counts are reported separately because they mean different things --
    libsndfile failures are expected on ASVspoof FLACs and are handled, whereas
    a clip neither decoder can read would silently become a zero vector and must
    block the run.
    """
    paths = df.path.values if not limit else df.path.values[:limit]
    sf_fails, hard_fails, rates = [], [], {}
    for p in paths:
        counted = False
        try:
            info = sf.info(p)
            rates[info.samplerate] = rates.get(info.samplerate, 0) + 1
            counted = True   # header read; don't count this clip's rate twice
            # sf.info reads the header only; force a real decode of the head,
            # which is where libsndfile's FLAC failures actually surface.
            sf.read(p, frames=1024, dtype="float32")
        except Exception as e:
            sf_fails.append(p)
            try:
                from public_ckpt_tta import _ffmpeg_decode
                x, sr = _ffmpeg_decode(p)
                if not counted:
                    rates[sr] = rates.get(sr, 0) + 1
                if len(x) == 0 or not np.any(x):
                    hard_fails.append((p, "ffmpeg returned silence"))
            except Exception as e2:
                hard_fails.append((p, f"{type(e2).__name__}: {e2}"))
    print(f"  audit {name}: {len(paths)} clips, {len(sf_fails)} need the FFmpeg "
          f"fallback, {len(hard_fails)} undecodable by either, "
          f"sample rates {dict(sorted(rates.items()))}")
    if hard_fails:
        print(f"    !! UNDECODABLE (would become silence): {hard_fails[:3]}")
        raise SystemExit(
            f"{len(hard_fails)} clips in {name} cannot be decoded at all. Zero-filling "
            f"them would teach 'silence => real' -- fix or drop them before running.")
    return sf_fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=list(EXPECTED_N))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--no-audit", action="store_true")
    ap.add_argument("--audit_limit", type=int, default=0, help="0 = audit every clip")
    args = ap.parse_args()

    walk_targets = [t for t in args.targets
                    if t not in FROM_MANIFEST and t not in FROM_KEYS]
    manifest = build_manifest() if walk_targets else pd.DataFrame(
        columns=["path", "label", "corpus", "generator", "language"])
    for target, spec in FROM_MANIFEST.items():
        if target in args.targets:
            manifest = pd.concat([manifest, load_from_manifest(target, spec)],
                                 ignore_index=True)
    for target, spec in FROM_KEYS.items():
        if target in args.targets:
            manifest = pd.concat([manifest, load_from_keys(target, spec)],
                                 ignore_index=True)
    print("manifest:\n" + manifest.groupby(["corpus", "label"]).size().to_string())

    for target in args.targets:
        corpus_df = manifest[manifest.corpus == target]
        if corpus_df.empty:
            raise SystemExit(f"no clips found for target {target!r} -- is data/ populated?")
        capped = cap_per_class(corpus_df, MAX_PER_CORPUS_CLASS)

        for seed in args.seeds:
            tgt = sample_target(capped, TARGET_PER_CLASS, seed).reset_index(drop=True)
            exp = EXPECTED_N.get(target)
            if exp is not None and len(tgt) != exp:
                raise SystemExit(
                    f"POOL SIZE GATE FAILED for {target} seed {seed}: built {len(tgt)} "
                    f"clips, results_ext.csv recorded {exp}. The corpus on disk no "
                    f"longer reproduces the pool the grid was run on -- stop and "
                    f"reconcile before adding any number to a plot.")

            out = pd.DataFrame({
                "path": tgt.path.values,
                "label": tgt.label.map(LBL).values.astype(np.int64),
                "label_name": tgt.label.values,
                "corpus": tgt.corpus.values,
                "generator": tgt.generator.values,
            })
            path = os.path.join(args.out_dir, f"manifest_tgt_{target}_seed{seed}.csv")
            out.to_csv(path, index=False)
            counts = out.label_name.value_counts().to_dict()
            print(f"wrote {path}: {len(out)} clips {counts}"
                  + (f"  [size gate ok, matches results_ext n={exp}]" if exp else ""))

            if not args.no_audit and seed == args.seeds[0]:
                audit(out, target, args.audit_limit)


if __name__ == "__main__":
    main()
