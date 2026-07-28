"""
Unified manifest builder for the 3 deepfake-audio datasets.

Datasets:
  dataset_1 = ASVspoof2021 DF eval (3 parts) -> flac files + trial_metadata.txt labels
  dataset_2 = SeedTTS-style set -> real_samples/ (real) + generator folders (fake)
  dataset_3 = MLAAD v5 -> fake/<lang>/<model>/meta.csv (all fake, no real audio bundled)

Output: a single CSV manifest with columns:
  filepath, label, dataset_source, generator, extra
where label is one of {"real", "fake"}.

Run:
  python build_manifest.py --root /home/general/Desktop/deepfake_project/data --out manifest.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# Some MLAAD meta.csv transcript fields exceed Python's default 131072-byte
# per-field limit. Raise it (fall back gracefully on platforms where the
# platform C long is 32-bit and sys.maxsize is too large for the C int field).
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


def load_asvspoof_labels(ds1_root: Path):
    """
    Scan all ASVspoof2021_DF_eval_partXX folders under dataset_1, find flac/ dirs,
    and pull labels from trial_metadata.txt (column index 5: 'spoof' or 'bonafide').
    trial_metadata.txt is expected once, under DF-keys-full/keys/DF/CM/trial_metadata.txt
    """
    rows = []

    # 1. find the DF trial_metadata.txt label file specifically.
    # There are near-identical files under DF-keys-full, LA-keys-full, and
    # PA-keys-full — we must not just take the first rglob match, since glob
    # order is filesystem-dependent and the wrong one silently gives 0 matches.
    meta_path = None
    candidates = list(ds1_root.rglob("trial_metadata.txt"))
    for p in candidates:
        if "/DF/" in str(p) or "DF-keys" in str(p):
            meta_path = p
            break
    if meta_path is None and candidates:
        print(f"[dataset_1] WARNING: no DF-specific trial_metadata.txt found among "
              f"{len(candidates)} candidates; falling back to first match: {candidates[0]}")
        meta_path = candidates[0]
    if meta_path is None:
        print("[dataset_1] WARNING: trial_metadata.txt not found, skipping labels.")
        return rows

    label_map = {}
    speaker_map = {}
    with open(meta_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            speaker = parts[0]           # e.g. LA_0023 / TEF2 — shared by real & fake
            filename = parts[1]          # e.g. DF_E_2000011
            raw_label = parts[5]         # 'spoof' or 'bonafide'
            label_map[filename] = "fake" if raw_label == "spoof" else "real"
            speaker_map[filename] = speaker

    print(f"[dataset_1] loaded {len(label_map)} labels from {meta_path}")

    # 2. find every flac/ dir under any *_DF_eval_part* folder and match to labels
    n_found, n_unmatched = 0, 0
    for flac_dir in ds1_root.rglob("flac"):
        if "DF_eval" not in str(flac_dir):
            continue
        for wav_file in flac_dir.glob("*.flac"):
            stem = wav_file.stem
            label = label_map.get(stem)
            if label is None:
                n_unmatched += 1
                continue
            rows.append({
                "filepath": str(wav_file),
                "label": label,
                "dataset_source": "dataset_1_asvspoof2021_DF",
                "generator": "unknown",  # could parse col 8/9 of metadata if needed
                "speaker": f"asv:{speaker_map.get(stem, stem)}",
                "extra": "",
            })
            n_found += 1

    print(f"[dataset_1] matched {n_found} files, {n_unmatched} unmatched")
    return rows


def load_seedtts_style(ds2_root: Path):
    """
    dataset_2 structure:
      real_samples/         -> real
      seedtts_files/        -> treated as fake (TTS-generated pairs, not in real_samples)
      NaturalSpeech3/, FlashSpeech/, OpenAI/, VALLE/, xTTS/, PromptTTS2/, VoiceBox/  -> fake
    """
    rows = []
    if not ds2_root.exists():
        print("[dataset_2] root not found, skipping.")
        return rows

    for sub in sorted(p for p in ds2_root.iterdir() if p.is_dir()):
        name = sub.name
        if name == "real_samples":
            label = "real"
            generator = "real"
        else:
            label = "fake"
            generator = name  # e.g. NaturalSpeech3, VALLE, seedtts_files, etc.

        count = 0
        for wav_file in sub.rglob("*.wav"):
            if label == "real":
                # real_samples are LibriSpeech: '<speaker>_<chapter>_<utt>_...wav'
                speaker = f"ds2:{wav_file.stem.split('_')[0]}"
            else:
                # generator filenames don't share a consistent speaker scheme;
                # give each its own group so it never forces a leak, and never
                # co-locates with a mismatched real speaker.
                speaker = f"ds2g:{wav_file.stem}"
            rows.append({
                "filepath": str(wav_file),
                "label": label,
                "dataset_source": "dataset_2_seedtts_style",
                "generator": generator,
                "speaker": speaker,
                "extra": "",
            })
            count += 1
        print(f"[dataset_2] {name}: {count} files -> label={label}")

    return rows


def load_mlaad(ds3_root: Path):
    """
    dataset_3 (MLAAD v5) structure:
      fake/<lang>/<model_folder>/meta.csv  with columns:
        path|original_file|language|is_original_language|duration|training_data|model_name|architecture|transcript
      No bundled 'real' folder -> everything here is fake.
    """
    def mlaad_speaker(original_file: str, fallback: str) -> str:
        # original_file looks like
        #   '<lang>/by_book/<gender>/<reader>/<book>/wavs/<file>.wav'
        # The reader is the group we want (its utterances are re-synthesised by
        # every TTS model, so they must not straddle train/test). Fall back to
        # the model name when the path doesn't follow that layout.
        parts = [p for p in original_file.split("/") if p]
        if len(parts) >= 4 and parts[-2] == "wavs":
            return f"mlaad:{parts[-4]}"
        if "by_book" in parts:
            i = parts.index("by_book")
            if i + 2 < len(parts):
                return f"mlaad:{parts[i + 2]}"
        return f"mlaad:{fallback}"

    rows = []
    fake_root = ds3_root / "Mlaad_v5" / "mlaad_v5" / "fake"
    if not fake_root.exists():
        print(f"[dataset_3] fake root not found at {fake_root}, skipping.")
        return rows

    n_meta_rows = 0
    for meta_csv in fake_root.rglob("meta.csv"):
        model_dir = meta_csv.parent
        with open(meta_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                # 'path' column looks like './fake/<lang>/<model>/filename.wav'.
                # Don't trust the <lang> segment: ~1000 MLAAD v5 rows have it
                # wrong (e.g. 'zh' instead of the real 'zh-cn' dir), which
                # silently points at a nonexistent file. model_dir is known
                # correct since it came from actually walking the filesystem,
                # and every audio file lives directly under its model_dir, so
                # just join the filename onto it.
                filename = Path(row.get("path", "")).name
                candidate = model_dir / filename
                model_name = row.get("model_name", model_dir.name)
                rows.append({
                    "filepath": str(candidate),
                    "label": "fake",
                    "dataset_source": "dataset_3_mlaad_v5",
                    "generator": model_name,
                    "speaker": mlaad_speaker(row.get("original_file", ""), model_dir.name),
                    "extra": row.get("language", ""),
                })
                n_meta_rows += 1

    print(f"[dataset_3] loaded {n_meta_rows} rows from meta.csv files")
    return rows


def load_asvspoof2019_la(root: Path):
    """ASVspoof2019 LA (train/dev/eval) from the CM protocol files.

    Protocol line: '<speaker> <utt_id> - <attack> <label>'
    (label at the last column is 'bonafide' or 'spoof'). Speaker (LA_00xx) is
    shared by real & fake, so we keep it for speaker-disjoint splitting.
    """
    base = root / "asvspoof2019_LA"
    if not base.exists():
        print("[asvspoof2019_LA] not found, skipping.")
        return []
    proto_dir = base / "ASVspoof2019_LA_cm_protocols"
    splits = {
        "train": ("ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train"),
        "dev": ("ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev"),
        "eval": ("ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval"),
    }
    rows = []
    for subset, (pfile, adir) in splits.items():
        ppath = proto_dir / pfile
        flac_dir = base / adir / "flac"
        if not ppath.exists() or not flac_dir.exists():
            print(f"[asvspoof2019_LA] {subset}: protocol or flac dir missing, skipping subset.")
            continue
        n = 0
        with open(ppath) as f:
            for line in f:
                p = line.split()
                if len(p) < 5:
                    continue
                spk, utt, attack, label = p[0], p[1], p[-2], p[-1]
                rows.append({
                    "filepath": str(flac_dir / f"{utt}.flac"),
                    "label": "fake" if label == "spoof" else "real",
                    "dataset_source": "asvspoof2019_LA",
                    "generator": attack if attack != "-" else "bonafide",
                    "speaker": f"asv19:{spk}",
                    "extra": subset,
                })
                n += 1
        print(f"[asvspoof2019_LA] {subset}: {n} rows")
    return rows


def load_in_the_wild(root: Path):
    """In-the-Wild (OOD test corpus), folder-based real/fake .wav.

    Used as a held-out cross-corpus test set, so each file is its own speaker
    group (per-file) — it is never in the training pool, so no leakage concern.
    """
    base = root / "in_the_wild"
    if not base.exists():
        print("[in_the_wild] not found, skipping.")
        return []
    rows = []
    for wav in base.rglob("*.wav"):
        low = [part.lower() for part in wav.parts]
        if "fake" in low or "spoof" in low:
            label = "fake"
        elif "real" in low or "bonafide" in low:
            label = "real"
        else:
            continue
        rows.append({
            "filepath": str(wav),
            "label": label,
            "dataset_source": "in_the_wild",
            "generator": "in_the_wild",
            "speaker": f"itw:{wav.stem}",
            "extra": "",
        })
    n_real = sum(1 for r in rows if r["label"] == "real")
    print(f"[in_the_wild] {len(rows)} rows ({n_real} real / {len(rows) - n_real} fake)")
    return rows


def load_arabic_arad(root: Path):
    """Arabic Audio Deepfake (ArAD) — folder-based real/fake, extracted by
    download_arabic.py to arabic_arad/<split>/<real|fake>/*.wav. Multilingual OOD
    target (Arabic); per-file speaker since it is only ever a held-out test corpus.
    """
    base = root / "arabic_arad"
    if not base.exists():
        print("[arabic_arad] not found, skipping.")
        return []
    rows = []
    for wav in base.rglob("*.wav"):
        parts = [p.lower() for p in wav.parts]
        label = "real" if "real" in parts else "fake" if "fake" in parts else None
        if label is None:
            continue
        split = "test" if "test" in parts else "train" if "train" in parts else ""
        rows.append({
            "filepath": str(wav),
            "label": label,
            "dataset_source": "arabic_arad",
            "generator": "arabic_arad",
            "speaker": f"arad:{wav.parent.parent.name}:{wav.stem}",
            "extra": split,
        })
    n_real = sum(1 for r in rows if r["label"] == "real")
    print(f"[arabic_arad] {len(rows)} rows ({n_real} real / {len(rows) - n_real} fake)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to the 'data' folder containing dataset_1/2/3")
    ap.add_argument("--out", default="manifest.csv", help="Output CSV path")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    ds1 = root / "dataset_1"
    ds2 = root / "dataset_2"
    ds3 = root / "dataset_3"

    all_rows = []
    all_rows += load_asvspoof_labels(ds1)
    all_rows += load_seedtts_style(ds2)
    all_rows += load_mlaad(ds3)
    # extra corpora for the DA-RAD cross-corpus study (skipped gracefully if absent)
    all_rows += load_asvspoof2019_la(root)
    all_rows += load_in_the_wild(root)
    all_rows += load_arabic_arad(root)

    if not all_rows:
        print("No rows collected — check your --root path.")
        return

    fieldnames = ["filepath", "label", "dataset_source", "generator", "speaker", "extra"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    n_real = sum(1 for r in all_rows if r["label"] == "real")
    n_fake = sum(1 for r in all_rows if r["label"] == "fake")
    print(f"\nWrote {len(all_rows)} rows to {args.out}")
    print(f"  real: {n_real}")
    print(f"  fake: {n_fake}")


if __name__ == "__main__":
    main()
