"""Fetch an independent audio-deepfake corpus from HuggingFace and build a target pool.

Why: the study is ASVspoof-bound on both axes. Seven of nine third-party
checkpoints are ASVspoof-trained, four of seven targets are ASVspoof tracks, and
ASVspoof2019 sits inside the source pool for most folds. The finding that most
depends on corpus diversity -- that extending the target set erases the pooled EER
benefit -- currently rests on three ASVspoof relatives, which a reviewer can
fairly discount.

These three are independent of ASVspoof lineage:

  wavefake     vocoder artefacts over LJSpeech; a different generation family
               entirely. IN-DOMAIN for ssl_aasist_wavefake -- exclude there.
  commercialtts 2024 commercial voice cloning (ElevenLabs, Polly, Kokoro, Hume,
               Speechify); the most deployment-relevant distribution available.
               IN-DOMAIN for hf_xlsr_stafford -- exclude there.
  romanian     a third language, non-ASVspoof.

Audio is decoded to 16 kHz mono wav so the existing loaders need no special
cases, and pools are capped by the same rule as every other target
(MAX_PER_CORPUS_CLASS then TARGET_PER_CLASS per seed) so cell sizes stay
comparable.

    python fetch_hf_corpus.py --corpus wavefake
    python fetch_hf_corpus.py --corpus all --seeds 0 1 2 3 4 5 6 7 8 9
"""

import argparse
import os

import numpy as np
import pandas as pd
import soundfile as sf

SR = 16000
MAX_PER_CORPUS_CLASS = 6000
TARGET_PER_CLASS = 3000
LBL = {"real": 0, "fake": 1}

CORPORA = {
    "wavefake": dict(
        repo="ajaykarthick/wavefake-audio",
        label_col="real_or_fake",
        # Values are "R" (13,100 LJSpeech originals) and WF1..WF7 (7 vocoders,
        # 13,100 each). Read from the data -- the first guess of real/fake matched
        # nothing and produced an empty corpus.
        map={"R": "real", **{f"WF{i}": "fake" for i in range(1, 8)}},
        note="LJSpeech + 6 vocoders; in-domain for ssl_aasist_wavefake",
    ),
    "commercialtts": dict(
        repo="garystafford/deepfake-audio-detection",
        label_col="label",
        map={0: "real", 1: "fake", "real": "real", "fake": "fake"},
        note="ElevenLabs/Polly/Kokoro/Hume/Speechify; in-domain for hf_xlsr_stafford",
    ),
    "romanian": dict(
        repo="alexlicuriceanu/ro-dia-deepfake-audio",
        label_col=None,          # labelled by path: fake/... vs real/...
        map=None,
        note="Romanian; adds a third language",
    ),
}


def to_wav(arr, sr, path):
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        import torch, torchaudio
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
    sf.write(path, x, SR)


def fetch(name, limit_per_class=MAX_PER_CORPUS_CLASS, retries=5):
    """Download and decode one corpus.

    Three failure modes hit on the first attempt, all handled here:

    * `NonMatchingSplitsSizesError` -- ajaykarthick/wavefake-audio ships metadata
      claiming 64,800 examples while the shards hold 104,800. The data is fine;
      only the recorded count is stale, so verification is disabled rather than
      the download abandoned.
    * `ImportError: torchcodec` -- datasets 5.0 decodes audio through torchcodec.
      Installing it risks the pinned torch/torchaudio stack this repo depends on,
      so we take the audio as raw bytes (`Audio(decode=False)`) and decode with
      soundfile instead. Same route adaptive_pipeline already uses for Arabic.
    * HTTP 429 -- unauthenticated Hub requests get rate limited; retried with
      exponential backoff rather than failing the corpus.
    """
    import io, time as _t
    from datasets import load_dataset, Audio
    spec = CORPORA[name]
    root = f"data/hf_{name}"
    if os.path.exists(f"{root}/.complete"):
        print(f"  {name}: already fetched")
        return root
    for c in ("real", "fake"):
        os.makedirs(f"{root}/{c}", exist_ok=True)

    ds = None
    for attempt in range(retries):
        try:
            ds = load_dataset(spec["repo"], split="train", verification_mode="no_checks")
            break
        except Exception as e:
            if "429" not in str(e) and "Too Many Requests" not in str(e):
                raise
            wait = 30 * (2 ** attempt)
            print(f"  {name}: rate limited, retrying in {wait}s ({attempt+1}/{retries})")
            _t.sleep(wait)
    if ds is None:
        raise RuntimeError(f"{name}: still rate limited after {retries} attempts")
    # decode=False -> raw bytes, decoded by soundfile below; avoids torchcodec
    ds = ds.cast_column("audio", Audio(decode=False))
    kept = {"real": 0, "fake": 0}
    for i, ex in enumerate(ds):
        if spec["label_col"] is None:
            p = str((ex.get("audio") or {}).get("path", "") or "")
            lab = "fake" if "/fake/" in p or p.startswith("fake/") else \
                  "real" if "/real/" in p or p.startswith("real/") else None
        else:
            lab = spec["map"].get(ex.get(spec["label_col"]))
        if lab is None or kept[lab] >= limit_per_class:
            if min(kept.values()) >= limit_per_class:
                break
            continue
        a = ex["audio"]
        try:
            arr, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=False)
        except Exception:
            continue                       # unreadable clip: skip, never zero-fill
        to_wav(arr, sr, f"{root}/{lab}/{kept[lab]:06d}.wav")
        kept[lab] += 1
        if (kept["real"] + kept["fake"]) % 1000 == 0:
            print(f"  {name}: {kept}")
    print(f"  {name}: fetched {kept}")
    if min(kept.values()) == 0:
        raise RuntimeError(f"{name}: one class is empty ({kept}) -- labelling rule is wrong, "
                           f"do not build a pool from this")
    open(f"{root}/.complete", "w").write(str(kept))
    return root


def build_pools(name, seeds):
    root = f"data/hf_{name}"
    rows = []
    for c in ("real", "fake"):
        for f in sorted(os.listdir(f"{root}/{c}")):
            if f.endswith(".wav"):
                rows.append((f"{root}/{c}/{f}", c))
    df = pd.DataFrame(rows, columns=["path", "label"])
    capped = pd.concat([g.sample(min(MAX_PER_CORPUS_CLASS, len(g)), random_state=0)
                        for _, g in df.groupby("label")])
    for seed in seeds:
        tgt = pd.concat([g.sample(min(TARGET_PER_CLASS, len(g)), random_state=seed)
                         for _, g in capped.groupby("label")]).sample(frac=1, random_state=seed)
        out = pd.DataFrame({"path": tgt.path.values,
                            "label": tgt.label.map(LBL).values.astype(np.int64),
                            "label_name": tgt.label.values,
                            "corpus": f"hf_{name}"})
        p = f"manifest_tgt_hf_{name}_seed{seed}.csv"
        out.to_csv(p, index=False)
        print(f"  wrote {p}: {len(out)} clips {out.label_name.value_counts().to_dict()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="all", choices=list(CORPORA) + ["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    args = ap.parse_args()
    names = list(CORPORA) if args.corpus == "all" else [args.corpus]
    for n in names:
        print(f"=== {n}: {CORPORA[n]['note']}")
        try:
            fetch(n)
            build_pools(n, args.seeds)
        except Exception as e:
            print(f"  !! {n} failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
