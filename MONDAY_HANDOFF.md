# Handoff: adaptive q/E cloud run

Self-contained runbook for a fresh session on a cloud box with **no data and no
repo**. Everything needed is either in this repo or listed under Uploads.

Paste the block in §0 as the opening message of a new session, then work through
§2–§7. `cloud_monday.ipynb` is the same sequence as a notebook if you prefer
cells; this file is the shell version.

---

## 0. Opening prompt for a new session

> I'm running the Monday GPU session for the audio-deepfake TTA project on a
> fresh cloud box (~35 GB GPU, no data). Follow `MONDAY_HANDOFF.md` in the repo
> at github.com/RohamIzadidoost/Kaggle_deepFakeAudio — clone it first, then work
> through sections 2–7 in order. Two rules: `verify_reduction.py` in §4 is a
> hard gate, do not start the grid if it fails; and never let anything download
> MLAAD (57 GB) — with `ckpt_ext/` uploaded nothing trains, so it is not needed.
> Report back the table from §7.

---

## 1. What this is, in one paragraph

The TTA method's hyperparameters (`q=0.3`, `λ=0.3`, `E=4`) were inherited, not
tuned. This run evaluates making **q** adapt (curriculum ramp + prevalence-split
budget) and **E** adapt (run to 8, stop on score movement). λ stays fixed —
three mechanisms for adapting it were falsified on CPU first, see
`PROJECT_LOG.md` §11. Two questions: does `ours_adaptive` beat `ours_fixed` on
the 4-target × 3-seed grid, and does adaptive `q` fix Protocol A, where a fixed
symmetric `q` drives EER from 26.33% to 42.45% on a 97.5%-spoof pool.

**Nothing here trains a source model.** All 12 source checkpoints are uploaded.

---

## 2. Uploads — do this first

| file | size | consequence if missing |
|---|---|---|
| `ckpt_ext/` (12 `.pt`) | 2.3 GB | **every fold silently skips**; the grid "finishes" in 2 min with an empty CSV |
| `kaggle.json` | 1 KB | no corpora |
| `ckpt_protocol_a_rawboost.pt` | 193 MB | Protocol A retrains 16 epochs (+35 min) |
| `results_protocol_a.csv` | 1 KB | Protocol A runs 4 arms instead of 1 (+4 h) |

Put them in the repo root after cloning.

---

## 3. Setup (~40 min, mostly download)

Clone:

```bash
git clone https://github.com/RohamIzadidoost/Kaggle_deepFakeAudio.git ~/deepfake && cd ~/deepfake && mkdir -p data logs
```

Dependencies — librosa/numba are removed deliberately, they pull a numpy that
breaks the pandas ABI here:

```bash
pip uninstall -y -q librosa numba && pip install -q "numpy==1.26.4" pandas scikit-learn scipy torch torchaudio soundfile datasets tqdm kaggle && pip install -q "numpy==1.26.4"
```

Kaggle credentials:

```bash
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

Verify the uploads landed — this must print 12:

```bash
ls ckpt_ext/source_*_seed*.pt | wc -l
```

Target corpora only (~19 GB). **MLAAD is not in this list on purpose**: it is a
source-diversity corpus, and nothing here trains:

```bash
kaggle datasets download -d azkurniwan/asvspoof-2019-la -p data/asvspoof2019_LA --unzip
```

```bash
kaggle datasets download -d bhaveshkumars/release-in-the-wild -p data/in_the_wild --unzip
```

```bash
kaggle datasets download -d adarshsingh0903/audio-deepfake-detection-dataset -p data/dataset_2 --unzip
```

Arabic ArAD comes from HuggingFace, not Kaggle:

```bash
python -c "
from datasets import load_dataset, Audio
import os
ds = load_dataset('DeepFake-Audio-Rangers/Arabic_Audio_Deepfake').cast_column('audio', Audio(decode=False))
names = ds['train'].features['label'].names
for split in ds:
    for i, ex in enumerate(ds[split]):
        d = f'data/arabic_arad/{split}/{names[ex[\"label\"]]}'
        os.makedirs(d, exist_ok=True)
        open(f'{d}/{i}.wav','wb').write(ex['audio']['bytes'])
print('arabic done')"
```

Start the 61 GB Protocol A download **now** so it runs during the grid and costs
no serial time:

```bash
nohup kaggle datasets download -d mohammedabdeldayem/avsspoof-2021 -p data/dataset_1 --unzip > logs/df_download.log 2>&1 &
```

---

## 4. Hard gate: the reduction check (~5 min)

Asserts `adapt_adaptive` with its schedules off is **bitwise identical** to the
published `adapt()`.

```bash
python verify_reduction.py
```

Expect `max |difference| : 0.000e+00` and `PASS`.

**If it fails, stop.** A non-zero difference means adaptive-vs-fixed differences
come from the refactor rather than the schedules, and every number the grid
produces is uninterpretable. Debug this instead of starting the grid.

---

## 5. Smoke run (~2 min)

```bash
ADAPTIVE_SMOKE=1 python adaptive_pipeline.py
```

Writes `*_smoke` files only. **Ignore the EERs** — they are on ~128 clips, where
one clip is ~0.8% EER. You are checking two things in the log:

- `q=0.100 … q=0.300` climbing across epochs, `conf=` growing with it
- `shift=` decaying (e.g. 0.051 → 0.037 → 0.022 → 0.009)

If `q` is pinned at one value or `shift` is `nan` after epoch 1, something is
wrong with the schedule — do not start the grid.

---

## 6. The grid (~6 h) and Protocol A (~1 h)

`nohup` so a dropped connection can't kill it. Rows append as each arm finishes,
and a restart resumes per `(seed, target, method, setting)`:

```bash
nohup env ADAPTIVE_SMOKE=0 python adaptive_pipeline.py > logs/grid.log 2>&1 &
```

Watch it — expect ~24 min per fold, 12 folds:

```bash
tail -f run_log_adaptive.txt
```

**Only after the grid finishes** (both want the GPU). First confirm the DF eval
arrived and check comparability:

```bash
ls -d data/dataset_1/ASVspoof2021_DF_eval_part0* && ls data/dataset_1/DF-keys-full/keys/DF/CM/trial_metadata.txt
```

The existing Protocol A rows scored **400,435** trials because the local copy had
only `part00`–`part02`. If the count above is not 3 parts, the new adaptive row
lands on a *different pool* and cannot be compared to the `source_rawboost` /
`ours_rawboost` rows it is meant to sit beside — either restrict `DF_PARTS` in
`protocol_a.py` to `part0[0-2]`, or plan to re-run every arm.

```bash
nohup python protocol_a.py > logs/protocol_a.log 2>&1 &
```

It should skip `source_rawboost`, `ours_rawboost`, `ours_prior_rawboost` as
already recorded and run only `ours_adaptive_rawboost`. If it starts training a
source model, `ckpt_protocol_a_rawboost.pt` didn't upload.

---

## 7. Report this

```bash
python -c "
import pandas as pd, os
d = pd.read_csv('results_adaptive.csv')
t = d[d.setting=='transductive'].pivot_table(index=['target','seed'], columns='method', values='eer')
t['delta'] = t['ours_adaptive'] - t['ours_fixed']
print(t.round(3).to_string())
print()
print('mean delta by target (negative = adaptive better):')
print(t.groupby('target').delta.mean().round(3).to_string())
print(f'improved in {(t.delta<0).sum()}/{t.delta.notna().sum()} points')
if os.path.exists('results_protocol_a.csv'):
    print()
    print(pd.read_csv('results_protocol_a.csv').to_string(index=False))"
```

Three things to state explicitly when reporting:

1. **Does `ours_fixed` reproduce `results_ext.csv` seeds 0–2?** (`ours` there:
   asvspoof2019 ≈ 4.25/4.27/2.55, dataset2 ≈ 36.7/32.3/34.1, in_the_wild ≈
   13.2/9.4/9.9, arabic ≈ 19.9/21.9/20.4.) If it doesn't, the adaptive
   comparison is still internally valid but can't be dropped into the published
   table as-is.
2. **Sign consistency**, "improved in k/12 points", not just the mean — this is
   the convention the paper already uses. 3 seeds does not support a
   significance claim; don't make one.
3. **Protocol A**: whether adaptive `q` beats the 42.45% collapse *and* whether
   it beats the 26.33% do-nothing source baseline. Beating the collapse but not
   the baseline is still a negative result and should be reported as one.

Save everything off the box before the session ends:

```bash
tar czf adaptive_results.tar.gz results_adaptive.csv adaptive_trace.csv run_log_adaptive.txt results_protocol_a.csv run_log_protocol_a_rawboost.txt logs/
```

---

## 8. Traps

- **Never let anything download MLAAD (57 GB).** `extended_pipeline.py`'s data
  cell starts it automatically if `data/mlaad` is missing. `adaptive_pipeline.py`
  asserts instead of fetching. Don't run `extended_pipeline.py` on this box.
- **Don't edit `extended_pipeline.py` to add methods.** Its resume guard keys on
  `(seed, target, setting)` only, so it silently skips all 20 folds.
- **~8 near-duplicate copies of the TTA `adapt()` loop exist** in this repo with
  genuinely different defaults. `adaptive_pipeline.py` is the one for this run.
- **Smoke and real runs write to different files** (`*_smoke`). Don't report
  smoke numbers.
- `adaptive_pipeline.py` never trains and never downloads — if a fold is skipped,
  the checkpoint is missing, not broken.
