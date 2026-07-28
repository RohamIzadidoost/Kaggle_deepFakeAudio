# Cloud Sprint — Run Guide

Everything needed to produce the paper's results on the cloud resource. Keep it simple: upload two files, set one secret, Run All, copy numbers.

## 0. Request the compute
GPU 24 GB · 16 CPU cores · 64 GB RAM · 150 GB storage · 1 GPU · 3 days.

## 1. Upload to the cloud environment
- `cloud_pipeline.ipynb`  (the notebook)
- `kaggle.json`  (your Kaggle API token, from kaggle.com → Account → Create New Token)

**Kernel:** select **Python 3 (ipykernel)** — this is Python 3.10.12, confirmed
compatible. Do **not** use the Python 3.8 kernel (EOL; forces old package versions).
Cell 0 is an environment check that prints torch/torchaudio/CUDA/bf16 and verifies
the XLS-R bundle — run it first; it fails fast with a clear message if anything is off.

## 2. One-time setup (run once, in a terminal or the first cell)
```bash
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```
That is the only manual step. The notebook installs its own Python deps in cell 1.

## 3. Run
Kernel → **Restart & Run All**. Order of what happens:
1. install deps  2. download data (Kaggle: ASVspoof2019-LA, In-the-Wild; HF: Arabic)
3. build manifest  4. train source model  5. test-time adaptation  6. **results table**

First run downloads ~18 GB of data + the 1.2 GB XLS-R weights (automatic). Expect
roughly 1–2 h end-to-end at the default sizes; longer once you scale up (step 5).

## 4. Outputs → paper
The **last cell** prints a table like:

| target | source EER% | source AUC | Tent EER% | Tent AUC | ours EER% | ours AUC |
|---|---|---|---|---|---|---|
| in_the_wild | … | … | … | … | … | … |
| arabic | … | … | … | … | … | … |

Copy these into `main.tex`:
- **Table `tab:main`** ← the `source` and `ours` columns for each target (and Tent to show it collapses).
- Fill the **Arabic** row (currently a placeholder `X.XX / Y.YY`).
- The insight figure `fig_score_dist.png` is already generated from real scores — just upload it to Overleaf. (To regenerate at cloud scale: in the results loop, `np.savez("fig_score_dist.npz", y=y, s_src=<source scores>, s_adp=<adapted scores>)` for In-the-Wild, then run `make_figure.py`.)

## 5. Scale up for the final numbers (config at top of the notebook)
Defaults are set small for a fast first pass. For the paper run, raise:
- `SOURCE_PER_CLASS` → e.g. 8000–15000 (or all)
- `TARGET_PER_CLASS` → e.g. 4000
- `SOURCE_EPOCHS` → 8–10, `TTA_EPOCHS` → 4–6
- Run the whole notebook for **3 seeds** (change `torch.manual_seed(0)` → 0,1,2) and report mean ± std, as in the current tables.

Optional if time allows: add `asvspoof2021_DF` and a held-out MLAAD language as extra target rows (same pattern as In-the-Wild in the manifest + results loop).

## 6. Known gotchas
- **Kaggle 403 / "dataset not found":** the token isn't in `~/.kaggle/kaggle.json`, or you haven't accepted the dataset's rules once on the website.
- **torch / torchaudio mismatch:** if the cloud image lacks a working torch, `pip install torch torchaudio` matching the CUDA version, then restart the kernel.
- **DataLoader worker crash:** the notebook already forces the `fork` start method; if the platform forbids it, set `num_workers=0` in `loader()`.
- **OOM:** lower the batch size in `loader(df, bs=16 → 8)`.

## Method recap (one line)
Fine-tune XLS-R on a source corpus → adapt unsupervised to each unseen target with
confident-pseudo-label self-training + channel consistency. Ranking transfers across
corpora even when the threshold doesn't, which is what makes the self-training stable.
