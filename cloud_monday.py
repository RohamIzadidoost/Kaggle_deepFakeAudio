# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Monday cloud session — adaptive q/E grid + Protocol A
#
# Fresh cloud box, no data, ~35 GB GPU. This notebook builds every directory and
# fetches every corpus it needs.
#
# ## Before you start: upload these four things
#
# | upload | size | why |
# |---|---|---|
# | `ckpt_ext/` (12 `.pt` files) | 2.3 GB | **skips ~3 h of source training and the entire 57 GB MLAAD download.** MLAAD only ever fed source training; with these present nothing trains. Also makes `ours_fixed` exactly comparable to `results_ext.csv`. |
# | `kaggle.json` | 1 KB | all corpora come from Kaggle |
# | `ckpt_protocol_a_rawboost.pt` | 193 MB | skips 16 epochs of Protocol A source training |
# | `results_protocol_a.csv` | 1 KB | its resume guard already has `source_rawboost`, `ours_rawboost`, `ours_prior_rawboost` — so **only the new adaptive arm runs**, turning Protocol A from ~5 h into ~1 h |
#
# Put them in the working directory (the repo root, after the clone cell).
#
# ## Time budget
#
# | stage | wall clock |
# |---|---|
# | setup + target corpora (~19 GB) | ~40 min |
# | `verify_reduction.py` gate | ~5 min |
# | **4-target x 3-seed grid** | **~6 h** |
# | Protocol A (adaptive arm only; 61 GB downloads in background during the grid) | ~1 h |
#
# Long jobs run under `nohup` and write results incrementally, so a dropped
# notebook connection does not kill them or lose completed folds.

# %% [markdown]
# ## 1. Box check

# %%
import os, subprocess, sys, textwrap, time

print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
print(subprocess.run(["df", "-h", "."], capture_output=True, text=True).stdout)
# Need ~19 GB targets + 61 GB DF eval + ~10 GB working. Under ~110 GB free,
# drop Protocol A rather than discovering it mid-download.
print("python", sys.version.split()[0])

# %% [markdown]
# ## 2. Clone the repo
#
# Everything below runs from the repo root. `adaptive_tta.py`,
# `adaptive_pipeline.py`, `verify_reduction.py` and the patched `protocol_a.py`
# are all on `main`.

# %%
REPO = "https://github.com/RohamIzadidoost/Kaggle_deepFakeAudio.git"

# Find the upload directory by looking for the uploads themselves, rather than
# by guessing a path. Guessing failed on a real JupyterLab box: its file browser
# shows "/work/" but that breadcrumb is relative to the notebook server root, so
# the actual path was ~/work while a "/work" probe said no such directory. The
# repo is then cloned inside whichever directory really holds the uploads --
# usually the large persistent mount, which is also where the ~110 GB of corpora
# need to land.
CANDIDATES = ["/work", "/content", os.path.expanduser("~/work"),
              os.path.expanduser("~"), os.getcwd()]
UPLOADS = next((d for d in CANDIDATES if os.path.isdir(d) and (
    glob.glob(f"{d}/ckpt_ext/source_*_seed*.pt")
    or glob.glob(f"{d}/source_*_seed*.pt")
    or os.path.exists(f"{d}/kaggle.json"))), None)
assert UPLOADS, (
    "could not find your uploads. Looked for ckpt_ext/, source_*_seed*.pt or "
    f"kaggle.json in {CANDIDATES}. Set UPLOADS by hand to the directory the "
    "file browser is showing -- note its breadcrumb may be relative to the "
    "notebook server root, so check with `import os; os.getcwd()` first.")
WORK = os.path.join(UPLOADS, "deepfake")
print("uploads dir:", UPLOADS, "-> work dir:", WORK)

if not os.path.isdir(WORK):
    subprocess.run(["git", "clone", "--depth", "1", REPO, WORK], check=True)
os.chdir(WORK)
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)
print("cwd:", os.getcwd())
print(subprocess.run(["git", "log", "--oneline", "-3"], capture_output=True, text=True).stdout)

# %% [markdown]
# ## 3. Environment
#
# Same pins as `requirements.txt`. librosa/numba are removed deliberately — they
# pull a numpy that breaks the pandas ABI here.
#
# **Restart the kernel after this cell, then continue from section 4.**

# %%
def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "librosa", "numba"], check=False)
pip("numpy==1.26.4", "pandas", "scikit-learn", "scipy")
pip("torch", "torchaudio", "soundfile", "datasets", "tqdm", "kaggle")
pip("numpy==1.26.4")   # last word: datasets tends to bump it
print("done -- RESTART THE KERNEL now, then run section 4 onward")

# %% [markdown]
# ## 4. Confirm the uploads landed
#
# This is a hard gate on purpose. Without `ckpt_ext/` the grid silently skips
# every fold (`adaptive_pipeline.py` never trains), and you would not find out
# until the run "finished" in two minutes with an empty results file.

# %%
# Re-derived rather than inherited: the kernel restart in section 3 wipes every
# variable, and this is the first cell you run afterwards.
import glob, os, shutil, subprocess, sys, textwrap, time

CANDIDATES = ["/work", "/content", os.path.expanduser("~/work"),
              os.path.expanduser("~"), os.getcwd()]
UPLOADS = next((d for d in CANDIDATES if os.path.isdir(d) and (
    glob.glob(f"{d}/ckpt_ext/source_*_seed*.pt")
    or glob.glob(f"{d}/source_*_seed*.pt")
    or os.path.exists(f"{d}/kaggle.json"))), None)
assert UPLOADS, f"could not find your uploads in {CANDIDATES} -- set UPLOADS by hand"
WORK = os.path.join(UPLOADS, "deepfake")
os.chdir(WORK)
os.makedirs("logs", exist_ok=True)
print("uploads:", UPLOADS, "| cwd:", os.getcwd())

# Adopt the uploads wherever they are: flat in the upload dir (a browser cannot
# upload a directory) or already inside a ckpt_ext/ folder you made by hand.
os.makedirs("ckpt_ext", exist_ok=True)
for p in (glob.glob(f"{UPLOADS}/ckpt_ext/source_*_seed*.pt")
          + glob.glob(f"{UPLOADS}/source_*_seed*.pt")):
    dst = f"ckpt_ext/{os.path.basename(p)}"
    if not os.path.exists(dst):
        shutil.move(p, dst)
        print("adopted", os.path.basename(p))
for f in ("kaggle.json", "ckpt_protocol_a_rawboost.pt", "results_protocol_a.csv"):
    if os.path.exists(f"{UPLOADS}/{f}") and not os.path.exists(f):
        shutil.move(f"{UPLOADS}/{f}", f)
        print("adopted", f)

ck = sorted(glob.glob("ckpt_ext/source_*_seed*.pt"))
print(f"\nckpt_ext: {len(ck)} checkpoints")
for c in ck:
    print("   ", c, f"{os.path.getsize(c)/1e6:.0f} MB")

assert len(ck) == 12, (
    f"expected 12 checkpoints (4 targets x seeds 0-2), found {len(ck)}. "
    f"Upload them to {UPLOADS} (flat is fine) -- otherwise every fold is skipped.")

# Kaggle credentials
kdir = os.path.expanduser("~/.kaggle")
if os.path.exists("kaggle.json"):
    os.makedirs(kdir, exist_ok=True)
    shutil.copy("kaggle.json", f"{kdir}/kaggle.json")
    os.chmod(f"{kdir}/kaggle.json", 0o600)
assert os.path.exists(f"{kdir}/kaggle.json"), f"upload kaggle.json to {UPLOADS}"

PROTO_A = os.path.exists("ckpt_protocol_a_rawboost.pt")
print(f"\nProtocol A checkpoint present: {PROTO_A}")
print(f"Protocol A prior results present: {os.path.exists('results_protocol_a.csv')}")
print("\nall good" if PROTO_A else "\nProtocol A will retrain its source model (+~35 min)")

# %% [markdown]
# ## 5. Target corpora (~19 GB)
#
# Only the four EER targets. **MLAAD is deliberately not downloaded** — it is a
# source-diversity corpus, and with `ckpt_ext/` uploaded nothing trains a source
# model. That is 57 GB and ~3 h saved.

# %%
TARGET_SETS = [
    ("azkurniwan/asvspoof-2019-la", "data/asvspoof2019_LA", "ASVspoof2019_LA_train/flac/*.flac"),
    ("bhaveshkumars/release-in-the-wild", "data/in_the_wild", "**/*.wav"),
    ("adarshsingh0903/audio-deepfake-detection-dataset", "data/dataset_2", "**/*.wav"),
]

import kaggle
for slug, path, probe in TARGET_SETS:
    if glob.glob(os.path.join(path, probe), recursive=True):
        print(f"already present: {path}")
        continue
    print(f"downloading {slug} -> {path}")
    kaggle.api.dataset_download_files(slug, path=path, unzip=True, quiet=False)
print("\ntarget corpora done")

# %% [markdown]
# ## 6. Arabic ArAD (HuggingFace, ~1.6 GB)

# %%
def fetch_arabic(out="data/arabic_arad"):
    if glob.glob(f"{out}/**/*.wav", recursive=True):
        print(f"already present: {out}")
        return
    from datasets import load_dataset, Audio
    ds = load_dataset("DeepFake-Audio-Rangers/Arabic_Audio_Deepfake").cast_column("audio", Audio(decode=False))
    names = ds["train"].features["label"].names
    for split in ds:
        for i, ex in enumerate(ds[split]):
            d = f"{out}/{split}/{names[ex['label']]}"
            os.makedirs(d, exist_ok=True)
            open(f"{d}/{i}.wav", "wb").write(ex["audio"]["bytes"])
    print("arabic done")

fetch_arabic()

for p, probe in [("data/asvspoof2019_LA", "ASVspoof2019_LA_train/flac/*.flac"),
                 ("data/in_the_wild", "**/*.wav"), ("data/dataset_2", "**/*.wav"),
                 ("data/arabic_arad", "**/*.wav")]:
    print(f"{p:28s} {len(glob.glob(os.path.join(p, probe), recursive=True)):>7d} files")

# %% [markdown]
# ## 7. Kick off the 61 GB DF-eval download in the background
#
# Protocol A needs ASVspoof2021-DF eval + its CM keys. It downloads *while the
# grid trains*, so it costs no serial time. Run this now, check it in section 11.

# %%
RUN_PROTOCOL_A = True    # set False to skip Protocol A and its 61 GB entirely

if RUN_PROTOCOL_A:
    if glob.glob("data/dataset_1/ASVspoof2021_DF_eval_part0*/**/*.flac", recursive=True):
        print("DF eval already present")
    else:
        subprocess.Popen(
            "nohup kaggle datasets download -d mohammedabdeldayem/avsspoof-2021 "
            "-p data/dataset_1 --unzip > logs/df_download.log 2>&1 &",
            shell=True)
        print("DF eval download started in background -> logs/df_download.log")

# %% [markdown]
# ## 8. Gate: the reduction check
#
# Asserts `adapt_adaptive` with its schedules switched off is **bitwise
# identical** to the published `adapt()`. If this fails, every adaptive-vs-fixed
# difference is the refactor rather than the schedules, and the day should be
# spent here instead of on the grid. Locally it returns `max |difference| = 0`.

# %%
r = subprocess.run([sys.executable, "verify_reduction.py"], capture_output=True, text=True)
print(r.stdout[-3000:])
print(r.stderr[-2000:] if r.returncode else "")
assert r.returncode == 0 and "PASS" in r.stdout, "REDUCTION CHECK FAILED -- stop and investigate"

# %% [markdown]
# ## 9. Smoke run (~2 min)
#
# Tiny subsets, separate `*_smoke` outputs. Validates the whole path on this box
# before committing 6 h. Smoke EERs are on ~128 clips and carry **no signal** —
# you are checking that the q ramp climbs and `shift` decays, nothing more.

# %%
r = subprocess.run([sys.executable, "adaptive_pipeline.py"], capture_output=True, text=True,
                   env={**os.environ, "ADAPTIVE_SMOKE": "1"})
print(r.stdout[-4000:])
assert r.returncode == 0, r.stderr[-3000:]

# %% [markdown]
# ## 10. The real grid (~6 h)
#
# 4 targets x seeds 0-2. Arms: `source`, `ours_fixed` (the published config,
# re-run in-session as the control), `ours_adaptive`, plus `ours_aq` / `ours_ae`
# on seed 0 only, plus an inductive check. Rows append to
# `results_adaptive.csv` as they finish, so an interruption loses at most one
# arm and a restart resumes per method.
#
# Runs under `nohup` — safe against a dropped connection.

# %%
subprocess.Popen(
    f"nohup env ADAPTIVE_SMOKE=0 {sys.executable} adaptive_pipeline.py "
    "> logs/grid.log 2>&1 &", shell=True)
time.sleep(20)
print(open("logs/grid.log").read()[-2000:])

# %% [markdown]
# ### Watch it
# Re-run this cell whenever you want. Expect ~24 min per fold, 12 folds.

# %%
print(subprocess.run("tail -25 run_log_adaptive.txt", shell=True,
                     capture_output=True, text=True).stdout)
if os.path.exists("results_adaptive.csv"):
    import pandas as pd
    d = pd.read_csv("results_adaptive.csv")
    print(f"\n{len(d)} rows so far\n")
    print(d.pivot_table(index=["target", "seed"], columns="method",
                        values="eer").round(2).to_string())

# %% [markdown]
# ## 11. Protocol A — the skewed-pool arm
#
# **Only run this once the grid has finished** (both want the GPU).
#
# This is where fixed `q` is most exposed: on a 97.5 %-spoof pool, a symmetric
# `q=0.3` labels the bottom 30 % "confident real" regardless of the true balance,
# and the published run went 26.33 -> 42.45 % EER. The BIC-guarded prevalence
# estimate plus the curriculum ramp is the fix being tested.

# %%
print(subprocess.run("tail -3 logs/df_download.log", shell=True,
                     capture_output=True, text=True).stdout)
n_flac = len(glob.glob("data/dataset_1/ASVspoof2021_DF_eval_part0*/**/*.flac", recursive=True))
keys = "data/dataset_1/DF-keys-full/keys/DF/CM/trial_metadata.txt"
print(f"DF eval flacs: {n_flac}   keys present: {os.path.exists(keys)}")
assert n_flac > 0 and os.path.exists(keys), "DF eval not ready yet -- wait for the download"

# %% [markdown]
# ### Comparability check before spending an hour
#
# The existing `results_protocol_a.csv` rows scored **400,435** trials
# (`n_missing=133493`) because the local copy had only `part00`–`part02`. If the
# cloud copy has more parts, the new adaptive row would be scored on a *different
# pool* and would not be comparable to the `source_rawboost` / `ours_rawboost`
# rows it is meant to sit beside. Check the part count first.

# %%
parts = sorted({p.split("/")[2] for p in
                glob.glob("data/dataset_1/ASVspoof2021_DF_eval_part0*/**/*.flac", recursive=True)})
print("DF eval parts present:", parts)
if len(parts) != 3:
    print(textwrap.dedent(f"""
        WARNING: {len(parts)} parts here vs 3 locally. `n_scored` will differ from
        400435 and the new row will NOT be comparable to the existing ones.
        Either restrict DF_PARTS in protocol_a.py to part00-02, or plan to
        re-run every Protocol A arm on this larger pool.
    """))

# %%
subprocess.Popen(f"nohup {sys.executable} protocol_a.py > logs/protocol_a.log 2>&1 &", shell=True)
time.sleep(20)
print(open("logs/protocol_a.log").read()[-2000:])

# %%
print(subprocess.run("tail -20 run_log_protocol_a_rawboost.txt", shell=True,
                     capture_output=True, text=True).stdout)

# %% [markdown]
# ## 12. Results
#
# The three questions this session answers:
#
# 1. Does `ours_fixed` reproduce `results_ext.csv` for seeds 0-2? (If not, the
#    cloud box differs from the H200 run and the adaptive comparison is still
#    internally valid but cannot be dropped into the published table.)
# 2. Does `ours_adaptive` beat `ours_fixed`, and on how many of the 12 points?
# 3. On Protocol A, does adaptive `q` turn the 26.33 -> 42.45 collapse into a gain?

# %%
import pandas as pd

d = pd.read_csv("results_adaptive.csv")
t = d[d.setting == "transductive"].pivot_table(index=["target", "seed"],
                                               columns="method", values="eer")
if {"ours_fixed", "ours_adaptive"} <= set(t.columns):
    t["delta"] = t["ours_adaptive"] - t["ours_fixed"]
    print(t.round(3).to_string())
    print("\nmean delta by target (negative = adaptive better):")
    print(t.groupby("target").delta.mean().round(3).to_string())
    print(f"\nimproved in {(t.delta < 0).sum()}/{t.delta.notna().sum()} points")

print("\n--- vs the published run (seeds 0-2) ---")
try:
    e = pd.read_csv("results_ext.csv")
    e = e[(e.family == "xlsr") & (e.setting == "transductive") & (e.seed <= 2)]
    ref = e[e.method == "ours"].pivot_table(index=["target", "seed"], values="eer")
    print(ref.join(t[["ours_fixed"]], how="inner").round(3).to_string())
except FileNotFoundError:
    print("results_ext.csv not in the clone -- compare locally")

if os.path.exists("results_protocol_a.csv"):
    print("\n--- Protocol A ---")
    print(pd.read_csv("results_protocol_a.csv").to_string(index=False))

# %% [markdown]
# ## 13. Save everything off the box
#
# Download this archive before the session ends — the checkpoints are large and
# regenerable, but the CSVs and the per-epoch trace are not.

# %%
subprocess.run(
    "tar czf adaptive_results.tar.gz results_adaptive.csv adaptive_trace.csv "
    "run_log_adaptive.txt results_protocol_a.csv run_log_protocol_a_rawboost.txt "
    "logs/ 2>/dev/null; ls -la adaptive_results.tar.gz", shell=True)
print("\ndownload adaptive_results.tar.gz, then commit the CSVs to the repo")
