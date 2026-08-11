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
# # Tuesday session — ICASSP_PLAN.md P1
#
# **Run All Cells, then leave.** The last cell launches a detached supervisor
# that works a 23-hour job queue on its own and survives this notebook, the
# kernel, and your browser closing. Nothing below needs you after that.
#
# ## What changed since Monday, and why
#
# Monday's notebook ran the adaptive q/E grid and Protocol A. `ICASSP_PLAN.md`
# §P3 lists **both as deliberately excluded** from this paper, and P1 — the
# entire compute package — was untouched. This session does P1 only.
#
# | plan item | state | this session |
# |---|---|---|
# | P0.1–P0.5 blockers | done (commit `baa5ae3`) | — |
# | **P1.1** seeds 5→10 main grid | not started | **seeds 4–9, as many as fit** |
# | **P1.2** DANN + ASDG to 5 seeds | at 3 seeds | **seeds 3,4 for both** |
# | **P1.3** re-run stats on more seeds | — | **automatic at the end** |
# | **P1.4** Arabic 5th seed | missing | **first job in the queue** |
#
# ## Everything resumes; nothing is redone
#
# Each pipeline appends to its CSV per fold and has its own `(seed, target)`
# resume guard, so completed work is skipped on sight. The supervisor keeps its
# own `tuesday_state.json` too. Re-running this notebook after any interruption
# costs no repeated GPU time.
#
# The one exception is DANN, which has no resume guard — the supervisor passes
# it `DANN_SEEDS=3,4` explicitly so it cannot duplicate seeds 0–2.
#
# ## Order, by value per GPU-hour
#
# 1. **Arabic seed 4** — one fold, kills the 4-vs-5 seed asymmetry (P1.4).
# 2. **Main grid seed 5** — the highest-value single job in the queue. Wilcoxon
#    at n=5 *cannot* reach p<0.05 (floor `2/2^5 = 0.0625`, combinatorial, not a
#    power problem); n=6 gives `0.031`. One seed turns "sign consistency" into a
#    real per-target significance claim — reviewer objection R1.
# 3. **ASDG seeds 3,4** — cheap, closes R3's asymmetry.
# 4. **DANN seeds 3,4** — same, dearer.
# 5. **Main grid seeds 6–9** — power, and buffer against a collapsed seed.
#
# At 23 h the realistic landing point is **8 seeds on the main grid** and
# **5 on DANN/ASDG**, which satisfies R1, R3 and R5.

# %% [markdown]
# ## 1. Locate the working directory
#
# Same detection as Monday: find the uploads rather than guessing a path (a
# JupyterLab breadcrumb showing `/work/` is relative to the server root; the
# real path was `~/work`).

# %%
import glob, os, shutil, subprocess, sys, time

CANDIDATES = ["/work/deepfake", os.path.expanduser("~/work/deepfake"),
              "/content/deepfake", os.path.expanduser("~/deepfake"), os.getcwd()]
WORK = next((d for d in CANDIDATES
             if os.path.isdir(d) and os.path.exists(f"{d}/extended_pipeline.py")), None)
assert WORK, f"could not find the repo in {CANDIDATES}"
os.chdir(WORK)
os.makedirs("logs", exist_ok=True)
print("work dir:", os.getcwd())
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)

# %% [markdown]
# ## 2. Pull the latest code
#
# `tuesday_runner.py` and the `EXT_SEEDS` / `ASDG_SEEDS` / `DANN_SEEDS`
# overrides it depends on were added after Monday's clone.

# %%
print(subprocess.run(["git", "stash"], capture_output=True, text=True).stdout)
print(subprocess.run(["git", "pull", "--rebase", "origin", "main"],
                     capture_output=True, text=True).stdout[-1500:])
assert os.path.exists("tuesday_runner.py"), \
    "tuesday_runner.py missing -- push it from the local repo first, then re-run this cell"
print("code up to date")

# %% [markdown]
# ## 3. What is already on disk
#
# Nothing here downloads or trains — it just reports, so you can see what the
# supervisor will skip.

# %%
import pandas as pd

for f in ("results_ext.csv", "results_dann.csv", "results_asdg.csv"):
    if os.path.exists(f):
        d = pd.read_csv(f)
        per = d.groupby("target").seed.nunique().to_dict() if "target" in d else {}
        print(f"{f:20s} {len(d):4d} rows | seeds {sorted(d.seed.unique())} | per-target {per}")
    else:
        print(f"{f:20s} MISSING")

print()
for p, probe in [("data/asvspoof2019_LA", "ASVspoof2019_LA_train/flac/*.flac"),
                 ("data/in_the_wild", "**/*.wav"), ("data/dataset_2", "**/*.wav"),
                 ("data/arabic_arad", "**/*.wav"), ("data/mlaad", "**/*.wav")]:
    n = len(glob.glob(os.path.join(p, probe), recursive=True))
    print(f"{p:24s} {n:>7d} files" + ("   <- will be downloaded" if n == 0 else ""))

print("\nckpt_ext:", len(glob.glob("ckpt_ext/*.pt")), "checkpoints")
print(subprocess.run(["df", "-h", "."], capture_output=True, text=True).stdout)

# %% [markdown]
# ## 3b. Checkpoint integrity
#
# This is what killed Monday's run: one file in the 2.3 GB `ckpt_ext/` upload
# arrived truncated, `torch.load` raised *"failed finding central directory"*,
# and the exception propagated out of the fold loop and abandoned the ten
# remaining folds. Verifying now costs seconds.
#
# Today's grid trains seeds 5–9 from scratch and does not read these files, so a
# failure here does **not** block the session — but you want to know which
# uploads are corrupt before relying on them for anything else.

# %%
import torch

bad = []
for c in sorted(glob.glob("ckpt_ext/*.pt")):
    try:
        torch.load(c, map_location="cpu")
    except Exception as e:
        bad.append((c, f"{type(e).__name__}"))
print(f"checked {len(glob.glob('ckpt_ext/*.pt'))} checkpoints, {len(bad)} corrupt")
for c, e in bad:
    print("  CORRUPT:", c, e)
if bad:
    print("\nRe-upload just these files. Today's P1 jobs do not need them"
          "\n(seeds 5-9 train fresh), so the session can proceed regardless.")

# %% [markdown]
# ## 4. Kaggle credentials
#
# MLAAD (~57 GB) is required: every P1 job trains source models, and MLAAD is
# the source-diversity corpus in every source pool. Monday did not need it
# because nothing trained.

# %%
kdir = os.path.expanduser("~/.kaggle")
if os.path.exists("kaggle.json") and not os.path.exists(f"{kdir}/kaggle.json"):
    os.makedirs(kdir, exist_ok=True)
    shutil.copy("kaggle.json", f"{kdir}/kaggle.json")
    os.chmod(f"{kdir}/kaggle.json", 0o600)
assert os.path.exists(f"{kdir}/kaggle.json"), "kaggle.json missing -- upload it"
print("kaggle credentials ok")

# %% [markdown]
# ## 5. Make sure nothing else is holding the GPU
#
# Monday's adaptive grid may still be running. Two jobs sharing the GPU would
# slow both and confuse the timings the supervisor budgets against.

# %%
# "import kaggle" is in this list on purpose: the MLAAD download runs as an
# inline `python -c` child of the supervisor, so killing the supervisor orphans
# it rather than stopping it. It then keeps writing into data/mlaad and holds
# NFS handles that make `rm -rf data/mlaad` fail with "Device or resource busy".
out = subprocess.run("ps -eo pid,etime,args | grep -E '[a]daptive_pipeline|[e]xtended_pipeline|[a]nalysis_pipeline|[a]sdg_pipeline|[t]uesday_runner|[i]mport kaggle'",
                     shell=True, capture_output=True, text=True).stdout
print(out or "(nothing running)")
if out.strip():
    print("\n>>> Something is already running. Stop it AND its children -- killing")
    print(">>> a supervisor orphans the pipeline it launched, it does not stop it:")
    print(">>>")
    print(">>>   pkill -f tuesday_runner; sleep 3; \\")
    print(">>>     pkill -f 'extended_pipeline|analysis_pipeline|asdg_pipeline|import kaggle'")
    print(">>>")
    print(">>> then re-run this cell to confirm, and delete a stale lock if any:")
    print(">>>   rm -f tuesday_runner.lock")

# %% [markdown]
# ## 6. Launch and walk away
#
# Detached via `nohup`, so it survives the kernel, the browser and this
# notebook. Re-running this cell later resumes rather than restarting.

# %%
assert not out.strip(), "stop the running job first (see cell above), then re-run"

subprocess.Popen(
    f"nohup {sys.executable} tuesday_runner.py > logs/tuesday.out 2>&1 &",
    shell=True)
time.sleep(25)
print(open("logs/tuesday.out").read()[-2000:])
print("\n--- launched; you can close everything now ---")

# %% [markdown]
# ## 7. Optional: check in whenever you like
#
# Re-run this cell any time. Nothing below is required for the run to finish.

# %%
print(subprocess.run("tail -30 run_log_tuesday.txt", shell=True,
                     capture_output=True, text=True).stdout)

# Coverage for the row the paper actually reports. A plain nunique() over every
# row counts other methods, other families and the MLAAD held-out-language
# diagnostic too -- it showed "arabic: 5" while `ours` had only seeds 0-3, i.e.
# it overstated exactly the number this run exists to increase.
print("\n=== seed coverage (the reported configuration only) ===")
spec = [("results_ext.csv", dict(family="xlsr", setting="transductive", method="ours")),
        ("results_dann.csv", {}), ("results_asdg.csv", {})]
for f, filt in spec:
    if not os.path.exists(f):
        print(f"{f:20s} MISSING"); continue
    d = pd.read_csv(f)
    for k, v in filt.items():
        if k in d.columns:
            d = d[d[k] == v]
    age = (time.time() - os.path.getmtime(f)) / 60
    print(f"{f:20s} (updated {age:5.1f} min ago) " +
          str(d.groupby("target").seed.apply(lambda s: sorted(s.unique())).to_dict()))

# %% [markdown]
# ## 8. When it is finished
#
# `stats_tests.py` has already re-run automatically. The numbers that matter:
# per-target Wilcoxon should now be attainable at p<0.05 wherever n≥6, which is
# reviewer objection R1 closed.

# %%
# These last two cells are for *after* the run. On "Run All Cells" they fire a
# few seconds after launch, so they no-op while the supervisor is alive rather
# than reporting half-empty results and writing a premature archive.
def runner_alive():
    return bool(subprocess.run("ps -eo args | grep -c '[t]uesday_runner'",
                               shell=True, capture_output=True,
                               text=True).stdout.strip() not in ("0", ""))

if runner_alive():
    print("supervisor still running -- re-run this cell when it has finished.")
    print("progress: cell 7 above, or `tail -f run_log_tuesday.txt`")
else:
    print(subprocess.run([sys.executable, "stats_tests.py"],
                         capture_output=True, text=True).stdout[-4000:])

# %%
if runner_alive():
    print("supervisor still running -- re-run this cell to package results.")
else:
    subprocess.run("tar czf tuesday_results.tar.gz results_ext.csv results_dann.csv "
                   "results_asdg.csv stats_results.csv run_log_tuesday.txt "
                   "run_log_ext.txt logs/ 2>/dev/null; ls -la tuesday_results.tar.gz",
                   shell=True)
    print("\ndownload tuesday_results.tar.gz and commit the CSVs")
