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
# # Wednesday session — finish P1, close R3
#
# **Run All Cells, then leave.** A detached supervisor works a 24-hour queue on
# its own and survives this notebook, the kernel and your browser closing.
#
# ## Where Tuesday got to
#
# Per-target Wilcoxon is now significant on three of four targets — **R1 is
# closed**, and it was mathematically unreachable at 5 seeds:
#
# | target | n | source | ours | p | improved |
# |---|---|---|---|---|---|
# | asvspoof2019 | 10 | 5.03 | 3.47 | **0.0020** | 10/10 |
# | arabic | 9 | 22.91 | 21.31 | **0.0039** | 9/9 |
# | in_the_wild | 10 | 12.78 | 11.33 | **0.0293** | 8/10 |
# | dataset2 | 10 | 33.54 | 33.74 | 0.2324 | 5/10 |
#
# Tent's instability also got much better evidenced: 4 collapses in 10 seeds on
# ASVspoof2019 (mean 23.47, std 21.96 — the spread exceeds the mean).
#
# ## What is left, and why
#
# The main method has 10 seeds; DANN has 5 and ASDG has 3. Comparing a 10-seed
# mean against a 3-seed mean is exactly reviewer objection **R3**. Most of
# today's budget goes to removing that asymmetry.
#
# | # | job | est | why |
# |---|---|---|---|
# | 1 | `ext_seed4` | 0.7 h | Arabic only — restores the fold lost to `git stash`, making it 10 seeds everywhere |
# | 2 | `asdg_seed34` | 1.2 h | the other stash casualty |
# | 3 | `dann_seed59` | 11 h | DANN 5 → 10 seeds. The job that actually closes R3 |
# | 4 | `asdg_seed59` | 3 h | ASDG 5 → 10 seeds, cheap |
# | 5 | `protocol_a` | 2 h | the skewed-pool arm, **only if** its 61 GB re-download lands. Out of ICASSP_PLAN scope (P3), so it runs last on leftover time |
#
# ~18 h of work in a 24 h budget. Estimates use Tuesday's measured rates
# (~30 min per extended-pipeline fold, ~7 min per ASDG fold, ~30 min per DANN
# fold), not the earlier guesses.
#
# ## Protecting the results this time
#
# Tuesday lost an Arabic fold and two ASDG seeds — about 1.7 GPU-hours — to a
# `git stash`, because the results CSVs are tracked and stash reverted them.
# The supervisor now copies every results CSV into `results_backup/<timestamp>/`
# at startup and after each completed job.
#
# **Do not run `git stash`, `git checkout` or `git restore` on this working tree
# while results are uncommitted.** Commit and push instead.

# %% [markdown]
# ## 1. Locate the repo and pull

# %%
import glob, os, shutil, subprocess, sys, textwrap, time

CANDIDATES = ["/work/deepfake", os.path.expanduser("~/work/deepfake"),
              "/content/deepfake", os.path.expanduser("~/deepfake"), os.getcwd()]
WORK = next((d for d in CANDIDATES
             if os.path.isdir(d) and os.path.exists(f"{d}/extended_pipeline.py")), None)
assert WORK, f"repo not found in {CANDIDATES}"
os.chdir(WORK)
os.makedirs("logs", exist_ok=True)
print("work dir:", os.getcwd())

# Commit any uncommitted results BEFORE pulling. This is the step whose absence
# cost 1.7 GPU-hours on Tuesday: `git stash` silently reverted tracked result
# CSVs to their committed state, discarding finished folds.
print(subprocess.run("git add -A results_*.csv stats_results.csv run_log_*.txt 2>/dev/null; "
                     "git commit -m 'results: session output' 2>&1 | tail -2",
                     shell=True, capture_output=True, text=True).stdout)
print(subprocess.run(["git", "pull", "--rebase", "origin", "main"],
                     capture_output=True, text=True).stdout[-1200:])
assert os.path.exists("wednesday_runner.py"), \
    "wednesday_runner.py missing -- push it from the local repo, then re-run this cell"
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)

# %% [markdown]
# ## 2. Where things actually stand
#
# Filtered to the configuration the paper reports, so the counts mean what they
# look like.

# %%
import pandas as pd

spec = [("results_ext.csv", dict(family="xlsr", setting="transductive", method="ours")),
        ("results_dann.csv", {}), ("results_asdg.csv", {})]
for f, filt in spec:
    if not os.path.exists(f):
        print(f"{f:20s} MISSING"); continue
    d = pd.read_csv(f)
    for k, v in filt.items():
        if k in d.columns:
            d = d[d[k] == v]
    cov = d.groupby("target").seed.apply(lambda s: sorted(s.unique())).to_dict()
    print(f"{f:20s} " + " ".join(f"{t}:n={len(s)}" for t, s in cov.items()))

print("\nmlaad:", len(glob.glob("data/mlaad/**/meta.csv", recursive=True)), "meta.csv (154 = complete)")
print("DF eval:", len(glob.glob("data/dataset_1/ASVspoof2021_DF_eval_part0*/**/*.flac",
                                recursive=True)), "flacs (0 = will re-download)")
print(subprocess.run(["df", "-h", "."], capture_output=True, text=True).stdout)

# %% [markdown]
# ## 3. Nothing else may be holding the GPU

# %%
out = subprocess.run(
    "ps -eo pid,etime,args | grep -E "
    "'[t]uesday_runner|[w]ednesday_runner|[e]xtended_pipeline|[a]nalysis_pipeline"
    "|[a]sdg_pipeline|[a]daptive_pipeline|[i]mport kaggle'",
    shell=True, capture_output=True, text=True).stdout
print(out or "(nothing running)")
if out.strip():
    print("\n>>> Stop it AND its children -- killing a supervisor orphans the")
    print(">>> pipeline it launched rather than stopping it:")
    print(">>>   pkill -f wednesday_runner; pkill -f tuesday_runner; sleep 3; \\")
    print(">>>     pkill -f 'extended_pipeline|analysis_pipeline|asdg_pipeline|import kaggle'")
    print(">>>   rm -f wednesday_runner.lock")

# %% [markdown]
# ## 4. Launch and walk away

# %%
assert not out.strip(), "stop the running job first (cell above), then re-run"

subprocess.Popen(f"nohup {sys.executable} wednesday_runner.py "
                 f"> logs/wednesday.out 2>&1 &", shell=True)
time.sleep(25)
print(open("logs/wednesday.out").read()[-2500:])
print("\n--- launched; you can close everything now ---")

# %% [markdown]
# ## 5. Checking in
#
# The notebook kernel on this box drops regularly, so prefer the terminal:
#
# ```
# tail -f run_log_wednesday.txt
# grep -E "rows|FAILED|ABORT" run_log_wednesday.txt
# ```
#
# Every `+N rows` line is a job that provably did work. A job that exits cleanly
# without adding rows is reported `FAILED` and left out of the state file, so it
# will be retried rather than silently skipped.

# %%
print(subprocess.run("tail -25 run_log_wednesday.txt", shell=True,
                     capture_output=True, text=True).stdout)
for f, filt in spec:
    if os.path.exists(f):
        d = pd.read_csv(f)
        for k, v in filt.items():
            if k in d.columns:
                d = d[d[k] == v]
        age = (time.time() - os.path.getmtime(f)) / 60
        print(f"{f:20s} ({age:5.1f} min ago) " +
              str(d.groupby("target").seed.apply(lambda s: len(s.unique())).to_dict()))

# %% [markdown]
# ## 6. When it finishes
#
# Statistics re-run automatically. Then **commit and push** — the results live
# on an ephemeral box, and this is ~18 GPU-hours.

# %%
def runner_alive():
    return subprocess.run("ps -eo args | grep -c '[w]ednesday_runner'", shell=True,
                          capture_output=True, text=True).stdout.strip() not in ("0", "")

if runner_alive():
    print("supervisor still running -- re-run this cell when it has finished.")
else:
    print(subprocess.run([sys.executable, "stats_tests.py"],
                         capture_output=True, text=True).stdout[-4000:])
    print("\nNow commit and push:")
    print("  git add -A results_*.csv stats_results.csv run_log_*.txt analysis_res/")
    print("  git commit -m 'results: DANN/ASDG to 10 seeds; arabic seed 4' && git push origin main")
