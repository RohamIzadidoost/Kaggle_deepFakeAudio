"""Unattended supervisor for the Wednesday session.

    nohup python wednesday_runner.py > logs/wednesday.out 2>&1 &

Same proven structure as tuesday_runner.py -- pid lock, wall-clock deadline,
per-job subprocess with timeout, and the row-count check that catches a job
which exits 0 without doing anything. Only the queue is new.

## Where Tuesday left off

    results_ext.csv   seeds 0-9 on three targets, 0-9 minus 4 on Arabic
    results_dann.csv  seeds 0-4
    results_asdg.csv  seeds 0-2      <- 3,4 were lost to a `git stash`

Per-target Wilcoxon is already significant on three of four targets
(p = 0.0020 / 0.0039 / 0.0293), so R1 is closed. What remains is R3: the main
method now has 10 seeds while DANN has 5 and ASDG has 3, and comparing a
10-seed mean against a 3-seed mean is exactly the asymmetry a reviewer objects
to. Most of this queue is about removing it.

## Queue, by value

  1. ext_seed4        Arabic only (the others already have seed 4). Restores the
                      fold lost to the stash and makes it 10 seeds everywhere.
  2. asdg_seed34      Restores the other stash casualty. Back to 5 seeds.
  3. dann_seed59      DANN 5 -> 10 seeds. The expensive one (~30 min/fold) and
                      the one that actually closes R3.
  4. asdg_seed59      ASDG 5 -> 10 seeds. Cheap (~7 min/fold).
  5. protocol_a       The skewed-pool arm, if its 61 GB re-download finishes in
                      the background. Out of ICASSP_PLAN scope (P3), so it is
                      deliberately last: it runs only on time left over after
                      everything the paper actually needs.

Measured on Tuesday: ~30 min per extended-pipeline fold, ~7 min per ASDG fold,
~30 min per DANN fold. Estimates below use those, not the older guesses.
"""

import json
import os
import shutil
import subprocess
import sys
import time

BUDGET_H = float(os.environ.get("WED_BUDGET_H", "24"))
PY = sys.executable
T0 = time.time()
DEADLINE = T0 + BUDGET_H * 3600

os.makedirs("logs", exist_ok=True)
LOG = "run_log_wednesday.txt"
STATE = "wednesday_state.json"
LOCK = "wednesday_runner.lock"
BACKUP = "results_backup"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def left_h():
    return (DEADLINE - time.time()) / 3600


RESULT_FILES = ["results_ext.csv", "results_dann.csv", "results_asdg.csv",
                "results_protocol_a.csv", "stats_results.csv"]

# name, est_hours, env, command, csv that must gain rows
JOBS = [
    ("ext_seed4", 0.7, {"EXT_SEEDS": "4"}, [PY, "extended_pipeline.py"], "results_ext.csv"),
    ("asdg_seed34", 1.2, {"ASDG_SEEDS": "0,1,2,3,4"}, [PY, "asdg_pipeline.py"], "results_asdg.csv"),
    ("dann_seed59", 11.0, {"DANN_SEEDS": "5,6,7,8,9", "ANALYSIS_PARTS": "dann"},
     [PY, "analysis_pipeline.py"], "analysis_res/results_dann.csv"),
    ("asdg_seed59", 3.0, {"ASDG_SEEDS": "0,1,2,3,4,5,6,7,8,9"},
     [PY, "asdg_pipeline.py"], "results_asdg.csv"),
    ("protocol_a", 2.0, {}, [PY, "protocol_a.py"], "results_protocol_a.csv"),
]


def row_count(path):
    if not path or not os.path.exists(path):
        return 0
    with open(path) as f:
        return max(sum(1 for _ in f) - 1, 0)


def snapshot_results():
    """Copy every results CSV aside before touching anything.

    Tuesday lost an Arabic fold and two ASDG seeds -- about 1.7 GPU-hours -- to a
    `git stash`, because the results CSVs are tracked and stash reverted them to
    their committed state. Nothing here runs git, but a copy outside the work
    tree costs milliseconds and makes that class of accident recoverable.
    """
    stamp = time.strftime("%Y%m%d_%H%M%S")
    d = os.path.join(BACKUP, stamp)
    os.makedirs(d, exist_ok=True)
    for f in RESULT_FILES:
        if os.path.exists(f):
            shutil.copy(f, d)
    log(f"backed up {len([f for f in RESULT_FILES if os.path.exists(f)])} "
        f"results files -> {d}/")


def start_df_download():
    """Re-fetch the ASVspoof2021-DF eval set in the background, for protocol_a.

    Tuesday's attempt died with "avsspoof-2021.zip is corrupted or not a valid
    zip file" -- two concurrent downloaders had written the same zip. The
    partial file has to go first, or the kaggle client skips it as a "more
    recently modified local copy" and the corruption survives forever.
    """
    import glob
    if glob.glob("data/dataset_1/ASVspoof2021_DF_eval_part0*/**/*.flac", recursive=True):
        log("DF eval already present, no download needed")
        return
    for z in glob.glob("data/dataset_1/*.zip"):
        log(f"removing corrupt/partial {z}")
        os.remove(z)
    subprocess.Popen(
        [PY, "-c", "import kaggle; kaggle.api.dataset_download_files("
                   "'mohammedabdeldayem/avsspoof-2021', path='data/dataset_1',"
                   " unzip=True, quiet=True)"],
        stdout=open("logs/df_download.log", "w"), stderr=subprocess.STDOUT)
    log("DF eval download started in background -> logs/df_download.log")


def publish_dann():
    """Merge analysis_res/results_dann.csv into the root copy stats_tests reads.

    A merge, not a copy: analysis_pipeline writes only the seeds it was asked
    for, so the two files hold disjoint seed ranges.
    """
    src, dst = "analysis_res/results_dann.csv", "results_dann.csv"
    if not os.path.exists(src):
        return
    import pandas as pd
    new = pd.read_csv(src)
    old = pd.read_csv(dst) if os.path.exists(dst) else new.iloc[0:0]
    if len(old) and list(old.columns) != list(new.columns):
        log(f"!! {src} and {dst} columns differ -- NOT merging")
        return
    key = [c for c in ("seed", "target", "method") if c in new.columns]
    merged = (pd.concat([old, new], ignore_index=True)
                .drop_duplicates(subset=key, keep="last").sort_values(key))
    merged.to_csv(dst, index=False)
    log(f"published DANN: {len(old)} + {len(new)} -> {len(merged)} rows "
        f"(seeds {sorted(merged.seed.unique())})")


def acquire_lock():
    if os.path.exists(LOCK):
        try:
            pid = int(open(LOCK).read().strip())
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError, PermissionError):
            log("stale lock from a dead pid, taking over")
        else:
            log(f"ANOTHER SUPERVISOR IS ALREADY RUNNING (pid {pid}). Exiting.")
            log(f"  To take over:  pkill -f wednesday_runner && rm {LOCK}")
            return False
    open(LOCK, "w").write(str(os.getpid()))
    return True


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"done": []}


def main():
    if not acquire_lock():
        return
    log(f"=== Wednesday supervisor | budget {BUDGET_H:.1f} h | python {PY} ===")
    log(f"deadline {time.strftime('%H:%M:%S', time.localtime(DEADLINE))}")
    snapshot_results()
    start_df_download()
    st = load_state()

    for name, est, env, cmd, out_csv in JOBS:
        if name in st["done"]:
            log(f"{name}: already done in a previous run, skipping")
            continue
        if left_h() < est * 0.6:
            log(f"{name}: needs ~{est:.1f} h, only {left_h():.1f} h left -- skipping")
            continue
        if name == "protocol_a":
            import glob
            if not glob.glob("data/dataset_1/DF-keys-full/keys/DF/CM/trial_metadata.txt"):
                log("protocol_a: DF eval / keys not ready, skipping "
                    "(check logs/df_download.log)")
                continue

        log(f"--- {name} (est {est:.1f} h, {left_h():.1f} h left) ---")
        t0, before = time.time(), row_count(out_csv)
        try:
            timeout = min(est * 2.0, max(left_h(), 0.1)) * 3600
            r = subprocess.run(cmd, env={**os.environ, **env}, timeout=timeout)
            ok = r.returncode == 0
            after = row_count(out_csv)
            if ok and out_csv and after <= before:
                ok = False
                log(f"    !! {out_csv} still has {after} rows (was {before}) after a "
                    f"clean exit -- treating as FAILED. Check the pipeline's own log.")
            elif out_csv:
                log(f"    {out_csv}: {before} -> {after} rows (+{after - before})")
            log(f"--- {name}: {'ok' if ok else 'FAILED'} in {(time.time()-t0)/3600:.2f} h ---")
            if ok:
                st["done"].append(name)
                json.dump(st, open(STATE, "w"), indent=1)
                snapshot_results()      # a completed job is worth a fresh copy
        except subprocess.TimeoutExpired:
            log(f"--- {name}: TIMED OUT after {(time.time()-t0)/3600:.2f} h "
                f"(finished folds are written and will resume) ---")
        except Exception as e:
            log(f"--- {name}: {type(e).__name__}: {e} ---")

        if left_h() <= 0:
            log("budget exhausted")
            break

    publish_dann()
    snapshot_results()
    log("--- refreshing statistics ---")
    try:
        r = subprocess.run([PY, "stats_tests.py"], capture_output=True, text=True,
                           timeout=1800)
        log(r.stdout[-3000:])
    except Exception as e:
        log(f"stats_tests failed: {type(e).__name__}: {e}")

    if os.path.exists(LOCK):
        os.remove(LOCK)
    log(f"=== DONE after {(time.time()-T0)/3600:.2f} h ===")
    import pandas as pd
    for f in RESULT_FILES:
        if os.path.exists(f) and "seed" in open(f).readline():
            d = pd.read_csv(f)
            log(f"{f}: {len(d)} rows, seeds {sorted(d.seed.unique())}")


if __name__ == "__main__":
    main()
