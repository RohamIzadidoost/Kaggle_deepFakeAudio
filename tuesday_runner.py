"""Unattended supervisor for the Tuesday GPU session (ICASSP_PLAN.md P1).

Run it and walk away:

    nohup python tuesday_runner.py > logs/tuesday.out 2>&1 &

It works a priority-ordered job queue against a wall-clock deadline, skipping
any job whose estimate does not fit in the time left and trying the next one.
Every job is a separate subprocess, so one crash cannot take the session down,
and every underlying pipeline appends to its CSV per fold with its own
(seed, target) resume guard -- so re-running this script after any interruption
picks up exactly where it stopped and repeats no GPU work.

## Why these jobs, in this order

ICASSP_PLAN.md says P0 is done and P1 is the whole critical path. The ordering
is by marginal value per GPU-hour, not by plan numbering:

  1. ext seed 4   -- only Arabic is missing at seed 4; ~1 fold. Removes the
                     "4 seeds on Arabic, 5 everywhere else" asymmetry (P1.4)
                     that a reviewer notices for free.
  2. ext seed 5   -- takes every target to n=6. This is the single highest-value
                     job in the queue: Wilcoxon at n=5 CANNOT reach p<0.05 (the
                     floor is 2/2^5 = 0.0625, a combinatorial limit, not a power
                     problem), and n=6 gives 2/2^6 = 0.031. One seed converts
                     "sign consistency" into an actual per-target significance
                     claim, which is reviewer objection R1.
  3. asdg 3,4     -- cheap (~7 min/fold) and closes R3's seed asymmetry.
  4. dann 3,4     -- same, dearer (~30 min/fold).
  5. ext 6,7,8,9  -- more power and, per the risk register, buffer against one
                     collapsed seed destroying a claim.

Stats re-run last and unconditionally: it is seconds of CPU and makes whatever
landed immediately usable.

## What it deliberately does NOT run

The adaptive q/E grid and Protocol A. ICASSP_PLAN.md P3 lists both as
deliberately excluded from this paper, and Monday's session spent a full GPU day
on them. They are not in this queue.
"""

import json
import os
import subprocess
import sys
import time

BUDGET_H = float(os.environ.get("TUESDAY_BUDGET_H", "23"))
PY = sys.executable
T0 = time.time()
DEADLINE = T0 + BUDGET_H * 3600

os.makedirs("logs", exist_ok=True)
LOG = "run_log_tuesday.txt"
STATE = "tuesday_state.json"
LOCK = "tuesday_runner.lock"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def left_h():
    return (DEADLINE - time.time()) / 3600


# name, est_hours, env overrides, command
JOBS = [
    ("mlaad_download", 1.5, {}, None),          # special-cased below
    ("ext_seed4", 1.3, {"EXT_SEEDS": "4"}, [PY, "extended_pipeline.py"]),
    ("ext_seed5", 4.8, {"EXT_SEEDS": "5"}, [PY, "extended_pipeline.py"]),
    ("asdg_seed34", 1.2, {"ASDG_SEEDS": "0,1,2,3,4"}, [PY, "asdg_pipeline.py"]),
    ("dann_seed34", 4.5, {"DANN_SEEDS": "3,4", "ANALYSIS_PARTS": "dann"},
     [PY, "analysis_pipeline.py"]),
    ("ext_seed6", 4.8, {"EXT_SEEDS": "6"}, [PY, "extended_pipeline.py"]),
    ("ext_seed7", 4.8, {"EXT_SEEDS": "7"}, [PY, "extended_pipeline.py"]),
    ("ext_seed8", 4.8, {"EXT_SEEDS": "8"}, [PY, "extended_pipeline.py"]),
    ("ext_seed9", 4.8, {"EXT_SEEDS": "9"}, [PY, "extended_pipeline.py"]),
]


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"done": []}


def save_state(st):
    json.dump(st, open(STATE, "w"), indent=1)


# A complete MLAAD v5 is 154 meta.csv across 38 languages (57 GB), measured
# against the known-good local copy. The threshold is deliberately below 154 so
# a slightly different Kaggle packaging still passes, but far above the handful
# of files a part-finished download leaves behind.
MLAAD_MIN_META = int(os.environ.get("MLAAD_MIN_META", "100"))


def mlaad_meta_count():
    import glob
    return len(glob.glob("data/mlaad/**/meta.csv", recursive=True))


def have_mlaad():
    """Complete enough to train on.

    Emphatically not "does any file exist". An earlier version returned True on
    the first meta.csv it found, which would have accepted a 31 GB half-download
    left by two concurrent downloaders and trained seeds 5-9 on a partial corpus
    without a single error anywhere.
    """
    return mlaad_meta_count() >= MLAAD_MIN_META


def run_mlaad():
    """Source training needs MLAAD; every P1 job trains source models.

    Via the Python API, never the `kaggle` CLI -- pip --user puts that binary in
    ~/.local/bin, which is not on PATH on a stock JupyterLab image. That cost a
    silent hours-long failure on Monday.
    """
    n = mlaad_meta_count()
    if have_mlaad():
        log(f"mlaad already present ({n} meta.csv), skipping download")
        return True
    if n > 0:
        # Partial. Resuming onto it risks a mix of half-written files, and the
        # kaggle client will not repair what it already thinks it fetched.
        import shutil as _sh
        log(f"mlaad is PARTIAL ({n}/{MLAAD_MIN_META}+ meta.csv) -- removing and "
            f"re-downloading from scratch")
        _sh.rmtree("data/mlaad", ignore_errors=True)
    log("downloading MLAAD (~57 GB) -- required by every source-training job")
    r = subprocess.run(
        [PY, "-c", "import kaggle; kaggle.api.dataset_download_files("
                   "'trapka/mlaadthe-multi-languagaudioanti-spoofing-dataset',"
                   " path='data/mlaad', unzip=True, quiet=True)"],
        capture_output=True, text=True)
    ok = have_mlaad()
    log(f"mlaad download rc={r.returncode} present={ok}")
    if not ok:
        log(f"  stderr tail: {r.stderr[-500:]}")
    return ok


def acquire_lock():
    """Refuse to start if another supervisor is already running.

    Two instances is not a harmless duplicate: they both begin at job one, so
    they both drive a 57 GB MLAAD download into the same directory, and they
    both write tuesday_state.json. Observed live -- two runners 33 and 25
    minutes in. A corrupt corpus is worse than no corpus, because training
    proceeds on it silently.
    """
    if os.path.exists(LOCK):
        try:
            pid = int(open(LOCK).read().strip())
            os.kill(pid, 0)          # signal 0 = "does this pid exist?"
        except (ValueError, ProcessLookupError, PermissionError):
            log(f"stale lock from a dead pid, taking over")
        else:
            log(f"ANOTHER SUPERVISOR IS ALREADY RUNNING (pid {pid}). Exiting.")
            log(f"  To take over:  pkill -f tuesday_runner && rm {LOCK}")
            return False
    open(LOCK, "w").write(str(os.getpid()))
    return True


def main():
    if not acquire_lock():
        return
    log(f"=== Tuesday supervisor | budget {BUDGET_H:.1f} h | python {PY} ===")
    log(f"deadline {time.strftime('%H:%M:%S', time.localtime(DEADLINE))}")
    st = load_state()

    for name, est, env, cmd in JOBS:
        if name in st["done"]:
            log(f"{name}: already done in a previous run, skipping")
            continue

        # Hard gate. Every remaining job trains source models, and MLAAD is in
        # every source pool. If it is missing, build_manifest() simply yields no
        # mlaad rows and training proceeds happily on a DIFFERENT source
        # composition -- seeds 5-9 would then not be comparable to seeds 0-4 and
        # nothing would report an error. A wasted session is recoverable;
        # silently non-comparable seeds in the paper are not.
        if cmd is not None and not have_mlaad():
            log(f"ABORT before {name}: data/mlaad has {mlaad_meta_count()} meta.csv, "
                f"need >= {MLAAD_MIN_META} (a complete copy has 154).")
            log("  Every P1 job trains source models and MLAAD is in every source")
            log("  pool. Running without it would silently train seeds on a")
            log("  different composition than seeds 0-4, making them")
            log("  incomparable. Fix the download, then re-run this script --")
            log("  it resumes and repeats no completed work.")
            break
        if left_h() < est * 0.75:
            log(f"{name}: needs ~{est:.1f} h, only {left_h():.1f} h left -- skipping to next")
            continue

        log(f"--- {name} (est {est:.1f} h, {left_h():.1f} h left) ---")
        t0 = time.time()
        try:
            if cmd is None:
                ok = run_mlaad()
            else:
                # Hard-cap each job so one hung fold cannot eat the whole budget.
                timeout = min(est * 2.0, max(left_h(), 0.1)) * 3600
                r = subprocess.run(cmd, env={**os.environ, **env},
                                   timeout=timeout)
                ok = r.returncode == 0
            dt = (time.time() - t0) / 3600
            log(f"--- {name}: {'ok' if ok else 'FAILED'} in {dt:.2f} h ---")
            if ok:
                st["done"].append(name)
                save_state(st)
        except subprocess.TimeoutExpired:
            log(f"--- {name}: TIMED OUT after {(time.time()-t0)/3600:.2f} h "
                f"(partial folds are already written and will resume) ---")
        except Exception as e:
            log(f"--- {name}: {type(e).__name__}: {e} ---")

        if left_h() <= 0:
            log("budget exhausted")
            break

    # Always, even if nothing else ran: seconds of CPU, makes results usable.
    log("--- refreshing statistics on whatever landed ---")
    try:
        r = subprocess.run([PY, "stats_tests.py"], capture_output=True, text=True,
                           timeout=1800)
        log(r.stdout[-3000:])
    except Exception as e:
        log(f"stats_tests failed: {type(e).__name__}: {e}")

    if os.path.exists(LOCK):
        os.remove(LOCK)
    log(f"=== DONE after {(time.time()-T0)/3600:.2f} h ===")
    for f in ("results_ext.csv", "results_dann.csv", "results_asdg.csv"):
        if os.path.exists(f):
            import pandas as pd
            d = pd.read_csv(f)
            log(f"{f}: {len(d)} rows, seeds {sorted(d.seed.unique())}")


if __name__ == "__main__":
    main()
