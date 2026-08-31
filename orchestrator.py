"""Single-owner GPU job queue for the autonomous run (2026-08-27 -> 08-31).

One process owns the GPU and runs jobs strictly sequentially. The previous
failure this prevents: an interactive sign-check was launched while the E-sweep
held 8.9 GB, and every adaptation arm in the sweep died with OOM while the
eval-only arms survived -- a failure that looks like "the experiment ran and
mostly failed" rather than "two processes fought over one card".

Jobs are declared in `jobs.py` (a list of dicts) so the queue can be extended
between runs without touching this file. Each job is:

    dict(name="unique-id", cmd="shell command", est_min=45, tag="stage")

State lives in `queue_state.json`: a job whose name is recorded as done is
skipped, so re-launching resumes rather than repeating. A failing job is
recorded with its exit code and the queue continues -- one bad cell must not
cost the remaining days.

    python orchestrator.py            # run everything not yet done
    python orchestrator.py --status   # print progress and exit
    python orchestrator.py --only stageA   # run one tag
"""

import argparse
import json
import os
import subprocess
import time

STATE = "queue_state.json"
LOG = "queue_master.log"


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"done": {}, "failed": {}}


def save_state(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE)       # atomic: a kill mid-write must not corrupt state


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--only", default=None, help="run only jobs with this tag")
    ap.add_argument("--retry_failed", action="store_true")
    args = ap.parse_args()

    import jobs
    joblist = jobs.JOBS
    st = load_state()

    if args.status:
        done, failed = st["done"], st["failed"]
        todo = [j for j in joblist if j["name"] not in done]
        print(f"jobs: {len(joblist)} total | {len(done)} done | {len(failed)} failed "
              f"| {len(todo)} remaining")
        print(f"estimated remaining: {sum(j.get('est_min', 0) for j in todo)/60:.1f} h")
        for j in joblist:
            mark = "OK " if j["name"] in done else ("ERR" if j["name"] in failed else " . ")
            extra = ""
            if j["name"] in done:
                extra = f"  ({done[j['name']].get('minutes', 0):.0f} min)"
            elif j["name"] in failed:
                extra = f"  (exit {failed[j['name']].get('rc')})"
            print(f"  [{mark}] {j['tag']:<10} {j['name']}{extra}")
        return

    todo = [j for j in joblist if j["name"] not in st["done"]]
    if not args.retry_failed:
        todo = [j for j in todo if j["name"] not in st["failed"]]
    if args.only:
        todo = [j for j in todo if j["tag"] == args.only]

    log(f"queue start: {len(todo)} jobs, est {sum(j.get('est_min',0) for j in todo)/60:.1f} h")
    for i, j in enumerate(todo, 1):
        log(f"--> [{i}/{len(todo)}] {j['name']}  (tag={j['tag']}, est {j.get('est_min','?')} min)")
        t0 = time.time()
        rc = subprocess.run(j["cmd"], shell=True).returncode
        mins = (time.time() - t0) / 60
        st = load_state()                      # re-read: --status may have run meanwhile
        if rc == 0:
            st["done"][j["name"]] = dict(minutes=round(mins, 1), at=time.strftime("%F %T"))
            st["failed"].pop(j["name"], None)
            log(f"    done in {mins:.1f} min")
        else:
            st["failed"][j["name"]] = dict(rc=rc, minutes=round(mins, 1), at=time.strftime("%F %T"))
            log(f"    FAILED rc={rc} after {mins:.1f} min -- continuing")
        save_state(st)
    log("queue complete")


if __name__ == "__main__":
    main()
