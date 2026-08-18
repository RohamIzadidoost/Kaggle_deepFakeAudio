#!/usr/bin/env bash
# Re-run DeepFense seed 2, which the original grid never actually measured.
#
# A quoting bug in the download command (a preceding `for s in ...` loop left
# s=240 in the outer shell, and the inner URL expanded $s there) fetched Seed240
# into BOTH the s2 and s240 directories. The run labelled `s2` was therefore
# seed 240; its rows have been relabelled in results_public_ckpt.csv, and the
# duplicate s240 run was killed. `public_ckpt_tta.py` now verifies each
# checkpoint's identity against its upstream config.yaml before loading.
#
#   FOLLOWUP_PID=<pid> ./run_public_ckpt_seed2.sh

set -u
cd "$(dirname "$0")"

FOLLOWUP_PID="${FOLLOWUP_PID:-0}"
DEADLINE="${DEADLINE:-$(date -d '2026-08-18 13:30' +%s)}"
CK=deepfense_w2v2_aasist_s2
LOG=run_log_public_ckpt.txt

say() { echo "[$(date +%H:%M:%S)] [seed2] $*" | tee -a "$LOG"; }

# the re-download must have finished and be the right checkpoint
for _ in $(seq 1 120); do
  if [ -s "public_ckpt/$CK/best_model.pth" ] && \
     [ "$(stat -c%s "public_ckpt/$CK/best_model.pth")" -gt 3500000000 ]; then break; fi
  sleep 30
done
if ! grep -q "^seed: 2$" "public_ckpt/$CK/config.yaml" 2>/dev/null; then
  say "config.yaml is not seed 2 -- refusing to run"; exit 1
fi
say "seed-2 checkpoint present and verified"

if [ "$FOLLOWUP_PID" != "0" ]; then
  say "waiting for followup (PID $FOLLOWUP_PID)..."
  while kill -0 "$FOLLOWUP_PID" 2>/dev/null; do sleep 120; done
fi

source env/bin/activate
if [ "$(date +%s)" -ge "$DEADLINE" ]; then say "past deadline, skipping"; exit 0; fi

say ">>> $CK ours seed0 (the real seed 2)"
if timeout 10800 python public_ckpt_tta.py --mode ours --ckpt "$CK" --seed 0 --batch 16; then
  say "<<< done"
else
  say "<<< FAILED"
fi
say "SEED2 DONE"
