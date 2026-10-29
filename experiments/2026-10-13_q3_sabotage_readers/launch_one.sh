#!/bin/bash
# B3 light launcher (d9 18:0x: one watcher per job under $20, polls >= 120 s): launch_watched.sh with only the elapsed/cost
# watcher (watch_cost.py --poll 120); no billing watcher, no container guard. The original header:
# Local orchestration for one launch of this job (RULE.md, SPEND.md): start `modal run <script> <args>` in the background,
# read its app ID from the log as soon as it appears, and start the three guards on it: the strict billing watcher
# (stop at <stop $> billed for the app, or at the deadline), the elapsed watcher (both v2.5, main @ c44b5b46, by app ID; same stop at $5.4/h per container,
# per-container cap <cap min>), and guard_containers.py (<max containers>; the reader matched-FA folder's guard on main, --app-id/--max/--log). Fails closed: if no app ID appears within
# 180 s, the launch's process group is killed. Run from a frozen copy (bash reads scripts as it goes).
#   launch_watched.sh <label> <stop $> <max containers> <cap min> <deadline min> <script> [script args...]
# XREAD_RATE (env, $/h per container: 5.4 per H100; B3 copy: SABR_READER picks the reader) sets the elapsed watcher's rate; CPU steps pass their own rate.
set -u
LABEL=$1; STOP=$2; MAXC=$3; CAPMIN=$4; DEADMIN=$5; SCRIPT=$6; shift 6
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/results/launch_${LABEL}_$(date +%Y-%m-%d_%H%M%S)
mkdir -p "$OUT" || exit 1
M=(uv run --no-project --python 3.12 --with 'modal==1.6.1' modal)
cd "$HERE/../.." || exit 1
PYTHONPATH=src SABR_READER="${SABR_READER:?set SABR_READER}" perl -e 'setpgrp(0, 0); exec @ARGV' "${M[@]}" run "$SCRIPT" "$@" > "$OUT/run.log" 2>&1 < /dev/null &  # its own process group
RUNPID=$!
echo "$RUNPID" > "$OUT/run.pid"
APP=""
for i in $(seq 1 90); do
  # the run's own app ID; modal 1.6.1 prints the URL on the line after "View run at" (pattern tested on 1,642 captured logs)
  APP=$(python3 -I -c 'import re, sys; m = re.search(r"View run at\s+\S*?(ap-[A-Za-z0-9]+)", open(sys.argv[1], errors="replace").read()); print(m.group(1) if m else "")' "$OUT/run.log" 2>/dev/null || true)
  [ -n "$APP" ] && break
  kill -0 "$RUNPID" 2>/dev/null || break
  sleep 2
done
if [ -z "$APP" ]; then
  echo "$(date +%H:%M:%S) no app ID within 180 s: killing the launch" | tee -a "$OUT/guards.log"
  kill -TERM -- "-$RUNPID" 2>/dev/null; sleep 5; kill -KILL -- "-$RUNPID" 2>/dev/null
  exit 3
fi
echo "$APP" > "$OUT/app_id"
DEADLINE=$(( $(date +%s) + DEADMIN * 60 ))
APPNAME="probe-shift_aswin-surya_sub-q3sabr_${SABR_READER}_kill-2026-11-13"
( cd "$HERE" && nohup uv run --no-project --python 3.12 --with 'modal==1.6.1' python watch_cost.py --app-id "$APP" --stop "$STOP" --rate "${XREAD_RATE:-5.4}" --cap-min "$CAPMIN" --log "$OUT/watch_cost.log" --poll 120 > "$OUT/watch_cost.out" 2>&1 < /dev/null & echo $! > "$OUT/watch_cost.pid" )
# each watcher runs as shell -> uv -> python: record the whole chain, so a step's watchers are killed by PID only
for w in watch_cost; do  # fail closed: a watcher that never reaches python stops the app
  if ! "$HERE/record_chain.sh" "$OUT/$w.pid" "$OUT/$w.tree.pid"; then
    echo "$(date +%H:%M:%S) $w did not start: stopping app $APP" | tee -a "$OUT/guards.log"
    uv run --no-project --python 3.12 --with 'modal==1.6.1' modal app stop "$APP" --yes >> "$OUT/guards.log" 2>&1
    exit 4
  fi
done
echo "$(date +%H:%M:%S) watcher PID chain (shell uv python): cost $(cut -d" " -f1 "$OUT/watch_cost.tree.pid" | tr "\n" " ")"
echo "$(date +%H:%M:%S) launched $LABEL: app $APP ($APPNAME), stop \$$STOP, max $MAXC containers, cap ${CAPMIN} min, deadline ${DEADMIN} min; logs $OUT"
