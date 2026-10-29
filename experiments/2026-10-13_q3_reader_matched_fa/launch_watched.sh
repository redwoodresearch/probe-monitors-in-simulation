#!/bin/bash
# Launch one `modal run` of this experiment with its three watchers (SPEND.md): the frozen v2.1 watch_billing_strict.py
# (billing by app ID, stop and deadline) and watch_cost.py (elapsed containers x rate), and guard_containers.py (live
# containers above MAX, or a failed poll, stops the app). The watchers start as soon as the new app ID appears in
# `modal app list` (an app of this name created after this script started); if no app appears within 180 s the run is
# killed. Run it from a copy outside the repo (bash reads scripts while they run).
#   usage: [RATE=<$/h per container, default 4.93 (an H100)>] launch_watched.sh <label> <app name> <stop $> <max containers> <deadline minutes> <log dir> -- <modal run args...>
set -u
LABEL=$1; APP=$2; STOP=$3; MAX=$4; DEADLINE_MIN=$5; LOGS=$6; shift 6
[ "$1" = "--" ] || { echo "usage: ... -- <modal run args>" >&2; exit 2; }
shift
HERE=$(cd "$(dirname "$0")" && pwd)
EXP=experiments/2026-10-13_q3_reader_matched_fa
MODAL=(uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy --with pyyaml modal)
mkdir -p "$LOGS"
if [ "$LABEL" = "extract" ]; then  # the extraction's stop may not exceed the fresh price record's remaining budget (SPEND.md)
  python3 - "$STOP" "${PRICE_PREFIX:-price_olmo}" <<'PY' || { echo "STOP exceeds the latest price record's extraction stop; not launching" >&2; exit 6; }
import glob, json, sys
recs = sorted(glob.glob(f"experiments/2026-10-13_q3_reader_matched_fa/results/{sys.argv[2]}_*.json"))  # PRICE_PREFIX: price_olmo or price_llama
cap = json.load(open(recs[-1]))["extraction watcher stop $ (the stop less the spend so far and the allowance)"] if recs else -1
sys.exit(0 if 0 < float(sys.argv[1]) <= cap else 1)
PY
fi
T0=$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S')  # app list's created_at is local time, '2026-10-13 01:04:27-07:00'
STAMP=$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S)
RUNLOG="$LOGS/run_${LABEL}_${STAMP}.log"
PYTHONPATH=src nohup "${MODAL[@]}" run "$@" > "$RUNLOG" 2>&1 < /dev/null &
RUN_PID=$!
echo "$(date +%T) launched $LABEL (pid $RUN_PID), log $RUNLOG"
APP_ID=""
DISC_END=$(( $(date +%s) + 180 ))  # a wall-clock discovery deadline
while [ "$(date +%s)" -lt "$DISC_END" ]; do
  sleep 5
  LEFT=$(( DISC_END - $(date +%s) ))
  [ "$LEFT" -gt 2 ] || break
  APP_ID=$(python3 - "$APP" "$T0" "$LEFT" <<'PY' 2>/dev/null
import json, subprocess, sys
try:  # each discovery call is bounded by the time left before the deadline: a hung CLI counts as no app yet
    out = subprocess.run(["uv", "run", "--no-project", "--python", "3.12", "--with", "modal==1.6.1", "modal", "app", "list", "--json"],
                         capture_output=True, text=True, timeout=min(40, int(sys.argv[3]))).stdout
    rows = [a for a in json.loads(out) if a.get("description") == sys.argv[1] and a.get("state") in ("ephemeral", "deployed", "ephemeral (detached)")
            and str(a.get("created_at", "")) >= sys.argv[2]]
    rows.sort(key=lambda a: str(a.get("created_at", "")))
    print(rows[-1]["app_id"] if rows else "")
except Exception:
    print("")
PY
)
  [ "$(date +%s)" -gt "$DISC_END" ] && APP_ID=""  # a late answer counts as none
  [ -n "$APP_ID" ] && break
  kill -0 $RUN_PID 2>/dev/null || break
done
stop_apps() {  # $1: an app ID, or "new" for every app of this name created since this script started. Each CLI call is bounded;
  # retried until `modal app list` shows none of them running (up to 10 minutes), else a loud failure for the operator
  python3 - "$APP" "$T0" "$1" <<'PY'
import json, subprocess, sys, time
CLI = ["uv", "run", "--no-project", "--python", "3.12", "--with", "modal==1.6.1", "modal"]
end = time.time() + 600
while time.time() < end:
    try:
        rows = json.loads(subprocess.run(CLI + ["app", "list", "--json"], capture_output=True, text=True, timeout=60).stdout)
        live = [a["app_id"] for a in rows if a.get("state") != "stopped" and (a["app_id"] == sys.argv[3] if sys.argv[3] != "new" else
                (a.get("description") == sys.argv[1] and str(a.get("created_at", "")) >= sys.argv[2]))]
        if not live:
            print("stop confirmed"); sys.exit(0)
        for i in live:
            subprocess.run(CLI + ["app", "stop", i, "--yes"], capture_output=True, timeout=120)
    except Exception as e:
        print(f"stop attempt failed: {type(e).__name__}")
    time.sleep(15)
print("ALARM: stop NOT confirmed after 10 minutes; stop the app by hand"); sys.exit(1)
PY
}
if [ -z "$APP_ID" ]; then
  echo "$(date +%T) no app ID within 180 s (or the run ended first); killing the run and stopping any new app of this name" >&2
  kill $RUN_PID 2>/dev/null
  stop_apps new
  exit 4
fi
echo "$(date +%T) app $APP_ID"
DEADLINE=$(( $(date +%s) + DEADLINE_MIN * 60 ))
nohup python3 "$HERE/watch_billing_strict.py" "$APP_ID" "$STOP" "$DEADLINE" > "$LOGS/billing_${LABEL}_${STAMP}.log" 2>&1 < /dev/null &
W1=$!; echo "billing watcher pid $W1"
nohup uv run --no-project --python 3.12 --with 'modal==1.6.1' python "$HERE/watch_cost.py" --app-name "$APP" --stop "$STOP" --rate "${RATE:-4.93}" \
  --cap-min "$DEADLINE_MIN" --log "$LOGS/cost_${LABEL}_${STAMP}.log" > /dev/null 2>&1 < /dev/null &
W2=$!; echo "cost watcher pid $W2"
nohup python3 "$HERE/guard_containers.py" --app-id "$APP_ID" --max "$MAX" --log "$LOGS/guard_${LABEL}_${STAMP}.log" > /dev/null 2>&1 < /dev/null &
W3=$!; echo "guard pid $W3"
echo "$APP_ID" > "$LOGS/app_${LABEL}_${STAMP}.txt"
READY_END=$(( $(date +%s) + 150 ))  # fail closed: each watcher must log a successful first poll within 150 s
READY=0
while [ "$(date +%s)" -lt "$READY_END" ]; do
  sleep 10
  B=$(grep -c ' state=' "$LOGS/billing_${LABEL}_${STAMP}.log" 2>/dev/null)
  C=$(grep -c '"fails": 0' "$LOGS/cost_${LABEL}_${STAMP}.log" 2>/dev/null)
  G=$(grep -c ' state .*live containers' "$LOGS/guard_${LABEL}_${STAMP}.log" 2>/dev/null)
  if [ "${B:-0}" -ge 1 ] && [ "${C:-0}" -ge 1 ] && [ "${G:-0}" -ge 1 ] && kill -0 $W1 2>/dev/null && kill -0 $W2 2>/dev/null && kill -0 $W3 2>/dev/null; then READY=1; break; fi
  # an app already confirmed stopped (a short job that finished) has nothing left to watch
  if grep -q -E 'stop confirmed|app stopped; guard exits' "$LOGS/guard_${LABEL}_${STAMP}.log" 2>/dev/null; then READY=1; break; fi
done
if [ "$READY" != 1 ]; then
  echo "$(date +%T) the watchers did not all report a good first poll: killing the run, stopping $APP_ID" >&2
  kill $RUN_PID 2>/dev/null
  stop_apps "$APP_ID"
  exit 5
fi
echo "$(date +%T) all three watchers report a good first poll"
wait $RUN_PID
RC=$?
echo "$(date +%T) $LABEL run exited rc $RC"
exit $RC
