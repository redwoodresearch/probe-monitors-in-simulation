#!/bin/bash
# The guards of launch_watched.sh for an app that already exists (a deployed app whose functions are spawned by a local
# script): the strict billing watcher and the container guard by app ID, the elapsed watcher by app ID (v2.5, main @ c44b5b46); the guard is the reader matched-FA folder's on main.
# Records each watcher's whole PID chain (shell, uv, python), so kill_step.sh kills them by PID only.
#   watch_app.sh <label> <app id> <app name> <stop $> <max containers> <cap min> <deadline min>
set -u
LABEL=$1; APP=$2; APPNAME=$3; STOP=$4; MAXC=$5; CAPMIN=$6; DEADMIN=$7
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/results/launch_${LABEL}_$(date +%Y-%m-%d_%H%M%S)
mkdir -p "$OUT" || exit 1
echo "$APP" > "$OUT/app_id"
DEADLINE=$(( $(date +%s) + DEADMIN * 60 ))
( cd "$HERE" && nohup uv run --no-project --python 3.12 python watch_billing_strict.py "$APP" "$STOP" "$DEADLINE" > "$OUT/watch_billing.log" 2>&1 < /dev/null & echo $! > "$OUT/watch_billing.pid" )
( cd "$HERE" && nohup uv run --no-project --python 3.12 --with 'modal==1.6.1' python watch_cost.py --app-id "$APP" --stop "$STOP" --rate "${XREAD_RATE:-5.4}" --cap-min "$CAPMIN" --log "$OUT/watch_cost.log" > "$OUT/watch_cost.out" 2>&1 < /dev/null & echo $! > "$OUT/watch_cost.pid" )
( cd "$HERE" && nohup uv run --no-project --python 3.12 python guard_containers.py --app-id "$APP" --max "$MAXC" --log "$OUT/guard.log" > "$OUT/guard.out" 2>&1 < /dev/null & echo $! > "$OUT/guard.pid" )
for w in watch_billing watch_cost guard; do  # fail closed: a watcher that never reaches python stops the app
  if ! "$HERE/record_chain.sh" "$OUT/$w.pid" "$OUT/$w.tree.pid"; then
    echo "$(date +%H:%M:%S) $w did not start: stopping app $APP" | tee -a "$OUT/guards.log"
    uv run --no-project --python 3.12 --with 'modal==1.6.1' modal app stop "$APP" --yes >> "$OUT/guards.log" 2>&1
    exit 4
  fi
done
echo "$(date +%H:%M:%S) watching $LABEL: app $APP ($APPNAME), stop \$$STOP, max $MAXC containers; chains: billing $(cut -d" " -f1 "$OUT/watch_billing.tree.pid" | tr "\n" " "); cost $(cut -d" " -f1 "$OUT/watch_cost.tree.pid" | tr "\n" " "); guard $(cut -d" " -f1 "$OUT/guard.tree.pid" | tr "\n" " "); logs $OUT"
