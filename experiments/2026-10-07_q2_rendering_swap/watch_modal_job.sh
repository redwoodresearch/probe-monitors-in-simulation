#!/usr/bin/env bash
# Watch a detached Modal extraction: every POLL seconds, read the app's billed cost (billing report from START, a UTC date
# on or before launch, to START + 2 days, so a run crossing UTC midnight keeps its earlier cost) and its live container
# count. Stop the app at MAX_COST dollars, at more than MAX_CONTAINERS containers (the orchestrating function plus one GPU
# shard is 2), or when billing or the container count is unreadable 3 polls running (fail closed), then confirm its
# containers are gone. The limits are checked before completion, and completion waits for a readable cost. When the
# run's stats/summary.json appears, exit 0 only if every shard finished (n_crashed and n_unreachable 0), stopping any
# containers still listed; else stop the app (a crashed shard's container can still be running; a relaunch resumes)
# and exit 6.
# Exits 10, after stopping the app, when it has no containers for two polls after running (or none for ten polls from
# the start: still queued) and the done file never appeared.
# Refuses to start if a summary already exists (a previous attempt's). Logs each poll. Python is uv's, never the system's.
#   watch_modal_job.sh <app id> <volume> <run> [max_cost=3.5] [max_containers=2] [poll=180] [start=today, UTC] [done=summary.json]
# done=shard_0.json watches a run launched without --remote (one GPU container, no orchestrating function): that shard's
# stats file marks completion, and it is ok when it lists its chunks.
#   watch_modal_job.sh --check-summary <volume> <run>     # only the summary verdict (exit 0 ok, 6 crashed, 5 none)
set -uo pipefail
py() { uv run python -c "$@"; }
verdict() {  # the run's summary (or shard stats file): ok, crashed, none, or unparsable (a failure too)
  local raw; raw=$(uv run modal volume get "$1" "$2/stats/${3:-summary.json}" - 2>/dev/null || true)
  printf '%s' "$raw" | py "import json,sys
t = sys.stdin.read(); t = t[:t.rfind('}') + 1]
if not t: print('none'); sys.exit()
d = json.loads(t)
if 'chunks' in d: print('ok' if d['chunks'] else 'crashed (a shard file with no chunks)'); sys.exit()
print('ok' if d.get('n_crashed', 1) == 0 and d.get('n_unreachable', 1) == 0 and d.get('n_ok', 0) >= 1 else
      f\"crashed n_ok={d.get('n_ok')} n_crashed={d.get('n_crashed')} n_unreachable={d.get('n_unreachable')}\")" 2>/dev/null || echo unparsable
}
if [ "${1:-}" = "--check-summary" ]; then
  v=$(verdict "$2" "$3"); echo "$v"; case "$v" in ok) exit 0;; none) exit 5;; *) exit 6;; esac
fi
APP=$1 VOL=$2 RUN=$3 MAX_COST=${4:-3.5} MAX_CONTAINERS=${5:-2} POLL=${6:-180} START=${7:-$(date -u +%Y-%m-%d)} DONE=${8:-summary.json}
END=$(py "import datetime; print(datetime.date.fromisoformat('$START') + datetime.timedelta(days=2))")  # --start alone returns []
[ -n "$END" ] || { echo "REFUSE: cannot compute the billing end date from START=$START"; exit 7; }
containers() { uv run modal container list --json 2>/dev/null | py "import json,sys; print(sum(c['app_id']=='$APP' for c in json.load(sys.stdin)))" 2>/dev/null || echo "?"; }
app_state() { uv run modal app list --json 2>/dev/null | py "import json,sys; print(next((a['state'] for a in json.load(sys.stdin) if a['app_id']=='$APP'), 'unlisted'))" 2>/dev/null || echo "?"; }
stop() {  # exits only once the app itself reads stopped with no containers: a queued call can't start unwatched later
  echo "KILL: $1"
  for i in $(seq 1 20); do
    uv run modal app stop "$APP" --yes >/dev/null 2>&1 || echo "  modal app stop failed (attempt $i)"
    n=$(containers) s=$(app_state); echo "  after stop: $n containers, app state $s"
    [ "$n" = "0" ] && [ "$s" = "stopped" ] && exit "$2"; sleep 15
  done
  echo "WARNING: app not confirmed stopped with no containers 5 min after stop"; exit 4
}
if [ "$(verdict "$VOL" "$RUN" "$DONE")" != "none" ]; then echo "REFUSE: $RUN/stats/$DONE exists before this watch (a previous attempt)"; exit 5; fi
unreadable=0 cunreadable=0 seen=0 gone=0 polls=0
while true; do
  polls=$((polls + 1))
  cost=$(uv run modal billing report --start "$START" --end "$END" --json 2>/dev/null | py "import json,sys; print(round(sum(float(r['cost']) for r in json.load(sys.stdin) if r['object_id']=='$APP'), 3))" 2>/dev/null || echo "?")
  n=$(containers)
  v=$(verdict "$VOL" "$RUN" "$DONE")
  echo "$(TZ=America/Los_Angeles date +%H:%M:%S) cost \$$cost containers $n summary $v"
  # Limits first, so a finished run never hides an extra container or a cost overrun.
  if [ "$n" = "?" ]; then cunreadable=$((cunreadable + 1)); echo "  container count unreadable ($cunreadable in a row)"
    [ "$cunreadable" -ge 3 ] && stop "container count unreadable 3 polls running (fail closed)" 9
  else cunreadable=0
    [ "$n" -gt "$MAX_CONTAINERS" ] && stop "$n containers > $MAX_CONTAINERS" 3
  fi
  if [ "$cost" = "?" ]; then unreadable=$((unreadable + 1)); echo "  billing unreadable ($unreadable in a row)"
    [ "$unreadable" -ge 3 ] && stop "billing unreadable 3 polls running (fail closed)" 8
  else unreadable=0
    py "import sys; sys.exit(0 if float('$cost') >= float('$MAX_COST') else 1)" && stop "cost \$$cost >= \$$MAX_COST" 2
  fi
  # A run that ends without its done file (a crashed shard writes no stats): containers seen, then none for two polls.
  if [ "$n" != "?" ] && [ "$n" -gt 0 ]; then seen=1; gone=0; elif [ "$n" = "0" ]; then gone=$((gone + 1)); fi
  # Stop the app before giving up: a launch still queued for GPU capacity lists no running containers and could start
  # later with nothing watching it.
  if [ "$v" = "none" ] && { { [ "$seen" = 1 ] && [ "$gone" -ge 2 ]; } || { [ "$seen" = 0 ] && [ "$polls" -ge 10 ]; }; }; then
    stop "no containers and no $DONE (the run ended without finishing, or is still queued)" 10; fi
  case "$v" in
    ok) if [ "$cost" = "?" ]; then echo "  summary ok; holding until billing is readable (or the fail-closed stop)"; sleep "$POLL"; continue; fi
        echo "DONE"
        if [ "$n" != "0" ]; then echo "  stopping the app's $n remaining containers (idle ones still bill)"; uv run modal app stop "$APP" --yes
          for i in $(seq 1 20); do m=$(containers); echo "  after stop: $m containers"; [ "$m" = "0" ] && exit 0; sleep 15; done
          echo "WARNING: containers still listed 5 min after stop"; exit 4; fi
        exit 0;;
    none) ;;
    *) stop "summary reports a failed shard ($v); a crashed shard's container may still be running" 6;;  # relaunch resumes
  esac
  sleep "$POLL"
done
