#!/bin/bash
# usage: watch_whitebox.sh <stop_dollars> <done_file> <rate_file> <app_id> [<app_id> ...]
# Run from a frozen copy (never edit a running script). Fails closed:
# - Every Modal CLI call is bounded to 40 s (perl alarm; macOS has no `timeout`).
# - Integrated cost (live containers x the highest rate seen while containers are up; the launcher's <rate_file> only
#   lowers it once tasks reach 0) is checked against the trigger BEFORE the billing call. A failed app-list call counts
#   as 1 task at 5.5/h.
# - Billing is summed over the apps since yesterday UTC, so UTC midnight isn't dropped; a failed read keeps the last value.
# - Trigger at stop - $0.35: worst-case detection is a 60 s sleep plus two 40 s calls plus about 50 s of shutdown,
#   about 190 s, at the H200's $5.5/h.
# - At the trigger, it keeps stopping every app until each is verified stopped or gone (exit 2). It exits 0 when
#   <done_file> exists.
STOP=$1; DONE=$2; RATEF=$3; shift 3; APPS="$*"; est=0; billed=0; cur=0; last=$(date +%s); HEAD=0.35
M() { perl -e 'alarm shift; exec @ARGV' 40 uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal "$@"; }
SINCE=$(date -u -v-1d +%Y-%m-%d 2>/dev/null || date -u -d yesterday +%Y-%m-%d); UNTIL=$(date -u -v+1d +%Y-%m-%d 2>/dev/null || date -u -d tomorrow +%Y-%m-%d)
over() { python3 -c "import sys; sys.exit(0 if float('$1')>=float('$STOP')-$HEAD else 1)"; }
stop_all() {
  echo "STOP threshold reached (\$$1 >= \$$STOP - \$$HEAD): stopping $APPS"
  pending="$APPS"
  while [ -n "$pending" ]; do  # each round: stop every pending app, then verify all, and retry only the unconfirmed
    for a in $pending; do M app stop "$a" --yes; done
    js=$(M app list --json 2>/dev/null); left=""
    for a in $pending; do
      st=$(echo "$js" | python3 -c "import json,sys; x=[a for a in json.load(sys.stdin) if a['app_id']=='$a']; print(x[0]['state'] if x else 'missing')" 2>/dev/null)
      echo "$(TZ=America/Los_Angeles date +%H:%M:%S) $a state: ${st:-unknown}"
      { [ "$st" = "stopped" ] || [ "$st" = "missing" ]; } || left="$left $a"
    done
    pending=$(echo $left); [ -n "$pending" ] && sleep 10
  done; exit 2
}
while true; do
  [ -f "$DONE" ] && { echo "done file present; exiting; integrated \$$est billed \$$billed"; exit 0; }
  want=$(cat "$RATEF" 2>/dev/null || echo 5.5)
  tasks=$(M app list --json 2>/dev/null | python3 -c "import json,sys; ids='$APPS'.split(); print(sum(int(a['tasks'] or 0) for a in json.load(sys.stdin) if a['app_id'] in ids))" 2>/dev/null)
  if [ -z "$tasks" ]; then tasks=1; cur=5.5; echo "app list failed: counting 1 task at 5.5"; fi
  cur=$(python3 -c "print(float('$want') if int('$tasks')==0 else max(float('$cur'), float('$want')))")
  now=$(date +%s); est=$(python3 -c "print(round($est + $tasks * $cur * ($now - $last) / 3600, 3))"); last=$now
  over "$(python3 -c "print(max(float('$est'), float('$billed')))")" && stop_all "$est"
  b=$(M billing report --start $SINCE --end $UNTIL --resolution h --json 2>/dev/null | python3 -c "import json,sys; ids='$APPS'.split(); print(round(sum(float(r['cost']) for r in json.load(sys.stdin) if r['object_id'] in ids),3))" 2>/dev/null)
  [ -n "$b" ] && billed=$b
  cost=$(python3 -c "print(max(float('$est'), float('$billed')))")
  echo "$(TZ=America/Los_Angeles date +%H:%M:%S) tasks $tasks rate $cur integrated \$$est billed \$$billed"
  over "$cost" && stop_all "$cost"
  sleep 60
done
