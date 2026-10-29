#!/bin/bash
# Kill one launch's watchers by their recorded PIDs only (never by name or pattern): each PID is signalled only if its
# start time still equals the recorded one (no PID reuse), deepest first; then confirm. Exit 1 if a verified watcher lives.
#   kill_step.sh <launch folder>
L=$1; left=""
norm() { awk '{ if (NF > 2) { for (i = 1; i <= NF; i++) print $i } else print }' "$1"; }  # an old one-line chain: one PID per line
for w in watch_billing watch_cost guard; do
  f="$L/$w.tree.pid"; [ -f "$f" ] || continue
  while read -r p t; do
    [ -n "$p" ] || continue
    now=$(ps -p "$p" -o lstart= 2>/dev/null | tr -s ' ' '_')
    if [ -n "$now" ] && { [ -z "$t" ] || [ "$now" = "$t" ]; }; then kill "$p" 2>/dev/null; fi
  done < <(norm "$f" | tail -r)
done
sleep 2
for w in watch_billing watch_cost guard; do
  f="$L/$w.tree.pid"; [ -f "$f" ] || continue
  while read -r p t; do
    now=$(ps -p "$p" -o lstart= 2>/dev/null | tr -s ' ' '_')
    [ -n "$now" ] && { [ -z "$t" ] || [ "$now" = "$t" ]; } && left="$left $p"
  done < <(norm "$f")
done
echo "$(basename "$L"): killed the recorded chains; still alive:${left:- none}"
[ -z "$left" ]
