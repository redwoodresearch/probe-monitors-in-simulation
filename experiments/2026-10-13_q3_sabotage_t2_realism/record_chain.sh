#!/bin/bash
# Record one watcher's PID chain (shell -> uv -> python watcher), stopping AT the watcher's python process (the process whose
# executable is python, not uv, whose own command line also holds "python watch_*.py"; never below it,
# so a CLI subprocess the watcher spawns is not recorded), with each PID's start time, for kill_step.sh's identity check.
# Exits 1 if no python watcher appears within 60 s (the caller then stops the app: fail closed).
#   record_chain.sh <pid file> <tree file>
PIDF=$1; TREE=$2
for i in $(seq 1 60); do
  chain=$(cat "$PIDF"); cur=$chain; found=""
  for d in 1 2 3 4; do
    if ps -p "$cur" -o command= 2>/dev/null | grep -q -E '^ *[^ ]*python[0-9.]* (watch_billing_strict|watch_cost|guard_containers)\.py'; then found=1; break; fi
    c=$(pgrep -P "$cur" | head -1); [ -z "$c" ] && break; chain="$chain $c"; cur=$c
  done
  if [ -n "$found" ]; then
    : > "$TREE"
    for p in $chain; do echo "$p $(ps -p "$p" -o lstart= | tr -s ' ' '_')" >> "$TREE"; done
    exit 0
  fi
  sleep 1
done
exit 1
