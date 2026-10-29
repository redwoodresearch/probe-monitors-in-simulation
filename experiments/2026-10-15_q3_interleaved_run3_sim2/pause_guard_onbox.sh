#!/usr/bin/env bash
# ON THE DEVBOX (d9 ~21:40, 10-15): a time guard for one S_paper493 sim launch (L1 box3 / L2 box4). At AT (Pacific time, today),
# if the launch's launch.json is still unfinished, touch <run dir>/PAUSE_SIMS (the frozen launcher then starts no new run; runs in
# flight finish) and write a one-line <run dir>/TIME_GUARD.txt. Touch only: it never kills or signals any process. Replaces the
# laptop guard (sp493/pause_2245.sh), which cannot fire while the laptop sleeps.
#   nohup setsid bash pause_guard_onbox.sh <run dir> 22:45 >> ~/phase2/sp493sim/time_guard_<L>.log 2>&1 < /dev/null &
set -euo pipefail
RUN=${1%/}; AT=$2
[[ -f $RUN/launch.json ]] || { echo "no $RUN/launch.json" >&2; exit 1; }
log() { echo "$(TZ=America/Los_Angeles date '+%F %T') PT [time guard $RUN] $*"; }
T=$(TZ=America/Los_Angeles date -d "$(TZ=America/Los_Angeles date +%F) $AT" +%s)
log "armed (pid $$): at $AT PT ($T) touch PAUSE_SIMS + TIME_GUARD.txt if launch.json is unfinished"
while (( $(date +%s) < T )); do sleep 15; done
FIN=unread  # Codex r50: the launcher rewrites launch.json in place, so a read can fail mid-write; retry (bounded), and an
for i in $(seq 120); do  # unreadable file never counts as finished (pausing a finished launch is harmless)
  if FIN=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["finished"] is None)' "$RUN/launch.json"); then break; fi
  FIN=unread; log "launch.json read failed (try $i); retrying"; sleep 1
done
if [[ $FIN == False ]]; then log "launch finished: nothing touched"; exit 0; fi
touch "$RUN/PAUSE_SIMS"
echo "$(TZ=America/Los_Angeles date '+%F %T') PT PAUSE_SIMS by pause_guard_onbox.sh: set hard timeout ~23:29; split for the tail relaunch" > "$RUN/TIME_GUARD.txt"
log "PAUSE_SIMS touched ($(stat -c %y "$RUN/PAUSE_SIMS")); TIME_GUARD.txt written; no process signalled"
