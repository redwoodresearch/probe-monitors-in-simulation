#!/usr/bin/env bash
# Laptop side: launch one new-mains split as soon as its box's scale-up MAIN launch has ended (RULE.md Amendment 1, item
# 4: no two REAL launches on one settings tree). Polls the box every 120 s, read-only: the scale-up's main launcher
# session (-L phase2, exact name) is gone AND its launch.json says "finished". Then runs launch_nm.sh once. A failed read
# is retried; a deadline (DEADLINE, epoch s) ends the wait with nothing started.
#   K=<split> DEVBOX=<id> H=<nm harness> RUN=<run> N=<main servers> WD=<log dir> MAIN_SESSION=<s311q<n>>
#   MAIN_RUN=<scale-up main run dir> DEADLINE=<epoch s> launch_when_free.sh
set -uo pipefail
for v in K DEVBOX H RUN N WD MAIN_SESSION MAIN_RUN DEADLINE; do [[ -n ${!v:-} ]] || { echo "set $v" >&2; exit 1; }; done
HERE=$(cd "$(dirname "$0")" && pwd)
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') [S$K wait] $*"; }
while (( $(date +%s) < DEADLINE )); do
  # ENDED only on two positive observations: tmux lists the socket's sessions (exit 0) without MAIN_SESSION, or says no
  # server runs on that socket; and launch.json parses with a "finished" time. Any other tmux error is a failed read.
  r=$(perl -e 'alarm 60; exec @ARGV' bash "$H/devbox.sh" "bash -s -- '$MAIN_SESSION' '$MAIN_RUN'" 2>/dev/null <<'EOF_REMOTE'
set -u
out=$(tmux -L phase2 list-sessions -F '#{session_name}' 2>&1); rc=$?
if (( rc == 0 )); then
  if grep -qxF -- "$1" <<< "$out"; then echo LIVE; exit 0; fi
elif ! grep -qE '^(no server running on |error connecting to .*\(No such file or directory\))' <<< "$out"; then
  echo TMUXERR; exit 0
fi
python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("finished") else 1)' "$2/launch.json" 2>/dev/null \
  && echo ENDED || echo UNFINISHED
EOF_REMOTE
)
  rc=$?; r=$(tail -1 <<< "$r")
  (( rc == 0 )) || r="exit $rc"  # a remote command that didn't exit 0 is a failed read, whatever it printed
  case "$r" in
    ENDED) log "scale-up main ended ($MAIN_SESSION gone, launch.json finished): launching"; K=$K DEVBOX=$DEVBOX H=$H RUN=$RUN N=$N WD=$WD bash "$HERE/launch_nm.sh"; exit $? ;;
    LIVE|UNFINISHED) log "scale-up main: $r" ;;
    *) log "read failed; retrying" ;;
  esac
  sleep 120
done
log "deadline reached: nothing started"; exit 1
