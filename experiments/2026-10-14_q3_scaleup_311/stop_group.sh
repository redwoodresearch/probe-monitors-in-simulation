#!/usr/bin/env bash
# Stop one launch phase and confirm it (RULE.md, Stops). Used by allin_stop.sh (on a hard stop, and to stop the apps
# when the phase ends) and on a failed gate. Targets are tmux PANE ids (%N, bound to their run when first seen; never
# reused by the tmux server), never session names. Every remote or Modal call runs in its own process group, killed
# whole when its bound expires (a child holding the output open can't outlive it); each bound is clipped to the time
# left before its phase's deadline.
#   1. Panes (deadline T0 + STOP_WAIT_S, default 300 s, T0 = before the first Ctrl+C): Ctrl+C into every pane in
#      parallel, then confirm every pane gone (re-sending Ctrl+C), until the deadline. An unreachable devbox = not gone.
#   2. Apps (if panes are not confirmed gone, or FORCE_APPS=1; deadline = start of this phase + APP_STOP_S, default 180 s):
#      `modal app stop` every app in APP_IDS in parallel, then confirm each app's state is exactly "stopped".
# Exit 0: panes gone (and apps stopped if requested); 2: panes not confirmed, apps stopped; 3: an app not confirmed
# stopped (stop it by hand). Worst case wall clock: STOP_WAIT_S + APP_STOP_S + a few seconds.
# Usage (laptop): H=<bbf0729 harness> DEVBOX=<user>@<id> APP_IDS=<id,...> [FORCE_APPS=1] stop_group.sh [<%id:pid>[,...]]
set -uo pipefail
H=${H:?}; APP_IDS=${APP_IDS:-}; WAIT=${STOP_WAIT_S:-300}; APPW=${APP_STOP_S:-180}; FORCE=${FORCE_APPS:-0}
MODAL_CMD=${MODAL_CMD:-uv run --no-project --python 3.12 --with modal==1.2.6 modal}  # overridable for the tests only
IFS=, read -ra PANES <<< "${1:-}"
IFS=, read -ra APPS <<< "$APP_IDS"
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') $*"; }
# bounded <seconds> <cmd...>: run cmd in a new process group; on expiry kill the whole group; exit 142
bounded() {
  perl -e 'my $t = shift; my $pid = fork(); die "fork" unless defined $pid;
           if ($pid == 0) { setpgrp(0, 0); exec @ARGV or exit 127 }
           local $SIG{ALRM} = sub { kill "KILL", -$pid; waitpid($pid, 0); exit 142 }; alarm $t;
           waitpid($pid, 0); exit($? & 127 ? 128 + ($? & 127) : $? >> 8)' "$@"
}
left() { local l=$(( $1 - $(date +%s) )); (( l > $2 )) && l=$2; (( l < 1 )) && l=1; echo $l; }
dbx() { bounded "$(left "$1" "$2")" "$H/devbox.sh" "$3"; }  # dbx <deadline> <max s> <remote command>
# panes are <id>:<pid>, matched exactly among all live panes (tmux 3.4's display-message falls back on a missing target)
gone() { [[ $(dbx "$1" 30 "tmux -L phase2 list-panes -a -F '#{pane_id}:#{pane_pid}' 2>/dev/null | grep -qxF '$2' && echo LIVE || echo GONE" 2>/dev/null) == GONE ]]; }
ctrlc() { dbx "$1" 30 "tmux -L phase2 list-panes -a -F '#{pane_id}:#{pane_pid}' 2>/dev/null | grep -qxF '$2' && tmux -L phase2 send-keys -t '${2%%:*}' C-c"; }
T0=$(date +%s); end=$(( T0 + WAIT )); confirmed=1
if [[ -n ${PANES[*]:-} ]]; then
  for p in "${PANES[@]}"; do
    [[ -n $p ]] && { ctrlc $end "$p" && log "Ctrl+C sent to $p" || log "Ctrl+C to $p failed (or the pane is gone)"; } &
  done
  wait
  confirmed=0
  while (( $(date +%s) < end )); do
    still=""
    for p in "${PANES[@]}"; do [[ -n $p ]] && ! gone $end "$p" && still="$still $p"; done
    [[ -z $still ]] && { confirmed=1; log "every pane gone: stop confirmed $(( $(date +%s) - T0 ))s after the first Ctrl+C"; break; }
    for p in $still; do ctrlc $end "$p" > /dev/null 2>&1; done
    (( $(date +%s) + 10 < end )) && sleep 10 || break
  done
  (( confirmed )) || log "NOT CONFIRMED within ${WAIT}s of the first Ctrl+C:$still"
fi
if (( ! confirmed || FORCE )) && [[ -n $APP_IDS ]]; then
  aend=$(( $(date +%s) + APPW ))
  for a in "${APPS[@]}"; do bounded "$(left $aend 120)" $MODAL_CMD app stop "$a" > /dev/null 2>&1 & done
  wait
  pending=1
  while (( $(date +%s) < aend )); do
    j=$(bounded "$(left $aend 60)" $MODAL_CMD app list --json 2>/dev/null) || j=""
    pending=$(printf '%s' "$j" | python3 -c "
import json, sys
ids = sys.argv[1].split(',')
try:
    apps = {a.get('App ID') or a.get('app_id'): (a.get('State') or a.get('state') or '').lower() for a in json.load(sys.stdin)}
except Exception:
    print(1); sys.exit()
print(int(any(apps.get(i) != 'stopped' for i in ids)))" "$APP_IDS")
    (( pending )) || break
    (( $(date +%s) + 10 < aend )) && sleep 10 || break
  done
  if (( pending )); then log "APPS NOT CONFIRMED STOPPED ($APP_IDS) within ${APPW}s: stop them by hand"; exit 3; fi
  log "apps confirmed stopped: $APP_IDS"
fi
(( confirmed )) && exit 0 || exit 2
