#!/usr/bin/env bash
# All-in hard stop and app owner for one launch phase (RULE.md, Caps and Stops). Start it BEFORE the phase's servers;
# launch_servers.py owns its own cleanup while it runs (any failure: it stops and confirms every app it created), and this
# script owns the apps from the moment their ids are in APP_IDS_FILE until they are confirmed stopped. Every POLL_S
# seconds (default 120) it adds up:
#   simulator: spend_now.py on the launch (priced spend; unpriced and running attempts at max(fresh or live reading,
#     estimate); a running attempt with no successful fresh read for 300 s at the worst case), over the devbox ssh;
#   GPU: N_SERVERS x the hours since GPU_START (taken before launch_servers.py) x $4.54; devbox: hours since BOX_START x
#     BOX_RATE (default $1.61; 0 for a phase that shares the box); plus EXTRA_USD.
# v4 (amendment 10, runs may go unwatched): HARD STOP (stop_group.sh: panes, then the apps, each confirmed) only when a
# FRESH reading of ACTUAL spend (field 5 of spend_now.py: priced plus live readings, no pre-charge) puts the total at
# STOP_USD - RESERVE_USD; a stale or unreadable reading (snapshot older than STALE_S while the session is live) never stops
# anything, it is logged and retried. Also stops when the session has not appeared STARTUP_S (default 1800) after the
# app ids appeared, or its pane start command does not name its run (`--out /home/aswin/<run> `). Before a hard
# stop it writes ~/<K>/STOPPED_<session> on the devbox; launch_simh.sh refuses to start a session with that marker.
# When every session has appeared and ended (a finished run, or an early exit such as a failed server check), it stops
# the apps too (stop_group.sh with FORCE_APPS=1) and exits with its code. Sessions are bound on first sight to their pane's
# id:pid (exact matches among `list-panes -a`; tmux 3.4's display-message falls back silently on a missing target).
# Usage (laptop): H=<bbf0729 harness> DEVBOX=<user>@<id> STOP_USD= RESERVE_USD= N_SERVERS= GPU_START=<epoch s>
#   BOX_START=<epoch s> [BOX_RATE=1.61] APP_IDS_FILE=<file> allin_stop.sh <session>=<run dir under ~ on the devbox>[,...]
set -uo pipefail
H=${H:?}; STOP=${STOP_USD:?}; RES=${RESERVE_USD:?}; NS=${N_SERVERS:?}; G0=${GPU_START:?}; B0=${BOX_START:?}
BR=${BOX_RATE:-1.61}; APPF=${APP_IDS_FILE:?}; POLL=${POLL_S:-120}; STALE_S=${STALE_S:-300}; STALE_MAX=${STALE_MAX:-3}
STARTUP=${STARTUP_S:-1800}; APPS_WAIT=${APPS_WAIT_S:-3600}; EXTRA=${EXTRA_USD:-0}; K=${K:-phase2/simh_checks}; HARN=${HARN:-phase2/attack_check_simh}
PYV=${PYV:-/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python}; TREE=${TREE:-/home/aswin/phase2/sims-sept-strict-off}
HERE=$(cd "$(dirname "$0")" && pwd)
IFS=, read -ra PAIRS <<< "$1"
LASTV=(); FAILS=(); PANE=(); TAPPS=0; T0=$(date +%s)  # bash 3.2: indexed arrays parallel to PAIRS; PANE "" = not seen yet, "gone" = ended
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') $*"; }
bounded() {
  perl -e 'my $t = shift; my $pid = fork(); die "fork" unless defined $pid;
           if ($pid == 0) { setpgrp(0, 0); exec @ARGV or exit 127 }
           local $SIG{ALRM} = sub { kill "KILL", -$pid; waitpid($pid, 0); exit 142 }; alarm $t;
           waitpid($pid, 0); my $s = $?; kill "KILL", -$pid; exit($s & 127 ? 128 + ($s & 127) : $s >> 8)' "$@"
}
dbx() { bounded "$1" "$H/devbox.sh" "$2"; }
apps() { tr -d ' \n' < "$APPF" 2>/dev/null; }
finish() {  # $1 = reason, $2 = 1 for a hard stop (panes first), 0 when every session already ended
  local panes="" i s
  if (( $2 )); then
    log "HARD STOP: $1"
    for i in "${!PAIRS[@]}"; do
      s=${PAIRS[$i]%%=*}; dbx 30 "touch ~/$K/STOPPED_$s" || log "could not write the STOPPED_$s marker"
      [[ -n ${PANE[$i]:-} && ${PANE[$i]} != gone ]] && panes="$panes${panes:+,}${PANE[$i]}"
    done
  else
    log "$1: stopping the phase's apps"
  fi
  H=$H DEVBOX=$DEVBOX APP_IDS="$(apps)" FORCE_APPS=1 "$HERE/stop_group.sh" "$panes"; exit $?
}
while true; do
  [[ $TAPPS == 0 && -n "$(apps)" ]] && { TAPPS=$(date +%s); log "app ids present: $(apps)"; }
  if (( TAPPS == 0 && $(date +%s) - T0 >= APPS_WAIT )); then  # launch_servers.py allows 40 min; it owns a failed start
    log "ALARM: no app ids ${APPS_WAIT}s after start; launch_servers.py owns the cleanup of a failed start: check it"; exit 4
  fi
  sim=0; all_ended=1; FRESH=0
  for i in "${!PAIRS[@]}"; do
    s=${PAIRS[$i]%%=*}; run=${PAIRS[$i]#*=}
    if [[ -z ${PANE[$i]:-} ]]; then  # not seen yet: bind on first sight
      # list-panes -a with an exact session-name match (tmux 3.4's display-message falls back silently on a missing target)
      b=$(dbx 60 "bash ~/$K/pane_state.sh --of $s" 2>/dev/null) || b=""
      if [[ -n $b ]]; then
        [[ $b == *"--out /home/aswin/$run "* ]] || finish "session $s is not the launcher of $run" 1
        PANE[$i]=${b%% *}; log "bound $s to pane ${PANE[$i]} ($run)"
      elif (( TAPPS > 0 && $(date +%s) - TAPPS >= STARTUP )); then
        finish "session $s has not appeared ${STARTUP}s after the app ids" 1
      fi
    fi
    rc=1  # 1 = ended or not started (no spend); 0 = live; else unknown (counts as live)
    if [[ -z ${PANE[$i]:-} ]]; then all_ended=0
    elif [[ ${PANE[$i]} != gone ]]; then  # the bound pane id:pid, matched exactly among all live panes
      st=$(dbx 60 "bash ~/$K/pane_state.sh ${PANE[$i]}" 2>/dev/null) || st=""  # UNKNOWN or no reply: live
      if [[ $st == GONE ]]; then PANE[$i]=gone; log "$s ended"; else rc=0; all_ended=0; fi
    fi
    out=$(dbx 90 "PYTHONPATH=$TREE/src $PYV ~/$K/spend_now.py ~/$run ~/$HARN ~/$K" 2>/dev/null) || out=""
    [[ $out == absent ]] && out="0 0 0 0 0"
    age=$(echo "$out" | cut -d' ' -f2); v=$(echo "$out" | cut -d' ' -f5)  # field 5: ACTUAL spend (no pre-charge)
    if [[ $v =~ ^[0-9.]+$ && $age =~ ^[0-9]+$ ]] && { [[ $rc == 1 ]] || (( age <= STALE_S )); }; then
      FAILS[$i]=0; LASTV[$i]=$v; FRESH=1
    else  # v4 (amendment 10): a stale or missing reading NEVER stops anything; it is logged and retried
      FAILS[$i]=$(( ${FAILS[$i]:-0} + 1 ))
      log "spend of $run unreadable or stale (${out:-no reply}), ${FAILS[$i]} in a row: not stopping (stale reads never stop)"
      v=${LASTV[$i]:-0}
    fi
    sim=$(python3 -c "print($sim + $v)")
  done
  now=$(date +%s)
  total=$(python3 -c "print(round($sim + $NS * ($now - $G0) / 3600 * 4.54 + ($now - $B0) / 3600 * $BR + $EXTRA, 2))")
  log "all-in so far \$$total (simulator, actual: \$$sim; fresh: $FRESH); stop at \$$STOP - reserve \$$RES"
  # v4: only a FRESH reading of ACTUAL spend can trip the stop (GPU and devbox are clock-based, always fresh)
  (( FRESH )) && python3 -c "import sys; sys.exit(0 if $total >= $STOP - $RES else 1)" \
    && finish "all-in \$$total (actual, fresh) reached \$$STOP - \$$RES" 1
  ((all_ended)) && finish "every launcher session has appeared and ended" 0
  sleep "$POLL"
done
