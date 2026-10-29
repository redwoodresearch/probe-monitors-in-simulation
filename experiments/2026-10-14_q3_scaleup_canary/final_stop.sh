#!/usr/bin/env bash
# Laptop side, at the end (RULE.md, Stops): wait for allin_stop.sh to exit, require its "apps confirmed stopped" line
# (else stop and confirm every app here with stop_group.sh FORCE_APPS=1), confirm both launcher panes are gone, copy both
# run folders back to the laptop's .run_cache (0d's move path: files only, never read here; free disk checked first;
# bandwidth limited; each copy bounded to 45 minutes), then stop the devbox through EC2 whatever the copies did, and
# confirm "stopped". Exit 1 on anything not confirmed; a failed copy stays on the stopped devbox's disk.
#   H=<harness> DEVBOX=<user>@<id> APP_IDS_FILE=<file> final_stop.sh <allin_stop pid> <allin_stop log> <run A> <run B>
set -uo pipefail
H=${H:?}; INSTANCE=${DEVBOX:?}; INSTANCE=${INSTANCE#*@}; APPS=$(tr -d ' \n' < "${APP_IDS_FILE:?}")
HERE=$(cd "$(dirname "$0")" && pwd); LOCAL=$H/../../.run_cache/phase2_attack_check/runs
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') $*"; }
dbx() { perl -e 'alarm shift; exec @ARGV' "$1" "$H/devbox.sh" "$2"; }
while kill -0 "$1" 2>/dev/null; do sleep 60; done
rc=0
if grep -q "apps confirmed stopped" "$2"; then log "allin_stop confirmed the apps stopped"
else
  log "allin_stop's log has no 'apps confirmed stopped': stopping them here"
  H=$H DEVBOX=$DEVBOX APP_IDS=$APPS FORCE_APPS=1 "$HERE/stop_group.sh" "" || { log "ALARM: apps not confirmed stopped"; rc=1; }
fi
n=$(dbx 60 "tmux -L phase2 list-panes -a -F '#{session_name}' 2>/dev/null | grep -cxE 'canA|canB'") || n=unknown
[[ $n == 0 ]] || { log "ALARM: launcher panes left: $n"; rc=1; }
for run in "$3" "$4"; do
  need=$(dbx 60 "du -sk ~/phase2/runs/$run | cut -f1") || need=""
  free=$(df -k "$H" | awk 'NR==2 {print $4}')
  if [[ $need =~ ^[0-9]+$ ]] && (( need * 2 < free )); then
    mkdir -p "$LOCAL/$run"
    perl -e 'alarm 2700; exec @ARGV' "$H/devbox.sh" rsync -a --bwlimit=20000 "box:phase2/runs/$run/" "$LOCAL/$run/" &&
      perl -e 'alarm 300; exec @ARGV' "$H/devbox.sh" rsync -a "box:phase2/runs/$run.launcher.log" "$LOCAL/$run/launcher.log" &&
      log "copied $run" ||
      { log "ALARM: copy of $run failed (it stays on the stopped devbox's disk)"; rc=1; }
  else
    log "ALARM: not copying $run (size ${need:-unknown} KiB, laptop free $free KiB)"; rc=1
  fi
done
state() { aws ec2 describe-instances --profile redwood-sandbox --region us-west-2 --instance-ids "$INSTANCE" \
  --query 'Reservations[].Instances[].State.Name' --output text; }
aws ec2 stop-instances --profile redwood-sandbox --region us-west-2 --instance-ids "$INSTANCE" >/dev/null
for _ in $(seq 20); do [[ $(state) == stopped ]] && break; sleep 30; done
[[ $(state) == stopped ]] && log "devbox $INSTANCE stopped" || { log "ALARM: devbox $INSTANCE is $(state)"; rc=1; }
exit $rc
