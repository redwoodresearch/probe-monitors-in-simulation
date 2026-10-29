#!/usr/bin/env bash
# Laptop side: release both canary launches (RULE.md, release and readiness). Before it: allin_stop.sh is already running
# over canA and canB with APP_IDS_FILE (started BEFORE the servers; it owns the apps from the moment their ids are in the
# file, and stops them on a hard stop and when every launcher has ended), and every server app has its billing watcher
# and guard. This script:
#   1. refuses unless each app in APP_IDS_FILE has its billing watcher and guard and allin_stop.sh is running;
#   2. on the devbox: check_b_copy.sh, then canary.sh for A, then for B (each: pre-launch gates, then its launcher only
#      if they pass), then the live settings watch for A (tmux -L gates), verified;
#   3. on any failure: Ctrl+C every launcher pane already started, and stop and confirm every app (stop_group.sh with
#      FORCE_APPS=1, the purchase fork's @ 343e73b5).
# Every devbox call is bounded.
#   H=<harness> DEVBOX=<user>@<id> RUN_A= RUN_B= BASES_A= BASES_B= APP_IDS_FILE= IMAGES=<abs json on the devbox> start_group.sh
set -uo pipefail
for v in H DEVBOX RUN_A RUN_B BASES_A BASES_B APP_IDS_FILE IMAGES; do [[ -n ${!v:-} ]] || { echo "set $v" >&2; exit 1; }; done
HERE=$(cd "$(dirname "$0")" && pwd)
CA=/home/aswin/phase2/attack_check_canary; CB=/home/aswin/phase2/attack_check_canary_b
APPS=$(tr -d ' \n' < "$APP_IDS_FILE")
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') $*"; }
dbx() { perl -e 'alarm shift; exec @ARGV' "$1" "$H/devbox.sh" "$2"; }
panes() { dbx 60 "tmux -L phase2 list-panes -a -F '#{session_name} #{pane_id}:#{pane_pid}' 2>/dev/null | awk '\$1==\"canA\" || \$1==\"canB\" {print \$2}' | paste -sd, -"; }
fail() { log "RELEASE FAILED: $*; stopping the launchers started and every app"
  H=$H DEVBOX=$DEVBOX APP_IDS=$APPS FORCE_APPS=1 "$HERE/stop_group.sh" "$(panes)"; exit 1; }
[[ -n $APPS ]] || { log "no app ids in $APP_IDS_FILE"; exit 1; }
pgrep -f "allin_stop.sh canA=" >/dev/null || fail "allin_stop.sh is not running"
for a in ${APPS//,/ }; do
  pgrep -f "watch_billing_strict_v26.py $a" >/dev/null && pgrep -f "guard_tasks_v26.py $a" >/dev/null || fail "app $a has no billing watcher or guard"
done
dbx 300 "bash $CA/check_b_copy.sh $CA $CB" || fail "B's copy is not A's (check_b_copy.sh)"
dbx 600 "SESSION=canA HARNESS=$CA bash $CA/canary.sh $RUN_A config/pilot_canary_a.yaml $BASES_A $IMAGES" || fail "canary.sh A"
dbx 600 "SESSION=canB HARNESS=$CB bash $CB/canary.sh $RUN_B config/pilot_canary_b.yaml $BASES_B $IMAGES" || fail "canary.sh B"
dbx 60 "cd $CA && tmux -L gates new -d -s gatecanA \"PYTHONPATH=/home/aswin/phase2/sims-sept-strict-off/src /home/aswin/sims/simulated-control-evals-sept/.venv/bin/python gate_settings.py --reference-json config/sim_settings_reference_2026-10-14_125923.json --out results/gate_settings_new_$RUN_A.json --new /home/aswin/phase2/runs/$RUN_A --session canA > results/gate_settings_new_$RUN_A.log 2>&1\"" ||
  fail "settings watch did not start"
sleep 10; dbx 30 "tmux -L gates has-session -t '=gatecanA'" || fail "settings watch is not running"
log "released: canA, canB, settings watch gatecanA; launcher panes $(panes)"
