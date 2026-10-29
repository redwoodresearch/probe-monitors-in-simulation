#!/usr/bin/env bash
# ON THE BOX (RULE.md Amendment 6): launch_nm.sh's box steps for a split whose Modal servers were started from the laptop
# beforehand (the box has no Modal token), behind an on-box wait. Run in its own tmux session (-L nmguards s<K>wait):
#   0. every 120 s, until DEADLINE (epoch s): wait until SIGNAL (a launch.json) parses with a non-null "finished"; a failed
#      read is a wait; at the deadline nothing starts;
#   1. the box: the isolated tree is present, the run folders don't exist, the sessions are free, PAUSE_SIMS written and
#      checked, the scratch image, inotify;
#   2. canary.sh (pre-launch gates: freeze, preflight, approach/template/provenance/r2, settings equality; then the
#      launcher in tmux -L nm snms<K>, which runs its own server checks before any run);
#   3. gate1_nm.py (gbnms<K>, deadline END_MAIN - 10 min) and end_main.py (emnms<K>; follow-up URLs already in
#      ~/phase2/nm/checks/fu_bases_s<K>.txt), both verified running.
# Any failure: log it, Ctrl+C the launcher if it started, stop. One line per step to ~/phase2/nm/checks/s<K>_onbox.log.
#   K=<split> RUN=<run name> BM=<main base URLs, comma-separated> END_MAIN=<epoch s> END_FU=<epoch s> SIGNAL=<launch.json>
#   DEADLINE=<epoch s> N=<main servers> launch_nm_onbox.sh
set -uo pipefail
for v in K RUN BM END_MAIN END_FU SIGNAL DEADLINE N; do [[ -n ${!v:-} ]] || { echo "set $v" >&2; exit 1; }; done
C=/home/aswin/phase2/nm/attack_check_nm; CH=/home/aswin/phase2/nm/checks; RUNP=/home/aswin/phase2/nm/runs/$RUN; FU=${RUNP}_fu
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python; TREE=/home/aswin/phase2/sims-sept-strict-off/src
IMAGES=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json
LOG=$CH/s${K}_onbox.log
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') [S$K onbox] $*" | tee -a "$LOG"; }
fail() { log "LAUNCH FAILED: $*"; tmux -L nm has-session -t "=snms$K" 2>/dev/null && tmux -L nm send-keys -t "=snms$K:" C-c; exit 1; }
refuse() { log "NOT LAUNCHED: $*"; exit 1; }  # before canary.sh: nothing of ours is running, so nothing is stopped
[[ $(tr ',' '\n' <<< "$BM" | grep -c https) == "$N" ]] || refuse "expected $N main URLs"
[[ $(tr ',' '\n' < $CH/fu_bases_s$K.txt | grep -c https) == 4 ]] || refuse "expected 4 follow-up URLs in fu_bases_s$K.txt"
log "waiting for $SIGNAL to be finished (deadline $(TZ=America/Los_Angeles date -d @$DEADLINE '+%T PT'))"
while :; do
  (( $(date +%s) < DEADLINE )) || { log "deadline reached: nothing started"; exit 1; }  # Codex r8: before the signal
  if python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("finished") else 1)' "$SIGNAL" 2>/dev/null; then
    log "$SIGNAL finished: launching"; break
  fi
  sleep 120
done
(( $(date +%s) < DEADLINE )) || { log "deadline reached: nothing started"; exit 1; }
# 1. the box
# Codex r8: every check fails explicitly (a subshell under `|| fail` would ignore set -e)
test -d $C && test -d /home/aswin/phase2/nm/approaches || refuse "tree or approaches missing"
! test -e $RUNP && ! test -e $FU || refuse "a run folder exists"
! tmux -L nm has-session -t "=snms$K" 2>/dev/null || refuse "session snms$K exists"
! tmux -L nmguards has-session -t "=emnms$K" 2>/dev/null || refuse "session emnms$K exists"
touch $C/results/PAUSE_SIMS && test -e $C/results/PAUSE_SIMS || refuse "PAUSE_SIMS not written"
docker image inspect control-tower-sim-scratch:latest --format '{{.Id}}' | grep -q '^sha256:ba1bed0c' || refuse "scratch image"
test "$(cat /proc/sys/fs/inotify/max_user_instances)" -ge 8192 || refuse "inotify"
log "box ready; PAUSE_SIMS in place"
(( $(date +%s) < END_MAIN - 4 * 3600 )) || refuse "too late: under 4 h left on the main servers"
# 2. gates and launcher (Codex r9: the deadline once more, right before anything starts)
(( $(date +%s) < DEADLINE )) || refuse "deadline reached during the box checks: nothing started"
SESSION=snms$K HARNESS=$C bash $C/canary.sh $RUN config/pilot_nm_s$K.yaml $BM $IMAGES >> "$LOG" 2>&1 || fail "canary.sh (gates or launcher)"
# 3. the watches
tmux -L nm new -d -s gbnms$K "cd $CH && PYTHONPATH=$TREE $PY gate1_nm.py $RUNP snms$K $((END_MAIN - 600)) main" || fail "gate1_nm"
tmux -L nmguards new -d -s emnms$K "$PY $CH/end_main.py $K $RUNP $FU config/pilot_nm_fu_s$K.yaml $END_FU iii" || fail "end_main"
sleep 10
tmux -L nm has-session -t "=snms$K" && tmux -L nm has-session -t "=gbnms$K" && tmux -L nmguards has-session -t "=emnms$K" ||
  fail "a watch or the launcher is not running"
log "LAUNCHED $RUN (session snms$K): main servers end $(TZ=America/Los_Angeles date -d @$END_MAIN '+%T PT'), gate deadline $(TZ=America/Los_Angeles date -d @$((END_MAIN - 600)) '+%T PT'); follow-up $FU via end_main emnms$K (servers end $(TZ=America/Los_Angeles date -d @$END_FU '+%T PT'))"
