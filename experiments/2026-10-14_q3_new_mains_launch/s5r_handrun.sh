#!/usr/bin/env bash
# ON SIMS-ASTRA, RULE.md Amendment 10: a HAND-RUN OF REVIEWED STEPS (s4d_handrun.sh's, 09's Append 35, d9 about 01:10 PT):
# end_main.py's steps 4-6 for the 6-cell sim-only RERUN "5r" of S5's follow-up (the cells whose simulated run failed with
# an infrastructure rate limit), references from the S5 main, in end_main's order, with end_main's commands. Only after
# S5's follow-up has ended and no other launcher runs on the box. Every check fails closed before anything starts
# (refuse: nothing touched); after the timers start, a failure is logged and the script stops (as end_main). One line per
# step to <S5 main>/HAND_RUN_s5r.txt.
#   bash s5r_handrun.sh <follow-up servers' earliest hard end, epoch s>
set -uo pipefail
END_FU=$1
C=/home/aswin/phase2/nm/attack_check_nm; CH=/home/aswin/phase2/nm/checks
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python; TREE=/home/aswin/phase2/sims-sept-strict-off/src
MAIN=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s5; FU=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s5r_fu
PREV=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s5_fu; Q=5r; CFG=config/pilot_nm_fu_s5r.yaml; LOG=$MAIN/HAND_RUN_s5r.txt
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') $*" | tee -a "$LOG"; }
refuse() { log "NOT STARTED: $*"; exit 1; }
stop_here() {  # Codex r21: once the launcher may be running, a missing guard PAUSES it (no new simulated start), never a kill
  # Codex r22: pause unless the launcher is POSITIVELY absent and fu_launch.sh was never invoked
  if [ "${LAUNCH_TRIED:-0}" = 1 ] || ! absent nm fnms$Q; then mkdir -p $FU && touch $FU/PAUSE_SIMS $C/results/PAUSE_SIMS && log "PAUSED: <run>/PAUSE_SIMS and harness PAUSE_SIMS written (no new simulated start)"; fi
  log "STOPPED (after the timers started): $*"; exit 1; }
live() { tmux -L "$1" has-session -t "=$2" 2>/dev/null; }
# Codex r19: a session must be POSITIVELY absent (tmux lists the socket without it, or reports no server there); any
# other tmux error is not absence
absent() { local o rc; o=$(tmux -L "$1" list-sessions -F '#{session_name}' 2>&1); rc=$?
  if (( rc == 0 )); then grep -qxF -- "$2" <<< "$o"; [ $? = 1 ]; else grep -qE '^(no server running on |error connecting to .*\(No such file or directory\))' <<< "$o"; fi; }
log "hand-run of end_main steps 4-6 for the 6-cell rate-limit rerun $Q (Append 35)"
# step 4: checks (end_main's, except the attempt check d9 overrode)
python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("finished") else 1)' $MAIN/launch.json || refuse "S4 main record not finished"
absent nm snms5 || refuse "the S5 main session is not positively absent"
python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("finished") else 1)' $PREV/launch.json || refuse "S5's follow-up record not finished"
absent nm fnms5 || refuse "S5's follow-up session is not positively absent"
pgrep -f 'run_pilot[.]py' >/dev/null; [ $? = 1 ] || refuse "a run_pilot.py runs on this box (one launcher per settings folder)"
for m in GATE1_STOP.txt IDENTICAL.txt PAIR_CONFIG.txt PAIR_CHECK_FAILED.txt; do ! test -e $MAIN/$m || refuse "stop marker $m"; done
! ls $MAIN/PAUSED_*.txt >/dev/null 2>&1 || refuse "a PAUSED marker"
! test -e $MAIN.STOP_REQUEST || refuse "a STOP_REQUEST"
docker image inspect control-tower-sim-scratch:latest --format '{{.Id}}' | grep -q '^sha256:ba1bed0c' || refuse "scratch image"
BASES=$(cat $CH/fu_bases_s5r.txt); [[ $(tr ',' '\n' <<< "$BASES" | grep -c https) == 6 ]] || refuse "expected 6 follow-up URLs in fu_bases_s5r.txt"
! test -e $FU || refuse "$FU exists"
for s in nm:fnms$Q nmguards:rpnms$Q nmguards:gsnms$Q nmguards:scnms$Q nmguards:gpnms$Q nmgates:gatesfnms$Q; do
  absent ${s%%:*} ${s#*:} || refuse "session ${s#*:} is not positively absent"; done
LAST_START=$((END_FU - 7200 - 3600 - 1200)); GRACE=$((END_FU - 600)); NOW=$(date +%s)
(( NOW < LAST_START )) || refuse "past the last start ($LAST_START)"
LIMIT=$(python3 -c 'import sys,re; print(re.search(r"^  worst_case_limit_usd: (\d+)$", open(sys.argv[1]).read(), re.M).group(1))' $C/$CFG) || refuse "no limit in $CFG"
# step 5: the time protections first, then sim_cap_stop, then the pause off and fu_launch.sh
tmux -L nmguards new -d -s rpnms$Q "sleep $((LAST_START - NOW)) && touch $C/results/PAUSE_SIMS"
tmux -L nmguards new -d -s gsnms$Q "sleep $((GRACE - NOW)) && tmux -L nm send-keys -t =fnms$Q: C-c"
live nmguards rpnms$Q && live nmguards gsnms$Q || stop_here "could not schedule the timers"
log "scheduled: re-pause at $LAST_START (rpnms$Q), graceful stop at $GRACE (gsnms$Q)"
tmux -L nmguards new -d -s scnms$Q "$PY $CH/sim_cap_stop.py $FU fnms$Q $((LIMIT + 100)) $((LIMIT + 160))"; sleep 3
live nmguards scnms$Q || stop_here "could not start sim_cap_stop.py"
log "sim_cap_stop.py running (scnms$Q, pause at \$$((LIMIT + 100)), hard stop at \$$((LIMIT + 160)))"
(( $(date +%s) < LAST_START )) || stop_here "past the last start"
rm -f $C/results/PAUSE_SIMS
LAUNCH_TRIED=1
out=$(bash $C/fu_launch.sh $Q $MAIN $FU $CFG $BASES 2>&1); rc=$?
log "fu_launch.sh exit $rc: $(tail -c 600 <<< "$out" | tr '\n' ' ')"
[[ $rc == 0 ]] && live nm fnms$Q || stop_here "the follow-up did not start"
# step 6: the usable pause gate and the live settings watch
tmux -L nmguards new -d -s gpnms$Q "cd $CH && $PY gate_pause_nm.py $FU fnms$Q" || stop_here "gpnms$Q not created: the follow-up runs WITHOUT its usable gate"
sleep 5; live nmguards gpnms$Q || stop_here "gpnms$Q not live: the follow-up runs WITHOUT its usable gate"
tmux -L nmgates new -d -s gatesfnms$Q "cd $C && PYTHONPATH=$TREE $PY gate_settings.py --reference-json config/sim_settings_reference_2026-10-14_125923.json --out results/gate_settings_new_${FU##*/}.json --new $FU --session fnms$Q > results/gate_settings_new_${FU##*/}.log 2>&1" || stop_here "gatesfnms$Q not created: no live settings watch"
sleep 3; live nmgates gatesfnms$Q || stop_here "gatesfnms$Q not live: no live settings watch"
log "FOLLOW-UP STARTED: session fnms$Q, gate gpnms$Q $(live nmguards gpnms$Q && echo up || echo 'NOT UP'), settings watch gatesfnms$Q $(live nmgates gatesfnms$Q && echo up || echo 'NOT UP')"
