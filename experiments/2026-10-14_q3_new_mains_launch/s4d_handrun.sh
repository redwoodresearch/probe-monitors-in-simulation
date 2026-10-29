#!/usr/bin/env bash
# ON BOX 4 (q3-scale-canary), RULE.md Amendment 8: a HAND-RUN OF REVIEWED STEPS (d9 about 22:45 PT): end_main.py's
# steps 4-6 for the 19-cell sim-only follow-up "4d" of the S4 main, in end_main's order, with end_main's commands. The
# S4 main's own handover was refused (ATTEMPT_STOP.txt: 30 < 50 cells and r2 22/30, mostly the dead server 4); d9 overrode
# that refusal for these 19 cells only. Every check fails closed before anything starts (refuse: nothing touched); after
# the timers start, a failure is logged and the script stops (as end_main). One line per step to <S4 main>/HAND_RUN_s4d.txt.
#   bash s4d_handrun.sh <follow-up servers' earliest hard end, epoch s>
set -uo pipefail
END_FU=$1
C=/home/aswin/phase2/nm/attack_check_nm; CH=/home/aswin/phase2/nm/checks
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python; TREE=/home/aswin/phase2/sims-sept-strict-off/src
MAIN=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s4; FU=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s4d_fu
Q=4d; CFG=config/pilot_nm_fu_s4d.yaml; LOG=$MAIN/HAND_RUN_s4d.txt
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') $*" | tee -a "$LOG"; }
refuse() { log "NOT STARTED: $*"; exit 1; }
stop_here() { log "STOPPED (after the timers started): $*"; exit 1; }
live() { tmux -L "$1" has-session -t "=$2" 2>/dev/null; }
log "hand-run of end_main steps 4-6 for the 19-cell follow-up $Q (d9's override of the S4 ATTEMPT_STOP for these cells)"
# step 4: checks (end_main's, except the attempt check d9 overrode)
python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("finished") else 1)' $MAIN/launch.json || refuse "S4 main record not finished"
! live nm snms4 || refuse "the S4 main session is live"
test -e $MAIN/ATTEMPT_STOP.txt || refuse "no ATTEMPT_STOP.txt: this hand-run is only for that refusal"
for m in GATE1_STOP.txt IDENTICAL.txt PAIR_CONFIG.txt PAIR_CHECK_FAILED.txt; do ! test -e $MAIN/$m || refuse "stop marker $m"; done
! ls $MAIN/PAUSED_*.txt >/dev/null 2>&1 || refuse "a PAUSED marker"
! test -e $MAIN.STOP_REQUEST || refuse "a STOP_REQUEST"
docker image inspect control-tower-sim-scratch:latest --format '{{.Id}}' | grep -q '^sha256:ba1bed0c' || refuse "scratch image"
BASES=$(cat $CH/fu_bases_s4.txt); [[ $(tr ',' '\n' <<< "$BASES" | grep -c https) == 4 ]] || refuse "expected 4 follow-up URLs"
! test -e $FU || refuse "$FU exists"
for s in nm:fnms$Q nmguards:rpnms$Q nmguards:gsnms$Q nmguards:scnms$Q nmguards:gpnms$Q nmgates:gatesfnms$Q; do
  ! live ${s%%:*} ${s#*:} || refuse "session ${s#*:} exists"; done
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
out=$(bash $C/fu_launch.sh $Q $MAIN $FU $CFG $BASES 2>&1); rc=$?
log "fu_launch.sh exit $rc: $(tail -c 600 <<< "$out" | tr '\n' ' ')"
[[ $rc == 0 ]] && live nm fnms$Q || stop_here "the follow-up did not start"
# step 6: the usable pause gate and the live settings watch
tmux -L nmguards new -d -s gpnms$Q "cd $CH && $PY gate_pause_nm.py $FU fnms$Q" || stop_here "gpnms$Q not created: the follow-up runs WITHOUT its usable gate"
sleep 5; live nmguards gpnms$Q || stop_here "gpnms$Q not live: the follow-up runs WITHOUT its usable gate"
tmux -L nmgates new -d -s gatesfnms$Q "cd $C && PYTHONPATH=$TREE $PY gate_settings.py --reference-json config/sim_settings_reference_2026-10-14_125923.json --out results/gate_settings_new_${FU##*/}.json --new $FU --session fnms$Q > results/gate_settings_new_${FU##*/}.log 2>&1" || stop_here "gatesfnms$Q not created: no live settings watch"
sleep 3; live nmgates gatesfnms$Q || stop_here "gatesfnms$Q not live: no live settings watch"
log "FOLLOW-UP STARTED: session fnms$Q, gate gpnms$Q $(live nmguards gpnms$Q && echo up || echo 'NOT UP'), settings watch gatesfnms$Q $(live nmgates gatesfnms$Q && echo up || echo 'NOT UP')"
