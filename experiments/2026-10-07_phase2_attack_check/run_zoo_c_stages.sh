#!/usr/bin/env bash
# Zoo C stages 2-6 back to back on the devbox (Aswin, relayed by the orchestrator, 23:16:03 PT 2026-10-11: "Continue as
# approved"). Each stage is launch_zoo_c.sh <name> <stage> [config], one at a time; the next starts only after the last
# one's launcher and watcher have exited and every one of its runs is done. The chain stops, and nothing more
# launches, if a stage leaves POLICY_HALT, ends by a signal, has any run not done (its spend cap binding), if
# its launcher exits without its completion line, or if the zoo C API spend so far plus the next stage's stop would
# pass $650 (the $718 approval less devbox and extraction). STAGES (default "2 3 4 5 6") picks the stages; WAIT_FOR
# names a stage already running (as <run name>:<stage>) to see through first. Log times are Pacific. Run from a
# frozen copy (bash reads scripts live):
#   cp run_zoo_c_stages.sh ~/phase2/run_zoo_c_stages.frozen.sh; tmux -L driver new -d -s zcd "bash ~/phase2/run_zoo_c_stages.frozen.sh"
set -euo pipefail
C=/home/aswin/phase2/attack_check_zoo_c
RUNS=/home/aswin/phase2/runs
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
LOG=$RUNS/zooc_stages_driver.log
log() { echo "$(TZ=America/Los_Angeles date -Is) $*" >> $LOG; }
spent() {  # API so far, conservatively: every zoo C run folder's last committed reading (spend plus any unresolved worst case)
  $PY -c "import json, glob; print(round(sum(json.load(open(f))['committed_usd'] for f in glob.glob('$RUNS/*zooc*/policy_spend.json')), 2))"
}
see_through() {  # wait for a stage's launcher and watcher to end, then check it fully completed
  local NAME=$1 STAGE=$2
  until grep -q '^-> ' $RUNS/$NAME.launcher.log 2>/dev/null; do
    if ! tmux -L phase2 has-session -t zc 2>/dev/null; then
      sleep 30
      grep -q '^-> ' $RUNS/$NAME.launcher.log 2>/dev/null || { log "stop: stage $STAGE's launcher exited without its completion line"; exit 1; }
    fi
    sleep 60
  done
  while tmux -L canary has-session -t zcw 2>/dev/null || tmux -L phase2 has-session -t zc 2>/dev/null; do sleep 10; done
  # POLICY_STOP is written whenever one start is refused, even passingly (many runs reserving their worst case at
  # once); a stop that binds leaves runs not done, which the check below catches. POLICY_HALT (a dead or mismatched
  # watcher) always stops the chain.
  if [[ -e $RUNS/$NAME/POLICY_HALT ]]; then
    log "stop: stage $STAGE's watcher halted it (POLICY_HALT); see $RUNS/$NAME"; exit 1
  fi
  [[ -e $RUNS/$NAME/POLICY_STOP ]] && log "stage $STAGE: POLICY_STOP was written (a start was refused at least once)"
  if ! $PY -c "import json, sys; d = json.load(open(sys.argv[1])); sys.exit(1 if d.get('stopped_by_signal') or any(r['state'] != 'done' for r in d['runs']) else 0)" $RUNS/$NAME/launch.json; then
    log "stop: stage $STAGE ended by a signal or with runs not done; see $RUNS/$NAME"; exit 1
  fi
  if ! $PY -c "import json, sys; d = json.load(open(sys.argv[1])); sys.exit(0 if d['actual_usd'] < d['stop_usd'] else 1)" $RUNS/$NAME/policy_spend.json; then
    log "stop: stage $STAGE's final spend reached its stop (runs can finish past it between polls); see $RUNS/$NAME"; exit 1
  fi
  log "stage $STAGE done: $(tail -1 $RUNS/$NAME/policy_spend.jsonl | cut -c1-160)"
}
if [[ -n ${WAIT_FOR:-} ]]; then
  log "driver restarted: seeing ${WAIT_FOR%:*} (stage ${WAIT_FOR#*:}) through first"
  see_through ${WAIT_FOR%:*} ${WAIT_FOR#*:}
fi
for STAGE in ${STAGES:-2 3 4 5 6}; do
  if [[ -n ${WAIT_FOR:-} ]] && (( STAGE <= ${WAIT_FOR#*:} )); then log "skip stage $STAGE: at or before the stage seen through"; continue; fi
  CFG=config/pilot_zoo_c_oct11c.yaml; [[ $STAGE == 6 ]] && CFG=config/pilot_zoo_c_oct11c_stage6.yaml
  STOP=$($PY -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['policy_stop_usd'])" $C/$CFG)
  SO_FAR=$(spent)
  if ! $PY -c "import sys; sys.exit(0 if float(sys.argv[1]) + float(sys.argv[2]) <= 650 else 1)" "$SO_FAR" "$STOP"; then
    log "stop: API so far \$$SO_FAR plus stage $STAGE's \$$STOP would pass \$650"; exit 1
  fi
  NAME=$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S)_zooc_stage$STAGE
  log "stage $STAGE: launching $NAME ($CFG, API stop \$$STOP; API so far \$$SO_FAR)"
  bash $C/launch_zoo_c.sh $NAME $STAGE $CFG >> $LOG 2>&1 || { log "stop: stage $STAGE did not start"; exit 1; }
  see_through $NAME $STAGE
done
log "all stages done"
