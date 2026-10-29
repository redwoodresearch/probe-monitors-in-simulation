#!/usr/bin/env bash
# rl_pause_watch.sh (d9 09:27 PT 10-15): if any 10-min bin of the newest committed sim-error census shows >= 2%
# RateLimitError (census_rl.py pause, bins labelled at or after SINCE's 10-min bin), simulated-honest pauses FIRST:
# PAUSE_SIMS (plus a one-line RL_PAUSE.txt) is written in each registered simh run dir only (simh_running.txt; start_q.sh
# registers a run dir BEFORE its on-box start; nothing harness-wide, no other launch touched). The launcher starts no new
# simulated run while <run>/PAUSE_SIMS exists (run_pilot.py:605); in-flight runs finish; never a Ctrl+C. The local flag
# RL_PAUSED blocks start_q.sh. After a trip the watcher keeps going until DEADLINE (epoch seconds): every minute it pauses
# any registered run dir that exists and is not yet confirmed (one registered while the trip happened included); an absent
# dir has nothing to pause yet and is retried. Exit 1 at the deadline if any entry was malformed or a pause failed.
# Reads only committed censuses (git fetch); talks to the boxes only to write the pause.
#   nohup bash rl_pause_watch.sh 'YYYY-mm-dd HH:MM' <deadline epoch> > log 2>&1 &
set -uo pipefail
SINCE=$1 DEADLINE=$2
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
E=$SP/wt_simh311/experiments/2026-10-15_q3_simhonest_311; H=$SP/simh_launch_X/h; REPO=/Users/aswinsurya/probe-distribution-shift-from-sims
REG=${SIMH_REG:-$SP/simh_drive/simh_running.txt}; FLAG=${SIMH_FLAG:-$SP/simh_drive/RL_PAUSED}  # env overrides: dry-run tests only
B=origin/aswin/q3-new-mains-launch; D=experiments/2026-10-14_q3_new_mains_launch/results/
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') $*"; }
DONE=$'\n'  # newline-delimited "<instance> <run>" lines whose PAUSE_SIMS is confirmed (bash 3.2: no associative arrays)
NDONE=0
BAD=0 PENDING=0     # in the latest pass: malformed entries or failed pauses; registered dirs not yet present
pause_pass() {  # $1 the tripping census file name (validated); pauses every registered, existing, unconfirmed run dir
  local inst run r line; BAD=0 PENDING=0
  while read -r inst run || [[ -n ${inst:-} ]]; do
    line="$inst $run"
    [[ $inst =~ ^i-[0-9a-f]+$ && $run =~ ^/home/aswin/phase2/runs/2026-10-15_simh_q[1-4]$ ]] || { log "MALFORMED registry line: $line"; BAD=$((BAD + 1)); continue; }
    [[ $DONE != *$'\n'"$line"$'\n'* ]] || continue
    r=$( (cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$inst perl -e 'alarm 120; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh \
      "if [[ ! -d $run ]]; then echo ABSENT; else touch $run/PAUSE_SIMS && { [[ -e $run/RL_PAUSE.txt ]] || echo \"\$(TZ=America/Los_Angeles date '+%F %T PT') rl_pause_watch.sh: >= 2% RateLimitError in a 10-min bin of $1\" > $run/RL_PAUSE.txt; } && [[ -e $run/PAUSE_SIMS ]] && echo PAUSED_OK; fi") < /dev/null 2> /dev/null | grep -E '^(ABSENT|PAUSED_OK)$')
    case $r in
      PAUSED_OK) DONE+="$line"$'\n'; NDONE=$((NDONE + 1)); log "PAUSE_SIMS written: $line" ;;
      ABSENT) PENDING=$((PENDING + 1)); log "not started yet (no run dir), will retry: $line" ;;
      *) BAD=$((BAD + 1)); log "PAUSE FAILED, will retry: $line" ;;
    esac
  done < "$REG"
}
log "armed: since $SINCE, deadline $(TZ=America/Los_Angeles date -r "$DEADLINE" '+%F %T PT'), pid $$"
last="" tripped=""
while (( $(date +%s) < DEADLINE )); do
  if [[ -n $tripped ]]; then pause_pass "$tripped"; sleep 60; continue; fi
  if git -C "$REPO" fetch -q origin aswin/q3-new-mains-launch; then
    CF=$(git -C "$REPO" ls-tree --name-only "$B" "$D" | grep 'sim_error_census_' | sort | tail -1)
    if [[ -n $CF && $CF != "$last" ]]; then
      last=$CF
      out=$(git -C "$REPO" show "$B:$CF" | python3 -I "$E/census_rl.py" pause --since "$SINCE" 2>&1); rc=$?
      log "$(basename "$CF"): $out (rc $rc)"
      if (( rc == 3 )); then
        tripped=$(basename "$CF"); [[ $tripped =~ ^sim_error_census_[0-9_-]+\.json$ ]] || tripped=unnamed_census  # Codex r8: no free text to a shell
        echo "$(TZ=America/Los_Angeles date '+%F %T PT') $tripped: $out" > "$FLAG"
        log "RL PAUSE TRIGGERED: start_q.sh blocked ($FLAG); pausing every registered simh run, until the deadline"
        continue
      elif (( rc != 0 )); then log "census parse failed; waiting for the next census"; fi
    fi
  else log "git fetch failed"; fi
  sleep 120
done
if [[ -n $tripped ]]; then
  pause_pass "$tripped"  # Codex r8b: a final pass, so a dir that appeared during the last sleep is paused or reported
  log "deadline: paused $NDONE run dirs:$(echo "$DONE" | tr '\n' ' '); malformed or failed in the last pass: $BAD; registered but absent: $PENDING"
  (( BAD == 0 && PENDING == 0 )) || exit 1
fi
log "deadline reached"; exit 0
