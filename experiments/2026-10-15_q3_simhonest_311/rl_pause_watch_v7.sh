#!/usr/bin/env bash
# rl_pause_watch_v7.sh (new file, d9 ~17:55: also accepts 2026-10-15_simh_t196fill, the A25 fill-in; otherwise v6).
# rl_pause_watch_v6.sh (new file, d9 ~17:40: also accepts the S_paper493 run dirs 2026-10-15_simh_sp493_t196 and the design
# fork's sims-only 2026-10-15_sp493_sim_L1/L2/L3; otherwise v5).
# rl_pause_watch_v5.sh (new file, d9 ~15:45: also accepts <...>_rlr2, for q1_rlr2; otherwise v4).
# rl_pause_watch_v4.sh (new file: also accepts the replacement run dirs <...>_rlr; otherwise v3).
# rl_pause_watch_v3.sh (new file, nm360: also accepts the nm sim-honest run dirs 2026-10-15_simh_nm_<g>; otherwise v2).
# rl_pause_watch_v2.sh (new file, d9 ~10:20: adds df/q2's Modal replay as a pause target; otherwise rl_pause_watch.sh,
# Codex r8c AGREE). rl_pause_watch.sh (d9 09:27 PT 10-15): if any 10-min bin of the newest committed sim-error census shows >= 2%
# RateLimitError (census_rl.py pause, bins labelled at or after SINCE's 10-min bin), simulated-honest pauses FIRST:
# PAUSE_SIMS (plus a one-line RL_PAUSE.txt) is written in each registered simh run dir only (simh_running.txt; start_q.sh
# registers a run dir BEFORE its on-box start; nothing harness-wide, no other launch touched). The launcher starts no new
# simulated run while <run>/PAUSE_SIMS exists (run_pilot.py:605); in-flight runs finish; never a Ctrl+C. The local flag
# RL_PAUSED blocks start_q.sh. After a trip the watcher keeps going until DEADLINE (epoch seconds): every minute it pauses
# any registered run dir that exists and is not yet confirmed (one registered while the trip happened included); an absent
# dir has nothing to pause yet and is retried. Exit 1 at the deadline if any entry was malformed or a pause failed.
# Reads only committed censuses (git fetch); talks to the boxes only to write the pause.
# Registry lines: "<instance> <simh run dir>", or "modal <fixed name>" (replay_t196_v1 -> volume VOL /replay/PAUSE, confirmed
# by /replay/status/live.json "paused": true; replay_watch_test -> /replay/_watch_test/, for tests only). Unpausing is
# never automatic.
#   nohup bash rl_pause_watch_v2.sh 'YYYY-mm-dd HH:MM' <deadline epoch> > log 2>&1 &
set -uo pipefail
SINCE=$1 DEADLINE=$2
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
E=$SP/wt_simh311/experiments/2026-10-15_q3_simhonest_311; H=$SP/simh_launch_X/h; REPO=/Users/aswinsurya/probe-distribution-shift-from-sims
REG=${SIMH_REG:-$SP/simh_drive/simh_running.txt}; FLAG=${SIMH_FLAG:-$SP/simh_drive/RL_PAUSED}  # env overrides: dry-run tests only
B=origin/aswin/q3-new-mains-launch; D=experiments/2026-10-14_q3_new_mains_launch/results/
MODAL=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal; VOL=probe-shift_aswin-surya_sub-q2-onpolicysim_data_kill-2026-11-14
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') $*"; }
DONE=$'\n'  # newline-delimited "<instance> <run>" lines whose PAUSE_SIMS is confirmed (bash 3.2: no associative arrays)
NDONE=0
BAD=0 PENDING=0     # in the latest pass: malformed entries or failed pauses; registered dirs not yet present
pause_modal() {  # $1 fixed target name, $2 validated census name -> PAUSED_OK | PENDING | FAILED (d9 ~10:20: df/q2's replay)
  local P S T rc
  case $1 in  # fixed names only; nothing free-form reaches a command line
    replay_t196_v1) P=/replay/PAUSE; S=/replay/status/live.json ;;
    replay_watch_test) P=/replay/_watch_test/PAUSE; S=/replay/_watch_test/status/live.json ;;
    *) echo FAILED; return ;;
  esac
  T=$(mktemp -d "$SP/simh_drive/w2_XXXXXX") || { echo FAILED; return; }
  echo "$(TZ=America/Los_Angeles date '+%F %T PT') rl_pause_watch_v7.sh: >= 2% RateLimitError in a 10-min bin of $2" > "$T/PAUSE"
  perl -e 'alarm 60; exec @ARGV' "$MODAL" volume put --force "$VOL" "$T/PAUSE" "$P" < /dev/null > /dev/null 2>&1; rc=$?
  if (( rc != 0 )); then rm -rf "$T"; echo FAILED; return; fi
  # confirmed only by the replay's own status: live.json with "paused": true (absent or false = not yet; re-put next pass)
  perl -e 'alarm 60; exec @ARGV' "$MODAL" volume get --force "$VOL" "$S" "$T/live.json" < /dev/null > /dev/null 2>&1
  if [[ -f $T/live.json ]] && python3 -I -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get("paused") is True else 1)' "$T/live.json" 2> /dev/null; then
    echo PAUSED_OK; else echo PENDING; fi
  rm -rf "$T"
}
pause_pass() {  # $1 the tripping census file name (validated); pauses every registered, existing, unconfirmed run dir
  local inst run r line; BAD=0 PENDING=0
  while read -r inst run || [[ -n ${inst:-} ]]; do
    line="$inst $run"
    if [[ $inst == modal ]]; then
      [[ $run == replay_t196_v1 || $run == replay_watch_test ]] || { log "MALFORMED registry line: $line"; BAD=$((BAD + 1)); continue; }
      [[ $DONE != *$'\n'"$line"$'\n'* ]] || continue
      case $(pause_modal "$run" "$1") in
        PAUSED_OK) DONE+="$line"$'\n'; NDONE=$((NDONE + 1)); log "replay PAUSE put and confirmed (live.json paused: true): $line" ;;
        PENDING) PENDING=$((PENDING + 1)); log "replay PAUSE put, not yet confirmed by live.json, will re-put: $line" ;;
        *) BAD=$((BAD + 1)); log "replay PAUSE put FAILED, will retry: $line" ;;
      esac
      continue
    fi
    [[ $inst =~ ^i-[0-9a-f]+$ && $run =~ ^/home/aswin/phase2/runs/(2026-10-15_simh_((q[1-4]|nm_(s1|s2|s3|s4d|s4r|s5))(_rlr|_rlr2)?|sp493_t196|t196fill)|2026-10-15_sp493_sim_L[1-3])$ ]] || { log "MALFORMED registry line: $line"; BAD=$((BAD + 1)); continue; }
    [[ $DONE != *$'\n'"$line"$'\n'* ]] || continue
    r=$( (cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$inst perl -e 'alarm 120; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh \
      "if [[ ! -d $run ]]; then echo ABSENT; else touch $run/PAUSE_SIMS && { [[ -e $run/RL_PAUSE.txt ]] || echo \"\$(TZ=America/Los_Angeles date '+%F %T PT') rl_pause_watch_v7.sh: >= 2% RateLimitError in a 10-min bin of $1\" > $run/RL_PAUSE.txt; } && [[ -e $run/PAUSE_SIMS ]] && echo PAUSED_OK; fi") < /dev/null 2> /dev/null | grep -E '^(ABSENT|PAUSED_OK)$')
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
