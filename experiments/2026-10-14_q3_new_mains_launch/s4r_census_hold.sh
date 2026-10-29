#!/usr/bin/env bash
# Laptop side (d9's condition 4 for the 4r swap, 2026-10-15 about 03:00 PT): 4r's follow-up starts only behind a committed
# census <= 30 min old with RateLimitError < 5% over the last 30 min and < 60 simulated runs running. end_main.py can't
# read the census, so the hold is a marker it already obeys: <main>/PAUSED_census_hold.txt. While any PAUSED_*.txt is in
# the main run, end_main's step 1 waits ("main PAUSED ...: waiting for a person", end_main.py:117-121: no calm pass, no
# Ctrl+C, no follow-up), and step 4 refuses if one appears later (:165). gate1_nm.py only writes PAUSED markers.
#   bash s4r_census_hold.sh place     put the marker (atomic rename) and wait for end_main to log the pause
#   bash s4r_census_hold.sh release   A. every 300 s until 4r's main has no real run open (or its session is gone);
#     B. census_commit.sh; the committed file must PASS checks/s4r_census_verdict.py (all 8 boxes, exact window, age
#        <= 5 min); then, before the release deadline (below), rename the marker to census_hold_released_<epoch>.txt (no longer PAUSED_*.txt);
#        a FAIL censuses again in 600 s;
#     C. until the follow-up's first simulated attempt, every 120 s: a new passing census whenever the last is 10 min
#        old; the first attempt must follow a passing census observed before it by <= 30 min with no failed census;
#        otherwise (or on a failed census, 3 unreadable states, 90 min) the marker goes back and the harness PAUSE_SIMS
#        is written (step 1 waits; step 4 refuses; the launcher starts no new simulated run). Each is an ALERT for d9;
#        nothing is released again.
#   Never releases after the deadline (LAST_START - 15 min: 09:05 PT with the swapped end_main) or on anything
#   unreadable: the hold stays (fail-closed).
#   One line per poll to nm_launch/s4r_census_hold.log.
set -uo pipefail
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
HERE=$(cd "$(dirname "$0")" && pwd); W=$SP/wt_nm; LOG=$SP/nm_launch/s4r_census_hold.log
MAIN=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s4r; H=$MAIN/PAUSED_census_hold.txt; EL=/home/aswin/phase2/nm/checks/end_main_s4r.log
PS=/home/aswin/phase2/nm/attack_check_nm/results/PAUSE_SIMS
# d9: the release deadline is LAST_START - 15 min of the end_main actually running: 09:05 PT after the 03:38 swap
# (servers_end 12:40), 05:25 PT had the swap failed (servers_end 09:00). Read from the box at every check; exactly one
# end_main 4r must run, else nothing is released.
release_by() { local e; e=$(field "$1" em_servers_end); [[ $e =~ ^\[([0-9]+)\]$ ]] && echo $(( BASH_REMATCH[1] - 12000 - 900 )) || echo 0; }
export DEVBOX=aswin@i-07bce857a818b6f69
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') [4r hold] $*" | tee -a "$LOG"; }
box() { perl -e 'alarm shift; exec @ARGV' "$1" bash "$SP/nm_launch/devbox.sh" "$2" < /dev/null; }
state() { box 120 "python3 /home/aswin/phase2/nm/checks/s4r_hold_state.py"; }
field() { python3 -c 'import json,sys; v=json.loads(sys.argv[1])[sys.argv[2]]; print(v if isinstance(v, str) else json.dumps(v))' "$1" "$2"; }
MARK="printf '%s\n' 'census hold (d9 condition 4, 2026-10-15): end_main 4r waits until a committed census <= 30 min old shows RateLimitError < 5% (last 30 min) and < 60 simulated runs running; released only by s4r_census_hold.sh release (renamed census_hold_released_<epoch>.txt)' > $H.tmp && mv $H.tmp $H && test -e $H"
census() {  # sets V (verdict line) and C (census epoch); returns 0 only on a committed, pushed PASS no older than $1 min
  local out f sha rel cpid k; V="FAIL census did not run"; C=0
  # Codex r31 item 1: the whole census (eight boxes, commit, push) is bounded at 600 s; past that it is a FAIL (phase C
  # then protects) and the census process is sent SIGTERM (its own calls carry alarms; a late file is ignored)
  bash "$SP/nm_launch/census_commit.sh" > "$SP/nm_launch/s4r_census_out.txt" 2>&1 & cpid=$!
  for k in $(seq 1 120); do kill -0 $cpid 2>/dev/null || break; sleep 5; done
  if kill -0 $cpid 2>/dev/null; then pkill -TERM -P $cpid; kill -TERM $cpid; V="FAIL census did not finish in 600 s"; return 1; fi
  out=$(cat "$SP/nm_launch/s4r_census_out.txt"); f=$(grep -oE 'results/sim_error_census_2026-10-15_[0-9]{4}[.]json' <<< "$out" | head -1)
  sha=$(grep -oE 'committed @ [0-9a-f]{9} \(remote [0-9a-f]{9}\)' <<< "$out"); rel=experiments/2026-10-14_q3_new_mains_launch/$f
  [[ -n $f && $sha =~ \(remote\ ([0-9a-f]{9})\) && $sha == "committed @ ${BASH_REMATCH[1]} "* && -n $(git -C "$W" log -1 --format=%H -- "$rel") &&
     -z $(git -C "$W" status --porcelain -- "$rel") ]] || { V="FAIL census not committed and pushed ($f; $sha)"; return 1; }
  local r; r=$(python3 "$HERE/checks/s4r_census_verdict.py" "$W/$rel" "$1") || { V="FAIL verdict script error ($f)"; return 1; }
  V="$(head -1 <<< "$r") [$f $sha]"; C=$(tail -1 <<< "$r")
  [[ $V == PASS* ]]
}
perl -e 'alarm 120; exec @ARGV' bash "$SP/nm_launch/devbox.sh" rsync -a "$HERE/checks/s4r_hold_state.py" box:phase2/nm/checks/ < /dev/null ||
  { log "could not copy s4r_hold_state.py"; exit 1; }
protect() {  # the marker (blocks end_main before step 4) and PAUSE_SIMS (the launcher starts no new simulated run); each checked (r30 item 2)
  local m=written p=written
  box 60 "$MARK" > /dev/null 2>&1 || m="COULD NOT BE WRITTEN"
  box 60 "touch $PS && test -e $PS" > /dev/null 2>&1 || p="COULD NOT BE WRITTEN"
  log "ALERT: $1; marker $m, harness PAUSE_SIMS $p; d9 decides$([[ $m$p == writtenwritten ]] || echo ': ACT BY HAND')"
}
case "${1:-}" in
place)
  s=$(state) || { log "PLACE refused: state unreadable"; exit 1; }
  [[ $(field "$s" hold) == false && $(field "$s" fu_live) == false ]] || { log "PLACE refused: $s"; exit 1; }
  box 60 "$MARK" || { log "PLACE failed"; exit 1; }
  log "hold placed: $H"
  for i in 1 2 3; do sleep 130
    box 60 "tail -1 $EL" | grep -q "main PAUSED (\[.*'PAUSED_census_hold.txt'.*\]): waiting for a person" && { log "end_main logs the pause"; exit 0; }
  done
  log "WARNING: end_main has not logged the pause after 390 s: $(box 60 "tail -2 $EL" | tr '\n' '|')"; exit 1;;
release)
  log "release watcher armed"
  HARD_END=$(TZ=America/Los_Angeles date -j -f '%Y-%m-%d %H:%M:%S' '2026-10-15 09:20:00' +%s)  # the latest LAST_START of any end_main 4r
  while :; do  # A and B
    if ! s=$(state); then
      (( $(date +%s) < HARD_END )) || { protect "09:20 PT and the state is unreadable"; exit 1; }  # Codex r32
      log "state unreadable; holding"; sleep 300; continue
    fi
    # Codex r31 item 2: a missing marker this watcher did not release is a possible release: protect, whatever the time
    [[ $(field "$s" hold) == true ]] || { protect "the hold marker is missing and this watcher did not release it (fu_live $(field "$s" fu_live))"; exit 1; }
    (( $(date +%s) < HARD_END )) || { log "09:20 PT: watcher ends without a release; the marker is present"; exit 1; }
    RB=$(release_by "$s")
    # past the release deadline: never release, but keep watching the marker until 09:20 (a missing one is protected above)
    (( $(date +%s) < RB )) || { log "past the release deadline ($RB; end_main servers_end $(field "$s" em_servers_end)): no release; marker present"; sleep 300; continue; }
    if [[ $(field "$s" open_real) != 0 && $(field "$s" main_live) == true ]]; then log "main running: $s"; sleep 300; continue; fi
    census 5 || { log "$V; holding, next census in 600 s"; sleep 600; continue; }
    s=$(state) || { log "state unreadable after the census; holding"; sleep 300; continue; }
    [[ $(field "$s" hold) == true && $(field "$s" fu_live) == false &&
       $(field "$s" launchers_in_harness) -le $([[ $(field "$s" main_live) == true ]] && echo 1 || echo 0) ]] ||
      { log "not released (state changed): $s"; sleep 300; continue; }
    RB=$(release_by "$s"); LIMIT=$(( RB < C + 300 ? RB : C + 300 ))
    # Codex r29 items 2 and 6: on the box, at the rename: before the release deadline AND <= 5 min after the census's
    # oldest observation; otherwise nothing is renamed
    if ! box 60 "[ \$(date +%s) -lt $LIMIT ] && mv $H $MAIN/census_hold_released_\$(date +%s).txt && test ! -e $H"; then
      s2=$(state) && [[ $(field "$s2" hold) == true ]] && { log "not released (deadline or census age at the rename; limit $LIMIT): $V"; sleep 60; continue; }
      protect "the rename's result is unknown (marker absent or state unreadable): treated as a possible release"; exit 1  # Codex r30 item 1
    fi
    log "RELEASED on $V; state before: $s"
    break
  done
  # C (Codex r26 item 1, r29 items 3-5): until the follow-up's first simulated attempt, keep a passing census <= 10 min
  # old. HIST = observation epochs of passing censuses. The first attempt is accepted only against a passing census
  # observed BEFORE it and at most 30 min earlier, with no failed census since; anything else (a failed census, 3
  # unreadable states in a row, the 90-minute limit) puts the marker back and writes the harness PAUSE_SIMS, verified.
  HIST="$C"; alarm=0; bad=0; CEND=$(( $(date +%s) + 5400 ))  # Codex r30 item 4: an absolute 90 min (+ at most one census and two box calls)
  while :; do
    sleep 120
    (( $(date +%s) < CEND )) || { protect "no simulated attempt 90 min after the release"; exit 1; }
    if ! s=$(state); then bad=$((bad + 1)); (( bad >= 3 )) && { protect "state unreadable 3 times in the handover"; exit 1; }; log "state unreadable (handover, $bad)"; continue; fi
    bad=0
    case "$(field "$s" em_last)" in *"NO FOLLOW-UP"*) protect "end_main: $(field "$s" em_last)"; exit 1;; esac  # r30 item 3
    st=$(field "$s" fu_first_sim_start)
    if [[ $st != null ]]; then
      S=$(python3 -c 'import sys; from datetime import datetime; print(int(datetime.fromisoformat(sys.argv[1]).timestamp()))' "$st")
      best=0; for h in $HIST; do (( h <= S && h > best )) && best=$h; done
      if (( alarm || best == 0 || S - best > 1800 )); then protect "first simulated attempt $st; last passing census observed before it: $best; failed census: $alarm"; exit 1; fi
      log "follow-up's first simulated attempt $st, $(( (S - best) / 60 )) min after a passing census observed before it: OK; $(field "$s" em_last)"; exit 0
    fi
    if (( alarm )); then
      [[ $(field "$s" harness_pause_sims) == true ]] || protect "end_main removed PAUSE_SIMS after a failed census"
      log "held after a failed census (d9 decides)"; continue
    fi
    last=${HIST##* }
    if (( $(date +%s) - last > 600 )); then
      if census 5; then HIST="$HIST $C"; log "handover census: $V"; else alarm=1; protect "handover census $V"; fi
    fi
    log "handover: $(field "$s" em_last)"
  done;;
*) echo "usage: $0 place|release"; exit 2;;
esac
