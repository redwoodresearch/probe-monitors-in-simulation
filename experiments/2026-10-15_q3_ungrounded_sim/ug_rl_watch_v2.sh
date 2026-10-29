#!/usr/bin/env bash
# ug_rl_watch_v2.sh (RULE.md; v2: reads censuses with ug_census_check_v2.py, which counts a box the census records as powered off as complete, d9 ~10:35; d9 10-15 ~06:10-06:35, 0d's choice (d): ug starts now and YIELDS to 4r's follow-up): FROM THE
# LAPTOP, the ug pause/resume watch. It touches nothing on the box but ug's own PAUSE_SIMS
# (/home/aswin/phase2/runs/2026-10-15_ug/PAUSE_SIMS: the launcher starts no new simulated run while it exists, in-flight
# ones finish; never a kill). Every 300 s it fetches aswin/q3-new-mains-launch and reads the newest committed census
# (ug_census_check.py) and, in phases run and yield, the 4r hold status (ug_yield_check.py: the last committed line of
# s4r_hold_status_2026-10-15.jsonl or the publisher's laptop copy, the newer). Every 3rd cycle one bounded box read: is
# the launch live, and are PAUSE_SIMS / SIM_CAP_PAUSE.txt / GATE_PAUSE.txt present. Each box call runs in its own process
# group (ug_wait.sh's bounded). Each pause or resume is logged with its PT time; the watch then exits and wakes the fork,
# which tells d9 and re-arms the next phase.
#   run   (from the start): PAUSE then exit 10 if any launch has a RateLimitError run ended since the start (d9: any
#         launch's bin >= 5%; that pause waits for d9); PAUSE then exit 20 at the first of open_real_4r_main <= 40,
#         main_real_not_started == 0, released, fu_live (d9's ruling, ~45 min before 4r's main ends); PAUSE then exit 24
#         if the 4r status is missing, unreadable or > 15 min old (d9: fail closed). Exit 22 if a PAUSE_SIMS someone else
#         wrote appears (left as is).
#   yield (after a 4r pause): stays paused; NOTIFY-ONLY resume (d9 ~06:45): exit 21 "resume condition met" when all of
#         4r's follow-up sims have started, or >= 75% have and the newest census is <= 30 min old, complete, with < 60
#         running and < 5% RateLimitError (ug_census_check.py --prestart), and the launch is live. The fork then
#         re-checks and removes ug's PAUSE_SIMS itself by a plain box command. Exit 23 instead if SIM_CAP_PAUSE.txt or
#         GATE_PAUSE.txt exists (a cap or gate pause shares the file). A RateLimitError run since the start: exit 10.
#   post  (after the resume): the RateLimitError rule only (PAUSE then exit 10).
# Fail closed: in run and post, every exit but "launch ended" (0) and 22 writes the pause first (3 tries; exit 15 if it
# cannot: exit 15 means NOT PAUSED, and the fork hands the pause at once to d9 and the design fork, d9's fallback writer of
# ug's PAUSE_SIMS). Setup runs after the trap, so a setup failure in run/post also pauses first. A PAUSE_SIMS already
# present when the watch would pause is someone else's: left as is, exit 22 (the log names the trigger). An incomplete
# census (a box missing, invalid counts) cannot clear: tolerated only within 30 min of the last CLEAR census this watch
# saw (none yet: not tolerated), else exit 11. The watch never removes anything.
# Exits: 0 launch ended (launch.json finished, ugsim gone), 1 error, 10 RateLimitError, 11 census > 45 min old or none clear,
# 12 ug missing from a census taken >= 20 min after the start, 13 ugsim gone but not finished, 14 deadline, 15 pause write
# failed (NOT paused: hand off), 20 paused for 4r, 21 resume condition met, 22 someone else's PAUSE_SIMS, 23 a cap/gate marker blocks the resume,
# 24 paused (4r status unknown or stale).
#   DEVBOX_SH=<devbox.sh> bash ug_rl_watch.sh <run|yield|post> <start, ISO with offset> <deadline HH:MM, PT today>
set -uo pipefail
D=${DEVBOX_SH:?set DEVBOX_SH}; BOX=aswin@i-0f2299f29dd296259; R=/home/aswin/phase2/runs/2026-10-15_ug
PHASE=${1:?phase}; START=${2:?start}; DL=${3:?deadline}
[[ $PHASE == run || $PHASE == yield || $PHASE == post ]] || { echo "phase: run, yield or post"; exit 1; }
BR=origin/aswin/q3-new-mains-launch; P=experiments/2026-10-14_q3_new_mains_launch/results
LATEST=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad/nm_launch/s4r_hold_status_latest.json
OUTF="${TMPDIR:-/tmp}/ug_rl_watch_$$.out"  # fixed before the trap: the pause path needs it
bounded() {  # bounded <seconds> <command...>: the command's exit status (128+N if killed by signal N), or 124 on
             # timeout; output in $OUTF
  local t=$1; shift
  perl -e '
    my $t = shift @ARGV;
    my $pid = fork(); die "fork failed\n" unless defined $pid;
    if ($pid == 0) { setpgrp(0, 0); exec @ARGV or exit 127; }
    local $SIG{ALRM} = sub { kill "KILL", -$pid; waitpid($pid, 0); exit 124; };
    alarm $t; waitpid($pid, 0); my $st = $?; alarm 0; kill "KILL", -$pid; exit(($st & 127) ? 128 + ($st & 127) : ($st >> 8));
  ' "$t" "$@" > "$OUTF" 2>&1
}
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') [$PHASE] $*"; }
box() {  # box <script>: run a bash script on xbox3 (base64, so no quoting), bounded 120 s; output in $OUTF
  DEVBOX=$BOX bounded 120 bash "$D" "echo $(printf '%s' "$1" | base64 | tr -d '\n') | base64 -d | bash"
}
STATE_SH="R=$R"'
[[ -d $R ]] || { echo NORUN; exit 0; }
if tmux -L phase2 has-session -t =ugsim 2>/dev/null; then s=LIVE
elif python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get(\"finished\") else 1)" "$R/launch.json" 2>/dev/null; then s=FINISHED
else s=GONE; fi
f() { [[ -e $R/$1 ]] && echo 1 || echo 0; }
echo "$s pause=$(f PAUSE_SIMS) cap=$(f SIM_CAP_PAUSE.txt) gate=$(f GATE_PAUSE.txt)"'
PAUSE_SH="R=$R"'
[[ -d $R ]] || { echo NORUN; exit 3; }
if [[ -e $R/PAUSE_SIMS ]]; then echo "PAUSED existing"; exit 0; fi
touch "$R/PAUSE_SIMS" && [[ -e $R/PAUSE_SIMS ]] && echo "PAUSED written"'
pause_exit() {  # pause_exit <code> <why>: write ug's PAUSE_SIMS (3 tries), then exit <code>; an existing one is someone
                # else's: exit 22; 15 = NOT paused (hand off)
  local i rc
  for i in 1 2 3; do
    box "$PAUSE_SH"; rc=$?
    # ownership only from the box script's own result line, matched exactly (Codex r14), wherever transport noise puts it:
    # it prints "PAUSED written" or "PAUSED existing" (each only once the file is confirmed), never both. A "written"
    # in an attempt whose transport failed still counts, so a retry cannot mistake this watch's pause for someone else's.
    if grep -qx "PAUSED written" "$OUTF"; then log "PAUSED (written): $R/PAUSE_SIMS; $2; exit $1 (transport rc $rc)"; exit "$1"; fi
    if grep -qx "PAUSED existing" "$OUTF"; then
      log "PAUSE_SIMS already present (not written by this watch; left as is); trigger was: $2; exit 22"; exit 22
    fi
    log "pause write failed (rc $rc): $(tail -1 "$OUTF")"; sleep 20
  done
  log "PAUSE NOT WRITTEN after 3 tries ($2): ug is NOT paused; hand off to d9 and the design fork now; exit 15"; exit 15
}
fail() {  # fail <code> <why>: run and post pause first (fail closed); yield is already paused
  if [[ $PHASE == yield ]]; then log "$2; exit $1 (still paused)"; exit "$1"; fi
  pause_exit "$1" "$2"
}
trap 'fail 1 "interrupted"' INT TERM
E=$(cd "$(dirname "$0")" && pwd) || fail 1 "cannot resolve the script dir"
WT=$(git -C "$E" rev-parse --show-toplevel) || fail 1 "not in a git checkout"
CEN=$(mktemp -t ug_cen) && ST=$(mktemp -t ug_st) && CK=$(mktemp -t ug_ck) || fail 1 "mktemp failed"
deadline=$(TZ=America/Los_Angeles date -j -f '%Y-%m-%d %H:%M:%S' "$(TZ=America/Los_Angeles date +%Y-%m-%d) $DL:00" +%s) \
  || fail 1 "cannot parse deadline $DL"
log "watch: start $START, deadline $DL PT, run $R"
prev=""; cycle=0; last_clear=0  # no CLEAR census seen yet
while :; do
  (( $(date +%s) < deadline )) || fail 14 "deadline $DL PT"
  bounded 120 git -C "$WT" fetch -q origin aswin/q3-new-mains-launch || log "fetch failed (rc $?): $(tail -1 "$OUTF")"
  newest=$(git -C "$WT" ls-tree --full-tree --name-only "$BR" "$P/" | grep -E '/sim_error_census_2026-10-15_[0-9]{4}\.json$' | sort | tail -1)
  [[ -n $newest ]] && git -C "$WT" show "$BR:$newest" > "$CEN" || fail 1 "no census readable on $BR"
  python3 "$E/ug_census_check_v2.py" "$CEN" --since "$START" > "$CK" 2>&1; rc=$?
  line="${newest##*/}: $(tail -1 "$CK")"
  case $rc in
    0) last_clear=$(date +%s) ;;
    13) if [[ $PHASE != yield ]] && (( last_clear == 0 || $(date +%s) - last_clear > 1800 )); then
          fail 11 "incomplete census and no CLEAR one in the last 30 min ($line)"; fi ;;
    10) cat "$CK"; [[ $PHASE == yield ]] && { log "RateLimitError since the start; exit 10 (still paused)"; exit 10; }
        pause_exit 10 "RateLimitError since the start ($line)" ;;
    11) if [[ $PHASE == yield ]]; then line="$line (stale; staying paused)"; else fail 11 "census stale ($line)"; fi ;;
    12) fail 12 "ug missing from the census ($line)" ;;
    *) cat "$CK"; fail 1 "ug_census_check.py failed (rc $rc)" ;;
  esac
  if [[ $PHASE != post ]]; then
    git -C "$WT" show "$BR:$P/s4r_hold_status_2026-10-15.jsonl" > "$ST" 2>/dev/null || : > "$ST"
    python3 "$E/ug_yield_check.py" "$PHASE" "$ST" "$LATEST" > "$OUTF" 2>&1; yrc=$?; yl=$(tail -1 "$OUTF")
    line="$line | $yl"
    case "$PHASE $yrc" in
      "run 0"|"yield 0") ;;
      "run 20") pause_exit 20 "4r: $yl" ;;
      "run 24") pause_exit 24 "4r status unknown or stale: $yl" ;;
      "yield 21"|"yield 26")
        if [[ $yrc == 26 ]]; then
          python3 "$E/ug_census_check_v2.py" "$CEN" --prestart > "$CK" 2>&1 || { line="$line | census gate: $(tail -1 "$CK")"; yrc=0; }
        fi
        if [[ $yrc != 0 ]]; then
          box "$STATE_SH"; rc=$?; s=$(tail -1 "$OUTF")
          [[ $rc == 0 && $s == LIVE* ]] || { log "state read before resume failed or not live (rc $rc): $s"; exit 1; }
          [[ $s == "LIVE pause=1"* ]] || { log "resume condition met but PAUSE_SIMS is already gone ($s); exit 1"; exit 1; }
          [[ $s == "LIVE pause=1 cap=0 gate=0" ]] || { log "resume blocked: SIM_CAP_PAUSE.txt or GATE_PAUSE.txt present ($s); exit 23"; exit 23; }
          log "RESUME CONDITION MET ($yl; $s); still paused until the fork removes PAUSE_SIMS; exit 21"; exit 21
        fi ;;
      *) cat "$OUTF"; fail 1 "ug_yield_check.py failed (rc $yrc)" ;;
    esac
  fi
  [[ $line != "$prev" ]] && { log "$line"; prev=$line; }
  cycle=$((cycle + 1))
  if (( cycle % 3 == 1 )); then
    box "$STATE_SH"; rc=$?; s=$(tail -1 "$OUTF")
    if (( rc != 0 )); then log "state read failed (rc $rc): $s (not an action)"
    else
      case $s in
        FINISHED*) log "launch ended: $s; exit 0"; exit 0 ;;
        GONE*) fail 13 "ugsim gone but launch.json not finished: $s" ;;
        NORUN) fail 1 "no run dir $R" ;;
        "LIVE pause=1"*) [[ $PHASE == yield ]] || { log "PAUSE_SIMS present, not written by this watch ($s); left as is; exit 22"; exit 22; } ;;
        "LIVE pause=0"*) [[ $PHASE != yield ]] || { log "yield phase but PAUSE_SIMS is gone ($s); exit 1"; exit 1; } ;;
        *) log "unexpected state: $s (not an action)" ;;
      esac
    fi
  fi
  left=$(( deadline - $(date +%s) ))
  (( left > 5 )) || fail 14 "deadline $DL PT"
  sleep $(( left < 300 ? left : 300 )) || fail 1 "sleep failed"
done
