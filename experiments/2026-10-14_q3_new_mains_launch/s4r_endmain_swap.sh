#!/usr/bin/env bash
# ON BOX 4, once (0d's GO relayed by d9, 2026-10-15 about 03:00 PT): swap 4r's end_main for one whose servers_end is
# 12:40 PT (1792093200; 0d's 6 H200s started 02:44:14 PT, hard 10 h) instead of 09:00 PT (1792080000; il_q4's servers,
# now dead). ONLY servers_end differs; same end_main.py (sha256 289892b4...), same python, same other arguments.
# Then LAST_START = 12:40 - 3 h 20 min = 09:20 PT and the follow-up's graceful stop = 12:30 PT (end_main.py:52,58,210).
# Order: refuse unless exactly one process runs the OLD command and end_main's log shows it in step 1; start the NEW one
# in its own session (tmux -L nmguards emnms4rb); confirm it is alive and in step 1 (its start line with last start
# 1792081200, then a step-1 line); only then SIGTERM the old one BY PID (its command re-read first). Any failure after
# the new one started, or an interruption, before the old one is signalled stops the new one (its session and PID,
# verified) and leaves the old one as it was; after the signal the new one is never touched (Codex r26 item 4, r27).
# Nothing else is touched.
#   bash ~/phase2/nm/checks/s4r_endmain_swap.sh
set -u
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
EM=/home/aswin/phase2/nm/checks/end_main.py
REST="4r /home/aswin/phase2/nm/runs/2026-10-14_newmains_s4r /home/aswin/phase2/nm/runs/2026-10-14_newmains_s4r_fu config/pilot_nm_fu_s4r.yaml"
OLD="$PY $EM $REST 1792080000 iii"
NEW="$PY $EM $REST 1792093200 iii"
L=/home/aswin/phase2/nm/checks/end_main_s4r.log
STEP1=' main: [1-9][0-9]* of 244 real runs not yet terminal; calm passes 0$'
t() { TZ=America/Los_Angeles date '+%F %T PT'; }
die() { echo "$(t) STOPPED: $*"; exit 1; }
cmd_of() { tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | sed 's/ $//'; }
pids_of() { for p in $(pgrep -f "end_main[.]py 4r "); do [ "$(cmd_of "$p")" = "$1" ] && echo "$p"; done; }
all_em() { for p in $(pgrep -f "end_main[.]py 4r "); do case "$(cmd_of "$p")" in "$PY $EM 4r "*) echo "$p";; esac; done; }
[ "$(sha256sum $EM | cut -c1-16)" = 289892b4ac8bccca ] || die "end_main.py is not the v3 file"
old=$(pids_of "$OLD"); [ "$(wc -w <<< "$old")" = 1 ] || die "expected exactly one old end_main 4r, found '$old'"
[ "$(all_em | wc -w)" = 1 ] || die "more than one end_main 4r process: $(all_em | tr '\n' ' ')"
tail -1 $L | grep -qE "$STEP1" || die "old end_main is not in step 1: $(tail -1 $L)"
! tmux -L nmguards has-session -t '=emnms4rb' 2>/dev/null || die "session emnms4rb exists"
! tmux -L nm has-session -t '=fnms4r' 2>/dev/null || die "a follow-up session fnms4r exists"
n0=$(wc -l < $L)
retiring=0
rollback() {  # before the old one is signalled: stop the new one (its session, then any NEW-command PID; verified for 20 s)
  # Codex r27: once the SIGTERM to the old one is sent (retiring=1) the new one is never touched (never leave none)
  (( retiring )) && { kill -0 "$old" 2>/dev/null && echo "$(t) ALERT: old $old signalled but alive; new $(pids_of "$NEW" | tr '\n' ' ')kept: decide by hand"; return; }
  kill -0 "$old" 2>/dev/null || { echo "$(t) old pid $old is gone by itself: new $(pids_of "$NEW" | tr '\n' ' ')kept"; return; }
  local i
  for i in $(seq 1 10); do
    tmux -L nmguards kill-session -t '=emnms4rb' 2>/dev/null
    for p in $(pids_of "$NEW"); do kill -TERM "$p"; done
    sleep 2
    [ -z "$(pids_of "$NEW")" ] && ! tmux -L nmguards has-session -t '=emnms4rb' 2>/dev/null &&
      { echo "$(t) rolled back: no new end_main, no emnms4rb; old pid $old still running, unchanged"; return; }
  done
  echo "$(t) ALERT: new end_main $(pids_of "$NEW" | tr '\n' ' ')still alive with old $old after 20 s: decide by hand"
}
trap rollback EXIT
trap 'exit 130' INT TERM HUP
tmux -L nmguards new -d -s emnms4rb "$NEW" || die "could not start emnms4rb"
sleep 15
new=$(pids_of "$NEW")
[ "$(wc -w <<< "$new")" = 1 ] || die "new end_main not running (found '$new')"
tail -n +$((n0 + 1)) $L > /tmp/s4r_swap_newlines.txt
grep -A1 "start: quarter 4r, .*, variant iii, last start 1792081200$" /tmp/s4r_swap_newlines.txt | tail -1 | grep -qE "$STEP1" ||
  die "new end_main's start line and step-1 line not found: $(tr '\n' '|' < /tmp/s4r_swap_newlines.txt)"
[ "$(cmd_of "$old")" = "$OLD" ] || die "pid $old no longer runs the old command"
retiring=1
kill -TERM "$old"
for i in 1 2 3 4 5; do sleep 2; kill -0 "$old" 2>/dev/null || break; done
kill -0 "$old" 2>/dev/null && die "old pid $old still alive 10 s after SIGTERM"
! tmux -L nmguards has-session -t '=emnms4r' 2>/dev/null || die "session emnms4r still listed (old pid $old gone; new $new kept)"
[ "$(all_em)" = "$new" ] || die "end_main 4r processes now: $(all_em | tr '\n' ' ') (expected only $new)"
echo "$(t) SWAPPED ($(date -u '+%F %T UTC')): old pid $old (servers_end 1792080000, emnms4r) stopped by SIGTERM;" \
  "new pid $new (servers_end 1792093200, emnms4rb) in step 1; last start 09:20 PT, graceful stop 12:30 PT"
