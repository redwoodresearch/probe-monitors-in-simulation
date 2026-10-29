#!/usr/bin/env bash
# ON SIMS-ASTRA, tmux -L nmguards s5rcleanup (d9, 10-15 about 00:40 PT): every 150 s until 05:00 PT, wait for 5r
# to END, read positively in one pass (its launch.json carries "finished" AND tmux lists -L nm without fnms5r or
# has no server there). Only then: stop 5r's finished-launch timers and guards (rpnms5r, gsnms5r, scnms5r, gpnms5r on
# -L nmguards; gatesfnms5r on -L nmgates; gbnms5r on -L nm), move the harness results/PAUSE_SIMS aside if present (renamed,
# never deleted), and verify: no nm sleeper, PAUSE_SIMS absent, no run_pilot.py. Nothing is touched while fnms5r runs.
# One line per poll to ~/phase2/nm/checks/s5r_cleanup.log. Deadline 05:00 PT (1792065600): nothing done after it.
set -u
L=/home/aswin/phase2/nm/checks/s5r_cleanup.log; FU=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s5r_fu
P=/home/aswin/phase2/nm/attack_check_nm/results/PAUSE_SIMS; DEADLINE=1792065600
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') [s5rcleanup] $*" >> $L; }
absent() { local o rc; o=$(tmux -L "$1" list-sessions -F '#{session_name}' 2>&1); rc=$?
  if (( rc == 0 )); then grep -qxF -- "$2" <<< "$o"; [ $? = 1 ]; else grep -qE '^(no server running on |error connecting to .*\(No such file or directory\))' <<< "$o"; fi; }
log "armed: waiting for fnms5r to end (deadline 05:00 PT)"
while :; do
  now=$(date +%s) || { log "date failed; retrying"; sleep 150; continue; }
  (( now < DEADLINE )) || { log "05:00 deadline: nothing done"; exit 1; }
  if python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("finished") else 1)' $FU/launch.json 2>/dev/null && absent nm fnms5r; then
    break
  fi
  log "fnms5r still running or unreadable: waiting"; sleep 150
done
log "fnms5r has ended (launch.json finished, session gone): cleaning up"
for s in nmguards:rpnms5r nmguards:gsnms5r nmguards:scnms5r nmguards:gpnms5r nmgates:gatesfnms5r nm:gbnms5r; do
  sock=${s%%:*}; n=${s#*:}
  if ! absent $sock $n; then tmux -L $sock kill-session -t "=$n" 2>/dev/null; sleep 1
    absent $sock $n && log "stopped $n" || log "COULD NOT STOP $n"; fi
done
if [ -e $P ]; then a=$P.moved_aside_$(date +%s); mv $P $a && log "PAUSE_SIMS moved aside to $a" || log "COULD NOT MOVE PAUSE_SIMS"; fi
sl=$(pgrep -fc 'sleep [0-9]+ && (touch /home/aswin/phase2/nm|tmux -L nm send-keys)'); rp=$(pgrep -fc 'run_pilot[.]py')
log "verify: nm sleepers $sl; PAUSE_SIMS $([ -e $P ] && echo PRESENT || echo absent); run_pilot.py processes $rp"
[[ $sl == 0 && ! -e $P && $rp == 0 ]] && log "CLEAN: sims-astra ready for the next launcher" || log "NOT CLEAN: see the lines above"
