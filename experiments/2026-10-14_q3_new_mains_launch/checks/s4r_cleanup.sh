#!/usr/bin/env bash
# ON BOX 4, one shot (d9's cleanup rule: a finished launch's timers are stopped; 10-15 ~09:25 PT): the s5r cleanup watcher
# made one-shot for 4r's follow-up. Only if 4r's follow-up has ENDED, read positively (its launch.json carries "finished"
# AND tmux lists -L nm without fnms4r or has no server there): stop its timers and guards (rpnms4r, gsnms4r, scnms4r,
# gpnms4r on -L nmguards; gatesfnms4r on -L nmgates; gbnms4r on -L nm), move the harness results/PAUSE_SIMS aside if
# present (renamed, never deleted), and verify: no nm sleeper, PAUSE_SIMS absent, no run_pilot.py. One line per step to
# ~/phase2/nm/checks/s4r_cleanup.log.
set -u
L=/home/aswin/phase2/nm/checks/s4r_cleanup.log; FU=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s4r_fu
P=/home/aswin/phase2/nm/attack_check_nm/results/PAUSE_SIMS
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') [s4rcleanup] $*" >> $L; }
absent() { local o rc; o=$(tmux -L "$1" list-sessions -F '#{session_name}' 2>&1); rc=$?
  if (( rc == 0 )); then grep -qxF -- "$2" <<< "$o"; [ $? = 1 ]; else grep -qE '^(no server running on |error connecting to .*\(No such file or directory\))' <<< "$o"; fi; }
python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("finished") else 1)' $FU/launch.json 2>/dev/null && absent nm fnms4r ||
  { log "fnms4r not positively ended: nothing done"; exit 1; }
log "fnms4r has ended (launch.json finished, session gone): cleaning up"
for s in nmguards:rpnms4r nmguards:gsnms4r nmguards:scnms4r nmguards:gpnms4r nmgates:gatesfnms4r nm:gbnms4r; do
  sock=${s%%:*}; n=${s#*:}
  if ! absent $sock $n; then tmux -L $sock kill-session -t "=$n" 2>/dev/null; sleep 1
    absent $sock $n && log "stopped $n" || log "COULD NOT STOP $n"; fi
done
if [ -e $P ]; then a=$P.moved_aside_$(date +%s); mv $P $a && log "PAUSE_SIMS moved aside to $a" || log "COULD NOT MOVE PAUSE_SIMS"; fi
sl=$(pgrep -fc 'sleep [0-9]+ && (touch /home/aswin/phase2/nm|tmux -L nm send-keys)'); rp=$(pgrep -fc 'run_pilot[.]py')
log "verify: nm sleepers $sl; PAUSE_SIMS $([ -e $P ] && echo PRESENT || echo absent); run_pilot.py processes $rp"
[[ $sl == 0 && ! -e $P && $rp == 0 ]] && log "CLEAN: box 4 clean for the next launcher" || log "NOT CLEAN: see the lines above"
