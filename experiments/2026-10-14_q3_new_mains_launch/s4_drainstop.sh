#!/usr/bin/env bash
# ON BOX q3-scale-canary, tmux -L nmguards s4drain (RULE.md Amendment 7; d9's decision about 21:15 PT, option (b)): the
# S4 main launcher is SIGSTOPped by gate1_nm.py (PAUSED_usable75.txt: policy server index 4 died). Drain, then stop it
# gracefully, as health_monitor.py @ fe62b0846 steps 2-3 do: every 30 s, until the launcher has no live (non-zombie)
# child, i.e. every in-flight run has ended, or until DRAIN_DEADLINE; then the stop record, then SIGINT and SIGCONT (the
# launcher wakes inside its sleep with its stop flag set and stops gracefully, starting nothing). The stop record is
# written as DEADLINE_STOP.txt because end_main.py's time-bound stop path keys on that name (a resource stop, not a
# quality failure; the file says so), and the PAUSED marker is renamed so end_main hands over to the follow-up for cells
# with a usable honest_a. One line per step to <run>/s4_drainstop.log.
set -uo pipefail
R=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s4; P=$1; DRAIN_DEADLINE=$2
L=$R/s4_drainstop.log
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') $*" | tee -a "$L"; }
tr '\0' ' ' < /proc/$P/cmdline | grep -q "run_pilot.py .*--out $R" || { log "pid $P is not the S4 launcher: nothing done"; exit 1; }
[[ $(awk '{print $3}' /proc/$P/stat) == T ]] || { log "launcher $P is not stopped (SIGSTOP): nothing done"; exit 1; }
kids() { local n=0; for s in /proc/[0-9]*/stat; do f=$(cat "$s" 2>/dev/null) || continue; f=${f##*) }; set -- $f
  [[ $2 == "$P" && $1 != Z ]] && n=$((n + 1)); done; echo $n; }
log "drain: launcher $P paused; waiting for its live children to end (deadline $(TZ=America/Los_Angeles date -d @$DRAIN_DEADLINE '+%T PT'))"
while :; do
  k=$(kids); [[ $k == 0 ]] && { log "drained: no live child"; break; }
  (( $(date +%s) < DRAIN_DEADLINE )) || { log "drain deadline with $k live children: stopping anyway (they are cut, recorded stopped)"; break; }
  sleep 30
done
cat > $R/DEADLINE_STOP.txt <<EOF
$(TZ=America/Los_Angeles date '+%F %T PT') DRAIN STOP of the S4 main (not its 00:12 deadline): d9's decision (about 21:15 PT,
option b) after gate1_nm.py paused it (PAUSED_usable75: policy server index 4, ap-Ptp5NWVwK27Gt7ksEsbhuB, died; 13
RetryError failures on it). Written under this name so end_main.py takes its time-bound stop path (DEADLINE_STOP no
longer blocks the follow-up); a resource stop, not a quality failure. Runs recorded "stopped" by this stop are relabelled
by the committed rule in RELABEL_RULE_drainstop.txt.
EOF
mv $R/PAUSED_usable75.txt $R/RESUMED_usable75_drainstop.txt
log "stop record written (DEADLINE_STOP.txt); PAUSED_usable75.txt renamed RESUMED_usable75_drainstop.txt"
kill -INT $P && kill -CONT $P && log "SIGINT then SIGCONT sent to $P"
for _ in $(seq 1 90); do sleep 10; [[ -e /proc/$P ]] || break; done
[[ -e /proc/$P ]] && log "launcher $P still alive after 15 min" || log "launcher $P exited"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("finished", d.get("finished"))' $R/launch.json 2>&1 | tee -a "$L"
