#!/usr/bin/env bash
# THE laptop step for the Q2 RERUN (RULE.md Amendment 11), ONE command, no arguments: q2r_launch.sh
#   1. box 2 gets the rerun's harness files (config/tasks_311_q2r_fu.json, config/pilot_311_fu_q2r_iii.yaml,
#      config/servers_311_fu_q2r.yaml, fu_launch_q2r.sh, LAUNCH_COMMIT_Q2R, config/harness_manifest_fu_q2r.tsv) and
#      gate_pause_311.py (~/phase2/s311_checks); the freeze check runs on the box BEFORE any server starts;
#   2. 4 servers from the laptop (launch_servers.py --server-config servers_311_fu_q2r.yaml: detached, 5.5 h Modal timeout,
#      60 minutes to the first request, 20-minute idle exit), in their own process group, waited for (URLs);
#   3. on box 2, the time protections FIRST: rp311q2r (the harness PAUSE_SIMS at LAST_START = the servers' end - 12,000 s)
#      and gs311q2r (one Ctrl+C to f311q2r at the servers' end - 600 s); sim_cap_stop sc311q2r at the rerun's admission
#      limit + $100 ($2,340); then fu_launch_q2r.sh (gates, launcher); then the pause-only usable gate gp311q2r and the
#      live settings watch gatef311q2r. Any failure after the servers start: launch_servers.py's apps are stopped by
#      tag from the laptop (stop_group.sh with FORCE_APPS=1) before exit.
# Prints the launch ID, app IDs and times. Box 2's shutdown (the first follow-up servers' end + 30 min) is later than
# this one's end and stays.
set -uo pipefail
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
HERE=$(cd "$(dirname "$0")" && pwd); WD=$SP/scaleup311/watch; H=$SP/scaleup311/harness_q2r/experiments/2026-10-07_phase2_attack_check  # its own laptop copy
B=aswin@i-0c02a38abcde32bfb; C=/home/aswin/phase2/attack_check_311; K=/home/aswin/phase2/s311_checks
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python; TREE=/home/aswin/phase2/sims-sept-strict-off/src
MAIN=/home/aswin/phase2/runs/$(cut -d' ' -f2 $WD/runs_q1q2.txt); FU1=$(cat $WD/fu_run_q2.txt)
FU=/home/aswin/phase2/runs/$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S)_scaleup311_q2r_fu
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') $*"; }
dbx() { DEVBOX=$B perl -e 'alarm shift; exec @ARGV' "$1" bash "$H/devbox.sh" "$2"; }
[[ $FU1 == */2026-10-14_173012_scaleup311_q2_fu && $MAIN == */2026-10-14_154401_scaleup311_q2 ]] || { log "unexpected runs $MAIN $FU1"; exit 1; }
[[ ! -e $WD/fu_run_q2r.txt ]] || { log "a rerun was already started ($(cat $WD/fu_run_q2r.txt)): refusing"; exit 1; }
# 1. files and the freeze check, before any spend
for f in config/tasks_311_q2r_fu.json config/pilot_311_fu_q2r_iii.yaml config/servers_311_fu_q2r.yaml fu_launch_q2r.sh LAUNCH_COMMIT_Q2R config/harness_manifest_fu_q2r.tsv; do
  cmp -s "$HERE/$f" "$H/$f" || { log "the laptop harness copy of $f differs from the experiment folder's"; exit 1; }
  DEVBOX=$B perl -e 'alarm 120; exec @ARGV' bash "$H/devbox.sh" rsync -a "$H/$f" "box:phase2/attack_check_311/$f" || { log "copy of $f failed"; exit 1; }
done
DEVBOX=$B perl -e 'alarm 120; exec @ARGV' bash "$H/devbox.sh" rsync -a "$HERE/gate_pause_311.py" box:phase2/s311_checks/ || exit 1
dbx 120 "cd $C && diff <(find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './attempt/results/*' ! -path './config/harness_manifest.tsv' ! -path './config/harness_manifest_fu.tsv' ! -path './config/harness_manifest_fu_q2r.tsv' | LC_ALL=C sort | xargs -d '\n' sha256sum) config/harness_manifest_fu_q2r.tsv > /dev/null && test ! -e results/PAUSE_SIMS && ! tmux -L phase2 has-session -t '=f311q2r' 2>/dev/null && ! tmux -L phase2 has-session -t '=f311q2' 2>/dev/null" \
  || { log "freeze, pause-file or session check failed on box 2"; exit 1; }
log "rerun files in place on box 2, freeze PASS, no harness pause, no f311q2/f311q2r session"
echo "$FU" > "$WD/fu_run_q2r.txt"
# 2. servers. Codex round 21: the last start leaves the full 12,000 s (7,200 + 3,600 + 1,200) before the graceful stop;
#    a signal to this script while launch_servers.py runs sends it SIGINT (its own cleanup stops every app it created,
#    then every app with its tag), and one after the servers are up but before the launcher exists stops the apps.
S=$(date +%s); END=$((S + 19800)); GRACE=$((END - 600)); LAST=$((GRACE - 12000)); echo "$S $END" > "$WD/fu_servers_q2r_start_end.txt"
APPS=""; LP=""; LAUNCHED=0
stop_apps() { log "STOPPING the rerun's apps: $1"; H=$H DEVBOX=$B APP_IDS="$APPS" FORCE_APPS=1 "$HERE/stop_group.sh" ""; exit 1; }
on_signal() {
  if [[ -n $LP ]] && kill -0 "$LP" 2>/dev/null; then
    log "signal during server startup: SIGINT to launch_servers.py (it stops its own apps)"; kill -INT -- "-$LP"; wait "$LP"; exit 1
  fi
  (( LAUNCHED )) && { log "signal after the launch: the launch keeps running under its on-box protections"; exit 1; }
  [[ -n $APPS ]] && stop_apps "signal before the launcher started"
  exit 1
}
trap on_signal INT TERM HUP
set -m
(cd "$H" && exec uv run --no-project --python 3.12 --with modal==1.2.6 --with pyyaml python launch_servers.py --count 4 \
  --server-config servers_311_fu_q2r.yaml) > "$WD/fu_launch_servers_q2r.log" 2>&1 &
LP=$!; set +m
wait $LP; rc=$?; LP=""
APPS=$(grep "^app IDs" "$WD/fu_launch_servers_q2r.log" | sed 's/.*: //'); BASES=$(grep "^base URLs" "$WD/fu_launch_servers_q2r.log" | sed 's/.*: //')
echo "$APPS" > "$WD/fu_app_ids_q2r.txt"
(( rc == 0 )) || { log "launch_servers.py exit $rc (it stops its own apps)"; exit 1; }
[[ $(tr ',' '\n' <<< "$BASES" | grep -c https) == 4 && $(tr ',' '\n' <<< "$APPS" | grep -c '^ap-') == 4 ]] || stop_apps "expected 4 URLs and 4 app IDs, got '$BASES' / '$APPS'"
log "servers up: $APPS; end $(date -r $END '+%H:%M:%S'); last start $(date -r $LAST '+%H:%M:%S'); graceful stop $(date -r $GRACE '+%H:%M:%S')"
# 3. on box 2: time protections, spend backstop, launch, gates
dbx 60 "echo '$BASES' > $K/fu_bases_q2r.txt && for s in rp311q2r gs311q2r sc311q2r gp311q2r; do ! tmux -L guards has-session -t \"=\$s\" 2>/dev/null || exit 1; done \
  && tmux -L guards new -d -s rp311q2r 'sleep \$(( $LAST - \$(date +%s) )) && touch $C/results/PAUSE_SIMS' \
  && tmux -L guards new -d -s gs311q2r 'sleep \$(( $GRACE - \$(date +%s) )) && tmux -L phase2 send-keys -t =f311q2r: C-c' \
  && tmux -L guards new -d -s sc311q2r '$PY $K/sim_cap_stop.py $FU f311q2r 2340' && sleep 3 \
  && tmux -L guards has-session -t =rp311q2r && tmux -L guards has-session -t =gs311q2r && tmux -L guards has-session -t =sc311q2r" \
  || stop_apps "the time protections or sim_cap_stop did not start"
log "rp311q2r, gs311q2r and sc311q2r (\$2,340) running on box 2"
# Codex round 24: from the handoff on, nothing destructive while the launcher's state is unknown. A signal now only exits;
# the apps are stopped only on CONFIRMED absence (the box answered, no f311q2r session, no fu_launch_q2r.sh still running,
# and the launch call didn't end on its own time bound, which could leave it running on the box).
LAUNCHED=1
out=$(dbx 1800 "bash $C/fu_launch_q2r.sh $MAIN $FU1 $FU $BASES"); rc=$?
echo "$out"
st=$(dbx 60 "if tmux -L phase2 has-session -t =f311q2r 2>/dev/null && test -s $FU.meta.json; then echo LIVE; elif pgrep -f [f]u_launch_q2r.sh >/dev/null; then echo STARTING; else echo ABSENT; fi"); src=$?
if [[ $st == LIVE ]]; then
  log "the rerun launcher is running (fu_launch_q2r.sh reported exit $rc)"
elif [[ $src == 0 && $st == ABSENT && $rc != 142 ]]; then
  stop_apps "confirmed: the rerun launcher is not running (fu_launch_q2r.sh exit $rc)"
else
  log "launcher state UNKNOWN or still starting (state '$st', ssh exit $src, launch exit $rc): nothing stopped; check box 2 by hand (unused servers exit 60 min after start)"; exit 1
fi
N=$(basename "$FU")
# The pause-only gate and the live settings watch: two tries; if they still aren't up, a CLEAN pause (no new simulated
# start; in-flight runs finish; never a Ctrl+C), and this script exits 1 (Codex round 21: no silent success).
for try in 1 2; do
  dbx 60 "{ tmux -L guards has-session -t =gp311q2r 2>/dev/null || tmux -L guards new -d -s gp311q2r '$PY $K/gate_pause_311.py $FU f311q2r'; } \
    && { tmux -L gates has-session -t =gatef311q2r 2>/dev/null || tmux -L gates new -d -s gatef311q2r 'cd $C && PYTHONPATH=$TREE $PY gate_settings.py --reference-json config/sim_settings_reference_2026-10-14_125923.json --out results/gate_settings_new_$N.json --new $FU --session f311q2r > results/gate_settings_new_$N.log 2>&1'; } \
    && sleep 5 && tmux -L guards has-session -t =gp311q2r && tmux -L gates has-session -t =gatef311q2r" && { log "gp311q2r and gatef311q2r running"; break; }
  if (( try == 2 )); then
    dbx 60 "touch $FU/PAUSE_SIMS" && log "the gate or the settings watch did not start: PAUSE_SIMS written (clean pause, in-flight runs finish)" \
      || log "the gate or the settings watch did not start AND the pause could not be written: act by hand"
    exit 1
  fi
done
log "RERUN LAUNCHED: $FU (session f311q2r); apps $APPS"
