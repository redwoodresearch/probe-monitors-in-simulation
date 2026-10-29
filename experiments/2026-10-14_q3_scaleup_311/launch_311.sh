#!/usr/bin/env bash
# Laptop side: launch ONE quarter on one box (RULE.md, Launch; Amendments 1-2). One launch per quarter, never two at once
# on one box. In order; each step must succeed:
#   0. the Modal day guard's status is OK and at most 300 s old (else nothing starts); 8 servers below $1,600, else 6;
#   1. on the box: the run folder must not exist; <harness>/results/PAUSE_SIMS is written and checked BEFORE anything
#      else (the patched launcher starts no simulated run while it exists; only unpause removes it), and the devbox's own
#      shutdown is scheduled (now + 10.5 h): both REQUIRED before anything spends;
#   2. allin_stop.sh (the canary's, GPU_RATE 5.3, POLL_S 120, STALE_MAX 5, RESERVE 250) over this quarter's launcher
#      session, started BEFORE the servers, STARTUP_S 3000; then final_stop.sh, which waits for allin_stop.sh to exit,
#      confirms the apps stopped (retrying until confirmed), copies the run folder and stops the devbox through EC2;
#   3. launch_servers.py in its OWN process group (job control), so a cancel reaches the Python process itself: each app id
#      it records goes into APP_IDS_FILE at once, and the guard is re-read every 10 s (not OK -> SIGINT to the group; the
#      server launcher stops and confirms every app it created; then every app named with its tag is stopped too);
#      billing watcher v2.6b ($45, S + 9.25 h) and guard v2.6c per app (the scale-up's copies: the pinned modal binary);
#   4. the guard again (OK, fresh) and allin_stop.sh alive, then on the devbox: canary.sh (pre-launch gates, then the
#      launcher in tmux -L phase2 session s311q<N>); the live settings watch (tmux -L gates); gate1_311.py (tmux -L phase2
#      session gate311q<N>; graceful stop at S + 8.5 h); all verified running.
# Any failure after step 2: Ctrl+C the launcher pane if it started, stop and confirm every app (stop_group.sh, then every
# app with the launch's tag), then end allin_stop.sh so final_stop.sh stops the devbox.
#   Q=<quarter 1-4> DEVBOX=<user>@<id> H=<local frozen harness> RUN=<run name> BOX_START=<epoch s: the box's start, or
#   this launch's start for a box's second quarter> STATUS=<guard status file> WD=<laptop log dir> launch_311.sh
set -uo pipefail
for v in Q DEVBOX H RUN BOX_START STATUS WD; do [[ -n ${!v:-} ]] || { echo "set $v" >&2; exit 1; }; done
HERE=$(cd "$(dirname "$0")" && pwd); W=$HERE/watchers; MODAL=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal
B=q$Q; SESSION=s311q$Q; GATE=gate311q$Q; C=/home/aswin/phase2/attack_check_311; RUNP=phase2/runs/$RUN
IMAGES=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json
APPF=$WD/app_ids_box$B.txt; : > "$APPF"
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') [Q$Q] $*"; }
dbx() { perl -e 'alarm shift; exec @ARGV' "$1" bash "$H/devbox.sh" "$2"; }
guard_ok() {  # the day guard's latest status: OK and at most 300 s old; prints the total
  local level total t; read -r level total _ _ _ t < "$STATUS" 2>/dev/null || return 1
  (( $(date +%s) - t <= 300 )) && [[ $level == OK ]] && echo "$total"; }
TOTAL=$(guard_ok) || { log "Modal day guard not OK or stale ($STATUS): nothing started"; exit 1; }
N=6; log "Modal day \$$TOTAL: $N servers (Amendment 3: 6 per quarter, each with a 6 h Modal timeout)"
dbx 60 "mkdir -p ~/phase2/s311_checks && ! test -e ~/$RUNP && touch $C/results/PAUSE_SIMS && test -e $C/results/PAUSE_SIMS" ||
  { log "devbox unreachable, $RUNP exists, or PAUSE_SIMS not written: nothing started"; exit 1; }
log "PAUSE_SIMS in place ($C/results/PAUSE_SIMS)"
dbx 60 "sudo shutdown -h +${SHUTDOWN_MIN:-390}" || { log "could not schedule the devbox's own shutdown: nothing started"; exit 1; }  # S + 6.5 h: the servers' 6 h timeout + 30 min
DEVBOX=$DEVBOX perl -e 'alarm 120; exec @ARGV' bash "$H/devbox.sh" rsync -a "$HERE/spend_now.py" box:phase2/s311_checks/ || exit 1
S=$(date +%s); echo "$S" > "$WD/gpu_start_box$B.txt"
H=$H DEVBOX=$DEVBOX STOP_USD=1850 RESERVE_USD=250 N_SERVERS=$N GPU_START=$S BOX_START=$BOX_START BOX_RATE=4.1 GPU_RATE=5.3 \
  APP_IDS_FILE=$APPF K=phase2/s311_checks HARN=phase2/attack_check_311 POLL_S=120 STALE_MAX=5 STARTUP_S=3000 \
  nohup bash "$HERE/allin_stop.sh" "$SESSION=$RUNP" > "$WD/allin_stop_box$B.log" 2>&1 &
AP=$!; echo $AP > "$WD/allin_stop_box$B.pid"
H=$H DEVBOX=$DEVBOX APP_IDS_FILE=$APPF SESSION=$SESSION nohup bash "$HERE/final_stop.sh" "$AP" "$WD/allin_stop_box$B.log" "$RUN" \
  > "$WD/final_stop_box$B.log" 2>&1 &
echo $! > "$WD/final_stop_box$B.pid"
sleep 5; kill -0 "$AP" 2>/dev/null || { log "allin_stop.sh did not stay up"; exit 1; }
log "allin_stop.sh pid $AP (trigger \$1600), final_stop.sh pid $(cat "$WD/final_stop_box$B.pid")"
TAG=""
stop_tagged() {  # every app named with this launch's tag, whether or not it reached APP_IDS_FILE
  [[ -n $TAG ]] || return 0
  local ids; ids=$(perl -e 'alarm 120; exec @ARGV' "$MODAL" app list --json 2>/dev/null | python3 -I -c "import json,sys; print(' '.join(a['app_id'] for a in json.load(sys.stdin) if '$TAG' in str(a.get('description','')) and str(a.get('state','')).lower() not in ('stopped',)))") || ids=""
  for a in $ids; do perl -e 'alarm 120; exec @ARGV' "$MODAL" app stop "$a" >/dev/null 2>&1; log "stopped tagged app $a"; done; }
fail() { log "LAUNCH FAILED: $*"
  p=$(dbx 60 "tmux -L phase2 list-panes -a -F '#{session_name} #{pane_id}:#{pane_pid}' 2>/dev/null | awk '\$1==\"$SESSION\" {print \$2}'") || p=""
  H=$H DEVBOX=$DEVBOX APP_IDS="$(tr -d ' \n' < "$APPF")" FORCE_APPS=1 bash "$HERE/stop_group.sh" "$p"
  stop_tagged
  kill "$AP" 2>/dev/null; log "allin_stop.sh ended; final_stop.sh now confirms the apps and stops the devbox"; exit 1; }
guard_ok >/dev/null || fail "Modal day guard not OK before the servers"
publish() { local f
  [[ -n $TAG ]] || TAG=$(grep -o 'srv[0-9a-f]\{6\}' "$WD/launch_servers_box$B.log" | head -1)
  [[ -n $TAG ]] || return 0
  f=$(ls -t "$H"/results/servers_*_"${TAG#srv}".json 2>/dev/null | head -1); [[ -n $f ]] || return 0
  python3 -I -c "import json,sys; print(','.join(s['app_id'] for s in json.load(open(sys.argv[1]))['servers'] if s.get('app_id')))" "$f" > "$APPF.tmp" &&
    [[ -s $APPF.tmp ]] && mv "$APPF.tmp" "$APPF"; }
set -m  # the server launcher gets its own process group: a SIGINT to the group reaches the Python process, not only uv
(cd "$H" && exec uv run --no-project --python 3.12 --with modal==1.2.6 --with pyyaml python launch_servers.py --count "$N" \
  --server-config servers_311.yaml) > "$WD/launch_servers_box$B.log" 2>&1 &
LP=$!
set +m
while kill -0 "$LP" 2>/dev/null; do
  publish
  if ! guard_ok >/dev/null; then
    kill -INT -- "-$LP" 2>/dev/null
    for _ in $(seq 60); do kill -0 "$LP" 2>/dev/null || break; sleep 5; done  # its own cleanup: stop and confirm its apps
    kill -0 "$LP" 2>/dev/null && kill -TERM -- "-$LP" 2>/dev/null
    publish; fail "Modal day guard not OK during server start"
  fi
  sleep 10
done
wait "$LP" || { publish; fail "launch_servers.py"; }
publish
APPS=$(grep "^app IDs" "$WD/launch_servers_box$B.log" | sed 's/.*: //'); BASES=$(grep "^base URLs" "$WD/launch_servers_box$B.log" | sed 's/.*: //')
[[ $APPS == "$(tr -d ' \n' < "$APPF")" ]] || fail "app ids in the log ($APPS) differ from APP_IDS_FILE"
[[ $(tr ',' '\n' <<< "$APPS" | grep -c .) == "$N" && $(tr ',' '\n' <<< "$BASES" | grep -c .) == "$N" ]] || fail "expected $N apps and URLs"
echo "$BASES" > "$WD/bases_box$B.txt"; log "apps $APPS"
# Amendment 3 (unwatched): no per-app laptop watchers (q2's v2.6 fail closed on failed polls, and a sleeping laptop runs
# nothing); each server's own 6 h Modal timeout and 20-minute idle exit bound it.
guard_ok >/dev/null || fail "Modal day guard not OK at release"
kill -0 "$AP" 2>/dev/null || fail "allin_stop.sh is not running at release"
dbx 60 "test -e $C/results/PAUSE_SIMS" || fail "PAUSE_SIMS missing at release"
dbx 1800 "SESSION=$SESSION HARNESS=$C bash $C/canary.sh $RUN config/pilot_311_q$Q.yaml $BASES $IMAGES" || fail "canary.sh (gates or launcher)"
dbx 60 "cd $C && tmux -L gates new -d -s gate${SESSION} \"PYTHONPATH=/home/aswin/phase2/sims-sept-strict-off/src /home/aswin/sims/simulated-control-evals-sept/.venv/bin/python gate_settings.py --reference-json config/sim_settings_reference_2026-10-14_125923.json --out results/gate_settings_new_$RUN.json --new /home/aswin/$RUNP --session $SESSION > results/gate_settings_new_$RUN.log 2>&1\"" || fail "settings watch did not start"
dbx 60 "cd $C && tmux -L phase2 new -d -s $GATE \"PYTHONPATH=/home/aswin/phase2/sims-sept-strict-off/src /home/aswin/sims/simulated-control-evals-sept/.venv/bin/python gate1_311.py /home/aswin/$RUNP $SESSION $((S + 30600))\"" || fail "gate1_311 did not start"
sleep 10
dbx 30 "tmux -L gates has-session -t '=gate$SESSION' && tmux -L phase2 has-session -t '=$GATE' && tmux -L phase2 has-session -t '=$SESSION'" || fail "a watch or the launcher is not running"
log "LAUNCHED $RUN: session $SESSION, $N servers ($APPS), gates gate$SESSION and $GATE; GPU start $S"
