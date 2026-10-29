#!/usr/bin/env bash
# Laptop side: launch ONE new-mains split on one box (RULE.md, Launch). Derived from the scale-up's launch_311b.sh and
# fu_servers_311.sh, with the scale-up's laptop stops removed (Aswin 15:35, "run unwatched"): the main launch runs real
# arms only (simulated attacks paused by the harness's results/PAUSE_SIMS); every bound is on Modal or on the box. Steps,
# each must succeed:
#   1. on the box: the isolated tree ~/phase2/nm is present, the run folder doesn't exist, PAUSE_SIMS is written and
#      checked, the scratch image is present, inotify is 8192; the box's shutdown is moved LATER if needed (never
#      earlier: the box may be shared with a scale-up follow-up) to the follow-up servers' end + 30 min;
#   2. from the laptop, at once: the main servers (N, servers_nm.yaml: 5 h) and the follow-up servers (4, Amendment 5;
#      servers_nm_fu.yaml: 10 h, a 5 h wait for their first request), each launcher in its own process group; on any
#      failure every app it created is stopped by the launcher itself and every app id recorded here is stopped too;
#   3. on the box: the follow-up URLs (~/phase2/nm/checks/fu_bases_s<K>.txt) and the out-of-harness checks;
#   4. on the box: canary.sh (pre-launch gates, then the launcher in tmux -L nm session snms<K>); gate1_nm.py on the main run (tmux -L nm gbnms<K>; Amendment 4: pause, never kill; graceful stop at the
#      main servers' end - 10 min); end_main.py (tmux -L nmguards emnms<K>), which hands the box to the follow-up
#      launch once every real run has ended; all verified running.
#   K=<split> DEVBOX=<user>@<id> H=<local nm harness> RUN=<run name> N=<main servers> WD=<laptop log dir> launch_nm.sh
set -uo pipefail
for v in K DEVBOX H RUN N WD; do [[ -n ${!v:-} ]] || { echo "set $v" >&2; exit 1; }; done
HERE=$(cd "$(dirname "$0")" && pwd); MODAL=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal
C=/home/aswin/phase2/nm/attack_check_nm; RUNP=/home/aswin/phase2/nm/runs/$RUN; FU=${RUNP}_fu
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python; TREE=/home/aswin/phase2/sims-sept-strict-off/src
IMAGES=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json
export AWS_PROFILE=redwood-sandbox
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') [S$K] $*"; }
dbx() { perl -e 'alarm shift; exec @ARGV' "$1" bash "$H/devbox.sh" "$2"; }
APPS_ALL=""
stop_apps() { for a in $(tr ',' ' ' <<< "$APPS_ALL"); do perl -e 'alarm 120; exec @ARGV' "$MODAL" app stop "$a" >/dev/null 2>&1; log "stopped app $a"; done; }
fail() { log "LAUNCH FAILED: $*"
  dbx 60 "tmux -L nm has-session -t '=snms$K' 2>/dev/null && tmux -L nm send-keys -t '=snms$K:' C-c" || true
  stop_apps; exit 1; }
mkdir -p "$WD"
# 1. the box
dbx 120 "set -e; test -d $C && test -d /home/aswin/phase2/nm/settings && test -d /home/aswin/phase2/nm/approaches
  mkdir -p /home/aswin/phase2/nm/runs /home/aswin/phase2/nm/checks; ! test -e $RUNP; ! test -e $FU
  ! tmux -L nm has-session -t '=snms$K' 2>/dev/null; ! tmux -L nmguards has-session -t '=emnms$K' 2>/dev/null
  touch $C/results/PAUSE_SIMS && test -e $C/results/PAUSE_SIMS
  docker image inspect control-tower-sim-scratch:latest --format '{{.Id}}' | grep -q '^sha256:ba1bed0c'
  test \$(cat /proc/sys/fs/inotify/max_user_instances) -ge 8192" || { log "box not ready (tree, run folder, session, PAUSE_SIMS, scratch image or inotify): nothing started"; exit 1; }
log "box ready; PAUSE_SIMS in place"
S=$(date +%s); END_MAIN=$((S + 5 * 3600)); END_FU=$((S + 10 * 3600)); WANT=$((END_FU + 1800))
# Codex r1 item 2: an existing schedule that can't be read or parsed fails closed (nothing started); only a MISSING
# schedule file means no shutdown is scheduled.
dbx 60 "set -e; f=/run/systemd/shutdown/scheduled; if [ -e \$f ]; then cur=\$(sed -n 's/^USEC=//p' \$f); [[ \$cur =~ ^[0-9]+\$ ]] || { echo unparseable; exit 1; }; cur=\$((cur / 1000000)); else cur=0; fi
  if (( cur < $WANT )); then sudo shutdown -h +$(( (WANT - S) / 60 )) >/dev/null 2>&1 && echo \"shutdown moved from \$cur to $WANT\"; else echo \"shutdown kept at \$cur (>= $WANT)\"; fi" \
  | tee -a "$WD/shutdown_s$K.log" || { log "could not check or move the box's shutdown: nothing started"; exit 1; }
# 2. the servers
set -m
(cd "$H" && exec uv run --no-project --python 3.12 --with modal==1.2.6 --with pyyaml python launch_servers.py --count "$N" \
  --server-config servers_nm.yaml) > "$WD/launch_servers_main_s$K.log" 2>&1 &
LM=$!
(cd "$H" && exec uv run --no-project --python 3.12 --with modal==1.2.6 --with pyyaml python launch_servers.py --count 4 \
  --server-config servers_nm_fu.yaml) > "$WD/launch_servers_fu_s$K.log" 2>&1 &
LF=$!
set +m
wait "$LM"; rm_=$?; wait "$LF"; rf_=$?
AM=$(grep "^app IDs" "$WD/launch_servers_main_s$K.log" | sed 's/.*: //'); BM=$(grep "^base URLs" "$WD/launch_servers_main_s$K.log" | sed 's/.*: //')
AF=$(grep "^app IDs" "$WD/launch_servers_fu_s$K.log" | sed 's/.*: //'); BF=$(grep "^base URLs" "$WD/launch_servers_fu_s$K.log" | sed 's/.*: //')
APPS_ALL="$AM,$AF"; echo "$AM" > "$WD/app_ids_main_s$K.txt"; echo "$AF" > "$WD/app_ids_fu_s$K.txt"
[[ $rm_ == 0 && $rf_ == 0 ]] || fail "launch_servers.py exit main $rm_ fu $rf_"
[[ $(tr ',' '\n' <<< "$BM" | grep -c https) == "$N" && $(tr ',' '\n' <<< "$BF" | grep -c https) == 4 ]] || fail "expected $N main and 4 follow-up URLs"
log "servers: main $AM; follow-up $AF"
# 3. URLs and checks
dbx 60 "echo '$BF' > /home/aswin/phase2/nm/checks/fu_bases_s$K.txt" || fail "follow-up URLs"
DEVBOX=$DEVBOX perl -e 'alarm 120; exec @ARGV' bash "$H/devbox.sh" rsync -a "$HERE/checks/end_main.py" "$HERE/checks/gate1_nm.py" "$HERE/checks/gate_pause_nm.py" \
  "$HERE/checks/sim_cap_stop.py" "$HERE/checks/spend_now.py" box:phase2/nm/checks/ || fail "checks copy"
# 4. gates and launcher, then the watches
dbx 1800 "SESSION=snms$K HARNESS=$C bash $C/canary.sh $RUN config/pilot_nm_s$K.yaml $BM $IMAGES" || fail "canary.sh (gates or launcher)"
# (no live settings watch on the main: no simulated run by design; end_main.py starts it on the follow-up, Codex r1 item 5)
dbx 60 "tmux -L nm new -d -s gbnms$K \"cd /home/aswin/phase2/nm/checks && PYTHONPATH=$TREE $PY gate1_nm.py $RUNP snms$K $((END_MAIN - 600)) main\"" || fail "gate1_nm"
dbx 60 "tmux -L nmguards new -d -s emnms$K \"$PY /home/aswin/phase2/nm/checks/end_main.py $K $RUNP $FU config/pilot_nm_fu_s$K.yaml $END_FU iii\"" || fail "end_main"
sleep 10
dbx 60 "tmux -L nm has-session -t '=snms$K' && tmux -L nm has-session -t '=gbnms$K' && tmux -L nmguards has-session -t '=emnms$K'" ||
  fail "a watch or the launcher is not running"
log "LAUNCHED $RUN (session snms$K): main servers $AM (end $(date -r $END_MAIN '+%H:%M')); follow-up servers $AF (end $(date -r $END_FU '+%H:%M')); follow-up run $FU via end_main emnms$K"
