#!/usr/bin/env bash
# Laptop side of a launch: stop what costs money as soon as the runs end, without anyone watching. Every minute it
# asks the devbox whether the launcher's tmux session still exists. Once it is gone (the runs ended, failed or were
# stopped) or the deadline passes, it stops the Modal apps (the pilot's two servers), copies the run folder, its
# <run>.launcher.log and the live timeout checks back, and stops the devbox. Run it in the background right after
# launching (one launcher, one tmux session: the pilot's run_pilot.py schedules every run of every arm):
#   DEVBOX=aswin@i-05b1d910fd70016e6 experiments/2026-10-07_phase2_attack_check/watch_and_stop.sh \
#       <modal app id>[,<app id>...] <run dir under ~ on the devbox, e.g. phase2/runs/<stamp>_pilot> <deadline in min>
set -uo pipefail  # not -e: one failed poll (an ssh blip) must not end the watcher
E=$(cd "$(dirname "$0")" && pwd)
IFS=, read -ra APPS <<< "$1"; RUN=$2 DEADLINE=$(($(date +%s) + $3 * 60))
INSTANCE=${DEVBOX:?set DEVBOX=<user>@<instance-id>}; INSTANCE=${INSTANCE#*@}
LOCAL="$E/../../.run_cache/phase2_attack_check/runs/$(basename "$RUN")"
log() { echo "$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') $*"; }

M=(uv run --no-project --python 3.12 --with modal==1.2.6 modal)
# Modal-side checks each minute (the devbox launcher has no Modal credentials): every watched app must run at most
# EXPECTED_CONTAINERS containers (1: each vLLM server has max_containers=1 and no retries); more means a duplicate is
# billing (q2's incident, 10-08: a retried input started a second H100 beside the first). On the first sighting it
# stops the newest containers beyond that (the original keeps serving the runs); if a duplicate is still there at the
# next check (two sightings in a row; a clean check resets the count), it stops the launcher (Ctrl+C: its runs are interrupted and recorded) and tears everything down. Each
# step is logged here and written into the run folder (watcher_stop_reason.txt). Every 10 minutes it also logs the
# apps' cost from Modal's billing report (full hours only, so it lags; the 1.2.6 client has no billing).
EXPECTED=${EXPECTED_CONTAINERS:-1}
ALARMS="$LOCAL.modal_alarms.log"
bounded() { perl -e 'alarm shift; exec @ARGV' "$@"; }  # macOS has no `timeout`: a hung CLI can't stall the watcher
modal_check() {
  bounded 90 "${M[@]}" container list --json 2>/dev/null | uv run --no-project --python 3.12 python -c "import collections, json, sys
apps, expected = sys.argv[1].split(','), int(sys.argv[2])
by_app = collections.defaultdict(list)
for c in json.load(sys.stdin):
    by_app[c['App ID']].append(c)
bad = [a for a in apps if len(by_app[a]) > expected]
# Oldest first. Start times are rounded to the second and the list comes newest first, so a tie goes to the one
# listed later (older).
order = {c['Container ID']: i for i, c in enumerate(c for a in by_app.values() for c in a)}
extra = [c['Container ID'] for a in bad
         for c in sorted(by_app[a], key=lambda c: (c['Start Time'], -order[c['Container ID']]))[expected:]]
print('ALARM duplicate containers: ' + '; '.join(f'{a}: {len(by_app[a])} (expected at most {expected})' for a in bad)
      + ' | extra: ' + ','.join(extra) if bad else 'containers ok: ' + ', '.join(f'{a}={len(by_app[a])}' for a in apps))" \
    "$(IFS=,; echo "${APPS[*]}")" "$EXPECTED"
}
billed() {
  bounded 120 uv run --no-project --python 3.12 --with modal==1.6.1 modal billing report --for today --resolution h \
    --tz America/Los_Angeles --json 2>/dev/null |
    uv run --no-project --python 3.12 python -c "import json, sys
apps = sys.argv[1].split(',')
rows = [r for r in json.load(sys.stdin) if r['object_id'] in apps]
print(f'billed today so far (full hours): \${sum(float(r[\"cost\"]) for r in rows):.2f} over {len(rows)} app-hours')" \
    "$(IFS=,; echo "${APPS[*]}")"
}
mkdir -p "$(dirname "$LOCAL")"
log "watching $RUN on $INSTANCE; apps ${APPS[*]}; deadline in $3 min"
POLL=0 STRIKES=0
alarm() {  # logged here, and written into the run folder (best effort: a hung ssh can't stall the watcher)
  log "$1"; log "$1" >> "$ALARMS"
  bounded 30 "$E/devbox.sh" "echo \"$(date '+%F %T %Z') $1\" >> $RUN/watcher_stop_reason.txt" 2>/dev/null
}
# Request-log rescue (10-08 pilot): servers started in the same second shared one log name on the Volume, where
# commits are last-write-wins, but each container keeps its own full stream locally. With RESCUE=1, every
# RESCUE_EVERY_S (and once more before any app is stopped) each watched container copies the file its own proxy is
# writing (read from the proxy's --log argument, so a replacement container can't pass off an inherited copy) to
# <that name>_<app id>.jsonl.gz, which only it writes and which only grows, and commits from inside. A copy taken while
# the proxy still writes lacks the gzip trailer: every record up to the copy is there; read it trailer-tolerantly.
RESCUE=${RESCUE:-0}
RESCUE_EVERY_S=${RESCUE_EVERY_S:-600}
REQUEST_LOG_VOLUME=probe-shift_aswin-surya_sub-phase2_request-log_kill-2026-11-06
rescue() {
  [[ "$RESCUE" == 1 ]] || return 0
  local pairs pair app ct out
  pairs=$(bounded 90 "${M[@]}" container list --json 2>/dev/null | uv run --no-project --python 3.12 python -c "import json, sys
apps = set(sys.argv[1].split(','))
print(' '.join(c['App ID'] + ':' + c['Container ID'] for c in json.load(sys.stdin) if c['App ID'] in apps))" \
    "$(IFS=,; echo "${APPS[*]}")") || { alarm "request-log rescue: container list failed"; return 1; }
  for pair in $pairs; do
    app=${pair%%:*}; ct=${pair##*:}
    out=$(bounded 240 "${M[@]}" container exec "$ct" -- python -c "import glob, os, shutil, modal
logs = [a[6:] for f in glob.glob('/proc/[0-9]*/cmdline') for a in open(f, 'rb').read().decode(errors='replace').split(chr(0))
        if a.startswith('--log=') and 'request_log_proxy' in open(f, 'rb').read().decode(errors='replace')]
assert len(set(logs)) == 1, f'proxy log: {sorted(set(logs))}'
src = logs[0]; dst = src.removesuffix('.jsonl.gz') + '_$app.jsonl.gz'
shutil.copy2(src, dst); modal.Volume.from_name('$REQUEST_LOG_VOLUME').commit(); print(os.path.basename(dst), os.path.getsize(dst))" 2>&1 | tail -1)
    if [[ "$out" =~ \ [0-9]+$ ]]; then log "request log rescued: $app ($ct) -> $out"
    else alarm "request-log rescue FAILED for $app ($ct): $out"; fi
  done
}
LAST_RESCUE=0
while (($(date +%s) < DEADLINE)); do
  if (($(date +%s) - LAST_RESCUE >= RESCUE_EVERY_S)); then rescue; LAST_RESCUE=$(date +%s); fi
  line=$(modal_check) || line="modal container check failed"
  [[ $line == *failed ]] && { log "$line"; log "$line" >> "$ALARMS"; }
  if [[ $line == ALARM* ]]; then
    ((STRIKES++))
    if ((STRIKES == 1)); then  # stop only the extra containers; the oldest of each app keeps serving
      alarm "$line; stopping the extra containers"
      for c in $(echo "${line##*extra: }" | tr ',' ' '); do bounded 60 "${M[@]}" container stop "$c" || alarm "could not stop $c"; done
    else  # still duplicated: stop the launcher (its runs are interrupted and recorded), then tear everything down
      alarm "$line again after stopping the extras; stopping the launcher and every app"
      bounded 30 "$E/devbox.sh" 'tmux -L phase2 send-keys -t run C-c' 2>/dev/null
      GRACE=$(($(date +%s) + 180))  # for the launcher to interrupt its runs and write launch.json
      while (($(date +%s) < GRACE)); do
        bounded 30 "$E/devbox.sh" 'tmux -L phase2 has-session -t run' 2>/dev/null
        (($? == 1)) && break  # the session has ended; any other code (ssh or the bound failing): keep waiting
        sleep 10
      done
      break
    fi
  elif [[ $line == "containers ok"* ]]; then
    STRIKES=0  # escalation needs two duplicate sightings in a row; a failed check (unknown) leaves the count as is
  fi
  ((POLL++ % 10 == 0)) && log "$line; $(billed || echo 'billing report unavailable')"
  bounded 60 "$E/devbox.sh" 'tmux -L phase2 has-session -t run' 2>/dev/null
  case $? in
    0) sleep 60 ;;                                   # still running
    1) log "the launcher's session has ended"; break ;;
    *) log "devbox unreachable this minute"; sleep 60 ;;  # ssh's 255: try again, until the deadline
  esac
done
(($(date +%s) >= DEADLINE)) && log "deadline reached"
app_stopped() {  # not listed as live: `modal app list` shows recently stopped apps as "stopped", then drops them
  bounded 120 "${M[@]}" app list --json | uv run --no-project --python 3.12 python -c "import json, sys
sys.exit(1 if any(x['App ID'] == sys.argv[1] and x['State'] != 'stopped' for x in json.load(sys.stdin)) else 0)" "$1"
}
box_state() { aws ec2 describe-instances --profile redwood-sandbox --region us-west-2 --instance-ids "$INSTANCE" \
  --query 'Reservations[].Instances[].State.Name' --output text; }
FAILED=()
stop_app() {  # stop and confirm; a deployed app (launch_servers.py --server-mode deployed) never stops by itself
  for i in 1 2 3 4 5; do
    log "stopping Modal app $1 (try $i)"
    bounded 120 "${M[@]}" app stop "$1" && app_stopped "$1" && return 0
    sleep 30
  done
  app_stopped "$1" && return 0
  FAILED+=("Modal app $1")
  log "ALARM: Modal app $1 is not confirmed stopped; stop it by hand (modal app stop $1)"
  log "ALARM: Modal app $1 is not confirmed stopped" >> "$ALARMS"
  return 1
}
rescue  # the last copy of each server's request log, before anything is stopped
for APP in "${APPS[@]}"; do stop_app "$APP"; done  # the GPUs first: they are the expensive part
log "copying $RUN back to $LOCAL"
mkdir -p "$LOCAL"
"$E/devbox.sh" rsync -az "box:$RUN/" "$LOCAL/" || FAILED+=("copy of $RUN")
"$E/devbox.sh" rsync -az "box:$RUN.launcher.log" "$LOCAL/launcher.log" ||  # the launch writes its log beside its run
  FAILED+=("copy of $RUN.launcher.log")
"$E/devbox.sh" rsync -az 'box:phase2/attack_check/results/timeout_check_*' "$E/results/"
for i in 1 2 3 4 5; do
  log "stopping devbox $INSTANCE (try $i)"
  aws ec2 stop-instances --profile redwood-sandbox --region us-west-2 --instance-ids "$INSTANCE" >/dev/null
  [[ $(box_state) =~ stopping|stopped ]] && break
  sleep 30
done
[[ $(box_state) =~ stopping|stopped ]] || FAILED+=("devbox $INSTANCE")
if ((${#FAILED[@]})); then
  log "FAILED, stop or copy by hand: ${FAILED[*]}"
  exit 1
fi
log "done: apps stopped, devbox $(box_state)"
