#!/usr/bin/env bash
# Fresh-task honest runs, devbox side (experiments/2026-10-13_q2_fresh_honest/PLAN.md; a copy of launch_zoo_c.sh): one stage (smoke, parity, 1..7), with the
# docker event and resource logs, the launcher (tmux -L phase2 session zc) and policy_spend_watch.py (tmux -L canary
# session fhw) at the pilot config's policy_stop_usd. The launcher starts no run without the watcher's fresh reading.
# Usage: launch_fresh.sh <run dir name> <stage: smoke|parity|1..7> [pilot config, default config/pilot_fresh_honest_parity.yaml]
#   [worst-case $ per run the watcher reserves for each running run, default 2.0; full stages pass the largest per-run cost seen so far]
set -euo pipefail
NAME=$1; STAGE=$2; CFG=${3:-config/pilot_fresh_honest_parity.yaml}; WORST=${4:-2.0}
C=/home/aswin/phase2/attack_check_fresh
R=/home/aswin/phase2/runs/$NAME
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
TASKS_FILE=$C/config/pilot_tasks_fresh_honest.json
[[ $(sha256sum $TASKS_FILE | cut -d' ' -f1) == $(cat $C/TASKS_SHA256) ]] ||
  { echo "the task file is not the committed one (TASKS_SHA256)" >&2; exit 1; }
[[ ! -e $R ]] || { echo "$R exists" >&2; exit 1; }
LC=$(cat $C/LAUNCH_COMMIT 2>/dev/null || true)
[[ $LC =~ ^[0-9a-f]{8,40}$ ]] || { echo "LAUNCH_COMMIT is missing or not a commit: write it after the rsync" >&2; exit 1; }
for s in "phase2 fh" "canary fhw"; do
  tmux -L ${s% *} has-session -t ${s#* } 2>/dev/null && { echo "tmux session ${s#* } is already running" >&2; exit 1; }
done
STOP=$($PY -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['policy_stop_usd'])" $C/$CFG)
TASKS=$($PY -c "import json, sys; print(','.join(json.load(open(sys.argv[1]))['stages'][sys.argv[2]]))" $TASKS_FILE "$STAGE")
nohup docker events --filter type=container --filter type=image --filter event=create --filter event=pull \
  --filter event=load --filter event=import --filter event=tag --filter event=untag --filter event=delete \
  --format '{{json .}}' > ${R}.docker_events.jsonl 2>&1 < /dev/null &
echo $! > ${R}.docker_events.pid
nohup bash -c "while true; do echo \"\$(date -Is) \$(free -m | awk '/Mem:/{print \$3, \$7}') \$(docker ps -q | wc -l) \$(cat /proc/loadavg | cut -d' ' -f1)\"; sleep 60; done" > ${R}.resources.log 2>&1 < /dev/null &
echo $! > ${R}.resources.pid
tmux -L phase2 new -d -s fh "cd $C && PATH=$C/fixed_digest_lookup:\$PATH IMAGES_JSON=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json RR_OWNER_ID=aswin-surya rr-env run --env-file config/env.aws.tpl -- $PY $C/run_pilot.py --pilot-config $CFG --tasks $TASKS --base-urls none --commit $LC --approaches /home/aswin/phase2/approaches --out $R > ${R}.launcher.log 2>&1; kill $(cat ${R}.docker_events.pid) $(cat ${R}.resources.pid)"  # loggers end with it
for i in $(seq 1 60); do  # the launcher's own Python process (anchored: tmux and rr-env carry its command line too)
  PID=$(pgrep -f "^$PY $C/run_pilot.py" || true); [[ -n $PID ]] && break; sleep 1
done
stop() {  # the launcher, and the watcher if it started, so a retry isn't blocked by a leftover session
  tmux -L phase2 send-keys -t fh C-c 2>/dev/null || true; tmux -L canary kill-session -t fhw 2>/dev/null || true
  echo "$1: launcher and watcher stopped" >&2; exit 1
}
[[ -n $PID && $PID != *$'\n'* ]] || stop "launcher pid not found (or more than one)"
tmux -L canary new -d -s fhw "cd $C && $PY policy_spend_watch.py --run $R --launcher-pid $PID --stop-usd $STOP --worst-per-run $WORST > ${R}.policy_spend_watch.log 2>&1"
for i in $(seq 1 120); do [[ -s $R/policy_spend.json ]] && break; sleep 1; done
[[ -s $R/policy_spend.json ]] || stop "no reading from policy_spend_watch.py in 120 s (see ${R}.policy_spend_watch.log)"
echo "launched $R (stage $STAGE, launcher $PID, API stop \$$STOP, worst per run \$$WORST)"
