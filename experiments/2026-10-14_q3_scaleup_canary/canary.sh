#!/usr/bin/env bash
# The canary, devbox side (RULE.md): scaleup.sh @ bbf0729 with the harness folder, images manifest and sessions as
# arguments. Docker event log, resource log, then the launcher (tmux -L phase2 session $SESSION) under rr-env with the
# vLLM and OpenAI keys (config/env.aws.tpl: the devbox machine role reads the owner's vault; no value in argv).
# It first runs prelaunch_gates.sh (freeze manifest, preflight, approach and template, settings equality) and starts
# nothing unless every gate passes.
# Usage: SESSION=<name> HARNESS=<harness dir> canary.sh <run dir name> <pilot config> <base URLs> <absolute images.json>
set -euo pipefail
SESSION=${SESSION:?}
NAME=$1; CFG=$2; BASES=$3; IMAGES=$4
C=${HARNESS:?}
R=/home/aswin/phase2/runs/$NAME
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
[[ $IMAGES == /* && -s $IMAGES ]] || { echo "images manifest $IMAGES missing or not absolute" >&2; exit 1; }
[[ ! -e $R ]] || { echo "$R exists" >&2; exit 1; }
tmux -L phase2 has-session -t "$SESSION" 2>/dev/null && { echo "launcher session $SESSION is already running" >&2; exit 1; }
mkdir -p /home/aswin/phase2/runs
bash "$C/prelaunch_gates.sh" "$C" "$CFG" "$C/config/harness_manifest.tsv" || { echo "pre-launch gates failed: nothing launched" >&2; exit 1; }
nohup docker events --filter type=container --filter type=image --filter event=create --filter event=pull \
  --filter event=load --filter event=import --filter event=tag --filter event=untag --filter event=delete \
  --format '{{json .}}' > "${R}.docker_events.jsonl" 2>&1 < /dev/null &
echo $! > "${R}.docker_events.pid"
nohup bash -c "while true; do echo \"\$(date -Is) \$(free -m | awk '/Mem:/{print \$3, \$7}') \$(docker ps -q | wc -l) \$(cat /proc/loadavg | cut -d' ' -f1)\"; sleep 60; done" > "${R}.resources.log" 2>&1 < /dev/null &
echo $! > "${R}.resources.pid"
tmux -L phase2 new -d -s "$SESSION" "cd $C && PATH=$C/fixed_digest_lookup:\$PATH IMAGES_JSON=$IMAGES RR_OWNER_ID=aswin-surya rr-env run --env-file config/env.aws.tpl -- $PY $C/run_pilot.py --pilot-config $CFG --base-urls $BASES --commit \$(cat $C/LAUNCH_COMMIT) --approaches /home/aswin/phase2/approaches --out $R > ${R}.launcher.log 2>&1"
echo "launched $R in session $SESSION"
