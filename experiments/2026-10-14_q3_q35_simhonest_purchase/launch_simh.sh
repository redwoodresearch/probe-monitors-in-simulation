#!/usr/bin/env bash
# Devbox side of one simulated-honest launch. A copy of the September scale-up's scaleup.sh @ bbf0729 with two changes:
# (1) --references-from <prior run> (the task's honest_a is reused, not rerun) and (2) no canary watcher (there are no
# real arms, so canary_watch.py's str_replace gate has nothing to open on). The launcher line, environment, interpreter,
# image list and key path are scaleup.sh line 20's: RR_OWNER_ID=aswin-surya rr-env run --env-file config/env.aws.tpl.
# Refuses to start if allin_stop.sh has hard-stopped this session (~/phase2/simh_checks/STOPPED_<session>).
# Usage (on the devbox): SESSION=<s> launch_simh.sh <run dir name> <pilot config> <comma-separated task keys>
#                                                   <comma-separated base URLs> <prior run dir name>
set -euo pipefail
SESSION=${SESSION:?set SESSION (a distinct tmux session per launch)}
[[ ! -e /home/aswin/phase2/simh_checks/STOPPED_$SESSION ]] || { echo "$SESSION was hard-stopped (allin_stop.sh marker): not starting" >&2; exit 1; }
NAME=$1; CFG=$2; TASKS=$3; BASES=$4; PRIOR=$5
C=/home/aswin/phase2/attack_check_simh
R=/home/aswin/phase2/runs/$NAME
P=/home/aswin/phase2/runs/$PRIOR
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
[[ ! -e $R ]] || { echo "$R exists" >&2; exit 1; }
[[ -f $P/launch.json ]] || { echo "no $P/launch.json" >&2; exit 1; }
[[ -d $C/results ]] || { echo "$C/results missing (the server checks write there)" >&2; exit 1; }
tmux -L phase2 has-session -t "=$SESSION" 2>/dev/null && { echo "launcher session $SESSION is already running" >&2; exit 1; }
nohup docker events --filter type=container --filter type=image --filter event=create --filter event=pull \
  --filter event=load --filter event=import --filter event=tag --filter event=untag --filter event=delete \
  --format '{{json .}}' > ${R}.docker_events.jsonl 2>&1 < /dev/null &
echo $! > ${R}.docker_events.pid
nohup bash -c "while true; do echo \"\$(date -Is) \$(free -m | awk '/Mem:/{print \$3, \$7}') \$(docker ps -q | wc -l) \$(cat /proc/loadavg | cut -d' ' -f1)\"; sleep 60; done" > ${R}.resources.log 2>&1 < /dev/null &
echo $! > ${R}.resources.pid
tmux -L phase2 new -d -s $SESSION "cd $C && PATH=$C/fixed_digest_lookup:\$PATH IMAGES_JSON=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json RR_OWNER_ID=aswin-surya rr-env run --env-file config/env.aws.tpl -- $PY $C/run_pilot.py --pilot-config $CFG --tasks $TASKS --base-urls $BASES --commit $(cat $C/LAUNCH_COMMIT) --approaches /home/aswin/phase2/approaches --references-from $P --out $R > ${R}.launcher.log 2>&1"
echo "launched $R (session $SESSION, references from $P)"
