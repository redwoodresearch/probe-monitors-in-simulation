#!/usr/bin/env bash
# The gpt_oss_120b acting pilot's smoke (t17's 2 runs), devbox side, as the Seed pilot's seed_smoke.sh: docker
# event log, the launcher (tmux -L phase2g session run), the canary watcher (tmux -L canaryg session watch).
# Usage: gptoss_smoke.sh <stamp> <base URL>
set -euo pipefail
STAMP=$1; BASES=$2
C=/home/aswin/phase2/gptoss_check
R=/home/aswin/phase2/runs/${STAMP}_gpt_oss_120b_smoke
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
[[ ! -e $R ]] || { echo "$R exists" >&2; exit 1; }
tmux -L phase2g has-session -t run 2>/dev/null && { echo "a launcher is already running" >&2; exit 1; }
nohup docker events --filter type=container --filter type=image --filter event=create --filter event=pull \
  --filter event=load --filter event=import --filter event=tag --filter event=untag --filter event=delete \
  --format '{{json .}}' > ${R}.docker_events.jsonl 2>&1 < /dev/null &
echo $! > ${R}.docker_events.pid
tmux -L phase2g new -d -s run "cd $C && PATH=$C/fixed_digest_lookup:\$PATH IMAGES_JSON=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json RR_OWNER_ID=aswin-surya rr-env run --env-file config/env.aws.tpl -- $PY $C/run_pilot.py --pilot-config config/pilot_gpt_oss_120b_smoke.yaml --tasks t17_slack_clone --base-urls $BASES --commit $(cat $C/LAUNCH_COMMIT) --approaches /home/aswin/phase2/approaches --out $R > ${R}.launcher.log 2>&1"
tmux -L canaryg new -d -s watch "cd $C && $PY canary_watch.py $R t17_slack_clone --open-on-clean-finish --abort-missing-only > ${R}.canary_watch.log 2>&1"
echo "launched $R"
