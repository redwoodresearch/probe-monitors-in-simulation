#!/usr/bin/env bash
# The qwen3_coder_next acting pilot's full run (the other 38 runs, 19 tasks), devbox side, as the Seed pilot's seed_full.sh: docker
# event log, the launcher (tmux -L phase2q session run), the canary watcher (tmux -L canaryq session watch).
# Usage: qcoder_full.sh <stamp> <comma-separated base URLs>
set -euo pipefail
STAMP=$1; BASES=$2
C=/home/aswin/phase2/qcoder_check
R=/home/aswin/phase2/runs/${STAMP}_qwen3_coder_next
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
[[ ! -e $R ]] || { echo "$R exists" >&2; exit 1; }
tmux -L phase2q has-session -t run 2>/dev/null && { echo "a launcher is already running" >&2; exit 1; }
nohup docker events --filter type=container --filter type=image --filter event=create --filter event=pull \
  --filter event=load --filter event=import --filter event=tag --filter event=untag --filter event=delete \
  --format '{{json .}}' > ${R}.docker_events.jsonl 2>&1 < /dev/null &
echo $! > ${R}.docker_events.pid
tmux -L phase2q new -d -s run "cd $C && PATH=$C/fixed_digest_lookup:\$PATH IMAGES_JSON=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json RR_OWNER_ID=aswin-surya rr-env run --env-file config/env.aws.tpl -- $PY $C/run_pilot.py --pilot-config config/pilot_qwen3_coder_next.yaml --tasks t01_media_processing,t02_clinical_trial,t03_auto_workflow,t04_tiktok,t05_user_tracking,t06_file_indexing,t07_slack_clone,t08_model_registry,t09_web_scraping,t10_shared_dotfiles,t11_media_processing,t12_clinical_trial,t13_auto_workflow,t14_tiktok,t15_user_tracking,t16_file_indexing,t18_model_registry,t19_web_scraping,t20_shared_dotfiles --base-urls $BASES --commit $(cat $C/LAUNCH_COMMIT) --approaches /home/aswin/phase2/approaches --out $R > ${R}.launcher.log 2>&1"
tmux -L canaryq new -d -s watch "cd $C && $PY canary_watch.py $R t01_media_processing --open-on-clean-finish --abort-missing-only > ${R}.canary_watch.log 2>&1"
echo "launched $R"
