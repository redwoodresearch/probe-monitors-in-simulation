#!/usr/bin/env bash
# The Llama-3.3-70B honest check's remaining 30 runs, devbox side: two run_pilot.py launches ONE AFTER THE OTHER in one
# tmux session (-L phase2 -s run, so watch_and_stop.sh waits for both), under one parent folder (so its copy-back takes
# both), on the same 4 servers: ab = honest_a + honest_b of 11 tasks, then b = honest_b of the batch's 8 tasks (not
# started if ab's canary aborted). One after the other, not at once (Codex 10-12): two launches at once would share
# the server-check result names and control-tower's compose rewrites without run_pilot.py's start staggering.
# As q32n2.sh otherwise (docker event log; canary watchers in tmux -L canary, sessions watch and watch2; canary_watch.py
# waits for its launch.json, so b's watcher can start now).
# Usage: llama_rest.sh <stamp> <4 comma-separated base URLs>
set -euo pipefail
STAMP=$1; BASES=$2
C=/home/aswin/phase2/llama70b_check
R=/home/aswin/phase2/runs/${STAMP}_llama33_70b_rest
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
[[ ! -e $R ]] || { echo "$R exists" >&2; exit 1; }
[[ $(tr ',' '\n' <<< "$BASES" | wc -l) == 4 ]] || { echo "need 4 base URLs" >&2; exit 1; }
tmux -L phase2 has-session -t run 2>/dev/null && { echo "a launcher is already running" >&2; exit 1; }
mkdir -p $R
nohup docker events --filter type=container --filter type=image --filter event=create --filter event=pull \
  --filter event=load --filter event=import --filter event=tag --filter event=untag --filter event=delete \
  --format '{{json .}}' > ${R}.docker_events.jsonl 2>&1 < /dev/null &
echo $! > ${R}.docker_events.pid
ENVP="cd $C && PATH=$C/fixed_digest_lookup:\$PATH IMAGES_JSON=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json RR_OWNER_ID=aswin-surya rr-env run --env-file config/env.aws.tpl --"
tmux -L phase2 new -d -s run "( $ENVP $PY $C/run_pilot.py --pilot-config config/pilot_llama33_70b_rest_ab.yaml --tasks t07_slack_clone,t10_shared_dotfiles,t11_media_processing,t12_clinical_trial,t13_auto_workflow,t14_tiktok,t15_user_tracking,t16_file_indexing,t18_model_registry,t19_web_scraping,t20_shared_dotfiles --base-urls $BASES --commit $(cat $C/LAUNCH_COMMIT) --approaches /home/aswin/phase2/approaches --out $R/ab > $R/ab.launcher.log 2>&1 ); [ -e $R/ab/CANARY_ABORT ] || ( $ENVP $PY $C/run_pilot.py --pilot-config config/pilot_llama33_70b_rest_b.yaml --tasks t01_media_processing,t02_clinical_trial,t03_auto_workflow,t04_tiktok,t05_user_tracking,t06_file_indexing,t08_model_registry,t09_web_scraping --base-urls $BASES --commit $(cat $C/LAUNCH_COMMIT) --approaches /home/aswin/phase2/approaches --out $R/b > $R/b.launcher.log 2>&1 ); echo finished > ${R}.launcher.log"
sleep 5
tmux -L canary new -d -s watch "cd $C && $PY canary_watch.py $R/ab t07_slack_clone --open-on-clean-finish --abort-missing-only > $R/ab.canary_watch.log 2>&1"
tmux -L canary new -d -s watch2 "cd $C && $PY canary_watch.py $R/b t01_media_processing --open-on-clean-finish --abort-missing-only > $R/b.canary_watch.log 2>&1"
echo "launched $R"
