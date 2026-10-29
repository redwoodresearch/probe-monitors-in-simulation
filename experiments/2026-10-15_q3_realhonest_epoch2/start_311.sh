#!/usr/bin/env bash
# Laptop side: start the +311 epoch-2 launch on q3-scaleup4 (RULE.md amendment 1) once stage_ep2_311.sh has run and 0d's 4 server URLs
# exist. start_t196.sh with the +311 box, sessions and staged folder.
#   bash start_311.sh <comma-separated base URLs>
# Starts the launcher (tmux -L phase2 session ep2s311, under rr-env with the vLLM key only; launch_ep2_311.sh pins its inputs and
# runs the freeze gate before anything starts) and the ALERT-ONLY gate watch (session gateep2s311, deadline alert at now + 4.5 h),
# then prints the run folder, the sessions' pane PIDs and the start time. Refuses unless exactly 4 https URLs are given and
# neither session exists.
set -uo pipefail
URLS=${1:?urls}
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
export DEVBOX=aswin@i-0ad4140e97b11d30f
box() { perl -e 'alarm shift; exec @ARGV' "$1" bash "$SP/nm_launch/devbox.sh" "$2" < /dev/null; }
[[ $(tr ',' '\n' <<< "$URLS" | grep -c '^https://') == 4 && $(tr ',' '\n' <<< "$URLS" | wc -l) -eq 4 ]] || { echo "need exactly 4 https URLs"; exit 2; }
STAMP=$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S); RUN=/home/aswin/phase2/runs_ep2/${STAMP}_ep2_311
DL=$(( $(date +%s) + 16200 ))
box 60 "! tmux -L phase2 has-session -t '=ep2s311' 2>/dev/null && ! tmux -L phase2 has-session -t '=gateep2s311' 2>/dev/null" || { echo "a session exists"; exit 2; }
box 60 "tmux -L phase2 new -d -s ep2s311 'cd ~/phase2/attack_check_311 && RR_OWNER_ID=aswin-surya rr-env run --env-file ~/phase2/ep2_311/env.vllm-only.tpl -- bash ~/phase2/ep2_311/launch_ep2_311.sh $URLS $RUN'" || { echo "launcher start failed"; exit 1; }
sleep 20
box 60 "tmux -L phase2 new -d -s gateep2s311 '~/sims/simulated-control-evals-sept/.venv/bin/python ~/phase2/ep2_311/gate_ep2_arms.py watch $RUN ep2s311 $DL'" || echo "gate start failed"
sleep 5
box 60 "date '+box %F %T %Z'; TZ=America/Los_Angeles date '+%F %T PT'; echo RUN $RUN; tmux -L phase2 ls; for s in ep2s311 gateep2s311; do echo \"\$s pane pid \$(tmux -L phase2 list-panes -t =\$s -F '#{pane_pid}')\"; done; pgrep -af 'run_pilot.py --pilot-config /home/aswin/phase2/ep2_311/pilot_honest_ep2_311.yaml' | grep -v pgrep | cut -c1-140; tail -4 $RUN.launcher.log"
