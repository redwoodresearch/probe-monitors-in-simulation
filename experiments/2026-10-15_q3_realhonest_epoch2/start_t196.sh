#!/usr/bin/env bash
# Laptop side: start the T196 epoch-2 launch on q3-realhonest (RULE.md; Codex r61 AGREE @ 88bdf807e) once 0d's 4 server URLs exist.
#   bash start_t196.sh <comma-separated base URLs>
# Starts the launcher (tmux -L phase2 session ep2t196, under rr-env with the vLLM key only) and the ALERT-ONLY gate watch (session
# gateep2t196, deadline alert at now + 4.5 h), then prints the run folder, the sessions' pane PIDs and the start time. Refuses
# unless exactly 4 https URLs are given and neither session exists.
set -uo pipefail
URLS=${1:?urls}
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
export DEVBOX=aswin@i-0abea93b8ad99ee41
box() { perl -e 'alarm shift; exec @ARGV' "$1" bash "$SP/nm_launch/devbox.sh" "$2" < /dev/null; }
[[ $(tr ',' '\n' <<< "$URLS" | grep -c '^https://') == 4 && $(tr ',' '\n' <<< "$URLS" | wc -l) -eq 4 ]] || { echo "need exactly 4 https URLs"; exit 2; }
STAMP=$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S); RUN=/home/aswin/phase2/runs_ep2/${STAMP}_ep2_t196
DL=$(( $(date +%s) + 16200 ))
box 60 "! tmux -L phase2 has-session -t '=ep2t196' 2>/dev/null && ! tmux -L phase2 has-session -t '=gateep2t196' 2>/dev/null" || { echo "a session exists"; exit 2; }
box 60 "tmux -L phase2 new -d -s ep2t196 'cd ~/phase2/attack_check_rh && RR_OWNER_ID=aswin-surya rr-env run --env-file config/env.vllm-only.tpl -- bash launch_ep2_t196.sh $URLS $RUN'" || { echo "launcher start failed"; exit 1; }
sleep 20
box 60 "tmux -L phase2 new -d -s gateep2t196 '~/sims/simulated-control-evals-sept/.venv/bin/python ~/phase2/attack_check_rh/gate_ep2.py watch $RUN ep2t196 $DL'" || echo "gate start failed"
sleep 5
box 60 "date '+box %F %T %Z'; TZ=America/Los_Angeles date '+%F %T PT'; echo RUN $RUN; tmux -L phase2 ls; for s in ep2t196 gateep2t196; do echo \"\$s pane pid \$(tmux -L phase2 list-panes -t =\$s -F '#{pane_pid}')\"; done; pgrep -af 'run_pilot.py --pilot-config config/pilot_honest_ep2_t196.yaml' | grep -v pgrep | cut -c1-140; tail -3 $RUN.launcher.log"
