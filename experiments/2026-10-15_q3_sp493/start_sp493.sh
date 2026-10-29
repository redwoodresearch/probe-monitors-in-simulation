#!/usr/bin/env bash
# Laptop side: start S_paper493's real-arm launch on one box (start_311.sh @ 474c95fee with this box's folder and sessions),
# once stage_sp493.sh and prep_sp493.sh have run, 09 has acked the run list, the SPEND row is committed, and the box's 6
# server URLs exist. Starts the launcher (tmux -L phase2 session sp493<X>, under rr-env with the vLLM key only;
# launch_sp493.sh pins its inputs and runs the freeze gate before anything starts) and the ALERT-ONLY gate watch (session
# gatesp493<X>, deadline alert at now + 4 h), then prints the run folder, the pane PIDs and the start time. Refuses unless
# exactly 6 https URLs are given, the tasks sha is 64 hex, and neither session exists.
#   bash start_sp493.sh <A|B|C> <instance id> <comma-separated base URLs> <launch tasks sha256> [deadline alert, epoch s]
set -uo pipefail
X=${1:?box} I=${2:?instance} URLS=${3:?urls} TSHA=${4:?tasks sha} DLARG=${5:-}
[[ $X == A || $X == B || $X == C ]] || exit 2
[[ $URLS != *$'\n'* && $URLS != *$'\r'* ]] || { echo "URLs must not contain newlines"; exit 2; }  # Codex r2
[[ $TSHA =~ ^[0-9a-f]{64}$ ]] || { echo "tasks sha must be 64 hex"; exit 2; }
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
export DEVBOX=aswin@$I
box() { perl -e 'alarm shift; exec @ARGV' "$1" bash "$SP/nm_launch/devbox.sh" "$2" < /dev/null; }
[[ $(tr ',' '\n' <<< "$URLS" | grep -cE '^https://[A-Za-z0-9.-]+/v1$') == 6 && $(tr ',' '\n' <<< "$URLS" | wc -l) -eq 6 ]] || { echo "need exactly 6 URLs of the form https://<host>/v1"; exit 2; }  # Codex r1: no shell metacharacters
STAMP=$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S); RUN=/home/aswin/phase2/runs_sp493/${STAMP}_sp493_real_$X
DL=${DLARG:-$(( $(date +%s) + 14400 ))}  # 5th arg (09, box C): poweroff - 2 h (the per-run limit), so the alert fires while a run can still finish
[[ $DL =~ ^[0-9]+$ ]] || { echo "deadline must be epoch seconds"; exit 2; }
box 60 "! tmux -L phase2 has-session -t '=sp493$X' 2>/dev/null && ! tmux -L phase2 has-session -t '=gatesp493$X' 2>/dev/null && ! pgrep -f '[r]un_pilot.py --pilot-config' > /dev/null" || { echo "a session or launcher exists"; exit 2; }
box 60 "tmux -L phase2 new -d -s sp493$X 'cd ~/phase2/attack_check_311 && RR_OWNER_ID=aswin-surya rr-env run --env-file ~/phase2/sp493/env.vllm-only.tpl -- bash ~/phase2/sp493/launch_sp493.sh $X $URLS $RUN $TSHA'" || { echo "launcher start failed"; exit 1; }
sleep 20
box 60 "tmux -L phase2 new -d -s gatesp493$X '~/sims/simulated-control-evals-sept/.venv/bin/python ~/phase2/sp493/gate_sp493.py watch $RUN sp493$X $DL'" || echo "gate start failed"
sleep 5
box 60 "date '+box %F %T %Z'; TZ=America/Los_Angeles date '+%F %T PT'; echo RUN $RUN; tmux -L phase2 ls; for s in sp493$X gatesp493$X; do echo \"\$s pane pid \$(tmux -L phase2 list-panes -t =\$s -F '#{pane_pid}')\"; done; pgrep -af 'run_pilot.py --pilot-config /home/aswin/phase2/sp493/' | grep -v pgrep | cut -c1-140; tail -4 $RUN.launcher.log"
