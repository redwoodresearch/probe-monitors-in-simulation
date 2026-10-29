#!/usr/bin/env bash
# ug_ready_check.sh (RULE.md): ON q3-xbox3, one READ-ONLY readiness check for the ungrounded launch, under the box launch
# lock (taken non-blocking, released on exit): s2's record finished, its session (tmux -L nm ilnms2) gone, no
# run_pilot.py, no PAUSE_SIMS in the copy or the frozen harness. Prints READY and exits 0, else prints why and exits 1.
#   bash ~/phase2/ug_stage/ug_ready_check.sh
set -uo pipefail
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
PREV=/home/aswin/phase2/nm/runs/2026-10-14_il_s2
no() { echo "NOT READY: $*"; exit 1; }
exec 9> /home/aswin/phase2/.launch.lock || no "cannot open the launch lock"
flock -n 9 || no "the launch lock is held"
"$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('finished') else 1)" "$PREV/launch.json" || no "s2 not finished"
out=$(tmux -L nm ls -F '#S' 2>&1); rc=$?
if (( rc == 0 )); then [[ $'\n'"$out"$'\n' != *$'\n'ilnms2$'\n'* ]] || no "ilnms2 live"
elif [[ $out != *"no server running"* && $out != "error connecting to /tmp/tmux-$UID/nm (No such file or directory)" ]]; then no "tmux -L nm: $out"; fi
live=$(pgrep -af 'run_pilot\.py'); rc=$?
(( rc == 1 )) || no "a run_pilot.py is running, or pgrep failed (rc $rc): ${live:0:200}"
for f in /home/aswin/phase2/attack_check_ug/results/PAUSE_SIMS /home/aswin/phase2/attack_check_311/results/PAUSE_SIMS; do
  [[ ! -e $f && ! -L $f ]] || no "$f exists"
done
echo "READY: s2 finished and gone, no launcher, no PAUSE_SIMS"
