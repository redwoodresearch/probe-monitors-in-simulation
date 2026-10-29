#!/usr/bin/env bash
# Laptop side, RULE.md Amendment 4: swap one running split to the pause-never-kill checks. Copies the checks to the box,
# then ON THE BOX: (1) refuses unless end_main (emnms<K>) is still in its step 1 (its last line is a "main: ... not yet
# terminal" pass; nothing is handed over yet); (2) the main gate: stops gate1_311b.py's session if one is live, starts
# gate1_nm.py in gbnms<K> with the same deadline; (3) end_main: stops the old emnms<K> and starts the new end_main.py with
# the SAME arguments, read from the old process. Never an edit of a running script. Prints what it did, with times.
#   K=<split> DEVBOX=<user>@<id> E=<dir with devbox.sh> swap_gates.sh
set -euo pipefail
echo "swap_gates.sh is RETIRED: run once per split at 20:34-20:35 PT 10-14 (RULE.md Amendment 4); Codex r7 found" \
  "race and session-binding gaps that make it unsafe to reuse" >&2; exit 1
set -euo pipefail
: "${K:?}" "${DEVBOX:?}" "${E:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
perl -e 'alarm 120; exec @ARGV' bash "$E/devbox.sh" rsync -a "$HERE/checks/end_main.py" "$HERE/checks/gate1_nm.py" \
  "$HERE/checks/gate_pause_nm.py" "$HERE/checks/sim_cap_stop.py" "$HERE/checks/spend_now.py" box:phase2/nm/checks/
perl -e 'alarm 300; exec @ARGV' bash "$E/devbox.sh" "bash -s -- $K" <<'EOF_REMOTE'
set -euo pipefail
K=$1; C=/home/aswin/phase2/nm/checks; PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
TREE=/home/aswin/phase2/sims-sept-strict-off/src; RUNP=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s$K
t() { TZ=America/Los_Angeles date '+%F %T PT'; }
last=$(tail -1 $C/end_main_s$K.log)
[[ $last == *"real runs not yet terminal"* ]] || { echo "S$K REFUSED: end_main is not in step 1: $last"; exit 1; }
old=$(pgrep -af "^$PY $C/end_main.py $K $RUNP " || true)  # anchored: the python process, never a tmux or shell line
[[ $(wc -l <<< "$old") == 1 && -n $old ]] || { echo "S$K REFUSED: expected one end_main process: $old"; exit 1; }
ARGS=$(sed "s|.*end_main.py ||" <<< "$old")
read -r a1 a2 a3 a4 a5 a6 <<< "$ARGS"
[[ $a1 == "$K" && $a2 == "$RUNP" && $a3 == "${RUNP}_fu" && $a6 == iii && $a5 =~ ^[0-9]+$ ]] || { echo "S$K REFUSED: args $ARGS"; exit 1; }
STOP_AT=$(( a5 - 5 * 3600 - 600 ))  # the main servers' end - 10 min (launch_nm.sh: END_FU = S + 10 h, END_MAIN = S + 5 h)
gold=$(pgrep -af "^$PY gate1_311b.py $RUNP " || true)
if [[ -n $gold ]]; then
  old_at=$(awk '{print $NF}' <<< "$gold"); [[ $old_at == "$STOP_AT" ]] || { echo "S$K REFUSED: deadline $old_at != $STOP_AT"; exit 1; }
fi
tmux -L nm kill-session -t "=gbnms$K" 2>/dev/null && echo "$(t) S$K old gate session gbnms$K stopped" || echo "$(t) S$K no old gate session"
sleep 1; ! pgrep -f "^$PY gate1_311b.py $RUNP " >/dev/null || { echo "S$K gate1_311b.py still running"; exit 1; }
tmux -L nm new -d -s "gbnms$K" "cd $C && PYTHONPATH=$TREE $PY gate1_nm.py $RUNP snms$K $STOP_AT main"
echo "$(t) S$K gate1_nm.py started (gbnms$K, deadline $STOP_AT = $(TZ=America/Los_Angeles date -d @$STOP_AT '+%T PT'))"
tmux -L nmguards kill-session -t "=emnms$K"; sleep 1
! pgrep -f "^$PY $C/end_main.py $K $RUNP " >/dev/null || { echo "S$K old end_main still running"; exit 1; }
echo "$(t) S$K old end_main stopped (emnms$K)"
tmux -L nmguards new -d -s "emnms$K" "$PY $C/end_main.py $ARGS"
sleep 5
tmux -L nm has-session -t "=gbnms$K" && tmux -L nmguards has-session -t "=emnms$K" && tmux -L nm has-session -t "=snms$K"
echo "$(t) S$K new end_main started (emnms$K: $ARGS); sessions live: snms$K gbnms$K emnms$K"
tail -2 $C/end_main_s$K.log; tail -1 $RUNP/gate1_watch.log
EOF_REMOTE
