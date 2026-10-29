#!/usr/bin/env bash
# Laptop side: the ~17:50 PT stop of launch 1 (2026-10-15_ep2_311, q3-scaleup4) under RULE.md amendment 3 (0d's authorization via d9,
# executed by the new-mains fork). On the box, ONE read of launch.json checks, by key, that every run whose task is at 0-based index
# < P (265) in launch 1's tasks file (sha 3b645674...) is terminal (state "done" or "no policy call", last attempt's "end" set) and
# that every run in state "running" is at index >= P. Only then it sends ONE SIGINT to the run_pilot.py process (pid from the tmux
# pane's child, matched by its command line), never to the wrapper script, so launch_ep2_311.sh goes on to gate_ep2_arms.py final.
# Otherwise it sends nothing and prints HOLD with the counts (ping d9). It prints the check time, the SIGINT time and the
# check-time in-flight run keys (all at >= P: excluded by the rule); the audit of the runs actually stopped is launch.json's
# stopped_by_launcher records, read after the launcher exits (Codex r72).
#   bash sigint_311.sh            -> CHECK only (no signal)
#   bash sigint_311.sh --send     -> check, then SIGINT if and only if the condition holds
set -uo pipefail
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
SEND=${1:-}
DEVBOX=aswin@i-0ad4140e97b11d30f perl -e 'alarm 120; exec @ARGV' bash "$SP/nm_launch/devbox.sh" "bash -s -- $SEND" <<'B'  # Codex r72: "--" so --send is an argument, not a bash option
SEND=${1:-}
R=/home/aswin/phase2/runs_ep2/2026-10-15_122516_ep2_311
T=/home/aswin/phase2/ep2_311/pilot_tasks_ep2_311_309.json
PID=$(pgrep -f 'run_pilot[.]py --pilot-config /home/aswin/phase2/ep2_311/pilot_honest_ep2_31[1].yaml')
[[ $(wc -w <<< "$PID") == 1 ]] || { echo "HOLD: run_pilot pid not unique ('$PID')"; exit 3; }
/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python - "$T" "$R/launch.json" <<'PY' || exit 3
import hashlib, json, sys
from datetime import datetime
from zoneinfo import ZoneInfo
raw = open(sys.argv[1], "rb").read()
assert hashlib.sha256(raw).hexdigest().startswith("3b645674937459b6")
idx = {c["key"]: i for i, c in enumerate(json.loads(raw)["cells"])}
P = 265
d = json.load(open(sys.argv[2]))
assert not d.get("finished"), "launch already finished: nothing to stop"
below = [r for r in d["runs"] if idx[r["task"]] < P]
open_below = [f"{r['task']}/{r['arm']}:{r['state']}" for r in below
              if r["state"] not in ("done", "no policy call") or not (r["attempts"] and r["attempts"][-1].get("end"))]
running = [r for r in d["runs"] if r["state"] == "running"]
bad_running = [f"{r['task']}/{r['arm']}" for r in running if idx[r["task"]] < P]
now = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%F %T PT")
print(f"check {now}: runs below P {len(below)}, not terminal {len(open_below)}; running {len(running)}, below P {len(bad_running)}")
if open_below or bad_running:
    print("HOLD: not terminal below P:", open_below[:20])
    sys.exit(3)
print("check-time in-flight runs, all at >= P (excluded; the stopped-run audit uses launch.json's stopped_by_launcher records after the stop):", sorted(f"{r['task']}/{r['arm']}" for r in running))
PY
if [[ $SEND == --send ]]; then
  kill -INT "$PID" && echo "SIGINT sent to run_pilot pid $PID at $(TZ=America/Los_Angeles date '+%F %T PT')"
else
  echo "CHECK only: condition holds; no signal sent (pid $PID)"
fi
B
