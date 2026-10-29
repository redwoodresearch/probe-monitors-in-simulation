#!/usr/bin/env bash
# Laptop side: the stop of +311 launch 1 (2026-10-15_ep2_311, q3-scaleup4) under RULE.md amendment 3 (09 @ e00eff46d; record
# @ 91feb3525), as soon as the by-key condition holds (0d via d9 ~15:2x). v2 of sigint_311.sh with 0d's guards:
#   1. the by-key check runs on the box immediately before the signal, in the same box shell, and its full output is kept;
#   2. ONE pid: launch 1's run_pilot.py pid from its own start record (results/start_311_2026-10-15.json: 10155), verified by its
#      /proc cmdline naming launch 1's config and run dir (no pattern search);
#   3. a by-key snapshot of every run at 0-based index < P (265) is taken before the signal and again after the launcher has
#      exited; 0 changed is required; the before, the after, the diff, the SIGINT time and the stopped runs' keys (launch.json's
#      stopped_by_launcher attempts, read after exit) go into one record;
#   4. the condition failing means HOLD: nothing is sent, and there is no retry loop.
# The condition (the record's): every run at index < P is terminal (state "done" or "no policy call", its last attempt's "end"
# set) and every run in state "running" is at index >= P.
#   bash sigint_311_v2.sh            -> CHECK only (prints the check; no signal)
#   bash sigint_311_v2.sh --send     -> check, SIGINT if and only if it passes, wait for exit, snapshot again, write the record
set -uo pipefail
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
HERE=$(cd "$(dirname "$0")" && pwd)
SEND=${1:-}
export DEVBOX=aswin@i-0ad4140e97b11d30f
PID=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['pids_on_box']['run_pilot.py'])" "$HERE/results/start_311_2026-10-15.json") || exit 3
[[ $PID =~ ^[0-9]+$ ]] || { echo "HOLD: no pid in the start record"; exit 3; }
STAMP=$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S)
OUT=$SP/ep2/sigint_311_$STAMP; mkdir -p "$OUT"
perl -e 'alarm 60; exec @ARGV' bash "$SP/nm_launch/devbox.sh" rsync -a "$HERE/sigint_311_snap.py" box:/tmp/sigint_311_snap.py || { echo "HOLD: cannot copy the snapshot script"; exit 3; }
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
box() { perl -e 'alarm 120; exec @ARGV' bash "$SP/nm_launch/devbox.sh" "bash -s -- $*"; }  # the args are pids or --send: no spaces
# 1-2: the check and (with --send) the signal, in ONE box shell
box "$PID" "$SEND" > "$OUT/before.txt" 2>&1 <<B
PID=\$1 SEND=\${2:-}
TZ=America/Los_Angeles date '+check %F %T PT'
CMD=\$(tr '\0' ' ' < /proc/\$PID/cmdline 2>/dev/null)
[[ \$CMD == *run_pilot.py*--pilot-config\ /home/aswin/phase2/ep2_311/pilot_honest_ep2_311.yaml*--out\ /home/aswin/phase2/runs_ep2/2026-10-15_122516_ep2_311* ]] ||
  { echo "HOLD: pid \$PID is not launch 1's run_pilot.py ('\${CMD:0:120}')"; exit 3; }
CHECK_MIN=\$(TZ=America/Los_Angeles date '+%F %H:%M')
BJ=/tmp/sigint_311_before_\$\$.json
$PY /tmp/sigint_311_snap.py "\$BJ" --check; crc=\$?
cat "\$BJ" && echo || { echo "HOLD: snapshot not printable"; exit 3; }
(( crc == 0 )) || { echo HOLD; exit 3; }
if [[ \$SEND == --send ]]; then
  [[ \$(TZ=America/Los_Angeles date '+%F %H:%M') == "\$CHECK_MIN" ]] || { echo "HOLD: the check's minute (\$CHECK_MIN) has passed; no signal, no retry"; exit 3; }
  kill -INT "\$PID" && TZ=America/Los_Angeles date '+SIGINT sent to run_pilot pid '"\$PID"' at %F %T PT'
else
  echo "CHECK only: condition holds; no signal sent (pid \$PID)"
fi
B
rc=$?
cat "$OUT/before.txt" | grep -v '^{' | cut -c1-300
(( rc == 0 )) || { echo "not sent (rc $rc); output in $OUT/before.txt"; exit "$rc"; }
[[ $SEND == --send ]] || exit 0
grep -q '^SIGINT sent' "$OUT/before.txt" || { echo "the kill did not report success; output in $OUT/before.txt"; exit 4; }
# 3: wait for the launcher to exit (bounded), then the after snapshot
for i in $(seq 60); do
  box "$PID" > "$OUT/wait.txt" 2>&1 <<'B'
kill -0 "$1" 2>/dev/null && echo ALIVE || echo GONE
B
  grep -q GONE "$OUT/wait.txt" && break; sleep 20
done
grep -q GONE "$OUT/wait.txt" || { echo "launcher still alive after 20 min: no after-snapshot; ping d9"; exit 5; }
sleep 30  # the launch script's final gate runs after run_pilot exits; launch.json is final once run_pilot has exited
box > "$OUT/after.txt" 2>&1 <<B
TZ=America/Los_Angeles date '+after %F %T PT'
AJ=/tmp/sigint_311_after_\$\$.json
$PY /tmp/sigint_311_snap.py "\$AJ" > /dev/null || { echo "after snapshot FAILED"; exit 6; }
cat "\$AJ" && echo || exit 6
tail -4 /home/aswin/phase2/runs_ep2/2026-10-15_122516_ep2_311.launcher.log
B
arc=$?
(( arc == 0 )) || { echo "after-snapshot failed (rc $arc): no record; output in $OUT/after.txt; ping d9"; exit 6; }
python3 -I - "$OUT" "$HERE/results/sigint_311_$STAMP.json" <<'PY'
import json, sys
from pathlib import Path
out, rec = Path(sys.argv[1]), Path(sys.argv[2])
bt, at = (out / "before.txt").read_text().splitlines(), (out / "after.txt").read_text().splitlines()
before = json.loads(next(l for l in bt if l.startswith("{")))
after = json.loads(next(l for l in at if l.startswith("{")))
keys = set(before["below_P"]) | set(after["below_P"])
changed = {k: {"before": before["below_P"].get(k), "after": after["below_P"].get(k)} for k in sorted(keys)
           if after["below_P"].get(k) != before["below_P"].get(k)}
final_ok = bool(after.get("finished")) and any(str(x) in ("2", "SIGINT", "Signals.SIGINT") for x in (after.get("stopped_by_signal") or []))  # run_pilot stores str(signum)
assert not rec.exists()
rec.write_text(json.dumps({
    "what": "+311 launch 1 stop under RULE.md amendment 3 (09 @ e00eff46d, record @ 91feb3525; 0d: as soon as the by-key condition holds)",
    "check_and_signal_lines": [l for l in bt if not l.startswith("{")],
    "after_lines": [l for l in at if not l.startswith("{")],
    "before": before, "after": after,
    "below_P_changed": changed, "below_P_changed_count": len(changed),
    "stopped_by_launcher_keys": after["stopped_by_launcher"], "stopped_below_P": after["stopped_below_P"],
    "after_final": {"finished": after.get("finished"), "stopped_by_signal": after.get("stopped_by_signal"), "ok": final_ok}}, indent=1) + "\n")
print(f"record {rec.name}: below-P runs {len(before['below_P'])}, changed {len(changed)}; stopped by launcher "
      f"{len(after['stopped_by_launcher'])} (below P: {len(after['stopped_below_P'])})")
if changed or after["stopped_below_P"] or not final_ok:
    print(f"FAIL: below-P changed {len(changed)}, stopped below P {len(after['stopped_below_P'])}, finalized {final_ok}: ping d9")
    sys.exit(7)
PY
