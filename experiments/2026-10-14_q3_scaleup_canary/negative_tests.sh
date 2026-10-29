#!/usr/bin/env bash
# $0 negative tests for Codex rounds 3 and 4's fixes (RULE.md amendments 7 and 8). ON THE DEVBOX, from the folder that
# holds this job's scripts (~/phase2/simh_checks), with run_pilot.py and run_attack_check.py @ bbf0729 copied there as
# run_pilot_bbf0729.py and run_attack_check_bbf0729.py:
#   bash negative_tests.sh
# Fixtures are small hand-built launch.json files under ./negtest (no transcript). Two tests also read the canary
# launch's records and .evals (hashes only). Test tmux sessions are named negtest_* and are killed at the end.
# Prints PASS/FAIL per test; exits 1 if any test fails.
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
CANARY=/home/aswin/phase2/runs/2026-10-09_173815_scaleup_canary
T=$PWD/negtest; rm -rf "$T"; mkdir -p "$T/h"
printf '#!/usr/bin/env bash\nexec bash -c "$1"\n' > "$T/h/devbox.sh"; chmod +x "$T/h/devbox.sh"  # "ssh" = run locally
mkdir -p "$T/harn" && cp run_pilot_bbf0729.py "$T/harn/run_pilot.py" && cp run_attack_check_bbf0729.py "$T/harn/run_attack_check.py"
fails=0
ok() { echo "PASS $1"; }
no() { echo "FAIL $1"; fails=$((fails + 1)); }

python3 - "$T" <<'EOF'
import json, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
T = Path(sys.argv[1])
now = datetime.now(timezone.utc)
def att(clean, usable, sim_usd=20.0, called=True):
    return {"kind": "first", "start": now.isoformat(), "end": now.isoformat(), "sim_usd": sim_usd, "live_usd": sim_usd,
            "policy_called": called, "prestart": None,
            "outcome": {"clean": clean, "sample_limits": [], "sample_errors": [] if clean else ["x"],
                        "policy_calls": [{"usable": usable, "ended": "submit" if usable else "error"}]}}
def launch(d, runs, finished=True, spend=None):
    (T / d).mkdir(parents=True)
    (T / d / "launch.json").write_text(json.dumps({"finished": finished, "runs": runs, "tasks": [], "policy": "p",
                                                    "spend": spend or [], "trees": {}}))
run = lambda k, st, a: {"key": f"sim_honest@september/{k}", "arm": "sim_honest", "simulator": "september", "real": False,
                        "task": k, "state": st, "attempts": a}
launch("gate_fail", [run("tA", "done", [att(True, 1)]), run("tB", "done", [att(False, 0)])])
launch("gate_pass", [run("tA", "done", [att(True, 1)]), run("tB", "done", [att(True, 1)])])
(T / "keys_ab.txt").write_text("tA,tB")
(T / "keys_ac.txt").write_text("tA,tC")
launch("a_empty", [])
old = (now - timedelta(hours=1)).isoformat()
running = {"kind": "first", "start": old, "live_usd": 5.0, "policy_called": None}
launch("run_stale", [run("tA", "running", [running])], finished=False,
       spend=[{"time": old, "actual_usd": 5.0, "finished_unpriced_attempts": 0, "per_run_estimate_usd": 19.76,
               "worst_per_sim_run_usd": 60}])
EOF

# 1. gate12: FAIL fixture exits 1; PASS fixture exits 0 (positive control); wrong task set exits 1
python3 gate12.py --out "$T/g1.json" --expect-keys "$T/keys_ab.txt" "$T/gate_fail" > "$T/g1.log"; r1=$?
python3 gate12.py --out "$T/g2.json" --expect-keys "$T/keys_ab.txt" "$T/gate_pass" > "$T/g2.log"; r2=$?
python3 gate12.py --out "$T/g3.json" --expect-keys "$T/keys_ac.txt" "$T/gate_pass" > "$T/g3.log"; r3=$?
[[ $r1 == 1 && $r2 == 0 && $r3 == 1 ]] && ok "gate12 exits 1 on a FAIL fixture and on a wrong task set, 0 on a PASS one" \
  || no "gate12 exit codes $r1 $r2 $r3 (want 1 0 1)"

# 2. the A check fails on 0/0
$PY check_approach_hashes.py --launches "$CANARY" --gate tasks_simattack2_20.json \
  --harness-tasks pilot_tasks_amendment9_scaleup176.json --a-launches "$T/a_empty" --out "$T/a.json" > "$T/a.log" 2>&1; ra=$?
[[ $ra != 0 ]] && grep -q "A CHECK FAIL: 0/20" "$T/a.log" && ok "the A check fails on 0/0 (exit $ra)" \
  || no "the A check on 0/0: exit $ra, $(grep 'A CHECK' "$T/a.log" | head -1)"

# 3. a stale snapshot is rejected: a live session whose launch's last snapshot is an hour old -> allin_stop stops it
tmux -L phase2 new -d -s negtest_stale "sleep 600 # --out /home/aswin/phase2/simh_checks/negtest/run_stale "
: > "$T/apps.txt"
H=$T/h DEVBOX=local STOP_USD=99999 RESERVE_USD=0 N_SERVERS=0 GPU_START=$(date +%s) BOX_START=$(date +%s) \
  APP_IDS_FILE=$T/apps.txt POLL_S=1 STALE_S=300 STALE_MAX=2 STOP_WAIT_S=30 HARN=phase2/simh_checks/negtest/harn \
  timeout 120 bash allin_stop.sh negtest_stale=phase2/simh_checks/negtest/run_stale > "$T/stale.log" 2>&1; rs=$?
grep -q "HARD STOP: the spend of .* stale" "$T/stale.log" && grep -q "stop confirmed" "$T/stale.log" \
  && ! tmux -L phase2 has-session -t =negtest_stale 2>/dev/null && ok "a stale snapshot stops the group (exit $rs)" \
  || no "stale snapshot: exit $rs; $(tail -3 "$T/stale.log" | tr '\n' ' ')"

# 4. stop_group's deadline runs from the first Ctrl+C: two panes that ignore Ctrl+C, STOP_WAIT_S=20
tmux -L phase2 new -d -s negtest_deaf1 "bash -c 'trap \"\" INT; sleep 600'"
tmux -L phase2 new -d -s negtest_deaf2 "bash -c 'trap \"\" INT; sleep 600'"
p1=$(tmux -L phase2 display-message -p -t =negtest_deaf1: '#{pane_id}:#{pane_pid}'); p2=$(tmux -L phase2 display-message -p -t =negtest_deaf2: '#{pane_id}:#{pane_pid}')
t0=$(date +%s)
H=$T/h DEVBOX=local APP_IDS= STOP_WAIT_S=20 bash stop_group.sh "$p1,$p2" > "$T/deaf.log" 2>&1; rd=$?
el=$(( $(date +%s) - t0 ))
[[ $rd == 2 ]] && (( el <= 20 + 30 + 5 )) && grep -q "NOT CONFIRMED within 20s of the first Ctrl+C" "$T/deaf.log" \
  && ok "stop_group's deadline runs from the first Ctrl+C (exit 2 after ${el}s)" || no "stop_group deadline: exit $rd after ${el}s"

# 5. server numbering starts at 0 (the harness prints `server {i}` over enumerate(servers)); LAUNCH.md waits on server 0
grep -q 'for i, server in enumerate(servers):' run_pilot_bbf0729.py && grep -q 'server {i} checks' run_pilot_bbf0729.py \
  && grep -q '"server 0 checks: passed"' LAUNCH.md \
  && ok "server numbering starts at 0 in the harness and in LAUNCH.md" || no "server numbering"

# 6. the settings check fails on a done run with no .eval
python3 - "$T" "$CANARY" <<'EOF'
import json, sys
from pathlib import Path
T, C = Path(sys.argv[1]), Path(sys.argv[2])
rec = json.loads((C / "launch.json").read_text())
src = next(r for r in rec["runs"] if r["arm"] == "sim_attack" and r["simulator"] == "september" and r["state"] == "done")
a = dict(src["attempts"][-1])
d = T / "no_eval" / "sim_honest@september" / src["task"]
d.mkdir(parents=True)
a["log_dir"] = str(d)
r = {"key": f"sim_honest@september/{src['task']}", "arm": "sim_honest", "simulator": "september", "real": False,
     "task": src["task"], "state": "done", "attempts": [a]}
(T / "no_eval" / "launch.json").write_text(json.dumps({"runs": [r], "trees": rec["trees"], "tasks": []}))
EOF
$PY check_sim_settings.py --reference "$CANARY" --expect-reference 5 --new "$T/no_eval" --out "$T/s.json" > "$T/s.log" 2>&1; rv=$?
[[ $rv != 0 ]] && grep -q "done without exactly one .eval" "$T/s.log" && ok "the settings check fails on a done run with no .eval (exit $rv)" \
  || no "settings check on a run with no .eval: exit $rv"

# fake Modal CLI for tests 7-8: records its calls; `app list --json` reports FAKE_ID in state FAKE_STATE
cat > "$T/fakemodal" <<'FAKE'
#!/usr/bin/env bash
echo "$*" >> "$FAKE_LOG"
if [[ $1 == app && $2 == list ]]; then printf '[{"App ID": "%s", "State": "%s"}]' "$FAKE_ID" "$FAKE_STATE"; fi
FAKE
chmod +x "$T/fakemodal"

# 7. a launcher that exits early (e.g. a failed server check): allin_stop still stops and confirms the apps
tmux -L phase2 new -d -s negtest_early "sleep 4 # --out /home/aswin/phase2/simh_checks/negtest/run_early "
echo "ap-negtest7" > "$T/apps7.txt"
FAKE_LOG=$T/fake7.log FAKE_ID=ap-negtest7 FAKE_STATE=stopped MODAL_CMD=$T/fakemodal H=$T/h DEVBOX=local STOP_USD=99999 \
  RESERVE_USD=0 N_SERVERS=0 GPU_START=$(date +%s) BOX_START=$(date +%s) APP_IDS_FILE=$T/apps7.txt POLL_S=2 \
  HARN=phase2/simh_checks/negtest/harn timeout 120 bash allin_stop.sh negtest_early=phase2/simh_checks/negtest/run_early > "$T/early.log" 2>&1; r7=$?
[[ $r7 == 0 ]] && grep -q "appeared and ended: stopping the phase's apps" "$T/early.log" && grep -q "app stop ap-negtest7" "$T/fake7.log" \
  && ok "an early launcher exit still stops and confirms the apps (exit 0)" || no "early exit: exit $r7; $(tail -2 "$T/early.log" | tr '\n' ' ')"

# 8. stop_group accepts only "stopped": an app reported "stopping" is not confirmed (exit 3, within the app deadline)
t0=$(date +%s)
FAKE_LOG=$T/fake8.log FAKE_ID=ap-negtest8 FAKE_STATE=stopping MODAL_CMD=$T/fakemodal H=$T/h DEVBOX=local APP_IDS=ap-negtest8 \
  FORCE_APPS=1 APP_STOP_S=15 bash stop_group.sh "" > "$T/stopping.log" 2>&1; r8=$?; el=$(( $(date +%s) - t0 ))
[[ $r8 == 3 ]] && (( el <= 15 + 5 )) && ok "\"stopping\" is not stopped (exit 3 after ${el}s)" || no "stopping: exit $r8 after ${el}s"

# 9. a running attempt with no successful fresh price read for > 300 s is charged the worst case
python3 - "$T" <<'PYEOF'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
T = Path(sys.argv[1]); d = T / "run_fresh" / "sim_honest@september" / "tA"; d.mkdir(parents=True)
now = datetime.now(timezone.utc).isoformat()
a = {"kind": "first", "start": now, "live_usd": 3.0, "policy_called": None, "log_dir": str(d)}
(T / "run_fresh" / "launch.json").write_text(json.dumps({"finished": False, "policy": "vllm/Qwen/Qwen3.5-27B", "spend": [
    {"time": now, "actual_usd": 3.0, "finished_unpriced_attempts": 0, "per_run_estimate_usd": 19.76,
     "worst_per_sim_run_usd": 60}], "runs": [{"key": "sim_honest@september/tA", "real": False, "attempts": [a]}]}))
PYEOF
SP="env PYTHONPATH=/home/aswin/phase2/sims-sept-strict-off/src $PY spend_now.py $T/run_fresh $T/harn $T"
o1=$($SP 2>&1 | tail -1)
python3 -c "import json,sys; f=sys.argv[1]; s=json.load(open(f)); [v.update(seen=v['seen']-400) for v in s.values()]; json.dump(s,open(f,'w'))" "$T/spend_state_run_fresh.json"
o2=$($SP 2>&1 | tail -1)
[[ $o1 == "19.76 "* && $o2 == "60.00 "*" 1 1" ]] && ok "no fresh price read for > 300 s charges the worst case ('$o1' then '$o2')" \
  || no "fresh-read staleness: '$o1' then '$o2'"

# 10. positive control: on a finished launch, spend_now equals the launcher's own final actual
L2=/home/aswin/phase2/runs/2026-10-09_182505_scaleup
o3=$(env PYTHONPATH=/home/aswin/phase2/sims-sept-strict-off/src $PY spend_now.py $L2 $T/harn $T 2>&1 | tail -1)
want=$(python3 -c "import json; print('%.2f' % json.load(open('$L2/launch.json'))['spend'][-1]['actual_usd'])")
[[ ${o3%% *} == "$want" ]] && ok "spend_now equals a finished launch's final actual (\$$want)" || no "spend_now control: '$o3' vs $want"

# 11. wait_settled: a running A run without a policy call is not settled; a finished launch is
python3 - "$T" <<'PYEOF'
import json, sys
from pathlib import Path
T = Path(sys.argv[1])
for name, fin in (("a_unsettled", False), ("a_finished", True)):
    d = T / name / "sim_attack@september" / "tA"; d.mkdir(parents=True)
    a = {"kind": "first", "start": "2026-10-14T13:00:00-07:00", "log_dir": str(d)}
    (T / name / "launch.json").write_text(json.dumps({"finished": "2026-10-14T13:00:00-07:00" if fin else None,
        "policy": "vllm/Qwen/Qwen3.5-27B", "runs": [{"key": "sim_attack@september/tA", "real": False,
                                                     "state": "running", "attempts": [a]}]}))
PYEOF
tmux -L phase2 new -d -s negtest_a "sleep 600"
python3 wait_settled.py "$T/a_unsettled" negtest_a 5 > "$T/w1.log" 2>&1; w1=$?
python3 wait_settled.py "$T/a_finished" negtest_a 5 > "$T/w2.log" 2>&1; w2=$?
[[ $w1 == 2 && $w2 == 0 ]] && grep -q "not settled in time" "$T/w1.log" && ok "wait_settled holds an unstarted A run, passes a finished A" \
  || no "wait_settled: exits $w1 $w2"

# 12. launch_simh.sh refuses a hard-stopped session
touch STOPPED_negtest_marker
SESSION=negtest_marker bash launch_simh.sh x y z w v > "$T/m.log" 2>&1; rm_=$?
rm -f STOPPED_negtest_marker
[[ $rm_ == 1 ]] && grep -q "hard-stopped" "$T/m.log" && ok "launch_simh.sh refuses a hard-stopped session" || no "marker: exit $rm_"

for s in negtest_stale negtest_deaf1 negtest_deaf2 negtest_early negtest_a; do tmux -L phase2 kill-session -t "=$s" 2>/dev/null; done
echo "negative tests: $(( 12 - fails ))/12 pass"
exit $(( fails > 0 ))
