"""Negative tests of unpause_onbox.py's decision (RULE.md Amendment 6), $0, local: its checks() is loaded from the script
(everything before its loop) with the box-only calls (tmux, docker, pgrep, the attempt script) replaced by fakes, and run
on synthetic run folders. Only PASS may unpause; a missing or stale reading must be WAIT.
  python3 negative_tests_unpause.py"""
import json, subprocess, sys, tempfile, time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SRC = (Path(__file__).parent / "unpause_onbox.py").read_text().split("\nwhile True:")[0]
PT = ZoneInfo("America/Los_Angeles")
ok = True


def scenario(name, want, *, counts="good", pairs="good", scratch="sha256:ba1bed0cabc", guard=True, late=False,
             attempts=(50, 45, 44), marker=None, guard_run=None, attempt_rc=0, full=True):
    global ok
    d = Path(tempfile.mkdtemp()); run = d / "run"; run.mkdir(); harness = d / "h"; (harness / "attempt").mkdir(parents=True)
    now = datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT")
    arms = {k: {"runs": 30, "ended": 30, "usable": 30} for k in ("honest_a", "honest_b", "attack r1", "attack r2")}
    if counts == "edge":
        arms["attack r2"] = {"runs": 20, "ended": 20, "usable": 18}
    if counts == "old":
        arms["honest_a"]["runs_ended"] = 30
    if counts != "missing":
        (run / "gate1_counts.json").write_text(json.dumps({"time": now, "arms": arms}))
    s = {"pairs": 25, "identical_first_reply": 0, "identical_messages": 0, "config_mismatch": 0}
    if pairs == "identical":
        s["identical_messages"] = 1
    (run / "pairs_check.json").write_text(json.dumps({"summary": s}))
    if marker:
        (run / marker).write_text("x")
    n, k1, k2 = attempts
    att = {"cells_counted": n, **{f"{w}_{x}": {"attempted": k, "tasks": n} for w, k in (("r1", k1), ("r2", k2))
                                   for x in ("all", "strategy", "empty_approach")}}
    graceful = time.time() + (1000 if late else 4 * 3600)
    ns = {"__name__": "t"}
    sys.argv = ["unpause_onbox.py", str(run), "s311qX", "bg311qX", "config/x.json", str(graceful)]
    exec(SRC, ns)
    ns["HARNESS"] = harness
    ns["tmux_live"] = lambda sock, name: guard if name == "bg311qX" else True

    def fake_run(cmd, **kw):
        if cmd[0] == "docker":
            return subprocess.CompletedProcess(cmd, 0, scratch + "\n", "")
        if cmd[0] == "pgrep":
            r = (str(run) + "-old") if guard_run == "RUNOLD" else run
            return subprocess.CompletedProcess(cmd, 0, f"123 /venv/python box_guard.py --launch s311qX={r} --trigger 1600\n" if guard else "", "")
        return subprocess.CompletedProcess(cmd, attempt_rc, json.dumps(att) + "\n", "")
    ns["subprocess"] = type("S", (), {"run": staticmethod(fake_run), "DEVNULL": subprocess.DEVNULL})
    v, why, _ = ns["checks"](full)
    good = v == want
    ok &= good
    print(f"{'ok ' if good else 'BAD'} {name}: {v} ({why})")


scenario("all pass", "PASS")
scenario("gate1 counts missing", "WAIT", counts="missing")
scenario("old-format counts (call-level gate)", "WAIT", counts="old")
scenario("an arm at exactly 90% usable", "WAIT", counts="edge")
scenario("an identical pair", "WAIT", pairs="identical")
scenario("scratch image missing", "WAIT", scratch="")
scenario("box_guard not running", "WAIT", guard=False)
scenario("too late for the simulated tail", "TOO_LATE", late=True)
scenario("a stop marker", "WAIT", marker="GATE1_STOP.txt")
scenario("fewer than 50 labelled cells", "WAIT", attempts=(40, 40, 40))
scenario("attempts below 82% (r2 40 of 50)", "ATTEMPT_FAIL", attempts=(50, 48, 40))
scenario("a guard for ANOTHER run (<run>-old)", "WAIT", guard_run="RUNOLD")
scenario("attempt check exits nonzero with passing JSON", "WAIT", attempt_rc=1)
scenario("the recheck (checks 0-5) passes", "PASS", full=False)
scenario("the recheck after the time limit", "TOO_LATE", late=True, full=False)
print("ALL PASS" if ok else "SOME FAIL")
sys.exit(0 if ok else 1)
