"""Negative tests for the new-mains sim_cap_stop.py (RULE.md Amendment 2; 0d 18:42: the Ctrl+C fires ONLY on a FRESH
reading over the bound, never on a stale, failed or missing one). Runs anywhere with python3 (no harness, no network): a
fake spend_now.py replays readings ("<usd> <age_s> <running> <charged_worst> <long_unread>" or FAIL), a fake `tmux` on
PATH records every call and reports the session live while readings remain, and a fake clock advances 120 s per sleep
(a sleep after the last reading ends the case). Each case runs sim_cap_stop.py's source UNCHANGED except its directory
and interpreter constants (pointed at a temp dir). Then spend_now.py's fifth field (long_unread) on a stub harness.
  python3 test_sim_cap_stop_nm.py <path to sim_cap_stop.py> <path to spend_now.py>
"""
import json
import os
import stat
import sys
import tempfile
import time
from pathlib import Path

SRC, SPEND_NOW = Path(sys.argv[1]).read_text(), sys.argv[2]
CAP, HARD, WORST = 1000.0, 1060.0, 75.0
real_time = time.time
clock, queue = [time.time()], [None]


class Done(Exception):
    pass


real_sleep = time.sleep


def fake_sleep(s: float) -> None:
    if s < 1:  # subprocess's own polling waits (Popen.wait with a timeout): real, never the script's 120 s loop sleep
        real_sleep(s)
        return
    if json.loads(queue[0].read_text()) == []:
        raise Done
    clock[0] += s


time.time = lambda: clock[0]  # the script's own `import time` gets this module
time.sleep = fake_sleep


def case(name: str, readings: list[str], run_exists: bool = True) -> dict:
    d = Path(tempfile.mkdtemp(prefix=f"scs-{name}-"))
    harness, checks, run, bin_ = d / "h", d / "c", d / "run", d / "bin"
    for p in (harness / "results", checks, bin_) + ((run,) if run_exists else ()):
        p.mkdir(parents=True)
    if run_exists:
        (run / "launch.json").write_text(json.dumps({"spend": [{"worst_per_sim_run_usd": WORST}]}))
    q = queue[0] = d / "readings.json"
    q.write_text(json.dumps(readings))
    (checks / "spend_now.py").write_text(
        "import json,sys\nfrom pathlib import Path\n"
        f"q = Path({str(q)!r}); r = json.loads(q.read_text())\n"
        "x = r.pop(0) if r else 'absent'; q.write_text(json.dumps(r))\n"
        "if x == 'FAIL': sys.exit(1)\nprint(x)\n")
    calls = d / "tmux_calls.txt"
    (bin_ / "tmux").write_text(f"#!/bin/sh\necho \"$@\" >> {calls}\n"
                               f"case \"$*\" in *has-session*) grep -q '\"' {q} && exit 0; exit 1;; esac\nexit 0\n")
    (bin_ / "tmux").chmod((bin_ / "tmux").stat().st_mode | stat.S_IEXEC)
    src = SRC.replace('Path("/home/aswin/phase2/nm/attack_check_nm"), Path("/home/aswin/phase2/nm/checks")',
                      f"Path({str(harness)!r}), Path({str(checks)!r})")
    src = src.replace('PY, TREE = "/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python"', f"PY, TREE = {sys.executable!r}")
    src = src.replace('"PATH": "/usr/bin:/bin"', f'"PATH": {str(bin_) + ":/usr/bin:/bin"!r}')
    assert src.count(str(checks)) == 1 and src.count(sys.executable) == 1 and src.count(str(bin_)) == 1, "constants not found"
    old_path = os.environ["PATH"]
    os.environ["PATH"] = f"{bin_}:{old_path}"
    sys.argv = ["sim_cap_stop.py", str(run), "fnms9", str(CAP), str(HARD)]
    try:
        exec(compile(src, "sim_cap_stop.py", "exec"), {"__name__": "__main__"})
    except Done:
        pass
    finally:
        os.environ["PATH"] = old_path
    t = calls.read_text() if calls.exists() else ""
    assert run_exists or not run.exists(), "sim_cap_stop.py created the run folder"
    return {"ctrl_c": "send-keys" in t, "run_paused": (run / "PAUSE_SIMS").exists(),
            "harness_paused": (harness / "results" / "PAUSE_SIMS").exists(),
            "blind_marker": (run / "SIM_BLIND_PAUSE.txt").exists(), "hard_marker": (run / "SIM_HARD_STOP.txt").exists()}


F = dict(ctrl_c=False, run_paused=False, harness_paused=False, blind_marker=False, hard_marker=False)
CASES = {  # name: (readings, expected); known = usd - charged_worst x $75
    # 0d's rule: a stale or failed read is never a Ctrl+C, however high
    "stale_above_hard": (["2000 400 5 0 0"], F),
    "failed_reads_then_stale_above_hard": (["FAIL", "FAIL", "5000 900 5 0 0"], F),
    "stale_15min_above_hard": (["5000 400 5 0 0"] * 9, F | dict(run_paused=True, blind_marker=True)),
    # fresh readings
    "fresh_between_cap_and_hard": (["1030 60 5 0 0"], F | dict(run_paused=True, harness_paused=True)),
    "fresh_above_hard": (["1100 60 5 0 0"], F | dict(ctrl_c=True, run_paused=True, harness_paused=True, hard_marker=True)),
    # an unread attempt counts at the $75 worst case toward the hard stop (Codex r2), never toward the clean pause
    "unread_counted_for_hard": (["1065 60 5 1 0"], F | dict(ctrl_c=True, hard_marker=True)),
    "unread_below_hard": (["1050 60 5 1 0"], F),
    # blind pause: reversible, never a Ctrl+C
    "stale_15min_then_fresh_latches": (["500 400 5 0 0"] * 9 + ["500 60 5 0 0"], F | dict(run_paused=True, blind_marker=True)),
    "three_long_unread_pauses": (["300 60 5 3 3"], F | dict(run_paused=True, blind_marker=True)),
    "two_long_unread_no_pause": (["300 60 5 2 2"], F),
    "long_unread_then_read_latches": (["300 60 5 3 3", "300 60 5 0 0"], F | dict(run_paused=True, blind_marker=True)),
    "cap_pause_never_lifted": (["1030 60 5 0 0"] + ["1030 400 5 0 0"] * 9 + ["1030 60 5 0 0"],
                               F | dict(run_paused=True, harness_paused=True)),
    "cap_pause_kept_through_blind_recovery": (["300 60 5 3 3", "1030 60 5 3 3", "1030 60 5 0 0"],
                                              F | dict(run_paused=True, harness_paused=True, blind_marker=True)),
    # missing readings (Codex r3): failing from the first read, or launch.json gone after a reading: pause, never Ctrl+C
    "reads_fail_from_start_15min": (["FAIL"] * 9, F | dict(run_paused=True, blind_marker=True)),
    "reads_fail_14min_no_pause": (["FAIL"] * 8, F),
    "absent_after_reading_15min": (["300 60 5 0 0"] + ["absent"] * 10, F | dict(run_paused=True, blind_marker=True)),
    "absent_before_any_reading": (["absent"] * 12, F),
}
results = {k: case(k, r) for k, (r, _) in CASES.items()}
results["reads_fail_before_run_folder"] = case("norun", ["FAIL"] * 12, run_exists=False)
CASES["reads_fail_before_run_folder"] = (None, F)
bad = {k: {"got": results[k], "want": w} for k, (_, w) in CASES.items() if results[k] != w}

# spend_now.py: long_unread counts RUNNING attempts unread for >= 1,200 s (stub run_spend: None for every log)
d = Path(tempfile.mkdtemp(prefix="sn-"))
(d / "h").mkdir(); (d / "sim").mkdir(); (d / "run").mkdir(); (d / "st").mkdir()
(d / "h" / "run_pilot.py").write_text("def run_spend(log_dir, policy, prices):\n    return None\n")
(d / "sim" / "__init__.py").write_text(""); (d / "sim" / "prices.py").write_text("def load_prices():\n    return {}\n")
now = real_time()  # spend_now.py runs on the real clock
att = lambda i: {"key": f"k{i}", "real": False, "attempts": [{"log_dir": str(d / f"l{i}")}]}
(d / "run" / "launch.json").write_text(json.dumps({"policy": "p", "spend": [
    {"per_run_estimate_usd": 16.5, "worst_per_sim_run_usd": WORST, "time": __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).isoformat()}], "runs": [att(i) for i in range(5)]}))
ages = [0, 400, 1100, 1300, 5000]  # seconds since each attempt was last read (k0: just seen)
(d / "st" / "spend_state_run.json").write_text(json.dumps(
    {f"k{i}#0": {"seen": now - a, "ok": None} for i, a in enumerate(ages) if a}))
import subprocess  # noqa: E402
out = subprocess.run([sys.executable, SPEND_NOW, str(d / "run"), str(d / "h"), str(d / "st")], capture_output=True,
                     text=True, env={"PYTHONPATH": str(d), "PATH": "/usr/bin:/bin"})
fields = out.stdout.split()
sn_ok = out.returncode == 0 and len(fields) == 5 and fields[:1] + fields[2:] == ["316.50", "5", "4", "2"]  # 16.50 + 4 x $75; running 5; > 300 s: 4; >= 1,200 s: 2
if not sn_ok:
    bad["spend_now_long_unread"] = {"got": out.stdout + out.stderr[-500:], "want": "316.50 <age> 5 4 2"}
print(json.dumps({"cases": len(CASES) + 1, "failed": bad, "pass": not bad}, indent=1))
sys.exit(0 if not bad else 1)
