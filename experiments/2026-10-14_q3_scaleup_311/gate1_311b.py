"""gate1_311b.py: gate1_311.py with run-level counting and a timed graceful stop (RULE.md Amendment 4; d9 15:50, Codex via
0d). It runs from ~/phase2/s311_checks (outside the frozen harness), started after the launch, and replaces the launch's
gate1_311.py process once it has read the run (never an edit of a running script).
Gate 1, the usable tripwire and the pair check of one scale-up box's launch (experiments/2026-10-14_q3_scaleup_311/
RULE.md), ON THE DEVBOX, in its own tmux session (gate311b<N>; no session name may start with the launcher's). The real-
honest launch's gate1.py, per arm instead of per replicate. Every 120 s while the launcher's session exists (exact match):
  - per arm (honest_a, honest_b, attack r1, attack r2, sim_attack): DISTINCT ENDED RUNS ("ended": state "done" or "no
    policy call") and how many of them are usable ("usable": the run's LAST attempt with an outcome and no
    redrawn_because has at least one policy call and every one of them usable); k usable of n ended runs;
  - gate 1 (d9, 15:05: the scale-up is its own canary): any arm with >= 20 ended runs below 90% usable, or >= 100 ended
    runs in all below 90% -> Ctrl+C the launcher once (GATE1_STOP.txt);
  - every pass writes the per-arm counts to <run>/gate1_counts.json (read by unpause_311.sh);
  - pairs: once >= 20 cells have both real attacks done, check_pairs.py on every done pair, at most every 10 minutes;
    any byte-identical pair (exit 2) or a configuration mismatch (exit 3) -> PAUSE_SIMS, Ctrl+C the launcher and
    IDENTICAL.txt / PAIR_CONFIG.txt; any other failure 3 checks in a row -> the same (fail closed, PAIR_CHECK_FAILED.txt);
  - deadline: Ctrl+C at <stop at> (DEADLINE_STOP.txt), so the launcher records its runs before the laptop's deadline.
One line per check to <run>/gate1_watch.log. Exits when the session is gone or the launch has finished.
  <sept venv python> gate1_311b.py <run dir> <launcher session> <graceful stop at, epoch s>
"""

import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ENDED = ("done", "no policy call")
HERE = Path(__file__).resolve().parent
HARNESS = Path("/home/aswin/phase2/attack_check_311")  # check_pairs.py lives in the frozen harness
run, session, stop_at = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])


def stamp() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT")


def arm_of(r: dict) -> str:
    if r["arm"] == "attack":
        return "attack r2" if r["task"].endswith("_r2") else "attack r1"
    return r["arm"] if r["real"] else f"{r['arm']}@{r['simulator']}"


def run_usable(r: dict) -> bool:
    finals = [a for a in r["attempts"] if a.get("outcome") and not a.get("redrawn_because")]
    calls = finals[-1]["outcome"]["policy_calls"] if finals else []
    return bool(calls) and all(c["usable"] for c in calls)


def counts(launch: dict) -> dict:
    """Per arm: runs, distinct ended runs ("ended") and the usable ones among them ("usable")."""
    out = {}
    for r in launch["runs"]:
        c = out.setdefault(arm_of(r), Counter(runs=0, ended=0, usable=0))  # explicit zeros survive dict()
        c["runs"] += 1
        if r["state"] in ENDED:
            c["ended"] += 1
            c["usable"] += run_usable(r)
    return {k: dict(v) for k, v in sorted(out.items())}


def alive() -> bool:
    return subprocess.run(["tmux", "-L", "phase2", "has-session", "-t", f"={session}"], stderr=subprocess.DEVNULL).returncode == 0


log = None


def note(msg: str) -> None:
    global log
    if log is None and run.is_dir():
        log = open(run / "gate1_watch.log", "a")
    print(f"{stamp()} {msg}", file=log or sys.stdout, flush=True)


def mark(name: str, why: str) -> None:
    try:
        (run / name).write_text(f"{stamp()} {why}\n")
    except OSError as e:
        note(f"could not write {name}: {e!r}")


def interrupt(name: str, why: str) -> None:
    subprocess.run(["tmux", "-L", "phase2", "send-keys", "-t", f"={session}:", "C-c"], check=True)  # first: never blocked
    note(f"{why}; Ctrl+C sent to {session}")
    mark(name, why)


stopped = paused = False
last_pairs, pair_errors = 0.0, 0
while alive():
    if not stopped and time.time() >= stop_at:  # the deadline first, before anything that could stall (Codex round 5)
        interrupt("DEADLINE_STOP.txt", f"DEADLINE STOP at {stop_at:.0f} (epoch s)")
        stopped = True
    launch = None
    try:
        launch = json.loads((run / "launch.json").read_text())
    except FileNotFoundError:
        note("no launch.json yet")
    except json.JSONDecodeError:
        note("launch.json unreadable (mid-write?); next check")
    if launch is not None:
        if launch.get("finished"):
            note("launch finished")
            break
        c = counts(launch)
        ended, usable = sum(v["ended"] for v in c.values()), sum(v["usable"] for v in c.values())
        note(f"ended {ended}, usable {usable}; " + "; ".join(f"{k}: ended {v['ended']} usable {v['usable']}" for k, v in c.items()))
        try:
            (run / "gate1_counts.json").write_text(json.dumps({"time": stamp(), "arms": c}))
        except OSError as e:
            note(f"could not write gate1_counts.json: {e!r}")
        # >= 10% unusable stops (inclusive, integers: 18 of 20 usable stops; Codex round 5)
        low = [k for k, v in c.items() if v["ended"] >= 20 and 10 * v["usable"] <= 9 * v["ended"]]
        if not stopped and (low or (ended >= 100 and 10 * usable <= 9 * ended)):
            interrupt("GATE1_STOP.txt", f"GATE 1 STOP: below 90% usable in {low or 'all arms'} ({usable} of {ended} in all)")
            stopped = True
        both = sum(1 for r in launch["runs"] if r["arm"] == "attack" and r["task"].endswith("_r2") and r["state"] == "done"
                   and any(x["key"] == f"attack/{r['task'][:-3]}" and x["state"] == "done" for x in launch["runs"]))
        if both >= 20 and not stopped and time.time() - last_pairs >= 600:  # at most every 10 minutes (reads .eval files)
            last_pairs = time.time()
            left = stop_at - time.time() - 60  # never past the deadline: the pair check is bounded by the time left
            try:
                p = subprocess.run([sys.executable, str(HARNESS / "check_pairs.py"), str(run), str(run / "pairs_check.json")],
                                   capture_output=True, text=True, env=os.environ, timeout=max(1.0, min(900.0, left)))
            except subprocess.TimeoutExpired:
                p = subprocess.CompletedProcess([], 124, "", "pair check timed out")
            note(f"pairs ({both} done): {p.stdout.strip()[-300:]} exit {p.returncode}")
            pair_errors = 0 if p.returncode in (0, 2, 3) else pair_errors + 1
            bad = {2: ("IDENTICAL.txt", "IDENTICAL PAIR: r1 and r2 replayed"),
                   3: ("PAIR_CONFIG.txt", "PAIR CONFIG MISMATCH: r1 and r2 attacks configured differently")}.get(p.returncode)
            if bad is None and pair_errors >= 3:
                bad = ("PAIR_CHECK_FAILED.txt", f"PAIR CHECK FAILED {pair_errors} times in a row: {p.stderr.strip()[-200:]}")
            if bad:
                mark("PAUSE_SIMS", bad[1])
                interrupt(bad[0], bad[1] + " (pairs_check.json)")
                stopped = paused = True
    if not stopped and time.time() >= stop_at:
        interrupt("DEADLINE_STOP.txt", f"DEADLINE STOP at {stop_at:.0f} (epoch s)")
        stopped = True
    time.sleep(max(1.0, min(120 if launch is not None else 30, stop_at - time.time())) if not stopped else 120)
note(f"session {session} gone or launch finished")
