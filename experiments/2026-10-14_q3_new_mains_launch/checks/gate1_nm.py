"""gate1_nm.py: the new-mains gate (RULE.md Amendment 4; Aswin's rule via 0d, about 19:40-19:50 PT: in-job gates PAUSE,
never kill). gate1_311b.py with its quality trips changed; the counts, the pair check and the deadline are its own.
ON THE DEVBOX, its own tmux session, every 120 s while the launcher's session exists:
  - per arm: distinct ended runs and the usable ones (gate1_311b.py's run_usable()); <run>/gate1_counts.json each pass;
  - ALERT (GATE1_ALERT.txt, logged once, nothing else): any arm with >= 20 ended runs at or below 90% usable, or >= 100
    ended runs in all at or below 90%. New-main tasks whose setup script fails at sandbox start fail every arm of their
    cell before any policy call: a task property, so this is for a person to read;
  - PAUSE, on FRESH data only (launch.json written within 300 s, or the pair check just run): any arm with >= 20 ended
    runs at or below 75% usable (PAUSED_usable75.txt), a byte-identical r1/r2 pair (exit 2, PAUSED_identical_pair.txt)
    or an r1/r2 configuration mismatch (exit 3, PAUSED_pair_config.txt; the same construction-bug class). Three failed
    pair checks in a row are an ALERT only (PAIR_ALERT.txt: no data). How it pauses:
      the real-honest fork's safe-point pause (health_monitor.py @ fe62b0846, its pause half only): wait for
        a NEW "running N, done M of K;" line in <run>.launcher.log (printed after launch(), before the 30 s sleep), then
        SIGSTOP the launcher's main process only. In-flight runs (own sessions) keep going; nothing new starts; reversible
        by SIGCONT, by a person. No safe point within 120 s: try again next pass (the deadline is never delayed);
    end_main.py never starts a follow-up while a PAUSED_*.txt is in the main run: a person decides. While paused the
    main servers idle-exit after 20 min, so a resume needs fresh servers;
  - deadline: one Ctrl+C to the launcher session at <stop at> (DEADLINE_STOP.txt; a time bound, not a quality gate); a
    paused launcher also gets SIGCONT right after, so it handles the Ctrl+C and stops gracefully.
One line per check to <run>/gate1_watch.log. Exits when the session is gone or the launch has finished.
  <sept venv python> gate1_nm.py <run dir> <launcher session> <stop at, epoch s> main
(MAIN launches only; a follow-up's usable gate is gate_pause_nm.py, the scale-up's gate_pause_311.py @ ba21dea16.)
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ENDED = ("done", "no policy call")
HERE = Path(__file__).resolve().parent
HARNESS = Path("/home/aswin/phase2/nm/attack_check_nm")  # check_pairs.py lives in the frozen harness
run, session, stop_at = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
assert sys.argv[4:] == ["main"], "gate1_nm.py is for MAIN launches only"
PROGRESS = re.compile(rb" running \d+, done \d+ of \d+;")


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
    return subprocess.run(["tmux", "-L", "nm", "has-session", "-t", f"={session}"], stderr=subprocess.DEVNULL).returncode == 0


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
    subprocess.run(["tmux", "-L", "nm", "send-keys", "-t", f"={session}:", "C-c"], check=True)  # first: never blocked
    if stopped_pid is not None:  # a SIGSTOPped launcher handles the pending Ctrl+C only once continued
        try:
            os.kill(stopped_pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
    note(f"{why}; Ctrl+C sent to {session}" + (f", then SIGCONT to {stopped_pid}" if stopped_pid else ""))
    mark(name, why)


def launcher_pid():
    """The launcher: a python process running <harness>/run_pilot.py with --out <run dir> (health_monitor.py's test)."""
    found = []
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                a = [x.decode(errors="replace") for x in (p / "cmdline").read_bytes().split(b"\0")]
            except OSError:
                continue
            if (len(a) > 2 and "python" in Path(a[0]).name and a[1] in ("run_pilot.py", str(HARNESS / "run_pilot.py"))
                    and "--out" in a and a.index("--out") + 1 < len(a) and a[a.index("--out") + 1] == str(run)):
                found.append(int(p.name))
    assert len(found) <= 1, f"several launchers for {run}: {found}"
    return found[0] if found else None


def safe_point(log: Path, timeout: float) -> bool:
    """A progress line written after this call began: the launcher has just left launch() for its 30 s sleep."""
    start, end = log.stat().st_size, time.time() + timeout
    while time.time() < end:
        with open(log, "rb") as f:
            f.seek(start)
            if PROGRESS.search(f.read()):
                return True
        time.sleep(0.2)
    return False


def usable75_now() -> list:
    """Codex r5: the <= 75% trigger, re-read now: launch.json written within 300 s and an arm with >= 20 ended at or
    below 75% usable; [] if stale, unreadable or recovered."""
    try:
        f = run / "launch.json"
        if time.time() - f.stat().st_mtime > 300:
            return []
        c = counts(json.loads(f.read_text()))
    except (OSError, json.JSONDecodeError):
        return []
    return [k for k, v in c.items() if v["ended"] >= 20 and 4 * v["usable"] <= 3 * v["ended"]]


def try_pause(reason: str, why: str) -> bool:
    """One attempt at the pause; True once it has taken effect (the PAUSED marker is written only then). The usable75
    evidence is re-read just before and again right after the safe-point wait (pair evidence doesn't go stale)."""
    global stopped_pid
    if reason == "usable75" and not usable75_now():
        note(f"{why}: no longer true on fresh data; no pause")
        return False
    pid = launcher_pid()
    if pid is None:
        note(f"{why}: no launcher process found; pause pending")
        return False
    if not safe_point(Path(f"{run}.launcher.log"), max(1.0, min(120.0, stop_at - time.time() - 60))):
        note(f"{why}: no safe point within 120 s; pause pending")
        return False
    if reason == "usable75" and not usable75_now():
        note(f"{why}: no longer true on fresh data after the safe point; no pause")
        return False
    os.kill(pid, signal.SIGSTOP)
    stopped_pid = pid
    mark(f"PAUSED_{reason}.txt", f"{why}; launcher pid {pid} SIGSTOPped at a safe point (SIGCONT to resume)")
    note(f"PAUSED ({reason}): {why}")
    return True


stopped = paused = alerted = pair_alerted = False
pending = None  # (reason, why) of a pause not yet taken effect
stopped_pid = None
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
        fresh = time.time() - (run / "launch.json").stat().st_mtime <= 300
        low = [k for k, v in c.items() if v["ended"] >= 20 and 10 * v["usable"] <= 9 * v["ended"]]
        if not alerted and (low or (ended >= 100 and 10 * usable <= 9 * ended)):
            why = f"GATE 1 ALERT (no action): at or below 90% usable in {low or 'all arms'} ({usable} of {ended} in all)"
            mark("GATE1_ALERT.txt", why)
            note(why)
            alerted = True
        very_low = [k for k, v in c.items() if v["ended"] >= 20 and 4 * v["usable"] <= 3 * v["ended"]]
        if very_low and fresh and not paused and pending is None:
            pending = ("usable75", f"at or below 75% usable in {very_low} (fresh launch.json)")
        both = sum(1 for r in launch["runs"] if r["arm"] == "attack" and r["task"].endswith("_r2") and r["state"] == "done"
                   and any(x["key"] == f"attack/{r['task'][:-3]}" and x["state"] == "done" for x in launch["runs"]))
        if both >= 20 and not stopped and not paused and time.time() - last_pairs >= 600:  # at most every 10 minutes (reads .eval files)
            last_pairs = time.time()
            left = stop_at - time.time() - 60  # never past the deadline: the pair check is bounded by the time left
            try:
                p = subprocess.run([sys.executable, str(HARNESS / "check_pairs.py"), str(run), str(run / "pairs_check.json")],
                                   capture_output=True, text=True, env=os.environ, timeout=max(1.0, min(900.0, left)))
            except subprocess.TimeoutExpired:
                p = subprocess.CompletedProcess([], 124, "", "pair check timed out")
            note(f"pairs ({both} done): {p.stdout.strip()[-300:]} exit {p.returncode}")
            pair_errors = 0 if p.returncode in (0, 2, 3) else pair_errors + 1
            bad = {2: ("identical_pair", "IDENTICAL PAIR: r1 and r2 replayed (pairs_check.json)"),
                   3: ("pair_config", "PAIR CONFIG MISMATCH: r1 and r2 attacks configured differently (pairs_check.json)")
                   }.get(p.returncode)
            if bad and (pending is None or pending[0] == "usable75"):  # Codex r6: pair evidence supersedes usable75
                pending = bad
            if bad is None and pair_errors >= 3 and not pair_alerted:
                why = f"PAIR ALERT (no action): the pair check failed {pair_errors} times in a row: {p.stderr.strip()[-200:]}"
                mark("PAIR_ALERT.txt", why)
                note(why)
                pair_alerted = True
        if pending and not paused and not stopped:
            paused = try_pause(*pending)
            if paused or pending[0] == "usable75":  # a usable75 trigger is re-derived from fresh data every pass
                pending = None
    if not stopped and time.time() >= stop_at:
        interrupt("DEADLINE_STOP.txt", f"DEADLINE STOP at {stop_at:.0f} (epoch s)")
        stopped = True
    time.sleep(max(1.0, min(120 if launch is not None else 30, stop_at - time.time())) if not stopped else 120)
note(f"session {session} gone or launch finished")
