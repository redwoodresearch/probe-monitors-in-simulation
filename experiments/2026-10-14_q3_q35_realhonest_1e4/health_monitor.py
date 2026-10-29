"""On-box pause-and-drain for the relaunch (RULE.md, Amendments 9-10). The frozen launcher never re-checks a server and keeps
routing runs to a dead one, and it has no "stop starting new runs" control. This monitor gives it one from outside, all on the
devbox (never a laptop read), in its own tmux session under rr-env (the vLLM key from the environment, never argv):
    RR_OWNER_ID=aswin-surya rr-env run --env-file config/env.vllm-only.tpl -- ~/sims/simulated-control-evals-sept/.venv/bin/python \
        health_monitor.py <base urls> <run dir> <pause at, epoch s> <drain deadline, epoch s> [poll s]
Triggers, whichever comes first:
  - a dead server: every <poll> seconds (default 60) it GETs <base>/models for each server (15 s timeout); only HTTP 200 is
    healthy; 2 failed checks in a row confirm a dead server (each failure logged with its status or error type);
  - the time <pause at> (the server timeout minus the longest run): no new start may begin after it.
Then, once:
  1. it waits for a safe point (as long as it takes; it never pauses outside one): a new "running N, done M of K" progress line in <run>.launcher.log, which the launcher prints
     after launch() and before its 30 s sleep, then SIGSTOP to the launcher's main process only (its runs are in their own
     sessions and keep going): no new start can happen; <run>/PAUSED.txt says why;
  2. it waits until the launcher has no live child left (every in-flight run has ended; each child's exit time is logged,
     because the launcher, stopped, can't record it) or until the drain deadline;
  3. SIGINT, then SIGCONT: the launcher wakes inside its sleep with its stop flag already set and goes straight to its own
     graceful stop, starting nothing; finished runs it hadn't reaped are recorded "stopped" with their exit code and full
     outcome, ready ones "not run"; <run>/RESUMED_TO_STOP.txt.
Exits when the launcher process is gone. One line per check in <run>/health_monitor.log (stdout until the folder exists)."""
import os
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

urls, run = sys.argv[1].split(","), Path(sys.argv[2])
pause_at, drain_deadline = float(sys.argv[3]), float(sys.argv[4])
POLL = float(sys.argv[5]) if len(sys.argv) > 5 else 60.0
DEAD_AFTER = 2
PROGRESS = re.compile(rb" running \d+, done \d+ of \d+;")


def stamp() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT")


def note(msg: str) -> None:
    out = open(run / "health_monitor.log", "a") if run.is_dir() else sys.stdout
    print(f"{stamp()} {msg}", file=out, flush=True)


def launcher_pid():
    """The launcher itself: a python process whose first argument is exactly run_pilot.py, with --out <run dir>."""
    found = []
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                a = [x.decode(errors="replace") for x in (p / "cmdline").read_bytes().split(b"\0")]
            except OSError:
                continue
            if len(a) > 2 and "python" in Path(a[0]).name and a[1] == "run_pilot.py" and "--out" in a and a[a.index("--out") + 1] == str(run):
                found.append(int(p.name))
    assert len(found) <= 1, f"several launchers for {run}: {found}"
    return found[0] if found else None


def live_children(pid: int) -> set:
    out = set()
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                fields = (p / "stat").read_text().rsplit(")", 1)[1].split()
            except OSError:
                continue
            if fields[1] == str(pid) and fields[0] != "Z":  # ppid, state (zombies have ended)
                out.add(int(p.name))
    return out


def check(url: str):
    try:
        r = httpx.get(url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"}, timeout=15)
        return r.status_code == 200, f"HTTP {r.status_code}"
    except httpx.HTTPError as e:
        return False, type(e).__name__


def safe_point(log: Path, timeout: float = 120) -> bool:
    """Wait for a progress line written after this call began: the launcher has just left launch() for its 30 s sleep."""
    start = log.stat().st_size
    end = time.time() + timeout
    while time.time() < end:
        with open(log, "rb") as f:
            f.seek(start)
            if PROGRESS.search(f.read()):
                return True
        time.sleep(0.2)
    return False


fails = {u: 0 for u in urls}
while (pid := launcher_pid()) is None:  # the launcher starts a moment after this monitor
    if time.time() > drain_deadline:
        sys.exit("no launcher appeared")
    time.sleep(10)
note(f"watching launcher pid {pid}, {len(urls)} servers, poll {POLL:.0f} s, pause at {pause_at:.0f}, drain deadline {drain_deadline:.0f}")
log = Path(f"{run}.launcher.log")
while launcher_pid() == pid:
    results = [check(u) for u in urls]
    for u, (ok, _) in zip(urls, results):
        fails[u] = 0 if ok else fails[u] + 1
    note("checks: " + ", ".join(f"server {i} " + ("ok" if ok else f"FAIL x{fails[u]} ({why})") for i, (u, (ok, why)) in enumerate(zip(urls, results))))
    dead = [i for i, u in enumerate(urls) if fails[u] >= DEAD_AFTER]
    if dead or time.time() >= pause_at:
        why = f"server(s) {dead} failed {DEAD_AFTER} checks in a row" if dead else "the last-new-start time passed"
        while not safe_point(log):  # never pause outside a safe point (Codex round 11); a hung launcher meets the backstops
            note(f"{why}; no progress line within 120 s; still waiting for a safe point")
            if launcher_pid() != pid:
                break
        if launcher_pid() != pid:
            break
        os.kill(pid, signal.SIGSTOP)
        note(f"{why}; launcher pid {pid} paused (SIGSTOP) at a safe point; draining")
        (run / "PAUSED.txt").write_text(f"{stamp()} {why}; launcher paused, draining\n")
        kids = live_children(pid)
        note(f"in flight: {len(kids)} runs (pids {sorted(kids)})")
        while kids and time.time() < drain_deadline:
            time.sleep(min(POLL, 30))
            now = live_children(pid)
            for k in sorted(kids - now):
                note(f"run process {k} exited (seen within {min(POLL, 30):.0f} s of its end; the record's end is the resume time)")
            kids = now
        note(f"drained ({len(kids)} live children left); SIGINT then SIGCONT to {pid}")
        os.kill(pid, signal.SIGINT)
        os.kill(pid, signal.SIGCONT)
        (run / "RESUMED_TO_STOP.txt").write_text(f"{stamp()} resumed to its graceful stop ({why}); {len(kids)} runs still in flight\n")
        break
    time.sleep(POLL)
while launcher_pid() == pid:
    time.sleep(10)
note("launcher gone; monitor exits")
