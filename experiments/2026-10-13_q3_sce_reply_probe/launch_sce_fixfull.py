"""Launch run_sce_fixfull.py (RULE.md Amendment 14) under watchers v2.5 (by app ID) and the container guard. Fails closed.
launch_sce.py as run at f6d50a21; only the job and the stops differ: billing $1.80 (2× the $0.90 estimate with 30%), elapsed
$3 (it counts the CPU packs at the H100 rate, so $1.80 there could stop a normal run). Guard max 6 = 2 + max(4, 10% of 2), the code's 2
containers plus 0d's standing slack. Duplicate writers (0d's standing rule, refined): the job's leases stop two containers
live on one input, and the job exits nonzero unless exactly one call per input completed; either sends this launcher into
the stop-and-confirm cleanup. A retry, restart, re-sent-input or preemption line in run.log is logged as an ALERT.

  python3 -I launch_sce_fixfull.py <job dir holding frozen watch_cost.py, watch_billing_strict.py, guard_containers.py>

- The `modal run` starts in its own process group, with INT/TERM blocked until it is recorded. Its app ID is read from
  its own output ("View run at …/ap-…"), so the watchers can't attach to another launch. Discovery reads only the
  local log, with a hard 60 s monotonic deadline.
- The three watchers start in their own process groups, and their PIDs go to watchers.pid. Each must show its first
  poll of this app ID within 60 s (cost.log a record with fails 0, guard.log a container count, billing.out READY), or the run is
  stopped. While the run is alive, every watcher must stay alive; it is checked every 5 s.
- Any failure, exception, INT/TERM or nonzero exit of the run runs cleanup once, with further signals ignored:
  - TERM to the run's process group; KILL to the group after 20 s, whether or not the leader has exited; then reap,
    with every wait bounded.
  - With an app ID: `modal app stop <id> --yes`, up to 3 tries of at most 60 s each, then require the exact ID to be
    listed with state "stopped".
  - Exit 1 when cleanup is confirmed, 2 when it isn't (including when no app ID was ever seen), so the operator checks.
- Independent of this launcher: every function has timeout 1800 s, retries 0 and hard CPU and memory limits, and the
  code runs at most two containers at a time (two CPU packs, then two H100 fits, twice). That is not a hard cap:
  start-up and idle time and a preempted input's restart add to it. The watchers are the stop.
"""
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

UVM = ["uv", "run", "--no-project", "--python", "3.12", "--with", "modal==1.6.1"]
HERE = Path(__file__).resolve().parent
SIGS = {signal.SIGINT, signal.SIGTERM}
DUP = re.compile(r"(?i)\b(retry|retrying|retries|restart(ed|ing)?|re-?sent|resend(ing)?|preempt(ed|ion|ing)?)\b")  # duplicate-writer signs
RPC_RETRY = re.compile(r"for \S+ \(.*\)\. Retrying")  # modal's benign RPC throttling retries ("for <RPC> (...). Retrying...")


def restart_lines(J: Path) -> int:
    """How many retry, restart, re-sent-input or preemption lines run.log holds, benign RPC retries excluded. Since 0d's
    refinement these are ALERTS, not stops: the job's leases stop two live writers on one input, and the job exits
    nonzero (into the cleanup below) unless exactly one call per input completed."""
    return sum(1 for l in (J / "run.log").read_text(errors="replace").splitlines() if DUP.search(l) and not RPC_RETRY.search(l))


def alert(J: Path, seen: int) -> int:
    n = restart_lines(J)
    if n > seen:
        log(f"ALERT: {n - seen} new retry/restart/re-sent/preemption line(s) in run.log ({n} in all); not a stop: the job's "
            f"leases stop two live writers, and it fails unless each input completed exactly once")
    return n
state = {"run": None, "id": None, "cleaning": False, "armed": False}


class Abort(Exception):
    pass


def say(msg: str) -> None:
    """Diagnostics never raise."""
    line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
    try:
        with open(state["log"], "a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass
    try:
        print(line, flush=True)
    except Exception:
        pass


def log(msg: str) -> None:
    """Must succeed while the run is guarded; a failure aborts into cleanup."""
    with open(state["log"], "a") as fh:
        fh.write(f"{datetime.now().strftime('%H:%M:%S')} {msg}\n")


def cli(args, timeout):
    """One modal CLI call in its own process group; the group is killed on timeout. Returns (rc, stdout)."""
    try:
        p = subprocess.Popen(UVM + ["modal"] + args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                             start_new_session=True)
    except OSError:
        return 125, ""
    try:
        out, _ = p.communicate(timeout=timeout)
        return p.returncode, out
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            p.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return 124, ""


def app_state(app_id):
    """The exact app's state string from `modal app list --json`, or None if not listed or the listing failed."""
    rc, out = cli(["app", "list", "--json"], 60)
    if rc != 0:
        return None
    try:
        rows = json.loads(out)
        return next((str(a["state"]) for a in rows if isinstance(a, dict) and a.get("app_id") == app_id), None)
    except (ValueError, KeyError, TypeError):
        return None


def kill_group(pid: int, sig) -> None:
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def cleanup(reason: str) -> int:
    for s in SIGS:
        signal.signal(s, signal.SIG_IGN)
    if state["cleaning"]:
        return 2
    state["cleaning"] = True
    say(f"FAIL CLOSED: {reason}")
    ok = True
    run = state["run"]
    if run is not None:
        kill_group(run.pid, signal.SIGTERM)
        try:
            run.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pass
        kill_group(run.pid, signal.SIGKILL)  # descendants too, whether or not the leader has exited
        try:
            run.wait(timeout=20)
        except subprocess.TimeoutExpired:
            ok = False
    if state["id"]:
        stopped = False
        for _ in range(3):
            cli(["app", "stop", state["id"], "--yes"], 60)
            st = app_state(state["id"])
            if st is not None and st.startswith("stopped"):
                stopped = True
                break
        ok = ok and stopped
    else:
        ok = False  # no app ID seen: remote state can't be verified
    say(f"cleanup {'confirmed' if ok else 'NOT CONFIRMED: check the app by hand'}")
    return 1 if ok else 2


def cost_ok(path: Path) -> bool:
    """The cost watcher has logged a successful poll: a parsed record with fails == 0."""
    if not path.is_file():
        return False
    for line in path.read_text(errors="replace").splitlines():
        try:
            if json.loads(line).get("fails") == 0:
                return True
        except (ValueError, AttributeError):
            pass
    return False


def on_signal(signum, _frame):
    """Raises only while guarded() is running, and only once; otherwise the signal is ignored."""
    if state["armed"]:
        state["armed"] = False
        raise Abort(f"interrupted ({signal.Signals(signum).name})")


def guarded(J: Path) -> int:
    state["armed"] = True
    try:
        return _guarded(J)
    finally:
        state["armed"] = False


def _guarded(J: Path) -> int:
    t0 = time.monotonic()
    signal.pthread_sigmask(signal.SIG_BLOCK, SIGS)
    try:
        with open(J / "run.log", "x") as out:
            state["run"] = subprocess.Popen(UVM + ["--with", "numpy", "modal", "run", "run_sce_fixfull.py"], cwd=HERE,
                                            env=os.environ | {"PYTHONPATH": str(HERE.parent.parent / "src")},
                                            stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, SIGS)
    (J / "run.pid").write_text(f"{state['run'].pid}\n")
    log(f"launched modal run, PID {state['run'].pid}")
    # modal 1.6.1 prints the URL on the line after "View run at" (as run 2026-10-13 the one-line pattern missed it);
    # this pattern matched the right app in all 1,642 captured single-app `modal run` logs, the old one in none.
    pat = re.compile(r"View run at\s+\S*?(ap-[A-Za-z0-9]+)")
    while state["id"] is None:
        if time.monotonic() - t0 >= 60:
            raise Abort("no app ID in the run's own output within 60 s")
        if state["run"].poll() is not None:
            raise Abort(f"modal run exited (rc {state['run'].returncode}) before printing an app ID")
        m = pat.search((J / "run.log").read_text(errors="replace"))
        if m and time.monotonic() - t0 < 60:
            state["id"] = m.group(1)
        else:
            time.sleep(1)
    log(f"app {state['id']} (from the run's own output)")
    stop_at = str(int(time.time()) + 3600)
    cmds = {"cost": UVM + ["python", str(J / "watch_cost.py"), "--app-id", state["id"], "--stop", "3", "--rate", "4.9",
                           "--cap-min", "32", "--log", str(J / "cost.log")],
            "billing": ["uv", "run", "--no-project", "--python", "3.12", "python", str(J / "watch_billing_strict.py"),
                        state["id"], "1.8", stop_at],
            "guard": UVM + ["python", str(J / "guard_containers.py"), "--app-id", state["id"], "--max", "6", "--log",
                            str(J / "guard.log")]}
    procs = {}
    for name, cmd in cmds.items():
        with open(J / f"{name}.out", "x") as out:
            procs[name] = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        with open(J / "watchers.pid", "a") as fh:
            fh.write(f"{name} {procs[name].pid}\n")
    log("watchers started: " + ", ".join(f"{n} {p.pid}" for n, p in procs.items()))
    marks, t1 = {"guard": ("guard.log", ", live containers"), "billing": ("billing.out", "READY")}, time.monotonic()
    ready = set()
    while len(ready) < len(marks) + 1:  # every watcher must show its first poll of this app ID within 60 s
        if time.monotonic() - t1 >= 60:
            raise Abort(f"watchers not ready within 60 s: {sorted((set(marks) | {'cost'}) - ready)}")
        if [n for n, p in procs.items() if p.poll() is not None]:
            raise Abort("a watcher exited before its first poll")
        ready = {n for n, (f, mark) in marks.items() if (J / f).is_file() and mark in (J / f).read_text(errors="replace")}
        ready |= {"cost"} if cost_ok(J / "cost.log") else set()
        time.sleep(1)
    log(f"watchers ready after {time.monotonic() - t1:.0f} s")
    seen = 0  # retry/restart/preemption lines already alerted on
    while state["run"].poll() is None:
        seen = alert(J, seen)
        down = [n for n, p in procs.items() if p.poll() is not None]
        if down:
            try:  # a watcher may exit as the app stops at the end of a normal run; give the run 10 s to finish
                state["run"].wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                raise Abort(f"watcher(s) exited while the run was alive: {down}")
        time.sleep(5)
    seen = alert(J, seen)  # lines written after the last scan, before the run exited
    rc = state["run"].returncode
    log(f"modal run exited rc {rc}")
    return rc


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    J = Path(sys.argv[1]).resolve()
    state["log"] = J / "launch.log"
    for f in ("watch_cost.py", "watch_billing_strict.py", "guard_containers.py"):
        if not (J / f).is_file():
            print(f"missing {J / f}")
            return 2
    for s in SIGS:
        signal.signal(s, on_signal)
    try:
        rc = guarded(J)
    except BaseException as e:  # noqa: BLE001 - every failure after launch must clean up
        return cleanup(f"{type(e).__name__}: {e}")
    return 0 if rc == 0 else cleanup(f"modal run exited rc {rc}")


if __name__ == "__main__":
    sys.exit(main())
