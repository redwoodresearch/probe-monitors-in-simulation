"""Launch run_null_shuffle.py under watchers v2.5 (by app ID) and the container guard (NULL_RULE.md). Fails closed.
Copied from experiments/2026-10-12_q3_sabotage_adjudication/launch_persist.py at f2030549 (cross-line app-ID parse);
only the job, its arguments and the watcher limits differ.

  python3 -I launch_null.py <job dir holding frozen watch_cost.py, watch_billing_strict.py, guard_containers.py> \\
      <learning_result_2026-10-12_144211.json>

- The `modal run` starts in its own process group, with INT/TERM blocked until it is recorded. Its app ID is read from
  its own output ("View run at …/ap-…"), so the watchers can't attach to another launch. Discovery reads only the
  local log, with a hard 60 s monotonic deadline (0d's condition).
- The three watchers start in their own process groups, and their PIDs go to watchers.pid. While the run is alive,
  every watcher must stay alive; it is checked every 5 s.
- Any failure, exception or INT/TERM runs cleanup once, with further signals ignored:
  - TERM to the run's process group; KILL to the group after 20 s, whether or not the leader has exited; then reap,
    with every wait bounded.
  - With an app ID: `modal app stop <id> --yes`, up to 3 tries of at most 60 s each, then require the exact ID to be
    listed with state "stopped".
  - Exit 1 when cleanup is confirmed, 2 when it isn't (including when no app ID was ever seen), so the operator checks.
- Independent of this launcher: each container has timeout 1800 s and retries 0 (about $0.23 each at 4 cores). The 18
  jobs could reach about $4.1 only if every one hit its timeout, so the $3 stop rests on the watchers.
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


def on_signal(signum, _frame):
    """Raises only while guarded() is running, and only once; otherwise the signal is ignored."""
    if state["armed"]:
        state["armed"] = False
        raise Abort(f"interrupted ({signal.Signals(signum).name})")


def guarded(J: Path, obs: Path) -> int:
    state["armed"] = True
    try:
        return _guarded(J, obs)
    finally:
        state["armed"] = False


def _guarded(J: Path, obs: Path) -> int:
    t0 = time.monotonic()
    signal.pthread_sigmask(signal.SIG_BLOCK, SIGS)
    try:
        with open(J / "run.log", "x") as out:
            state["run"] = subprocess.Popen(UVM + ["--with", "numpy", "modal", "run", "run_null_shuffle.py", "--observed", str(obs)], cwd=HERE, stdout=out, stderr=subprocess.STDOUT,
                                            start_new_session=True)
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
    stop_at = str(int(time.time()) + 1800)
    cmds = {"cost": UVM + ["python", str(J / "watch_cost.py"), "--app-id", state["id"], "--stop", "2.5", "--rate", "0.5",
                           "--cap-min", "30", "--log", str(J / "cost.log")],
            "billing": ["uv", "run", "--no-project", "--python", "3.12", "python", str(J / "watch_billing_strict.py"),
                        state["id"], "3", stop_at],
            "guard": UVM + ["python", str(J / "guard_containers.py"), "--app-id", state["id"], "--max", "24", "--log",
                            str(J / "guard.log")]}
    procs = {}
    for name, cmd in cmds.items():
        with open(J / f"{name}.out", "x") as out:
            procs[name] = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        with open(J / "watchers.pid", "a") as fh:
            fh.write(f"{name} {procs[name].pid}\n")
    log("watchers started: " + ", ".join(f"{n} {p.pid}" for n, p in procs.items()))
    while state["run"].poll() is None:
        down = [n for n, p in procs.items() if p.poll() is not None]
        if down:
            try:  # a watcher may exit as the app stops at the end of a normal run; give the run 10 s to finish
                state["run"].wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                raise Abort(f"watcher(s) exited while the run was alive: {down}")
        time.sleep(5)
    rc = state["run"].returncode
    log(f"modal run exited rc {rc}")
    return rc


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    J, obs = (Path(a).resolve() for a in sys.argv[1:3])
    state["log"] = J / "launch.log"
    for f in ("watch_cost.py", "watch_billing_strict.py", "guard_containers.py"):
        if not (J / f).is_file():
            print(f"missing {J / f}")
            return 2
    for s in SIGS:
        signal.signal(s, on_signal)
    try:
        return guarded(J, obs)
    except BaseException as e:  # noqa: BLE001 - every failure after launch must clean up
        return cleanup(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
