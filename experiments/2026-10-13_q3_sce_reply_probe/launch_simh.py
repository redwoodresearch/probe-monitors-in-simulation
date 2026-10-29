"""Launch a Modal job of RULE.md Amendment 13 (run_extract_g3_simh.py, run_sce_seed_grid.py) under watchers v2.5 (by app
ID) and the container guard. Fails closed. launch_sce_seed.py as at 2224f30b with the job and the watcher limits taken
from the command line (stop $, rate $/h per container, minutes cap, guard maximum, deadline seconds); nothing else differs.

  python3 -I launch_simh.py <job dir with frozen watch_cost.py, watch_billing_strict.py, guard_containers.py> \
      <stop $> <rate $/h> <cap minutes> <guard max> <deadline s> <script path> [script args ...]

A duplicate-writer check (0d's standing rule as refined at about 15:50 PT; see classify() below) is added: two live writers stop the run, a preemption restart is an alert that must be confirmed. The rest is launch_sce_seed.py's: the run in its own process group, its app ID from its own output, the three watchers
by that app ID with a first-poll check, and fail-closed cleanup that stops the app by ID.
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
# 0d's standing rule, refined (10-13 about 15:50 PT): STOP only on two live writers of one input; a preemption restart (the same
# input on a new container after the old one died) is an ALERT: logged, the old container's death confirmed (a container of
# this app left the live list within ALERT_WINDOW s of the line), and after the run one completed call per input confirmed
# (SIMH_POST_CHECK, a job-specific command). Either check failing stops and reports.
DUP = re.compile(r"(?i)\b(retry|retrying|retries|restart(ed|ing)?|re-?sent|resend(ing)?|preempt(ed|ion|ing)?)\b")  # restart signs
RPC_RETRY = re.compile(r"for \S+ \(.*\)\. Retrying")  # modal's benign RPC throttling retries ("for <RPC> (...). Retrying...")
TWO_WRITERS = re.compile(r"(?i)(\b(two|multiple|several) (live )?(containers|writers|calls)\b.*\b(same|one) (input|shard)\b"
                         r"|\b(second|duplicate) writer\b|\balready (running|being processed) (by|on) (another|a different) container\b)")
ALERT_WINDOW = 120  # seconds either side of a restart line in which some container of this app must have died


def classify(line: str):
    """'stop' (two live writers), 'alert' (a restart: retry, restart, re-sent input, preemption), or None."""
    if RPC_RETRY.search(line):
        return None
    if TWO_WRITERS.search(line):
        return "stop"
    return "alert" if DUP.search(line) else None


def confirm_gone(alert_t: float, death_times: list, window: float = ALERT_WINDOW) -> bool:
    """A restart is benign only if some container of this app left the live list within `window` s of the restart line."""
    return any(abs(t - alert_t) <= window for t in death_times)


def duplicate_writer_lines(text: str) -> list:
    """Lines classified as a stop or an alert (for tests and audits)."""
    return [l for l in text.splitlines() if classify(l)]



state = {"run": None, "id": None, "cleaning": False, "armed": False, "lim": None, "job": None}


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
            state["run"] = subprocess.Popen(UVM + ["--with", "numpy", "modal", "run"] + state["job"], cwd=HERE.parent.parent,
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
    stop, rate, cap_min, gmax, deadline = state["lim"]
    stop_at = str(int(time.time()) + int(deadline))
    cmds = {"cost": UVM + ["python", str(J / "watch_cost.py"), "--app-id", state["id"], "--stop", stop, "--rate", rate,
                           "--cap-min", cap_min, "--log", str(J / "cost.log")],
            "billing": ["uv", "run", "--no-project", "--python", "3.12", "python", str(J / "watch_billing_strict.py"),
                        state["id"], stop, stop_at],
            "guard": UVM + ["python", str(J / "guard_containers.py"), "--app-id", state["id"], "--max", gmax, "--log",
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
    seen_lines, live, deaths, alerts, last_poll = 0, None, [], [], 0.0

    def scan_and_poll(final=False):
        nonlocal seen_lines, live, last_poll
        now = time.time()
        if final or now - last_poll >= 10:  # the live list of this app's containers; a container that leaves it has died
            rc_, outp = cli(["container", "list", "--json"], 60)
            if rc_ == 0:
                try:
                    cur = {r["container_id"] for r in json.loads(outp) if r.get("app_id") == state["id"]}
                    if live is not None:
                        deaths.extend(now for _ in (live - cur))
                    live, last_poll = cur, now
                except (ValueError, KeyError, TypeError):
                    pass
        lines = (J / "run.log").read_text(errors="replace").splitlines()
        for l in lines[seen_lines:]:
            kind = classify(l)
            if kind == "stop":
                raise Abort("duplicate-writer check: two live writers of one input (run.log)")
            if kind == "alert":
                alerts.append({"t": now, "confirmed": False})
                log(f"ALERT: a restart line in run.log (preemption, retry or re-sent input); confirming the old container died")
        seen_lines = len(lines)
        for a in alerts:
            if not a["confirmed"] and confirm_gone(a["t"], deaths):
                a["confirmed"] = True
                log("ALERT confirmed benign so far: a container of this app left the live list within the window")
            elif not a["confirmed"] and now - a["t"] > ALERT_WINDOW:
                raise Abort("duplicate-writer check: a restart with no container leaving the live list (possible second writer)")

    while state["run"].poll() is None:
        scan_and_poll()
        down = [n for n, p in procs.items() if p.poll() is not None]
        if down:
            try:  # a watcher may exit as the app stops at the end of a normal run; give the run 10 s to finish
                state["run"].wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                raise Abort(f"watcher(s) exited while the run was alive: {down}")
        time.sleep(5)
    scan_and_poll(final=True)  # lines written after the last scan, before the run exited
    if any(not a["confirmed"] for a in alerts):
        raise Abort("duplicate-writer check: a restart at the end of the run with no container death confirmed")
    if alerts:  # after the run: one completed call per input, by the job's own check
        chk = os.environ.get("SIMH_POST_CHECK", "")
        if not chk:
            raise Abort("duplicate-writer check: restart alerts, and no SIMH_POST_CHECK command to confirm one result per input")
        pc = subprocess.run(chk, shell=True, cwd=HERE.parent.parent, capture_output=True, text=True, timeout=300)
        log(f"post-check (one completed call per input) rc {pc.returncode}: {pc.stdout.strip()[-300:]}")
        if pc.returncode:
            raise Abort("duplicate-writer check: the post-check found other than one completed call per input")
    rc = state["run"].returncode
    log(f"modal run exited rc {rc}")
    return rc


def main() -> int:
    if len(sys.argv) < 8:
        print(__doc__)
        return 2
    J = Path(sys.argv[1]).resolve()
    state["lim"] = tuple(sys.argv[2:7])
    for v in state["lim"]:
        float(v)  # numbers only
    state["job"] = sys.argv[7:]
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
