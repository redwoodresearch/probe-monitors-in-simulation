"""Billing watcher, strict: watch_billing.py with the orchestrator's 10-12 rule (after a hung `modal` call blinded another
watcher for 13 minutes): every CLI call has a timeout, and the FIRST failed or erroring poll (billing or state) raises an
ALARM and stops the app. usage: python3 watch_billing_strict.py <app id> <stop at $, whole app> <deadline, epoch s>
       python3 watch_billing_strict.py --selftest   (must read the row 11 pilot app's settled $3.50 before any launch)
Every 60 s: the app's bill (modal 1.6.1 `billing report`, hourly, UTC-safe) and its state. Stops the app when the bill
reaches the threshold, at the deadline, or on any failed poll; a stop is retried until the app shows as stopped.
Exits once the app is idle for 3 polls after work."""
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

CLI = ["uv", "run", "--no-project", "--with", "modal==1.6.1", "modal"]


def say(msg):
    try:
        print(msg, flush=True)
    except Exception:  # v2 (10-12): diagnostics are never fatal
        pass


def cli(*args, timeout=60):
    """One CLI call in its own process group: on timeout the whole group (uv and the modal process under it) is killed,
    so a grandchild holding the pipes open can't hang the watcher; then TimeoutExpired is raised."""
    import os
    import signal
    p = subprocess.Popen(CLI + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.communicate()
        raise
    if p.returncode:
        raise subprocess.CalledProcessError(p.returncode, args, out, err)
    return out


def bill(app: str) -> float:
    # the report reads dates in UTC: end at the UTC date + 2 so every hour so far is inside (local + 1 cut off 17:00 PT)
    end = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%d")
    rows = json.loads(cli("billing", "report", "--start", "2026-10-11", "--end", end, "--resolution", "h", "--json"))
    return sum(float(r["cost"]) for r in (rows if isinstance(rows, list) else rows.get("data", rows)) if r.get("object_id") == app)


class AppMissing(Exception):
    """v2 (10-12): the app is not in `modal app list` (a dedicated type, so a KeyError is never read as "gone")."""


def state(app: str) -> tuple[str, str]:
    rows = json.loads(cli("app", "list", "--json"))
    a = next((a for a in (rows if isinstance(rows, list) else rows.get("data", rows)) if a.get("app_id") == app), None)
    if a is None:  # v2 (10-12): an app missing from the list is a failed lookup, never "gone" (it used to exit without a stop)
        raise AppMissing(f"{app} not in app list")
    return (a["state"], str(a["tasks"] if "tasks" in a else a["running_tasks"]))


def stop(app: str, why: str):
    say(f"{time.strftime('%H:%M:%S')} STOP: {why}")
    attempt = 0
    while True:  # never gives up: retried until the app shows as stopped
        attempt += 1
        try:
            cli("app", "stop", app, "--yes", timeout=120)
            try:
                st = state(app)[0]
            except AppMissing:  # stop was issued and the app is no longer listed
                st = "gone"
            if st in ("stopped", "gone"):
                sys.exit(3)
        except Exception as e:  # fail closed: logged, then retried (sys.exit's SystemExit is not an Exception)
            say(f"stop attempt {attempt} failed: {type(e).__name__}")
        time.sleep(20)


def main(app: str, cap: str, deadline: str):
    cap, deadline, last, seen, idle, ready = float(cap), float(deadline), 0.0, False, 0, False
    while True:
        try:
            last = bill(app)
            st, tasks = state(app)
        except Exception as e:  # any failed or erroring poll (a timeout included): alarm, then stop
            say(f"{time.strftime('%H:%M:%S')} ALARM: poll failed ({type(e).__name__}); stopping the app")
            stop(app, f"failed poll ({type(e).__name__}); last good bill {last:.3f}")
        say(f"{time.strftime('%H:%M:%S')} state={st} tasks={tasks} bill={last:.3f} left={int(deadline - time.time())}s")
        if not ready and st in ("deployed", "ephemeral") and last < cap and time.time() < deadline:  # modal run apps are ephemeral
            ready = True
            say("READY")
        if last >= cap or time.time() >= deadline:
            say(f"{time.strftime('%H:%M:%S')} ALARM: bill {last:.3f} (stop {cap}) or deadline reached; stopping the app")
            stop(app, f"bill {last:.3f} (stop {cap}), deadline passed {time.time() >= deadline}")
        if st in ("stopped", "gone"):
            sys.exit(0)
        seen, idle = (True, 0) if tasks != "0" else ((seen, idle + 1) if seen else (seen, 0))
        if idle >= 3:  # v2 (10-12): finished work still ends in a stop, so an app that respawns is never left unwatched
            stop(app, "no tasks for 3 polls: finished")
        time.sleep(60)


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        b = bill("ap-PcL8X4bXVggJqkQMhDdQg7")  # the row 11 pilot, billed $3.50 on 10-11 20:49-22:30 PT
        print(f"selftest bill {b:.4f}")
        sys.exit(0 if abs(b - 3.5008) < 0.01 else 5)
    main(*sys.argv[1:4])
