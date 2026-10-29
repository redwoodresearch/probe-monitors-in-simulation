"""The yardstick job's billing watcher, fail closed (q2's watch_e1.py with a UTC-safe window). Run from a frozen copy.
usage: python3 watch_yardstick.py <app id> <stop at $, whole app> <deadline, epoch s> <window start, UTC ISO hour>
Every 60 s: the app's bill, summed over its object_id in `modal billing report -r h --json` (modal 1.6.1) from the fixed
window start (the job's first launch minus an hour, UTC) to now + 2 h (UTC), and the app's state. Stops the app with
`modal app stop <id> --yes` when the bill reaches the stop, at the deadline, or after 3 billing reads in a row fail; a
stop is retried until the app shows as stopped. Exits once the app has had no tasks for 3 polls after working."""
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

CLI = ["uv", "run", "--no-project", "--with", "modal==1.6.1", "modal"]


def cli(*args, timeout=90):
    return subprocess.run(CLI + list(args), capture_output=True, text=True, timeout=timeout, check=True).stdout


def bill(app: str, start: str) -> float:
    end = (datetime.now(timezone.utc) + timedelta(hours=2)).strftime("%Y-%m-%dT%H:00:00")
    rows = json.loads(cli("billing", "report", "--start", start, "--end", end, "--resolution", "h", "--json"))
    return sum(float(r["cost"]) for r in (rows if isinstance(rows, list) else rows.get("data", rows)) if r.get("object_id") == app)


def state(app: str) -> tuple[str, str]:
    rows = json.loads(cli("app", "list", "--json"))
    a = next((a for a in (rows if isinstance(rows, list) else rows.get("data", rows)) if a.get("app_id") == app), None)
    return (a["state"], str(a["tasks"] if "tasks" in a else a["running_tasks"])) if a else ("gone", "0")


def stop(app: str, why: str):
    print(f"{time.strftime('%H:%M:%S')} STOP: {why}", flush=True)
    for _ in range(10):
        try:
            cli("app", "stop", app, "--yes", timeout=120)
            if state(app)[0] in ("stopped", "gone"):
                sys.exit(3)
        except (subprocess.SubprocessError, ValueError, KeyError) as e:  # fail closed: logged, then retried
            print(f"stop attempt failed: {type(e).__name__}", flush=True)
        time.sleep(20)
    print("could not confirm the stop after 10 attempts", flush=True)
    sys.exit(4)


def main(app: str, cap: str, deadline: str, start: str):
    cap, deadline, last, fails, seen, idle, ready = float(cap), float(deadline), 0.0, 0, False, 0, False
    while True:
        try:
            last, fails = bill(app, start), 0
        except (subprocess.SubprocessError, ValueError, KeyError) as e:  # counted; 3 in a row stop the app
            fails += 1
            print(f"billing read failed: {type(e).__name__}", flush=True)
        try:
            st, tasks = state(app)
        except (subprocess.SubprocessError, ValueError, KeyError):
            st, tasks = "unknown", "?"
        print(f"{time.strftime('%H:%M:%S')} state={st} tasks={tasks} bill={last:.3f} fails={fails} left={int(deadline - time.time())}s", flush=True)
        if not ready and fails == 0 and st == "deployed" and last < cap and time.time() < deadline:
            ready = True
            print("READY", flush=True)  # the launcher spawns only after this line
        if last >= cap or time.time() >= deadline or fails >= 3:
            stop(app, f"bill {last:.3f} (stop {cap}), deadline passed {time.time() >= deadline}, failed reads {fails}")
        if st in ("stopped", "gone"):
            sys.exit(0)
        seen, idle = (True, 0) if tasks not in ("0", "?") else (seen, idle + (seen and tasks == "0"))
        if idle >= 3:
            print("no tasks for 3 polls: finished", flush=True)
            sys.exit(0)
        time.sleep(60)


if __name__ == "__main__":
    main(*sys.argv[1:5])
