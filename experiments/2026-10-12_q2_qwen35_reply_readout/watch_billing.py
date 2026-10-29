"""Billing watcher, fail closed, UTC-safe. usage: python3 watch_billing.py <app id> <stop at $, whole app> <deadline, epoch s>
       python3 watch_billing.py --selftest   (must read the row 11 pilot app's settled $3.50 before any launch)
Every 60 s: the app's bill (modal 1.6.1 `billing report`, hourly) and its state, each CLI call with a timeout. Stops the
app when the bill reaches the threshold, at the deadline, or after 3 billing reads in a row fail (the last good bill is
kept meanwhile); a stop is retried until the app shows as stopped. Exits once the app is idle for 3 polls after work."""
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

CLI = ["uv", "run", "--no-project", "--with", "modal==1.6.1", "modal"]


def cli(*args, timeout=60):
    return subprocess.run(CLI + list(args), capture_output=True, text=True, timeout=timeout, check=True).stdout


def bill(app: str) -> float:
    # the report reads dates in UTC: end at the UTC date + 2 so every hour so far is inside (local + 1 cut off 17:00 PT)
    end = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%d")
    rows = json.loads(cli("billing", "report", "--start", "2026-10-11", "--end", end, "--resolution", "h", "--json"))
    return sum(float(r["cost"]) for r in (rows if isinstance(rows, list) else rows.get("data", rows)) if r.get("object_id") == app)


def state(app: str) -> tuple[str, str]:
    rows = json.loads(cli("app", "list", "--json"))
    a = next((a for a in (rows if isinstance(rows, list) else rows.get("data", rows)) if a.get("app_id") == app), None)
    return (a["state"], str(a["tasks"] if "tasks" in a else a["running_tasks"])) if a else ("gone", "0")


def stop(app: str, why: str):
    print(f"{time.strftime('%H:%M:%S')} STOP: {why}", flush=True)
    attempt = 0
    while True:  # never gives up: retried until the app shows as stopped
        attempt += 1
        try:
            cli("app", "stop", app, "--yes", timeout=120)
            if state(app)[0] in ("stopped", "gone"):
                sys.exit(3)
        except (subprocess.SubprocessError, ValueError, KeyError) as e:  # fail closed: logged, then retried
            print(f"stop attempt {attempt} failed: {type(e).__name__}", flush=True)
        time.sleep(20)


def main(app: str, cap: str, deadline: str):
    cap, deadline, last, fails, seen, idle, ready = float(cap), float(deadline), 0.0, 0, False, 0, False
    while True:
        try:
            last, fails = bill(app), 0
        except (subprocess.SubprocessError, ValueError, KeyError) as e:  # counted; 3 in a row stop the app
            fails += 1
            print(f"billing read failed: {type(e).__name__}", flush=True)
        try:
            st, tasks = state(app)
        except (subprocess.SubprocessError, ValueError, KeyError):
            st, tasks = "unknown", "?"
        print(f"{time.strftime('%H:%M:%S')} state={st} tasks={tasks} bill={last:.3f} fails={fails} left={int(deadline - time.time())}s", flush=True)
        if not ready and fails == 0 and st in ("deployed", "ephemeral") and last < cap and time.time() < deadline:  # modal run apps are ephemeral
            ready = True
            print("READY", flush=True)  # the launcher spawns only after this line
        if last >= cap or time.time() >= deadline or fails >= 3:
            stop(app, f"bill {last:.3f} (stop {cap}), deadline passed {time.time() >= deadline}, failed reads {fails}")
        if st in ("stopped", "gone"):
            sys.exit(0)
        seen, idle = (True, 0) if tasks not in ("0", "?") else ((seen, idle + 1) if seen and tasks == "0" else (seen, 0))  # an unknown state resets the count
        if idle >= 3:
            print("no tasks for 3 polls: finished", flush=True)
            sys.exit(0)
        time.sleep(60)


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        b = bill("ap-PcL8X4bXVggJqkQMhDdQg7")  # the row 11 pilot, billed $3.50 on 10-11 20:49-22:30 PT
        print(f"selftest bill {b:.4f}")
        sys.exit(0 if abs(b - 3.5008) < 0.01 else 5)
    main(*sys.argv[1:4])
