"""App-ID watchdog for the by-layer runs (RULE_POSTHOC_catch_by_layer.md; SPEND row 71893daa, stop at $2.50 for the job).
Every 30 s it stops the app (`modal app stop -y`, modal >= 1.4.3) when any of these holds:
- its wall time since this watchdog started passes max_minutes;
- this app's billed cost (billing report for today UTC, hourly; it lags) plus prior_usd reaches cap_usd;
- 3 polls in a row fail.
It exits once the app list shows the app stopped. Every CLI call is bounded (45 s). Run from a frozen copy:

  python watch_layers.py <app id> <prior usd> <cap usd> <max minutes>
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

app_id, prior, cap, max_min = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
M = [sys.executable, "-m", "modal"]
t0, fails = time.time(), 0


def log(msg: str):
    print(f"{datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(timespec='seconds')} {app_id} {msg}", flush=True)


def cli(args: list[str]) -> str:
    return subprocess.run(M + args, capture_output=True, text=True, timeout=45, check=True).stdout


def state() -> str:
    return next(a["state"] for a in json.loads(cli(["app", "list", "--json"])) if a["app_id"] == app_id)


def billed() -> float:
    return sum(float(r["cost"]) for r in json.loads(cli(["billing", "report", "--for", "today", "--resolution", "h", "--json"])) if r["object_id"] == app_id)


def stop(why: str):
    log(f"STOP ({why})")
    for _ in range(5):
        try:
            cli(["app", "stop", "-y", app_id])
            if state() == "stopped":
                log("stopped, confirmed by the app list")
                sys.exit(0)
        except Exception as e:  # noqa: BLE001 - logged; the stop is retried
            log(f"stop attempt failed: {e!r}")
        time.sleep(15)
    log("ALARM: could not confirm the stop")
    sys.exit(1)


log(f"watching: prior ${prior:.2f}, cap ${cap:.2f}, max {max_min} min")
while True:
    if (time.time() - t0) / 60 > max_min:  # checked first, whether or not the polls succeed
        stop(f"wall time {(time.time() - t0) / 60:.1f} min > {max_min}")
    try:
        st = state()
        cost = billed()
        fails = 0
    except Exception as e:  # noqa: BLE001 - counted; 3 in a row stop the app
        fails += 1
        log(f"poll failed ({fails}): {e!r}")
        if fails >= 3:
            stop("3 failed polls")
        time.sleep(30)
        continue
    mins = (time.time() - t0) / 60
    log(f"state {st}, {mins:.1f} min, billed ${cost:.3f} (+ prior ${prior:.2f})")
    if st == "stopped":
        log("app stopped on its own; done")
        sys.exit(0)
    if cost + prior >= cap:
        stop(f"billed ${cost + prior:.2f} >= ${cap:.2f}")
    time.sleep(30)
