"""App-ID watchdog for one planted-slip pair's apps (the by-layer run's watch_layers.py, extended to several app IDs whose
billed costs add up against one stop). Every 30 s it stops every app in the set (`modal app stop -y`, modal >= 1.4.3) when:
- the wall time since this watchdog started passes max_minutes;
- the apps' summed billed cost (billing report for today UTC, hourly; it lags) plus prior_usd reaches cap_usd;
- 3 polls in a row fail.
It exits once every app is listed as stopped; an app not (yet) in the list counts as running. A stop is attempted for
every app independently and retried until all are confirmed, with an ALARM line every 5 rounds. Every CLI call is bounded (45 s). Run from a frozen copy:

  python watch_apps.py <prior usd> <cap usd> <max minutes> <app id> [<app id> ...]
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

prior, cap, max_min, apps = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4:]
M = [sys.executable, "-m", "modal"]
t0, fails = time.time(), 0


def log(msg: str):
    print(f"{datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(timespec='seconds')} {','.join(apps)} {msg}", flush=True)


def cli(args: list[str]) -> str:
    return subprocess.run(M + args, capture_output=True, text=True, timeout=45, check=True).stdout


def states() -> dict:
    listed = {a["app_id"]: a["state"] for a in json.loads(cli(["app", "list", "--json"]))}
    return {a: listed.get(a, "not listed") for a in apps}


def billed() -> float:
    return sum(float(r["cost"]) for r in json.loads(cli(["billing", "report", "--for", "today", "--resolution", "h", "--json"])) if r["object_id"] in apps)


def stop(why: str):
    log(f"STOP ({why})")
    rounds = 0
    while True:
        rounds += 1
        for a in apps:  # each app's stop is attempted on its own, whatever happened to the others
            try:
                cli(["app", "stop", "-y", a])
            except Exception as e:  # noqa: BLE001 - logged; retried next round
                log(f"stop of {a} failed: {e!r}")
        try:
            st = states()
            if all(s == "stopped" for s in st.values()):
                log("all stopped, confirmed by the app list")
                sys.exit(0)
            log(f"not all stopped yet: {st}")
        except Exception as e:  # noqa: BLE001 - logged; retried next round
            log(f"state check failed: {e!r}")
        if rounds % 5 == 0:
            log(f"ALARM: stop not confirmed after {rounds} rounds; still retrying")
        time.sleep(15)


log(f"watching: prior ${prior:.2f}, cap ${cap:.2f}, max {max_min} min")
while True:
    if (time.time() - t0) / 60 > max_min:  # checked first, whether or not the polls succeed
        stop(f"wall time {(time.time() - t0) / 60:.1f} min > {max_min}")
    try:
        st, cost = states(), billed()
        fails = 0
    except Exception as e:  # noqa: BLE001 - counted; 3 in a row stop the apps
        fails += 1
        log(f"poll failed ({fails}): {e!r}")
        if fails >= 3:
            stop("3 failed polls")
        time.sleep(30)
        continue
    log(f"states {st}, {(time.time() - t0) / 60:.1f} min, billed ${cost:.3f} (+ prior ${prior:.2f})")
    if all(s == "stopped" for s in st.values()):
        log("all apps stopped on their own; done")
        sys.exit(0)
    if cost + prior >= cap:
        stop(f"billed ${cost + prior:.2f} >= ${cap:.2f}")
    time.sleep(30)
