"""Fix (b) cost watcher, v4. v2 and v3 added a fixed billing-lag allowance (5, then 3 minutes of every running
container) on top of billed spend that already covered most of it, so 12-13 freshly started containers tripped false
stops at a billed $6.15 (21:22 and 21:25 PT, 2026-10-11). v4, each poll (every 15 s):
- billed = Modal billing "today" (UTC, hourly, including the current partial hour; a fixed --start drops it) over the
  named apps, every hour;
- unbilled = max(0, container time seen since the watcher started at $5.5/h - billed growth in the hours from the
  watcher's start hour on). Late billing for older hours raises billed but offsets nothing. Start it in an hour with
  no earlier fix (b) runtime (the last run stopped at 04:26 UTC, 2026-10-12), before any spawn;
- stop every named live app when billed + unbilled >= stop - margin, margin = max(--reserve, 300 s of the running
  containers: the worst blocking chain from the last sample below the trigger to a stop, every command at its 45 s
  timeout, is about 240 s; stops run concurrently under one shared 45 s deadline); also when the app list
  or the billing report is unreadable 4 polls in a row, or the UTC date changes ("today" would drop earlier spend).
  Live includes deploying and initializing apps. It refuses to start if billing is unreadable.

Fix (b) cost watcher, v3: v2 with the billing-lag allowance cut from 5 to 3 minutes of the running containers. v2's
5 minutes of 12-13 containers (about $6) tripped a false stop at 21:22 PT on 2026-10-11, at a billed $6.15 of $10.75.

Fix (b) cost watcher, v2 (billing-based; replaces v1's container-time integral, which overstated billing by about a
third and, with its 4-minute margin, would have stopped the remaining jobs at a billed spend near ). Each poll: total =
Modal billing for today over all fix (b) apps + the running containers x .5/h x 5 minutes (billing lag); stop when
total >= stop - margin, margin = 1 minute of the running containers. Original v1 notes follow.

Fix (b) cost watcher. Every 30 s, the fix (b) apps' cost is the larger of Modal billing for today (hourly, lags) and
an integrated estimate: containers x $5.5/h (q3's watcher rate), counting each interval at the larger of its start and
end container counts. It stops every listed app when prior + cost reaches stop - margin, where the margin is the larger
of --reserve and 4 minutes of the current containers (covering command delays and sequential stops). It stops the
known app IDs even when the listing fails, retries, and exits non-zero unless the apps are confirmed down. Run from a frozen copy; one log line a poll.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' python watch_b.py --apps <name,...> --stop 10.75 --prior 0 \
      --reserve 0.5 --log <file>
  (--apps names every fix (b) app, stopped ones too, so billing covers the whole job; --prior stays 0.)
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone

RATE = 5.5 / 3600  # $ per container-second
LIVE = ("deployed", "running", "ephemeral", "stopping", "deploying", "initializing")


def run(cmd, timeout=45):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def g(a, *keys):  # modal's JSON keys differ by version: "App ID" / "app_id", "Description" / "description", ...
    for k in keys:
        if k in a:
            return a[k]
    return None


def live_apps(names):
    r = run(["modal", "app", "list", "--json"])
    if r is None or r.returncode != 0 or not (r.stdout or "").strip():
        return None  # unreadable or empty output counts as a failed listing, never as "no apps"
    try:
        rows = json.loads(r.stdout)
        if not isinstance(rows, list) or not rows:
            return None
        out = []
        for a in rows:
            if not isinstance(a, dict):
                return None
            d, st, i, t = g(a, "Description", "description"), g(a, "State", "state"), g(a, "App ID", "app_id"), g(a, "Tasks", "tasks")
            if not i or d is None or st is None or t is None:
                return None
            if isinstance(t, bool) or not ((isinstance(t, int)) or (isinstance(t, str) and t.isdigit())):
                return None  # modal 1.6.1 gives task counts as digit strings
            st, t = str(st).lower().split(" ")[0], int(t)  # e.g. "ephemeral (detached)" -> "ephemeral"
            if st not in LIVE + ("stopped", "disabled") or t < 0:
                return None
            if d in names and st in LIVE:
                out.append({"id": i, "tasks": t})
        return out
    except (ValueError, TypeError, AttributeError):
        return None


def billed_hours(names):  # {interval start: cost} over the named apps, or None if unreadable
    rep = run(["modal", "billing", "report", "--for", "today", "--resolution", "h", "--json"])
    try:
        out = {}
        for r in json.loads(rep.stdout):
            if r.get("description") in names:
                out[r["interval_start"]] = out.get(r["interval_start"], 0.0) + float(r["cost"])
        return out
    except Exception:
        return None


def log_line(log, text):
    with open(log, "a") as fh:
        fh.write(f"{datetime.now().strftime('%H:%M:%S')} {text}\n")


def main(apps: str, stop: float, prior: float, reserve: float, log: str):
    names = set(apps.split(","))
    integrated, last, prev_tasks, misses, bmisses, known = 0.0, time.time(), 0, 0, 0, set()
    start = datetime.now(timezone.utc)
    day, hour0, b0 = start.date(), start.strftime("%Y-%m-%dT%H:00:00"), billed_hours(names)
    if b0 is None:
        raise SystemExit("billing unreadable at start; not watching (launch nothing)")
    bh = b0
    log_line(log, f"v4 start: billed ${sum(b0.values()):.3f} by hour {b0}; start hour {hour0}; stop ${stop:.2f}, prior ${prior:.2f}; "
                  f"apps {sorted(names)}")
    while True:
        live = live_apps(names)
        now = time.time()
        if live is None:
            misses += 1
            tasks = prev_tasks  # unknown: assume no change, count it
        else:
            misses, tasks = 0, sum(a["tasks"] for a in live)
            known |= {a["id"] for a in live}
        integrated += max(prev_tasks, tasks) * RATE * (now - last)
        last, prev_tasks = now, tasks
        got = billed_hours(names)
        bmisses = bmisses + 1 if got is None else 0
        bh = bh if got is None else got
        grown = sum(max(0.0, c - b0.get(h, 0.0)) for h, c in bh.items() if h >= hour0)  # billing for time since the start hour
        unbilled = max(0.0, integrated - grown)
        cost = sum(bh.values()) + unbilled
        margin = max(reserve, tasks * RATE * 300)
        new_day = datetime.now(timezone.utc).date() != day
        log_line(log, f"apps {None if live is None else len(live)} tasks {tasks} integrated ${integrated:.3f} billed "
                      f"{'n/a' if got is None else f'${sum(got.values()):.3f}'} unbilled ${unbilled:.3f} total ${prior + cost:.3f} "
                      f"trigger ${stop - margin:.2f}")
        reason = ("cost" if prior + cost >= stop - margin else "UTC date changed" if new_day
                  else "app list unreadable 4 times" if misses >= 4 else "billing unreadable 4 times" if bmisses >= 4 else None)
        if reason:
            for attempt in range(6):
                listed = live_apps(names)
                ids = known if listed is None else {a["id"] for a in listed} | set()
                if listed is not None and not listed:
                    break
                procs = {i: subprocess.Popen(["modal", "app", "stop", i, "--yes"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                         for i in sorted(ids)}  # concurrent stops
                deadline = time.time() + 45  # one shared deadline for the concurrent stops
                for i, pr in procs.items():
                    try:
                        rc = pr.wait(timeout=max(0.0, deadline - time.time()))
                    except subprocess.TimeoutExpired:
                        pr.kill()
                        rc = None
                    log_line(log, f"stop {i} attempt {attempt + 1}: rc {rc}")
                time.sleep(10)
            left = live_apps(names)
            if left is None or left:
                log_line(log, f"SHUTDOWN UNCONFIRMED (reason: {reason}); still live or unreadable: {left}")
                raise SystemExit(2)
            log_line(log, f"STOPPED and confirmed down (reason: {reason})")
            return
        time.sleep(15)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps", required=True)
    ap.add_argument("--stop", type=float, required=True)
    ap.add_argument("--prior", type=float, default=0.0)
    ap.add_argument("--reserve", type=float, default=0.5)
    ap.add_argument("--log", required=True)
    main(**vars(ap.parse_args()))
