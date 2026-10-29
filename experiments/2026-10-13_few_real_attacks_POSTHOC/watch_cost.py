"""Elapsed-cost watcher for a Modal app (H2 PLAN.md, ece4fa14: "the stop is enforced on elapsed cost too"). Every 60 s
it lists the app's live containers and keeps every container it has ever seen, finished ones included. v2.3 (10-13):
the app is named by ID, not name (`modal app stop <name>` fails on ephemeral `modal run` apps, and steps sharing a name
were counted together), and every container of that app ID is counted. A fresh `modal run` app ID already separates
launches; for a deployed app shared by several steps, pass --since (epoch seconds, recorded BEFORE the launch) to count
only containers started at or after it. A container's
cost is (last time it was seen running, or now if still running) minus its start time, times the rate. When the total
reaches the stop, or any container passes its own time cap, it stops the app (`modal app stop <app id> --yes`). Posted billing
for the app is read every 10 minutes; the larger of billing and the estimate counts, so nothing is counted twice.

  uv run --no-project --python 3.12 --with 'modal==1.6.1' python watch_cost.py --app-id ap-… --stop 55 --rate 5.4 \\
      --cap-min 150 --log PATH [--since EPOCH_SECONDS]
"""

import argparse
import json
import math
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")  # v2.4: log stamps in Pacific Time (watchers run on the laptop, never in a container)


def run_group(cmd, timeout):
    """v2.5: one call in its own process group. On timeout the whole group (uv and the modal process under it) is killed,
    so a grandchild holding the pipes open can't hang the watcher (subprocess.run's timeout kills only the child), as in
    the strict watcher's cli(). A hung call returns rc 124 and a launch failure rc 125, both counted as failed calls."""
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    except OSError:
        return subprocess.CompletedProcess(cmd, 125, "", "launch failed")
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # the process group already exited between the timeout and the kill: nothing left to kill
        p.communicate()
        return subprocess.CompletedProcess(cmd, 124, "", "timeout")
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)


def sh(args, timeout=120):
    return run_group(["uv", "run", "--no-project", "--python", "3.12", "--with", "modal==1.6.1", "modal", *args], timeout)  # pinned, as strict


def say(msg):
    try:
        print(msg, flush=True)
    except Exception:  # v2: diagnostics are never fatal
        pass


def ts(x):
    """A container's start time, or None while it is still pending."""
    try:
        return datetime.fromisoformat(str(x).replace("Z", "+00:00")).timestamp() if isinstance(x, str) else float(x)
    except (TypeError, ValueError):
        return None


def billed(app_id, day):
    # v2.2 (10-13): the report reads dates in UTC and, without --end, left out the current hours (it read $0 for a whole
    # 14-minute run); end at the UTC date + 2 as the strict watcher does. `day` is the UTC date before the watcher started.
    end = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%d")
    r = sh(["billing", "report", "--start", day, "--end", end, "--resolution", "h", "--json"])
    if r.returncode:
        return None
    rows = json.loads(r.stdout)
    rows = rows.get("data") if isinstance(rows, dict) else rows
    if not isinstance(rows, list) or not all(isinstance(x, dict) for x in rows):  # v2.2: malformed is a failed read, never $0
        return None
    costs = [x.get("cost") for x in rows if x.get("object_id") == app_id]  # exact match, as the strict watcher
    if any(c is None for c in costs):  # a matching row without a cost is a failed read
        return None
    costs = [float(c) for c in costs]
    return sum(costs) if all(math.isfinite(c) and c >= 0 for c in costs) else None


def stop_until_done(app_id, log, why):
    """v2 (10-12): once a stop is due it is latched: retried every 20 s until `modal app stop` succeeds (v2.3: by app ID)."""
    while True:
        res = sh(["app", "stop", app_id, "--yes"])
        try:
            with open(log, "a") as fh:
                fh.write(json.dumps({"stop_attempted": True, "returncode": res.returncode, "reason": why}) + "\n")
        except OSError:  # a failed log write must never stop the watcher from enforcing its stop
            pass
        if res.returncode == 0:
            return
        lines = f"{res.stdout}\n{res.stderr}".splitlines()
        if res.returncode == 1 and any(x.strip().startswith("App is already stopped.") for x in lines):
            return  # v2.1 (10-12): another watcher stopped it first; Modal prints this only after reading the stopped state
        if app_state(app_id) == "stopped":  # v2.3: a failed stop on an app the list shows stopped is done (missing is not proof)
            return
        time.sleep(20)


def app_containers(app_id, timeout=120):
    """v2.4: this app ID's containers from `modal container list --json`; [] if none or if the lookup failed (no proof)."""
    r = sh(["container", "list", "--json"], timeout=timeout)
    try:
        rows = json.loads(r.stdout) if r.returncode == 0 else []
        return [c for c in rows if isinstance(c, dict) and c.get("app_id") == app_id and isinstance(c.get("container_id"), str)] if isinstance(rows, list) else []
    except Exception:
        return []


def app_listed(app_id, timeout=120):
    """v2.4: True if `modal app list --json` shows the ID, False if it worked and does not, None if the lookup failed."""
    r = sh(["app", "list", "--json"], timeout=timeout)
    try:
        rows = json.loads(r.stdout) if r.returncode == 0 else None
        rows = rows.get("data", rows) if isinstance(rows, dict) else rows
        if not isinstance(rows, list) or not all(isinstance(x, dict) and isinstance(x.get("app_id"), str) for x in rows):
            return None  # malformed rows are a failed lookup, never proof of absence
        return any(x["app_id"] == app_id for x in rows)
    except Exception:
        return None


def app_state(app_id):
    """v2.3: this app's state from `modal app list --json`, or None if the list fails or does not show it."""
    r = sh(["app", "list", "--json"])
    try:
        rows = json.loads(r.stdout) if r.returncode == 0 else None
        rows = rows.get("data", rows) if isinstance(rows, dict) else rows
        return next((x.get("state") for x in rows if isinstance(x, dict) and x.get("app_id") == app_id), None)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--app-id", required=True)  # v2.3: by ID; a name is ambiguous and fails to stop ephemeral apps
    p.add_argument("--since", type=float, default=None)  # v2.3: only for a shared deployed app: count containers started at or after this
    p.add_argument("--stop", type=float, required=True)
    p.add_argument("--rate", type=float, default=5.4)
    p.add_argument("--cap-min", type=float, default=150)
    p.add_argument("--log", required=True)
    p.add_argument("--startup-polls", type=int, default=10)  # v2.4: stop if no container of the app is ever seen in this many polls
    a = p.parse_args()
    if not re.fullmatch(r"ap-[a-zA-Z0-9]{22}", a.app_id):  # modal 1.6.1's own ID pattern; anything else is a name lookup
        raise SystemExit(f"--app-id must be a Modal app ID (ap- plus 22 letters or digits), got {a.app_id!r}")
    if not all(math.isfinite(v) for v in (a.stop, a.rate, a.cap_min) + ((a.since,) if a.since is not None else ())):
        raise SystemExit("--stop, --rate, --cap-min and --since must be finite")
    if a.startup_polls < 1:  # v2.4: 0 or less would stop the app before its first container could appear
        raise SystemExit("--startup-polls must be at least 1")
    if a.stop <= 0 or a.rate <= 0 or a.cap_min <= 0:  # v2.4: a zero or negative rate would switch off the elapsed estimate
        raise SystemExit("--stop, --rate and --cap-min must be positive")
    if a.since is not None and a.since > time.time():  # v2.4: a future cutoff would exclude the launch's containers
        raise SystemExit("--since is in the future")
    since = math.floor(a.since) if a.since is not None else -math.inf  # v2.4: Modal reports whole-second start times
    deadline, seen_listed, absent, gate_rows = time.monotonic() + 120, False, False, []  # v2.4: about 2 min to see the ID listed
    while time.monotonic() < deadline:
        cap = max(5.0, min(15.0, deadline - time.monotonic()))  # v2.4: gate calls are short, so a hung CLI can't blind it for minutes
        gate_rows = app_containers(a.app_id, timeout=cap)
        if gate_rows:  # v2.4: running containers prove the app exists; watch costs at once, no blind wait
            seen_listed = True
            break
        r = app_listed(a.app_id, timeout=cap)  # one lookup per try: True, False (a working list without it), or None (failed)
        if r:
            seen_listed = True
            break
        absent = absent or r is False  # v2.4: a working list's "absent" holds through later failed lookups (until a sighting)
        time.sleep(min(10, max(0.0, deadline - time.monotonic())))
    if not seen_listed and absent:  # every check that worked said the ID does not exist: there is nothing this watcher could stop
        say(f"{a.app_id} not in `modal app list` for about 2 min: not watching it (check the ID)")
        raise SystemExit(2)
    seen, app_id, last_bill, bill, idle, fails, live, bfails, unseen = {}, a.app_id, 0.0, 0.0, 0, 0, [], 0, 0
    running = set()  # v2.4: containers live at the last successful poll; only these are assumed still running after a failure
    for c in gate_rows:  # v2.4: containers seen at the gate count even if they are gone by the first poll
        start = ts(c.get("start_time")) or time.time()
        if start >= since:
            seen[c["container_id"]] = {"start": start, "last": time.time()}
            running.add(c["container_id"])  # seen live at the gate: assumed running until a good poll says otherwise
    day0 = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")  # v2.2: fixed start, so a run past UTC midnight keeps its earlier hours
    while True:
        mine = []  # this poll's containers of the app ID, whatever their start time
        try:  # v2 (10-12): any error in a poll counts as a failed lookup (fail closed), never an uncaught exit
            r = sh(["container", "list", "--json"])
            now = time.time()
            if r.returncode != 0:
                raise RuntimeError(f"container list rc {r.returncode}")
            rows = json.loads(r.stdout)
            if not isinstance(rows, list):
                raise RuntimeError("container list: unexpected schema")
            if any(not isinstance(c, dict) or not isinstance(c.get("container_id"), str) or not isinstance(c.get("app_id"), str) for c in rows):
                raise RuntimeError("container list: bad id")
            mine = [c for c in rows if c["app_id"] == a.app_id]
            live, ok = [c for c in mine if (ts(c.get("start_time")) or now) >= since], True  # v2.3: this app ID, this launch
            for c in live:
                start = ts(c.get("start_time")) or now  # a pending container is counted from when it is first seen
                seen.setdefault(c["container_id"], {"start": start})["last"] = now
            running = {c["container_id"] for c in live}
        except Exception as e:
            ok, now = False, time.time()
            for k in running:  # containers live at the last good poll are assumed still running; finished ones stay frozen
                seen[k]["last"] = now
            say(f"lookup failed: {type(e).__name__}")
        fails = 0 if ok else fails + 1
        est = sum((v["last"] - v["start"]) / 3600 * a.rate for v in seen.values())
        over = [k for k, v in seen.items() if v["last"] - v["start"] > a.cap_min * 60 and v["last"] == now]
        if est >= a.stop or over or fails >= 3:  # v2.4: a stop that is already due never waits for a billing read
            stop_until_done(a.app_id, a.log, "cost" if est >= a.stop else (f"time cap {over}" if over else f"{fails} failed lookups"))
            return
        if app_id and now - last_bill > 600:
            try:
                b = billed(app_id, day0)
            except Exception:
                b = None
            bfails = bfails + 1 if b is None else 0  # billing failures are tracked apart, so container polls can't reset them
            bill, last_bill = (b if b is not None else bill), now
        total = max(est, bill)
        try:
            with open(a.log, "a") as fh:
                fh.write(json.dumps({"t": datetime.now(PT).strftime("%H:%M:%S"), "live": len(live), "seen": len(seen), "est_usd": round(est, 2),
                                     "billed_usd": round(bill, 2), "total": round(total, 2), "fails": fails}) + "\n")
        except OSError:  # a failed log write must never stop the watcher from enforcing its stop
            pass
        if total >= a.stop or bfails >= 3:  # latched: retried until the stop succeeds
            stop_until_done(a.app_id, a.log, "cost" if total >= a.stop else f"{bfails} failed billing reads")
            return
        if ok and not seen:  # v2.4: successful polls with no container of the app at all; failed polls change nothing
            unseen = 0 if mine else unseen + 1  # containers from before --since mean the app is alive: never stop it for that
        if unseen >= a.startup_polls:  # v2.4: a listed app whose containers never appear is stopped, not watched forever
            stop_until_done(a.app_id, a.log, f"no container of this app seen in {unseen} polls")
            return
        idle = idle + 1 if (seen and not live and ok) else 0
        if idle >= 10:  # ten minutes with no live container after work was seen: done, and the app is stopped too
            stop_until_done(a.app_id, a.log, "done: no live container for 10 polls")
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
