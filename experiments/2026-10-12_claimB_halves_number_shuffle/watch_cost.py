"""Elapsed-cost watcher for a Modal app (H2 PLAN.md, ece4fa14: "the stop is enforced on elapsed cost too"). Every 60 s
it lists the app's live containers and keeps every container it has ever seen, finished ones included. A container's
cost is (last time it was seen running, or now if still running) minus its start time, times the rate. When the total
reaches the stop, or any container passes its own time cap, it stops the app (`modal app stop --yes`). Posted billing
for the app is read every 10 minutes; the larger of billing and the estimate counts, so nothing is counted twice.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' python watch_cost.py --app-name NAME --stop 55 --rate 5.4 \\
      --cap-min 150 --log PATH
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone


def sh(args):
    try:
        return subprocess.run(["uv", "run", "--no-project", "--python", "3.12", "--with", "modal>=1.4.3", "modal", *args],
                              capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:  # v2 (10-12): a hung CLI call counts as a failed call, never an uncaught crash
        return subprocess.CompletedProcess(args, 124, "", "timeout")
    except OSError:  # v2: a launch failure is also a failed call
        return subprocess.CompletedProcess(args, 125, "", "launch failed")


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
    r = sh(["billing", "report", "--start", day, "--resolution", "h", "--json"])
    if r.returncode:
        return None
    rows = json.loads(r.stdout or "[]")
    return sum(float(x.get("cost", x.get("Cost", 0)) or 0) for x in rows if app_id in json.dumps(x))


def stop_until_done(name, log, why):
    """v2 (10-12): once a stop is due it is latched: retried every 20 s until `modal app stop` succeeds."""
    while True:
        res = sh(["app", "stop", name, "--yes"])
        try:
            with open(log, "a") as fh:
                fh.write(json.dumps({"stop_attempted": True, "returncode": res.returncode, "reason": why}) + "\n")
        except OSError:
            pass
        if res.returncode == 0:
            return
        time.sleep(20)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--app-name", required=True)
    p.add_argument("--stop", type=float, required=True)
    p.add_argument("--rate", type=float, default=5.4)
    p.add_argument("--cap-min", type=float, default=150)
    p.add_argument("--log", required=True)
    a = p.parse_args()
    seen, app_id, last_bill, bill, idle, fails, live, bfails = {}, None, 0.0, 0.0, 0, 0, [], 0
    while True:
        try:  # v2 (10-12): any error in a poll counts as a failed lookup (fail closed), never an uncaught exit
            r = sh(["container", "list", "--json"])
            now = time.time()
            if r.returncode != 0:
                raise RuntimeError(f"container list rc {r.returncode}")
            rows = json.loads(r.stdout)
            if not isinstance(rows, list):
                raise RuntimeError("container list: unexpected schema")
            live, ok = [c for c in rows if c.get("app_name") == a.app_name], True
            for c in live:
                if not isinstance(c.get("container_id"), str) or not isinstance(c.get("app_id"), str):
                    raise RuntimeError("container list: bad id")
                app_id = c["app_id"]
                start = ts(c.get("start_time")) or now  # a pending container is counted from when it is first seen
                seen.setdefault(c["container_id"], {"start": start})["last"] = now
        except Exception as e:
            ok, now = False, time.time()
            for v in seen.values():  # every container seen so far is assumed still running, so the estimate keeps rising
                v["last"] = now
            say(f"lookup failed: {type(e).__name__}")
        fails = 0 if ok else fails + 1
        est = sum((v["last"] - v["start"]) / 3600 * a.rate for v in seen.values())
        if app_id and now - last_bill > 600:
            try:
                b = billed(app_id, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
            except Exception:
                b = None
            bfails = bfails + 1 if b is None else 0  # billing failures are tracked apart, so container polls can't reset them
            bill, last_bill = (b if b is not None else bill), now
        total = max(est, bill)
        over = [k for k, v in seen.items() if v["last"] - v["start"] > a.cap_min * 60 and v["last"] == now]
        try:
            with open(a.log, "a") as fh:
                fh.write(json.dumps({"t": datetime.now().strftime("%H:%M:%S"), "live": len(live), "seen": len(seen), "est_usd": round(est, 2),
                                     "billed_usd": round(bill, 2), "total": round(total, 2), "fails": fails}) + "\n")
        except OSError:
            pass
        if total >= a.stop or over or fails >= 3 or bfails >= 3:  # latched: retried until the stop succeeds
            stop_until_done(a.app_name, a.log, "cost" if total >= a.stop else (f"time cap {over}" if over else f"{fails} failed lookups, {bfails} failed billing reads"))
            return
        idle = idle + 1 if (seen and not live and ok) else 0
        if idle >= 10:  # ten minutes with no live container after work was seen: done, and the app is stopped too
            stop_until_done(a.app_name, a.log, "done: no live container for 10 polls")
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
