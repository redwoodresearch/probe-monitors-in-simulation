"""Cost and start-failure watcher for the Qwen3.5 halves re-run (PLAN.md; 0d's rules after row 8(a)'s crash loop): a copy
of experiments/2026-10-11_claimB_clock_mask/watch_clock_mask.py (aswin/claimb-clockmask-plan) that follows ONE app by its
ID (--app-id): the Qwen3.5 extractor's app name is shared by the whole qwen35 track, so containers are matched by app ID,
never by name, and only that app is ever stopped. Derived from H2's watch_cost.py. Every 60 s it lists the app's containers and keeps every container it has seen. A
container's cost is its seen running time times the rate; posted billing for the app (today, local time) is read every
10 minutes, and the larger of the two counts.

It stops the app BY ITS ID (`modal app stop <app id>`) and prints a FLAG line when:
- the cost reaches --stop;
- a container passes --cap-min;
- containers keep restarting: more than --max-containers seen, or 3 or more containers gone within 60 s of starting\n  (the job's 9 containers each load data or the model for minutes; no control calls are made).
It prints a FLAG line, without stopping, when no container has started --grace-min minutes after the watcher began,
when a container stays pending longer than that, and when a container listing fails (the last valid state is kept).
Every line also goes to --log. It exits after 10 idle minutes once work was seen.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' python watch_halves_qwen35.py --app-id AP_ID --stop 30 \\
      --rate 5.4 --cap-min 60 --max-containers 12 --grace-min 10 --log PATH
"""

import argparse
import json
import subprocess
import time
from datetime import datetime


def sh(args):
    try:
        return subprocess.run(["uv", "run", "--no-project", "--python", "3.12", "--with", "modal>=1.4.3", "modal", *args],
                              capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 124, "", "timeout")


def ts(x):
    try:
        return datetime.fromisoformat(str(x).replace("Z", "+00:00")).timestamp() if isinstance(x, str) else float(x)
    except (TypeError, ValueError):
        return None


def billed(app_id):
    r = sh(["billing", "report", "--for", "today", "--tz", "local", "--resolution", "h", "--json"])
    if r.returncode:
        return None
    return sum(float(x.get("cost") or 0) for x in json.loads(r.stdout or "[]") if x.get("object_id") == app_id)


def emit(log, rec):
    line = json.dumps({"t": datetime.now().strftime("%H:%M:%S")} | rec)
    print(line, flush=True)
    with open(log, "a") as fh:
        fh.write(line + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--app-id", required=True)
    p.add_argument("--stop", type=float, required=True)
    p.add_argument("--rate", type=float, default=5.4)
    p.add_argument("--cap-min", type=float, default=60)
    p.add_argument("--max-containers", type=int, default=12)
    p.add_argument("--grace-min", type=float, default=10)
    p.add_argument("--log", required=True)
    a = p.parse_args()
    began, seen, app_id, last_bill, bill, idle, flagged_grace = time.time(), {}, a.app_id, 0.0, 0.0, 0, False
    pending, flagged_pending = {}, set()
    while True:
        r = sh(["container", "list", "--json"])
        now = time.time()
        try:
            listed = json.loads(r.stdout or "[]") if r.returncode == 0 else None
        except json.JSONDecodeError:
            listed = None
        if listed is None:  # an observation failure: flag it, keep the last valid state, try again
            emit(a.log, {"FLAG": f"container listing failed (return code {r.returncode})"})
            time.sleep(60)
            continue
        live = [c for c in listed if c.get("app_id") == app_id]
        live_ids = {c["container_id"] for c in live}
        for c in live:
            start = ts(c.get("start_time"))
            if start is None:  # still pending: not started, so not counted as running
                pending.setdefault(c["container_id"], now)
                continue
            seen.setdefault(c["container_id"], {"start": start})["last"] = now
        for k, t0 in pending.items():
            if k not in seen and k not in flagged_pending and now - t0 > a.grace_min * 60:
                emit(a.log, {"FLAG": f"container {k} pending for over {a.grace_min} min (failed to start?)"})
                flagged_pending.add(k)
        short = [k for k, v in seen.items() if k not in live_ids and v["last"] - v["start"] < 60]
        est = sum((v["last"] - v["start"]) / 3600 * a.rate for v in seen.values())
        if app_id and now - last_bill > 600:
            b = billed(app_id)
            bill, last_bill = (b if b is not None else bill), now
        total = max(est, bill)
        over = [k for k, v in seen.items() if v["last"] - v["start"] > a.cap_min * 60 and v["last"] == now]
        emit(a.log, {"app_id": app_id, "live": len(live), "seen": len(seen), "short_lived": len(short),
                     "est_usd": round(est, 2), "billed_usd": round(bill, 2), "total": round(total, 2)})
        reason = ("cost" if total >= a.stop else f"time cap {over}" if over else
                  f"restart loop: {len(seen)} containers seen" if len(seen) > a.max_containers else
                  f"restart loop: {len(short)} short-lived containers" if len(short) >= 3 else None)
        if reason:
            res = sh(["app", "stop", app_id, "--yes"]) if app_id else None
            emit(a.log, {"FLAG": reason, "stop_by_app_id": app_id, "returncode": None if res is None else res.returncode})
            if res is not None and res.returncode == 0:
                return
        if not seen and not flagged_grace and now - began > a.grace_min * 60:
            emit(a.log, {"FLAG": f"no container started within {a.grace_min} min"})
            flagged_grace = True
        idle = idle + 1 if (seen and not live and r.returncode == 0) else 0
        if idle >= 10:
            emit(a.log, {"done": True, "est_usd": round(est, 2), "billed_usd": round(bill, 2)})
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
