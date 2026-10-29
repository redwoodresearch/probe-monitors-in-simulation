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
    return subprocess.run(["uv", "run", "--no-project", "--python", "3.12", "--with", "modal>=1.4.3", "modal", *args],
                          capture_output=True, text=True, timeout=120)


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--app-name", required=True)
    p.add_argument("--stop", type=float, required=True)
    p.add_argument("--rate", type=float, default=5.4)
    p.add_argument("--cap-min", type=float, default=150)
    p.add_argument("--log", required=True)
    a = p.parse_args()
    seen, app_id, last_bill, bill, idle = {}, None, 0.0, 0.0, 0
    while True:
        r = sh(["container", "list", "--json"])
        now = time.time()
        live = [c for c in json.loads(r.stdout or "[]") if c.get("app_name") == a.app_name] if r.returncode == 0 else []
        for c in live:
            app_id = c["app_id"]
            start = ts(c["start_time"]) or now  # a pending container is counted from when it is first seen
            seen.setdefault(c["container_id"], {"start": start})["last"] = now
        est = sum((v["last"] - v["start"]) / 3600 * a.rate for v in seen.values())
        if app_id and now - last_bill > 600:
            b = billed(app_id, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
            bill, last_bill = (b if b is not None else bill), now
        total = max(est, bill)
        over = [k for k, v in seen.items() if v["last"] - v["start"] > a.cap_min * 60 and v["last"] == now]
        line = {"t": datetime.now().strftime("%H:%M:%S"), "live": len(live), "seen": len(seen), "est_usd": round(est, 2),
                "billed_usd": round(bill, 2), "total": round(total, 2)}
        with open(a.log, "a") as fh:
            fh.write(json.dumps(line) + "\n")
        if total >= a.stop or over:
            res = sh(["app", "stop", a.app_name, "--yes"])
            with open(a.log, "a") as fh:
                fh.write(json.dumps({"stop_attempted": True, "returncode": res.returncode, "reason": "cost" if total >= a.stop else f"time cap {over}"}) + "\n")
            if res.returncode == 0:
                return
        idle = idle + 1 if (seen and not live and r.returncode == 0) else 0
        if idle >= 10:  # ten minutes with no live container after work was seen: the spawned calls are done
            with open(a.log, "a") as fh:
                fh.write(json.dumps({"done": True, "est_usd": round(est, 2)}) + "\n")
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
