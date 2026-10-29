"""Read-only, ON a host (python3, streamed over ssh): simulator spend of every launch on this box that started at or
after a cutoff, from each launch.json the frozen launcher (run_pilot.py) writes. Keys and numbers only.

  python3 - <host label> <cutoff ISO, e.g. 2026-10-14T17:00:00-07:00> < sim_spend_read.py   -> one JSON line per launch

Per launch: run dir, started/finished, simulated-run counts, the last spend snapshot's actual_usd, and the attempts'
own prices (sim_usd once a finished attempt is priced from its log; live_usd otherwise). spend_usd = max(snapshot,
attempt sum), so attempts priced after the last snapshot are counted. Source: launch.json priced by the harness's
run_spend, not OpenAI-billed. A launch.json that does not parse or lacks the launcher's fields is reported, not skipped.
"""

import hashlib
import json
import os
import sys
from datetime import datetime

host, cutoff = sys.argv[1], datetime.fromisoformat(sys.argv[2])
read_time = datetime.now().astimezone().isoformat(timespec="seconds")
paths = []
for root, dirs, files in os.walk("/home/aswin"):  # launch.json sits at a run dir's top: at most 5 levels down
    depth = root.count("/") - "/home/aswin".count("/")
    dirs[:] = [] if depth >= 5 else [x for x in dirs if not x.startswith(".") and x not in ("node_modules", "venv")]
    if "launch.json" in files:
        paths.append(os.path.join(root, "launch.json"))
        dirs[:] = []  # a run dir's subfolders hold attempt logs, not launches
paths.sort()
for p in paths:
    if datetime.fromtimestamp(os.path.getmtime(p)).astimezone() < cutoff:
        continue  # written before the cutoff, so it cannot have started after it
    raw = open(p, "rb").read()
    row = {"host": host, "run_dir": os.path.dirname(p), "launch_json_sha256": hashlib.sha256(raw).hexdigest(),
           "launch_json_mtime": datetime.fromtimestamp(os.path.getmtime(p)).astimezone().isoformat(timespec="seconds"),
           "read_time": read_time}
    try:
        d = json.loads(raw)
        started = datetime.fromisoformat(d["started"])
    except Exception as e:  # reported, never dropped
        print(json.dumps(row | {"error": f"unreadable: {type(e).__name__}"}))
        continue
    if started < cutoff:
        continue
    runs = d.get("runs") or []
    runs = list(runs.values()) if isinstance(runs, dict) else runs
    sims = [r for r in runs if not r.get("real", True)]
    atts = [a for r in sims for a in r.get("attempts") or []]
    snaps = d.get("spend") or []
    last = snaps[-1] if snaps else {}
    priced = sum(a["sim_usd"] for a in atts if "sim_usd" in a)
    live = sum(a.get("sim_usd", a.get("live_usd", 0.0)) for a in atts)
    snap = last.get("actual_usd")
    row |= {"started": d["started"], "finished": d.get("finished"), "n_runs": len(runs), "n_sim_runs": len(sims),
            "n_sim_started": sum(bool(r.get("attempts")) for r in sims),
            "n_sim_done": sum(r.get("state") == "done" for r in sims),
            "n_sim_running": sum(r.get("state") == "running" for r in sims),
            "n_sim_attempts": len(atts), "n_sim_attempts_priced": sum("sim_usd" in a for a in atts),
            "n_spend_snapshots": len(snaps), "last_snapshot_time": last.get("time"), "snapshot_actual_usd": snap,
            "attempts_priced_usd": round(priced, 2), "attempts_all_usd": round(live, 2),
            "spend_usd": round(max(snap or 0.0, live), 2),
            "spend_source": "launch.json priced (harness run_spend), not OpenAI-billed",
            "has_launcher_fields": "spend" in d and "runs" in d}
    print(json.dumps(row))
