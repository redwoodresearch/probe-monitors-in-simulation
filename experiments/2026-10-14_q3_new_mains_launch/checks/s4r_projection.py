"""ON BOX 4, read-only (d9's 4r end question, 10-15): project the last start of 4r's main from the frozen launcher's start
rule (strictly in launch.json order; an environment's launches stagger_s apart; a 30 s poll), starting from now and each
environment's latest real start. Real runs not yet started are taken in launch.json order. Also the observed start rate.
Keys and times only.  python3 s4r_projection.py <main run dir> <stagger_s>"""

import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
d = json.loads((Path(sys.argv[1]) / "launch.json").read_text())
stagger = float(sys.argv[2])
env = lambda r: r["key"].split("/")[-1].split("_", 1)[1].rsplit("_r2", 1)[0]
now = datetime.now(PT).timestamp()
last = {}
starts = []
for r in d["runs"]:
    for a in r["attempts"]:
        if a.get("start"):
            t = datetime.fromisoformat(a["start"]).timestamp()
            last[env(r)] = max(last.get(env(r), 0), t)
            if r["real"]:
                starts.append(t)
queue = [r for r in d["runs"] if r["real"] and r["state"] in ("ready", "waiting") and not r["attempts"]]
t = now
for r in queue:
    t = max(t, last.get(env(r), 0) + stagger)
    t = now + math.ceil((t - now) / 30) * 30
    last[env(r)] = t
recent = [s for s in starts if s >= now - 3600]
fmt = lambda x: datetime.fromtimestamp(x, PT).strftime("%H:%M")
print(json.dumps({"now": fmt(now), "real_not_started": len(queue), "projected_last_start_floor": fmt(t) if queue else None,
                  "observed_real_starts_last_60_min": len(recent),
                  "last_start_at_observed_rate": fmt(now + len(queue) / max(len(recent), 1) * 3600) if queue else None}))
