"""ON BOX 4, read-only (d9's 4r end estimate, 10-15): progress of 4r's main real runs from its launch.json: states, real
attempts started and ended per 15 minutes (PT), run lengths of ended attempts (median, p90, max, minutes), running and
not-yet-started runs per environment, and the oldest running attempt. Keys and times only. Prints one JSON object.
  python3 s4r_progress.py <main run dir>"""

import collections
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
d = json.loads((Path(sys.argv[1]) / "launch.json").read_text())
real = [r for r in d["runs"] if r["real"]]
now = datetime.now(PT)
states = collections.Counter(r["state"] for r in real)
started, ended, lengths, running_age = collections.Counter(), collections.Counter(), [], []
env = lambda r: r["key"].split("/")[-1].split("_", 1)[1].rsplit("_r2", 1)[0]  # nm123_<environment>[_r2]
left = collections.Counter(env(r) for r in real if r["state"] in ("ready", "waiting"))
run_env = collections.Counter(env(r) for r in real if r["state"] == "running")
for r in real:
    for a in r["attempts"]:
        if a.get("start"):
            s = datetime.fromisoformat(a["start"]).astimezone(PT)
            started[f"{s:%H}:{s.minute // 15 * 15:02d}"] += 1
            if a.get("end"):
                e = datetime.fromisoformat(a["end"]).astimezone(PT)
                ended[f"{e:%H}:{e.minute // 15 * 15:02d}"] += 1
                lengths.append((e - s).total_seconds() / 60)
            elif r["state"] == "running":
                running_age.append((now - s).total_seconds() / 60)
q = statistics.quantiles(lengths, n=10) if len(lengths) >= 10 else []
print(json.dumps({
    "now_pt": f"{now:%F %T}", "states": dict(states), "terminal": sum(states[s] for s in ("done", "no policy call", "not run")),
    "attempts_started_per_15min": dict(sorted(started.items())), "attempts_ended_per_15min": dict(sorted(ended.items())),
    "length_min": {"n": len(lengths), "median": round(statistics.median(lengths), 1) if lengths else None,
                   "p90": round(q[-1], 1) if q else None, "max": round(max(lengths), 1) if lengths else None},
    "running_oldest_min": round(max(running_age), 1) if running_age else None,
    "not_started_by_env": dict(left.most_common()), "running_by_env": dict(run_env.most_common())}, indent=1))
