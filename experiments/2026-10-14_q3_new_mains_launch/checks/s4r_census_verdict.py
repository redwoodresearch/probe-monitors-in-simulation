"""Laptop, read-only (d9's census rule for 4r's follow-up; Codex r26 items 2-3): PASS or FAIL for one committed census
file (results/sim_error_census_2026-10-15_<HHMM>.json, written by census_commit.sh). PASS needs: all 8 boxes present
with data; no launch read error; every launch carries the exact last-30-minute count (sim_error_census.py "last30");
RateLimitError runs < 5% of simulated runs ended in the exact last 30 min (0 of 0 passes: no simulator traffic); fewer
than 60 simulated runs running; every box non-empty (Codex r29 item 1); the census at most MAX_AGE_MIN old, its age taken
from the OLDEST per-launch observation time (last30.now), not the packaging time (r29 item 2). Prints "PASS|FAIL
<details>" and, on the second line, that oldest observation epoch.
  python3 s4r_census_verdict.py <census json> <max age, minutes>"""

import json
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
BOXES = {"box1", "box2", "box3", "box4", "xbox1", "xbox2", "xbox3", "simsastra"}
c = json.load(open(sys.argv[1]))
pb = c["per_box"]
problems = [f"boxes differ: {sorted(set(pb) ^ BOXES)}"] if set(pb) != BOXES else []
e30 = rl30 = running = 0
seen = []
for b, v in sorted(pb.items()):
    if not isinstance(v, dict) or not v:
        problems.append(f"{b}: no data")
        continue
    for launch, x in v.items():
        if "error" in x or "last30" not in x:
            problems.append(f"{b}/{launch}: {'read error' if 'error' in x else 'no exact count'}")
            continue
        e30, rl30, running = e30 + x["last30"]["ended"], rl30 + x["last30"]["rl_runs"], running + x["sim_running"]
        seen.append(datetime.fromisoformat(x["last30"]["now"]))
t = min(seen) if seen else datetime(1970, 1, 1, tzinfo=PT)
age = (datetime.now(PT) - t).total_seconds() / 60
ok = not problems and 100 * rl30 < 5 * max(e30, 1) and (rl30 == 0 or e30 > 0) and running < 60 and age <= float(sys.argv[2])
print(f"{'PASS' if ok else 'FAIL'} RateLimitError runs {rl30} of {e30} ended (exact last 30 min), simulated running {running}, "
      f"oldest observation {t:%T} PT, age {age:.1f} min{', problems ' + '; '.join(problems) if problems else ''}")
print(int(t.timestamp()))
