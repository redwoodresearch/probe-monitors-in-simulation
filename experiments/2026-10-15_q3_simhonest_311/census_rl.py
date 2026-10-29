"""census_rl.py: the +311 simulated-honest start gate and rate-limit pause test, read from one committed sim-error census
(origin/aswin/q3-new-mains-launch experiments/2026-10-14_q3_new_mains_launch/results/sim_error_census_*.json) on stdin.
Rule (d9 09:27 PT 10-15, 0d raised the start cap 60 -> 90 for speed):
  gate:  census <= 30 min old and complete (no box without data, no read errors, no launch without the exact count),
         < 90 sims running, and NO RateLimitError in the latest 10-min bin. The bin labels' edge convention is not relied
         on: every bin labelled within the 20 min before the census time (or later) must show 0 RateLimitError, which
         covers the latest complete bin and the one in progress under either convention.
  pause: any 10-min bin labelled at or after --since (rounded down to its 10-min bin) with ended > 0 and RateLimitError / ended >= 2%  -> exit 3.
Bins come from the census's own "bins (end time PT): HH:MM rl/ended, ..." line; a label later than the census time
+ 10 min is from the day before (the list wraps midnight).
  python3 census_rl.py gate < census.json            exit 0 pass, 1 fail
  python3 census_rl.py pause --since 'YYYY-mm-dd HH:MM' < census.json   exit 0 none, 3 pause
"""
import ast
import datetime as dt
import json
import re
import sys
from zoneinfo import ZoneInfo

POWERED_OFF_KEY = "powered_off_boxes"  # field name agreed with the new-mains fork
d = json.load(sys.stdin)
mode = sys.argv[1]
t = d["census_time_pt"]
ct = dt.datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
lines = d["rates_text"].split("\n")
head = lines[0]
(bins_line,) = [line for line in lines if line.startswith("bins (end time PT):")]
entries = bins_line.removeprefix("bins (end time PT):").strip().split(", ")
bad = [e for e in entries if not re.fullmatch(r"\d\d:\d\d \d+/\d+", e)]
assert not bad, f"malformed bin entries {bad[:3]}"  # Codex r8: never skip an entry
bins = []
for hhmm, rl, ended in (re.fullmatch(r"(\d\d:\d\d) (\d+)/(\d+)", e).groups() for e in entries):
    b = dt.datetime.combine(ct.date(), dt.time.fromisoformat(hhmm))
    if b > ct + dt.timedelta(minutes=10):
        b -= dt.timedelta(days=1)
    bins.append((b, int(rl), int(ended)))
assert bins, "no bins parsed"

if mode == "gate":
    now = dt.datetime.now(ZoneInfo("America/Los_Angeles")).replace(tzinfo=None)
    age = (now - ct).total_seconds() / 60
    running = int(re.search(r"simulated running (\d+)", head).group(1))
    lists = {k: re.search(rf"{k} (\[[^]]*\])", head).group(1) for k in ("boxes without data", "launch read errors", "launches without the exact count")}
    recent = [(b.strftime("%H:%M"), rl, n) for b, rl, n in bins if b >= ct - dt.timedelta(minutes=20)]
    rl_recent = sum(rl for _, rl, _ in recent)
    # d9 10:1x: a box without data counts as complete only if this census records it as powered off with 0 launches
    # (POWERED_OFF_KEY, the new-mains fork's fresh EC2 read at census time): state stopped/terminated, 0 running launches,
    # read within 30 min of the census time. Any other no-data box fails the gate, as before.
    nodata = ast.literal_eval(lists["boxes without data"])
    off = d.get(POWERED_OFF_KEY) or {}
    def off_ok(box):
        e = off.get(box)
        if not isinstance(e, dict) or e.get("state") not in ("stopped", "terminated") or e.get("running_launches") != 0:
            return False
        read = dt.datetime.strptime(e["read_pt"], "%Y-%m-%d %H:%M:%S")
        return abs((ct - read).total_seconds()) <= 1800
    nodata_ok = isinstance(nodata, list) and all(isinstance(b, str) and off_ok(b) for b in nodata)
    offrec = {b: {k: off.get(b, {}).get(k) for k in ("instance", "state", "read_pt", "running_launches")} for b in nodata}
    print(f"census {t} age {age:.1f} min | running {running} (< 90) | recent bins (label, RL, ended) {recent} | {lists} | powered off: {offrec}")
    # Codex r8: no future-dated census (2 min clock skew); r8b: the RL and completeness predicates stay in force
    ok = (-2 <= age <= 30 and running < 90 and rl_recent == 0 and nodata_ok
          and lists["launch read errors"] == "[]" and lists["launches without the exact count"] == "[]")
    sys.exit(0 if ok else 1)
elif mode == "pause":
    since = dt.datetime.strptime(sys.argv[sys.argv.index("--since") + 1], "%Y-%m-%d %H:%M")
    since = since.replace(minute=since.minute - since.minute % 10)  # Codex r8: include the bin holding --since
    hot = [(b.strftime("%F %H:%M"), rl, n) for b, rl, n in bins if b >= since and n > 0 and rl / n >= 0.02]
    print(f"census {t} | bins since {since:%H:%M} at >= 2% RateLimitError: {hot}")
    sys.exit(3 if hot else 0)
sys.exit(f"unknown mode {mode}")
