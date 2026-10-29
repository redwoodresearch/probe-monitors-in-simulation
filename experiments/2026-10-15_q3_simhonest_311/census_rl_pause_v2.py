"""census_rl_pause_v2.py (new file, d9 00:2x 10-16): the rate-limit pause test of census_rl.py, without its midnight bug.
census_rl.py dated the census's "bins (end time PT): HH:MM rl/ended" labels with the census date, but the bins carry no
date and pool runs by time of day, so after midnight last night's 00:20 bin (33/48, the 10-15 00:23-00:47 burst) read
as tonight's and tripped watcher v9 at 00:15:44 (census 2026-10-16 00:14:27, exact last 30 min 0 of 63). The bins cannot
be dated, so this test does not read them. It reads the census's own exact count instead, which is dated by construction:
  "exact last 30 min: RateLimitError runs R of N ended"  ->  pause (exit 3) if N > 0 and R / N >= 2%.
The window is about the 30 min before the census time (each box's window ends at its own read, which can precede the
census time by a few minutes: up to 3 min 11 s in the 00:14 census); a census taken before --since is not used (exit 0),
so the window can start a little over 30 min before --since. Limits: a burst between two censuses more than ~30 min
apart is missed, and a short burst is averaged over 30 min. A census without a census time or without the exact line
is a parse failure (exit 2: the watcher logs it and waits for the next census). For cross-checking only, it prints the latest dated RateLimitError
(per launch rate_limit_first_last, ISO with offset) in any launch.
  python3 census_rl_pause_v2.py pause --since 'YYYY-mm-dd HH:MM' < census.json   exit 0 none, 3 pause, 2 parse failure
"""
import datetime as dt
import json
import re
import sys

d = json.load(sys.stdin)
assert sys.argv[1] == "pause", f"unknown mode {sys.argv[1]}"
since = dt.datetime.strptime(sys.argv[sys.argv.index("--since") + 1], "%Y-%m-%d %H:%M")
m = re.search(r"exact last 30 min: RateLimitError runs (\d+) of (\d+) ended", d.get("rates_text", ""))
if not m or "census_time_pt" not in d:  # an older census format: no census time or no exact count
    print(f"census {d.get('census_time_pt')} | no census time or no exact last-30-min count")
    sys.exit(2)
ct = dt.datetime.strptime(d["census_time_pt"], "%Y-%m-%d %H:%M:%S")
rl, n = int(m[1]), int(m[2])
last = [v["rate_limit_first_last"][1] for box in (d.get("per_box") or {}).values() if isinstance(box, dict)
        for v in box.values() if isinstance(v, dict) and v.get("rate_limit_first_last")]
latest = max(last, key=lambda s: dt.datetime.fromisoformat(s)) if last else None
hot = ct >= since and n > 0 and rl / n >= 0.02
print(f"census {d['census_time_pt']} | exact last 30 min RateLimitError {rl} of {n} | used (census >= since {since:%F %H:%M}): "
      f"{ct >= since} | latest dated RateLimitError in any launch: {latest} | pause: {hot}")
sys.exit(3 if hot else 0)
