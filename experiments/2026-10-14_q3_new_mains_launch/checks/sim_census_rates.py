# Laptop: combine nm_launch/census/*.json into per-bin rates (RateLimitError failures / simulated runs ended, by the
# 10-minute bin of the attempt end, PT) across all launches, plus the last-30-minute rate and current running sims.
import collections, glob, json, os, sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
PT = ZoneInfo("America/Los_Angeles")
ended, rl, running, per_launch = collections.Counter(), collections.Counter(), 0, {}
e30 = r30x = 0; missing, read_errors, no_last30 = [], [], []  # Codex r26 items 2-3: completeness and the exact window
for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "census", "*.json"))):
    if f.endswith("summary.json"):
        continue
    if not open(f).read().strip():
        missing.append(os.path.basename(f)[:-5]); continue
    for l, v in json.load(open(f)).items():
        if "error" in v: read_errors.append(l); continue
        if "last30" in v: e30 += v["last30"]["ended"]; r30x += v["last30"]["rl_runs"]
        else: no_last30.append(l)
        running += v["sim_running"]
        for b, n in v.get("ended_by_10min", {}).items(): ended[b] += n
        for k, n in v["by_class_and_10min"].items():
            c, b = k.rsplit(" ", 1)
            if c == "RateLimitError": rl[b] += n
        per_launch[l] = (v["sim_ended"], v["errors_by_class"].get("RateLimitError", 0), v["sim_running"])
now = datetime.now(PT)
last30 = [f"{t:%H}:{t.minute // 10}0" for t in (now - timedelta(minutes=m) for m in (30, 20, 10, 0))]
print(f"exact last 30 min: RateLimitError runs {r30x} of {e30} ended = {100 * r30x / e30 if e30 else 0:.1f}%  | simulated running {running}"
      f"  | boxes without data {missing}  | launch read errors {read_errors}  | launches without the exact count {no_last30}")
e30, r30 = sum(ended[b] for b in set(last30)), sum(rl[b] for b in set(last30))
print("bins (end time PT): " + ", ".join(f"{b} {rl[b]}/{ended[b]}" for b in sorted(ended)))
print(f"last 30 min ({last30[0]}-{last30[-1]} bins): RateLimitError {r30} of {e30} ended = {100 * r30 / e30 if e30 else 0:.1f}%  | sims running now {running}")
print("per launch (ended, RL, running):", per_launch)
stalls = {}
for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "census", "*.json"))):
    if f.endswith("summary.json") or not open(f).read().strip():
        continue
    for l, v in json.load(open(f)).items():
        st = v.get("stall")
        if st: stalls[l] = st
for l, st in sorted(stalls.items()):
    tag = "ALARM (stalled)" if st["ALARM"] else st["state"]
    print(f"stall: {l}: {tag}; running {st['running_all_arms']}, pending {st['pending_all_arms']}, idle {st['idle_minutes']} min, markers {st['markers']}, next_task_fits {st['next_task_fits']}")
