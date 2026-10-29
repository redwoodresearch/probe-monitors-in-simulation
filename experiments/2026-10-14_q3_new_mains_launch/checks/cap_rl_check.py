"""Cap and RateLimitError check on one census JSON (d9 10-15 ~22:00): total simulated running must stay <= 95, and the exact
last-30-min RateLimitError count must be 0. Fails closed (Codex r84): a box without data that is not a confirmed powered-off
box, a launch read error, a launch entry without sim_running, or a launch without the exact last-30 count is an alert,
because the total or the RL count could then be undercounted. Counts and keys only.
    python3 cap_rl_check.py <census json>   -> one line; exit 0 ok, 4 alert. Any other exit = unreadable or malformed."""
import json
import re
import sys

CAP = 95
d = json.load(open(sys.argv[1]))
off = set(d["powered_off_boxes"])
no_data = sorted(b for b, v in d["per_box"].items() if v is None and b not in off)
bad = sorted(f"{b}/{k}" for b, v in d["per_box"].items() if isinstance(v, dict) for k, x in v.items()
             if not isinstance(x, dict) or "error" in x or not isinstance(x.get("sim_running"), int))
no_exact = sorted(f"{b}/{k}" for b, v in d["per_box"].items() if isinstance(v, dict) for k, x in v.items()
                  if isinstance(x, dict) and "error" not in x and "last30" not in x)
total = sum(x["sim_running"] for v in d["per_box"].values() if isinstance(v, dict) for x in v.values()
            if isinstance(x, dict) and isinstance(x.get("sim_running"), int))
rl = int(re.search(r"exact last 30 min: RateLimitError runs (\d+) of", d["rates_text"]).group(1))
alerts = ([f"total sims {total} > {CAP}"] * (total > CAP) + [f"RateLimitError runs {rl} in the last 30 min"] * (rl > 0)
          + [f"incomplete: boxes without data (not powered off) {no_data}"] * bool(no_data)
          + [f"incomplete: launch read errors {bad}"] * bool(bad) + [f"incomplete: no exact last-30 count {no_exact}"] * bool(no_exact))
print(f"{d['census_time_pt']}: total sims {total}, RL last 30 min {rl}; " + ("ALERT: " + "; ".join(alerts) if alerts else "ok"))
sys.exit(4 if alerts else 0)
