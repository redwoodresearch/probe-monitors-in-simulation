"""A launch's simulator spend so far, conservatively, for allin_stop.sh (RULE.md, Caps). Prints one line:
  "<usd> <age_s> <running> <running_charged_worst> <actual_usd>"  or  "absent"  (no launch.json yet).
actual_usd (what allin_stop.sh trips on, amendment 10): priced spend, plus each unpriced or running attempt's fresh
reading (else its last live reading), with no pre-charge.

usd: every attempt's priced spend (sim_usd); an ended attempt not priced (and not known to have skipped the policy) at
max(its live reading, the per-run estimate); a RUNNING attempt read FRESH here with the harness's own run_spend()
(run_pilot.py:106-145: its finished log, else Inspect's live sample buffer) and charged max(that, the estimate). The
harness keeps an old reading silently when run_spend() returns None (run_pilot.py:527-535), so this script records, per
attempt, when its last successful fresh read was (<state dir>/spend_state_<run>.json): a running attempt with no
successful read for more than FRESH_S (300 s) is charged max(live, the worst case). age_s: seconds since the launcher's
last spend snapshot. No snapshot yet means no simulated run has started: "0 0 0 0".
ON THE DEVBOX, with the September tree's python (run_pilot.py's imports):
  PYTHONPATH=<tree>/src $PY spend_now.py <run dir> <frozen harness dir> [<state dir>]
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

FRESH_S = 300
run, harness = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser()
state_dir = Path(sys.argv[3]).expanduser() if len(sys.argv) > 3 else Path(__file__).resolve().parent
f = run / "launch.json"
if not f.exists():
    print("absent")
    sys.exit()
rec = json.loads(f.read_text())
if not rec["spend"]:
    print("0 0 0 0 0")
    sys.exit()
sys.path.insert(0, str(harness))
from run_pilot import run_spend  # noqa: E402  (the frozen harness's own pricing)
from sim.prices import load_prices  # noqa: E402

prices = load_prices()
snap = rec["spend"][-1]
est, worst, now = snap["per_run_estimate_usd"], snap["worst_per_sim_run_usd"], time.time()
sf = state_dir / f"spend_state_{run.name}.json"
state = json.loads(sf.read_text()) if sf.exists() else {}
usd, actual, running, charged_worst = 0.0, 0.0, 0, 0
for r in rec["runs"]:
    if r["real"]:
        continue
    for i, a in enumerate(r["attempts"]):
        live = a.get("live_usd", 0.0)
        if "sim_usd" in a:
            usd += a["sim_usd"]
            actual += a["sim_usd"]
        elif "end" in a:
            usd += live if a.get("policy_called") is False else max(live, est)
            actual += live
        else:
            running += 1
            key = f"{r['key']}#{i}"
            first_seen = state.setdefault(key, {"seen": now, "ok": None})["seen"]
            try:
                fresh = run_spend(Path(a["log_dir"]), rec["policy"], prices)
            except Exception:  # noqa: BLE001 - a vanished buffer or a half-written log: not a successful read
                fresh = None
            if fresh is not None:
                state[key]["ok"] = now
                usd += max(fresh, live, est)
                actual += max(fresh, live)
            elif now - (state[key]["ok"] or first_seen) > FRESH_S:
                charged_worst += 1
                usd += max(live, worst)
                actual += live
            else:
                usd += max(live, est)
                actual += live
sf.write_text(json.dumps(state))
age = (datetime.now(timezone.utc) - datetime.fromisoformat(snap["time"])).total_seconds()
print(f"{usd:.2f} {age:.0f} {running} {charged_worst} {actual:.2f}")
