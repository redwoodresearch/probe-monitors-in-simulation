"""Start-rate replay for the S_paper493 sim tail + L3 launches (d9/0d: each launch file must pass it before launch). DERIVED from
the B fork's verify_v2_sp493.py (Codex r5's method): exec the FROZEN run_pilot.py's own launch() (sha256 ea8d7253...) every 30 s
on this launch's task order, every run held D minutes; here all runs are simulated (sims only: one sim_attack run per cell),
6 servers x max_runs_per_server 5, stagger 90 s. Run it ON A DEVBOX (no laptop compute), with the box's pinned run_pilot.py:
  python3 replay_sp493_tail.py <run_pilot.py> <tasks json> <start epoch> <poweroff epoch>
Gate (0d): the replayed last start at D = 38 min (the L1/L2 p90 run time) is before the poweroff minus 2 h."""

import ast
import hashlib
import json
import sys
import types
from collections import Counter
from pathlib import Path

PINNED = "ea8d7253ba26fbd8be0f60b050fa17caf992b086317c6b00f7345537c4491d90"
rp, tf, start_epoch, poweroff_epoch = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
src = Path(rp).read_bytes()
assert hashlib.sha256(src).hexdigest() == PINNED, "not the frozen run_pilot.py"
node = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "launch")
LAUNCH = compile(ast.Module(body=[node], type_ignores=[]), "<pinned-launch>", "exec")
cells = json.loads(Path(tf).read_text())["cells"]


def simulate(dur_min):
    tasks = [{"key": c["key"], "env": c["env"]} for c in cells]
    runs = [{"task": t, "server": None, "state": "ready", "attempts": [], "real": False, "arm": "sim_attack"} for t in tasks]
    clock, last, trace = types.SimpleNamespace(t=0), {}, []

    def start(r, kind=""):
        r["attempts"].append({"start": clock.t})
        r.update(state="running", end=clock.t + dur_min * 60)
        last[r["task"]["env"]] = clock.t
        trace.append(clock.t)
    nowhere = Path("/nonexistent_replay_dir")  # no PAUSE_SIMS, no CANARY_OK: an open launch
    ns = {"Counter": Counter, "runs": runs, "tasks": tasks, "servers": list(range(6)), "out": nowhere, "HERE": nowhere,
          "pilot": {"max_real_runs": 60, "max_runs_per_server": 5, "spend": {"cap_usd": 1e12, "worst_case_limit_usd": 1e12}},
          "run": {"stagger_s": 90}, "sim_arms": ["sim_attack"], "rerun_sims": ["september"], "rac": types.SimpleNamespace(STOP=False),
          "time": types.SimpleNamespace(time=lambda: clock.t), "last_launch": last, "start": start}
    exec(LAUNCH, ns)
    peak = 0
    while any(r["state"] != "done" for r in runs):
        for r in runs:
            if r["state"] == "running" and r["end"] <= clock.t:
                r["state"] = "done"
        ns["launch"]({"_projected": 0, "_est": 0, "_worst_case": 0, "_worst": 0})
        peak = max(peak, sum(r["state"] == "running" for r in runs))
        clock.t += 30
        assert clock.t < 86400
    starts = sorted(trace)
    return {"last_start_min": round(starts[-1] / 60, 1), "last_end_min": round(max(r["end"] for r in runs) / 60, 1),
            "peak_running": peak, "starts_first_10_min": sum(s < 600 for s in starts), "n": len(starts)}


out = {"tasks": tf, "tasks_sha256": hashlib.sha256(Path(tf).read_bytes()).hexdigest(), "run_pilot_sha256": PINNED,
       "D": {f"{d}": simulate(d) for d in (24, 27, 38)}}
allowed = (poweroff_epoch - 7200 - start_epoch) / 60
out["gate"] = {"assumed_run_min": 38, "last_start_after_min": out["D"]["38"]["last_start_min"], "latest_allowed_after_min": round(allowed, 1),
               "pass": out["D"]["38"]["last_start_min"] <= allowed and all(v["n"] == len(cells) for v in out["D"].values())}
print("REPLAY " + json.dumps(out))
raise SystemExit(0 if out["gate"]["pass"] else 1)  # Codex r54: a failed gate exits nonzero
