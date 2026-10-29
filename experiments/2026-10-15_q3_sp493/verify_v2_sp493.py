"""Verify a v2 (reordered) S_paper493 launch file against its v1 (d9 ~18:03): (1) set equality by (key, arm set, content sha256);
(2) honest keys unchanged; (3) r2 after r1; (4) same-environment gaps; (5) a REPLAY of the frozen launcher's own launch() (pinned run_pilot.py, sha256 ea8d7253...; Codex r5) with each run held
D minutes, for v1 and v2: the time of the last start
and of the last end. Names and counts only.
  python verify_v2_sp493.py <v1 launch json> <v2 launch json> [--window 10]
"""
import argparse
import hashlib
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("v1", type=Path)
ap.add_argument("v2", type=Path)
ap.add_argument("--window", type=int, default=9)
ap.add_argument("--start-epoch", type=float, help="planned start (epoch s): with --poweroff-epoch, the 0d start-rate gate")
ap.add_argument("--poweroff-epoch", type=float)
ap.add_argument("--box")
ap.add_argument("--run-pilot", required=True, help="the frozen run_pilot.py (sha256 ea8d7253...)")
a = ap.parse_args()
assert (a.start_epoch is None) == (a.poweroff_epoch is None), "give both --start-epoch and --poweroff-epoch, or neither"
v1, v2 = json.loads(a.v1.read_text())["cells"], json.loads(a.v2.read_text())["cells"]
DEFAULT = ("honest_a", "attack", "honest_b")


def ident(c):
    arms = tuple(c.get("real_arms", DEFAULT))
    body = {k: v for k, v in c.items()}
    return (c["key"], arms, hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest())


out = {"set_equal_by_key_arms_content": sorted(map(ident, v1)) == sorted(map(ident, v2)),
       "honest_keys_equal": sorted(c["key"] for c in v1 if "real_arms" not in c) == sorted(c["key"] for c in v2 if "real_arms" not in c)}
pos = {c["key"]: i for i, c in enumerate(v2)}
out["r2_after_r1"] = all(pos[k] > pos[k[:-3]] for k in pos if k.endswith("_r2"))


def gaps(order):
    last, short = {}, []
    for i, c in enumerate(order):
        if c["env"] in last and i - last[c["env"]] <= a.window:
            short.append(i)
        last[c["env"]] = i
    return short


out["v1_short_gaps"], out["v2_short_gaps"] = len(gaps(v1)), len(gaps(v2))
out["v2_short_gap_positions"] = gaps(v2)


PINNED_RUN_PILOT_SHA = "ea8d7253ba26fbd8be0f60b050fa17caf992b086317c6b00f7345537c4491d90"  # the frozen +311 harness's run_pilot.py


def _launch_code(path):
    """The pinned launcher's own launch() function (Codex r5): its strict order, its per-arm 90 s stagger, its priority for
    started tasks' remaining arms, its 6 servers x 10 runs and 60 real slots."""
    import ast
    src = Path(path).read_bytes()
    assert hashlib.sha256(src).hexdigest() == PINNED_RUN_PILOT_SHA, "not the frozen run_pilot.py"
    node = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "launch")
    return compile(ast.Module(body=[node], type_ignores=[]), "<pinned-launch>", "exec")


def simulate(order, dur_min):
    """Replay the pinned launch() every 30 s with every run held dur_min minutes (no startup overhead, retries or redraws):
    returns (minutes to the last arm start, minutes to the last end)."""
    import types
    from collections import Counter
    tasks = [{"key": c["key"], "env": c["env"]} for c in order]
    runs = [{"task": t, "server": None, "state": "ready", "attempts": [], "real": True, "arm": arm}
            for c, t in zip(order, tasks) for arm in c.get("real_arms", DEFAULT)]
    clock, last, trace = types.SimpleNamespace(t=0), {}, []

    def start(r, kind):
        r["attempts"].append({"start": clock.t})
        r.update(state="running", end=clock.t + dur_min * 60)
        last[r["task"]["env"]] = clock.t
        trace.append(clock.t)
    ns = {"Counter": Counter, "runs": runs, "tasks": tasks, "servers": list(range(6)),
          "pilot": {"max_real_runs": 60, "max_runs_per_server": 10, "spend": {"cap_usd": 0, "worst_case_limit_usd": 0}},
          "run": {"stagger_s": 90}, "sim_arms": [], "rerun_sims": [], "rac": types.SimpleNamespace(STOP=False),
          "time": types.SimpleNamespace(time=lambda: clock.t), "last_launch": last, "start": start}
    exec(LAUNCH, ns)
    while any(r["state"] != "done" for r in runs):
        for r in runs:
            if r["state"] == "running" and r["end"] <= clock.t:
                r["state"] = "done"
        ns["launch"]({"_projected": 0, "_est": 0, "_worst_case": 0, "_worst": 0})
        clock.t += 30
        assert clock.t < 86400
    return round(max(trace) / 60, 1), round(max(r["end"] for r in runs) / 60, 1)


LAUNCH = _launch_code(a.run_pilot)
out["simulated_minutes_after_start_last_start_last_end"] = {
    f"D={d}": {"v1": simulate(v1, d), "v2": simulate(v2, d)} for d in (12, 15, 20)}
# 09's by-key diff: v2 equals v1 as a multiset of (key, cell, arm list, honest flag, box); only the order changes
row = lambda c: (c["key"], c["cell"], tuple(c.get("real_arms", DEFAULT)), "real_arms" not in c and not c["key"].endswith("_r2"), a.box)  # noqa: E731
out["diff_by_key"] = {"multiset_equal": sorted(map(row, v1)) == sorted(map(row, v2)), "entries": len(v2),
                      "moved": sum(1 for i, c in enumerate(v2) if v1[i]["key"] != c["key"]),
                      "v1_sha256": hashlib.sha256(a.v1.read_bytes()).hexdigest(), "v2_sha256": hashlib.sha256(a.v2.read_bytes()).hexdigest()}
# 0d's start-rate gate: the replayed last start (D = 20 min, the slower case) must be before poweroff minus the 2 h run limit
if a.start_epoch and a.poweroff_epoch:
    last_start_min = simulate(v2, 20)[0]
    out["start_rate_gate"] = {"assumed_run_min": 20, "last_start_after_min": last_start_min,
                              "latest_allowed_after_min": round((a.poweroff_epoch - 7200 - a.start_epoch) / 60, 1),
                              "pass": a.start_epoch + last_start_min * 60 <= a.poweroff_epoch - 7200}
out["pass"] = (out["set_equal_by_key_arms_content"] and out["honest_keys_equal"] and out["r2_after_r1"]
               and out["diff_by_key"]["multiset_equal"] and out.get("start_rate_gate", {"pass": True})["pass"])
print(json.dumps(out, indent=1))
import sys  # noqa: E402
sys.exit(0 if out["pass"] else 1)  # Codex r5: a failed gate exits nonzero
