"""A2 serving check (PLAN_A2_honest_batch_C_2026-10-11.md, descriptive): did C's 3 servers serve it like launch 2's
servers served A and B? $0, local; launch.json only, no model text.

Deviation from the registration, stated: it names the servers' request logs, but launch 2's request logs also hold
attack-run text, which this session must not open. Every batch is therefore measured from launch.json's per-attempt
records. Recorded policy calls are completed steps plus failed calls. Per batch:
- seconds per recorded call (process seconds over recorded calls, so tool execution is included: a run-level latency
  proxy);
- the share of recorded calls that ended in the 1,800 s attempt timeout (the producer's `timeouts`: calls ending in
  AttemptTimeoutError), and the share of runs with one;
- the share of recorded calls retried.
Attempts without readable call records are counted as unknown and left out of these denominators; coverage is reported.

Counted attempt per task, as the scorer and judge use: A (honest_a) the first counted attempt, B (honest_b) the last,
C (honest_b in C's launch folder) its only one. Attempts redrawn before the policy's first call don't count.

  uv run --no-project --python 3.12 python analyze_POSTHOC_A2_serving.py <A and B launch dirs ...> <C launch dir (name ends in honest_c)>
"""

import argparse
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent


def main(dirs: list):
    runs, ins = {"A": [], "B": [], "C": []}, []
    for d in map(Path, dirs):
        lj = (d / "launch.json").read_bytes()
        ins.append({"launch": d.name, "sha256": hashlib.sha256(lj).hexdigest()})
        is_c = d.name.endswith("honest_c")
        for r in json.loads(lj)["runs"]:
            batch = ("C" if r["arm"] == "honest_b" else None) if is_c else {"honest_a": "A", "honest_b": "B"}.get(r["arm"])
            ok = [a for a in r["attempts"] if not a.get("redrawn_because")]
            if batch is None or not ok:
                continue
            if batch == "C" and len(ok) != 1:
                raise SystemExit(f"C {r['task']}: {len(ok)} counted attempts, not 1")
            a = ok[0] if batch == "A" else ok[-1]
            o = a.get("outcome") or {}
            pcs = o.get("policy_calls") or []
            known = bool(pcs) and not o.get("unreadable")
            calls = sum(p.get("steps", 0) + p.get("failed_calls", 0) for p in pcs)
            runs[batch].append({"task": r["task"], "known": known, "seconds": a.get("process_seconds"), "calls": calls,
                                "timeouts": sum(p.get("timeouts", 0) for p in pcs), "retried": sum(p.get("retried_calls", 0) for p in pcs)})
    for b, v in runs.items():
        if len({x["task"] for x in v}) != len(v):
            raise SystemExit(f"batch {b}: a task counted in two launch folders")
    out = {"label": "descriptive: A2 serving check from launch.json (deviation: not request logs; see the docstring)", "inputs": ins}
    for b, v in runs.items():
        k = [x for x in v if x["known"]]
        sps = sorted(x["seconds"] / x["calls"] for x in k if x["calls"] and x["seconds"])
        q = lambda p: sps[min(len(sps) - 1, int(p * len(sps)))] if sps else None
        n_calls = sum(x["calls"] for x in k)
        out[b] = {"runs": len(v), "runs_with_call_records": len(k), "runs_unknown": len(v) - len(k), "recorded_calls": n_calls,
                  "seconds_per_call_median": statistics.median(sps) if sps else None, "seconds_per_call_p90": q(0.9),
                  "timed_out_calls_share": sum(x["timeouts"] for x in k) / n_calls if n_calls else None,
                  "runs_with_a_timeout_share": sum(x["timeouts"] > 0 for x in k) / len(k) if k else None,
                  "retried_calls_share": sum(x["retried"] for x in k) / n_calls if n_calls else None}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"A2_serving_POSTHOC_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out))
    for b in runs:
        print(b, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[b].items()})
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("launches", nargs="+")
    main(ap.parse_args().launches)
