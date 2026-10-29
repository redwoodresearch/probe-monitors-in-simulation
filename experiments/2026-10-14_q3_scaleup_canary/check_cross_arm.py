"""The canary's cross-arm prompt check (RULE.md, after the runs; 09's check): per task, the real attack run 1 (launch
A's `attack`), the real attack run 2 (launch B's `attack`) and the simulated attack (launch A's `sim_attack@september`)
must have been sent the same approach, system prompt and first non-empty user message, and the approach must be the
task file's. Each run's first policy call is hashed by the purchase fork's hash_eval (check_approach_hashes.py @
a961cadc, copied unchanged as check_approach_hashes_purchase.py); tool definitions (`sent_tools_sha256`) are recorded
and compared, and reported, not failed (the known real-vs-simulated schema difference). Hashes, counts and flags only.
Exit 1 on any mismatch; tasks without all three done runs are listed as not covered.

ON THE DEVBOX, September python, from the harness folder:
  $PY check_cross_arm.py --a ~/phase2/runs/<A> --b ~/phase2/runs/<B> --tasks config/tasks_canary20_final.json --out <json>
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from check_approach_hashes_purchase import hash_eval

ARMS = {("a", "attack"): "real1", ("b", "attack"): "real2", ("a", "sim_attack"): "sim"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), f"{args.out} exists"
    want = {c["key"]: c["approach_sha256"] for c in json.loads(args.tasks.read_text())["cells"]}
    rows, jobs = [], []
    for side, launch in (("a", args.a), ("b", args.b)):
        rec = json.loads((launch / "launch.json").read_text())
        for r in rec["runs"]:
            arm = ARMS.get((side, r["arm"]))
            if arm is None or (arm == "sim" and r["simulator"] != "september") or r["state"] != "done":
                continue
            d = Path(r["attempts"][-1]["log_dir"])
            evals = sorted((launch / d.parent.name / d.name).glob("*.eval"))
            assert len(evals) == 1, f"{r['key']}: {len(evals)} .eval files"
            rows.append({"arm": arm, "task": r["task"], "run": r["key"], "eval": evals[0].name})
            jobs.append((str(evals[0]), rec["policy"]))
    with ProcessPoolExecutor(max_workers=16) as pool:
        for row, h in zip(rows, pool.map(hash_eval, jobs)):
            row |= {k: h[k] for k in ("approach_sha256", "sent_system_sha256", "sent_first_nonempty_user",
                                      "sent_tools_sha256", "policy_calls")}
    by_task = {}
    for r in rows:
        assert r["arm"] not in by_task.setdefault(r["task"], {}), f"{r['task']}: two {r['arm']} runs"
        by_task[r["task"]][r["arm"]] = r
    bad, tools_differ, covered = [], [], []
    for task, arms in sorted(by_task.items()):
        for arm, r in arms.items():
            if r["approach_sha256"] != want[task]:
                bad.append((task, arm, "approach is not the task file's"))
        for field in ("sent_system_sha256", "sent_first_nonempty_user"):
            if len({json.dumps(r[field], sort_keys=True) for r in arms.values()}) > 1:
                bad.append((task, sorted(arms), f"{field} differs across arms"))
        if len({r["sent_tools_sha256"] for r in arms.values()}) > 1:
            tools_differ.append(task)
        if set(arms) == {"real1", "real2", "sim"}:
            covered.append(task)
    missing = sorted(set(want) - set(covered))
    print(f"runs hashed {len(rows)}; tasks with all three arms done {len(covered)} of {len(want)}; not covered {missing}")
    print(f"mismatches {len(bad)}: {bad}")
    print(f"tasks whose tool definitions differ across arms (reported, not failed): {len(tools_differ)} of {len(by_task)}")
    print(f"{'COMPLETE' if not missing else 'INCOMPLETE'} coverage; {'PASS' if not bad else 'FAIL'} on the tasks covered")
    args.out.write_text(json.dumps({"rows": rows, "mismatches": bad, "covered": covered, "not_covered": missing,
                                    "complete": not missing, "tools_differ": tools_differ, "pass": not bad}, indent=1))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
