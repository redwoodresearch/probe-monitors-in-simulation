"""Count simulated-honest runs and distinct main tasks in one on-policy launch (#1465, PRRT_kwDOU9RNXc6sMmSR). POST HOC.

Reads the launch's launch.json on stdin (S3 phase2/runs/<launch>/launch.json); prints one JSON line: the acting policy,
runs by (arm, simulator, state), and the finished simulated-honest runs' distinct main tasks per simulator. Keys and
counts only.
  aws s3 cp s3://<bucket>/phase2/runs/<launch>/launch.json - | python3 -I count_simhonest_POSTHOC.py <launch>
"""
import collections
import json
import sys

L = json.load(sys.stdin)
runs = L["runs"]
by = collections.Counter((r["arm"], r.get("simulator"), r.get("state")) for r in runs)
sh = {}
for r in runs:
    if r["arm"] == "sim_honest" and r.get("state") == "done":
        sh.setdefault(str(r.get("simulator")), set()).add(r.get("task") or r["key"].split("/", 1)[1])
print(json.dumps({"launch": sys.argv[1], "policy": L.get("policy"), "n_runs": len(runs),
                  "runs_by_arm_simulator_state": {"|".join(map(str, k)): v for k, v in sorted(by.items(), key=str)},
                  "sim_honest_done_tasks": {s: {"n_runs": sum(1 for r in runs if r["arm"] == "sim_honest" and str(r.get("simulator")) == s and r.get("state") == "done"),
                                                "n_tasks": len(t), "tasks": sorted(t)} for s, t in sorted(sh.items())}}))
