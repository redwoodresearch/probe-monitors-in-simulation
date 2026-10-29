# Read-only probe of box4 (i-07bce857a818b6f69) for the S4 source map (09 @ bffe22e3e): the s4d follow-up's and the S4 main's launch.json
# sha256 and, per simulated run of the follow-up, its key, cell and launcher state at this read. Keys and states only.
# Run: ssm_run_i.sh i-07bce857a818b6f69 "$(cat probe_s4d_launch.sh)"
R=~/phase2/nm/runs
date '+probe at %Y-%m-%d %H:%M:%S %Z'
sha256sum $R/2026-10-14_newmains_s4d_fu/launch.json $R/2026-10-14_newmains_s4/launch.json
python3 - <<'PY'
import json, os
R = os.path.expanduser("~/phase2/nm/runs")
for run in ("2026-10-14_newmains_s4d_fu",):  # the S4 main: its sha256 only (above)
    d = json.load(open(f"{R}/{run}/launch.json"))
    rows = [{"key": r.get("key"), "task": r["task"] if isinstance(r["task"], str) else r["task"].get("key"), "state": r.get("state"), "real": r.get("real"),
             "attempts": len(r.get("attempts") or [])} for r in d.get("runs", [])]
    print(json.dumps({"run": run, "finished": d.get("finished"), "references_from": d.get("references_from"), "runs": rows}))
PY
