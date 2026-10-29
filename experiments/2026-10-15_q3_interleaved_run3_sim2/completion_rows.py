"""Fill a completion launch's per-arm supersession map (config/completion/ilc_supersession_perarm_<h>.json, 09's format)
after the launch has finished, ON the box that ran it (read-only on the run): per row the replacement's state and its last
attempt's .eval paths and sha256s; status = the replacement state; provenance = the launch.json sha256 read now. Originals
keep the map's recorded state and attempts; their .eval hashes are null here (they live on the original's box and in the
chain's manifests). Tier 2 rows have no original run ("not run", 0 attempts). Fails loudly if the launch has not finished
or a mapped run key is missing.
  python3 completion_rows.py <per-arm map json> <completion run dir>  > <run>.supersession_rows.json
"""
import hashlib, json, sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text()); run = Path(sys.argv[2])
raw = (run / "launch.json").read_bytes(); lj = json.loads(raw)
assert lj.get("finished"), f"{run} has not finished"
runs = lj["runs"]; runs = list(runs.values()) if isinstance(runs, dict) else runs
by = {r["key"]: r for r in runs}
def evals(r):
    d = Path(r["attempts"][-1]["log_dir"]) if r.get("attempts") else None
    fs = sorted(d.glob("*.eval")) if d and d.is_dir() else []
    return [{"path": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()} for f in fs]
prov = {"launch_json": str(run / "launch.json"), "launch_json_sha256": hashlib.sha256(raw).hexdigest(), "finished": lj["finished"]}
rows = []
for r in m["rows"]:
    rk = r["authoritative"]["run_key"]; assert rk in by, (r["key"], "missing in the completion launch")
    n = by[rk]; ev = evals(n)
    rows.append(r | {"replacement": {"launch_id": run.name, "run_key": rk, "state": n["state"], "attempts": len(n.get("attempts") or []), "evals": ev},
                     "replacement_state_done": n["state"] == "done", "status": n["state"], "provenance": prov,
                     "eval_sha256": [e["sha256"] for e in ev] or None})
print(json.dumps({"completion_launch": run.name, "original_launch": Path(m["original_launch"]).name, "map_revision": m["revision"][:120],
                  "fill_rules": m["fill_rules"], "rows": rows}, indent=1))
