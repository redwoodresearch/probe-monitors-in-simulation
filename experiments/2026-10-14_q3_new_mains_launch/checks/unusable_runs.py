"""Every ENDED run that gate1_311b.py counts as NOT usable (gate1_311b.py, run_usable(): a run is usable if its LAST
attempt with an outcome and no redrawn_because has >= 1 policy call and every call's `usable` is true; ENDED = state
"done" or "no policy call"), from <run>/launch.json only: keys and fields, no transcript or error text (error classes
only). ON THE BOX: python3 unusable_runs.py <run dir> <quarter>"""
import json, sys
from collections import Counter
from pathlib import Path
run, q = Path(sys.argv[1]), sys.argv[2]
d = json.loads((run / "launch.json").read_text())
ENDED = ("done", "no policy call")
servers = d.get("servers") or []


def arm(r):
    return ("attack r2" if r["task"].endswith("_r2") else "attack r1") if r["arm"] == "attack" else (r["arm"] if r["real"] else f"{r['arm']}@{r['simulator']}")


out, totals = [], {}
for r in d["runs"]:
    if r["state"] not in ENDED:
        continue
    t = totals.setdefault(arm(r), Counter())
    t["ended"] += 1
    finals = [a for a in r["attempts"] if a.get("outcome") and not a.get("redrawn_because")]
    calls = finals[-1]["outcome"]["policy_calls"] if finals else []
    if calls and all(c["usable"] for c in calls):
        t["usable"] += 1
        continue
    a = finals[-1] if finals else (r["attempts"][-1] if r["attempts"] else {})
    o = a.get("outcome") or {}
    cls = ("no outcome" if not finals else "no policy call" if not calls else
           ",".join(sorted({f"ended={c.get('ended')}" + (" step_limit" if c.get("step_limit_hits") else "") +
                            (" timeouts" if c.get("timeouts") else "") + (" failed_after_start" if c.get("failed_after_start") else "")
                            for c in calls if not c["usable"]})))
    out.append({"quarter": q, "key": r["key"], "arm": arm(r), "env": r["task"].split("_", 1)[1].removesuffix("_r2") if "_" in r["task"] else None,
                "state": r["state"], "class": cls, "eval_status": o.get("eval_status"), "sample_error_classes":
                sorted({str(e).split(":")[0][:40] for e in (o.get("sample_errors") or [])}),
                "sample_limits": o.get("sample_limits"), "steps": sum(c.get("steps", 0) for c in calls),
                "process_seconds": a.get("process_seconds"), "attempts": len(r["attempts"]),
                "server": servers.index(r["server"]) if r.get("server") in servers else None})
print(json.dumps({"quarter": q, "run": run.name, "unusable": out, "totals": {k: dict(v) for k, v in sorted(totals.items())}}))
