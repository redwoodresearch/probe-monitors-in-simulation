"""ON THE BOX, after the S4 main's drain stop: per run of <run>/launch.json (keys and states only): key, task, arm, real,
state, server index, attempts, last exit code, and the drain-stop relabel (relabel_drainstop.json: "done" or "stopped");
plus launch.json's sha256 and finished time. Prints one JSON object (for make_s4r.py).
  python3 s4_final_states.py <run dir>"""
import hashlib, json, sys
from pathlib import Path
run = Path(sys.argv[1])
raw = (run / "launch.json").read_bytes()
d = json.loads(raw)
assert d.get("finished"), "the S4 main's record is not finished"
relabel = {r["key"]: r["label"] for r in json.loads((run / "relabel_drainstop.json").read_text())}
servers = d.get("servers") or []
out = {}
for r in d["runs"]:
    a = r["attempts"][-1] if r["attempts"] else {}
    out[r["key"]] = {"task": r["task"], "arm": r["arm"], "real": r["real"], "state": r["state"],
                     "server": servers.index(r["server"]) if r.get("server") in servers else None,
                     "attempts": len(r["attempts"]), "exit_code": a.get("exit_code"), "relabel": relabel.get(r["key"])}
print(json.dumps({"run": run.name, "launch_json_sha256": hashlib.sha256(raw).hexdigest(), "finished": d["finished"],
                  "servers": len(servers), "runs": out}))
