"""ON SIMS-ASTRA, read-only (Append 35 item 2; d9 10-15 ~03:25 PT): the 5r supersession map with each replacement filled
from 5r's finished launch.json: its state, attempts, the sha256 of the one .eval in its last attempt's log_dir, the
outcome's eval_status / clean / sample-error count / usable_reference, and the authoritative run by the map's own rule
(the replacement if it is done with one readable .eval, every eval_status "success", clean and 0 sample errors; otherwise
none). The committed map is not changed; this prints a new JSON (results/ is append-only). Keys and hashes only.
  python3 s5r_map_fill.py <committed s5r_supersession_map.json> <5r run dir>"""

import hashlib
import json
import sys
from pathlib import Path

m = json.loads(Path(sys.argv[1]).read_text())
run = Path(sys.argv[2])
raw = (run / "launch.json").read_bytes()
d = json.loads(raw)
assert d.get("finished"), "5r has not finished"
runs = {r["key"]: r for r in d["runs"]}
out = {"rule": m["rule"], "from_map_sha256": hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),
       "replacement_launch_json_sha256": hashlib.sha256(raw).hexdigest(), "finished": d["finished"], "map": []}
for e in m["map"]:
    r = runs[e["replacement"]["run_key"]]
    a = r["attempts"][-1] if r["attempts"] else {}
    o = a.get("outcome") or {}
    ev = sorted(Path(a["log_dir"]).glob("*.eval")) if a.get("log_dir") else []
    rep = {**e["replacement"], "state": r["state"], "attempts": len(r["attempts"]), "eval_file": ev[0].name if len(ev) == 1 else None,
           "eval_sha256": hashlib.sha256(ev[0].read_bytes()).hexdigest() if len(ev) == 1 else None,
           "eval_status": o.get("eval_status"), "clean": o.get("clean"), "sample_errors": len(o.get("sample_errors") or []),
           "usable_reference": o.get("usable_reference"), "sim_usd": a.get("sim_usd")}
    es = o.get("eval_status")
    es_ok = es == "success" or (isinstance(es, list) and len(es) > 0 and all(x == "success" for x in es))  # a list, one per sample
    good = r["state"] == "done" and len(ev) == 1 and es_ok and o.get("clean") is True and not o.get("sample_errors")
    out["map"].append({**e, "replacement": rep, "authoritative": "replacement" if good else "none (replacement not usable)"})
print(json.dumps(out, indent=1))
