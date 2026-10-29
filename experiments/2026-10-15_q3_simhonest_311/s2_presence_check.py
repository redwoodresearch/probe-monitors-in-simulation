"""s2_presence_check.py (nm360 sim-honest S2, d9 ~11:25 step 1): READ-ONLY presence check of the S2 main's honest inputs
on xbox3 (box2 is down; xbox3 ran il_s2 with --references-from this path). Reads <main>/launch.json; for every
references[key]: is the copy present and does it hash to references[key].sha256; is the source (the honest_a .eval the
copy came from) present and what is its sha256. Hashes, counts and keys only; writes nothing. Prints one JSON line.
  python3 -I - <main run dir> < s2_presence_check.py
"""
import hashlib
import json
import sys
from pathlib import Path


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


main = Path(sys.argv[1])
lj = main / "launch.json"
out = {"main": str(main), "launch_json_present": lj.is_file()}
if lj.is_file():
    d = json.loads(lj.read_text())
    refs = d.get("references") or {}
    out |= {"launch_json_sha256": sha(lj), "finished": d.get("finished"), "n_references": len(refs),
            "n_honest_a_runs": sum(1 for r in d.get("runs") or [] if r.get("arm") == "honest_a")}
    rows = {}
    for k, r in sorted(refs.items()):
        cp, src = Path(r.get("copy", "")), Path(r.get("source", ""))
        row = {"pinned_sha256": r.get("sha256"), "copy_present": cp.is_file(), "source_present": src.is_file()}
        row["copy_ok"] = row["copy_present"] and sha(cp) == r.get("sha256")
        row["source_sha256"] = sha(src) if row["source_present"] else None
        row["source_ok"] = row["source_sha256"] == r.get("sha256")
        rows[k] = row
    out |= {"n_copy_ok": sum(v["copy_ok"] for v in rows.values()), "n_source_present": sum(v["source_present"] for v in rows.values()),
            "n_source_ok": sum(v["source_ok"] for v in rows.values()), "references": rows}
print(json.dumps(out))
