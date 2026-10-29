"""Grounding-reference manifest, read ON an IL host (read-only; structure only): for each IL launch dir given, its
launch.json `references` (key -> the pinned reference copy, its sha256, reused_from = the source in the main or index), the
launch.json sha256 at read time, and the references_from launch.json sha256. run_pilot.py copies each reference and
asserts its sha256 against the source record (lines 676-677 @ bbf07299b); this manifest pins gate (a)'s inputs.
  python3 grounding_references_read.py <IL run dir> ...  > host.json
"""
import hashlib, json, sys
from pathlib import Path
out = []
for run in sys.argv[1:]:
    p = Path(run); lj = p / "launch.json"
    d = json.loads(lj.read_text())
    refs = d.get("references") or {}
    rf = d.get("references_from")
    main_lj = Path(rf) / "launch.json" if rf else None
    out.append({"launch_run_dir": str(p), "launch_json_sha256_at_read": hashlib.sha256(lj.read_bytes()).hexdigest(),
                "finished": d.get("finished"), "references_from": rf,
                "references_from_launch_json_sha256": hashlib.sha256(main_lj.read_bytes()).hexdigest() if main_lj and main_lj.exists() else None,
                "n_references": len(refs),
                "references": {k: {"copy": v.get("copy"), "sha256": v.get("sha256"), "reused_from": v.get("reused_from")} for k, v in sorted(refs.items())}})
print(json.dumps(out))
