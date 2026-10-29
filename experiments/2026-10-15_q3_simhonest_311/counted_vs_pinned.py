"""By key, for one +311 main: each cell's COUNTED honest_a (the stager's rule, box_files.py @ 9be2fa9b6: the unique
attempt that is neither a pre-start redraw (`redrawn_because`) nor of kind "rerun"; its folder's one .eval) against the
PINNED reference (launch.json `references`: the simulated-honest fork source). Read-only, run ON the main's box with the
main's run dir; prints one JSON record (09's Append 25f census). Metadata and hashes only.
  python3 counted_vs_pinned.py <main run dir>
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
d = json.loads((main / "launch.json").read_text())
refs = d["references"]
rows, differ, problems = [], [], []
for r in d["runs"]:
    if r.get("arm") != "honest_a":
        continue
    task = r["task"]["key"] if isinstance(r["task"], dict) else r["task"]
    counted = [(i, a) for i, a in enumerate(r.get("attempts") or [])
               if not a.get("redrawn_because") and a.get("kind") != "rerun"]
    kinds = [a.get("kind") for a in r.get("attempts") or []]
    row = {"task": task, "attempt_kinds": kinds, "pinned": task in refs}
    if len(counted) != 1:
        problems.append(f"{task}: {len(counted)} counted attempts")
        row["counted"] = None
    else:
        i, a = counted[0]
        evs = sorted(Path(a["log_dir"]).glob("*.eval")) if a.get("log_dir") else []
        row |= {"counted_attempt": i, "counted_kind": a.get("kind"),
                "counted_eval_sha256": sha(evs[0]) if len(evs) == 1 else None}
        if task in refs:
            row["pinned_sha256"] = refs[task]["sha256"]
            row["pinned_source_dir"] = Path(refs[task]["source"]).parent.name
            row["equal"] = row["counted_eval_sha256"] == refs[task]["sha256"]
            if not row["equal"]:
                differ.append(task)
    rows.append(row)
redraw_rows = [x for x in rows if "redraw" in (x.get("counted_kind") or "") or any("redraw" in str(k) for k in x["attempt_kinds"])]
print(json.dumps({"main": str(main), "honest_a_runs": len(rows), "pinned": sum(x["pinned"] for x in rows),
                  "counted_differs_from_pinned": sorted(differ), "problems": problems,
                  "redraw_cells": redraw_rows, "rows": rows}, sort_keys=True))
