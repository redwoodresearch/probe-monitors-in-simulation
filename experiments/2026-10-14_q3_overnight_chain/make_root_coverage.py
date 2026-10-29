"""By-key coverage of a finished sim-honest quarter: every cell its final run table lists as ran, and the routed root(s) that hold
its sim-arm run (from the committed snapshot / final manifests) with staged or staging-excluded (from the committed staging records).
Keys only. Fails loudly unless every ran cell sits in exactly one root.

  python3 make_root_coverage.py <pool e.g. simh311> <q e.g. q2> <out.json> [note]
"""
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARM = "sim_honest@september"


def head(path: Path) -> str:
    return subprocess.run(["git", "log", "-1", "--format=%h", "--", str(path)], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()


def main():
    pool, q, out = sys.argv[1:4]
    note = sys.argv[4] if len(sys.argv) > 4 else ""
    rt_p = HERE / "results" / "hygiene" / f"run_table_{pool}_{q}.json"
    ran = sorted(c["cell"] for c in json.loads(rt_p.read_text())["cells"] if c.get("sim_status") == "ran")
    roots, where = [], {}
    pat = re.compile(rf"^manifest_{pool}_{q}_sim(_s(\d+))?\.json$")
    for p in sorted(glob.glob(str(HERE / "config" / f"manifest_{pool}_{q}_sim*.json"))):
        m = pat.match(Path(p).name)
        if not m:
            continue
        target = Path(p).stem.removeprefix("manifest_")
        rows = [f for f in json.loads(Path(p).read_text())["files"] if f["arm_dir"] == ARM]
        recs = sorted(glob.glob(str(HERE / "results" / f"chain_stage_{target}_2026-*.json")))
        if not rows:
            continue
        assert recs, f"{target}: no staging record"
        rec = json.loads(Path(recs[-1]).read_text())
        rec = rec[next(k for k in rec if k != "git")]
        excl = {k.split("#", 1)[0].split("/")[k.split("/").index(ARM) + 1] for k in rec["c_gate"]["excluded"]}
        roots.append({"target": target, "manifest": f"config/{Path(p).name} @ {head(Path(p))}", "staging_record": f"{Path(recs[-1]).name} @ {head(Path(recs[-1]))}",
                      "staged_set": rec["out"], "cells": len(rows), "staging_excluded": sorted(excl)})
        for f in rows:
            where.setdefault(f["task"], []).append({"root": target, "staged": f["task"] not in excl})
    missing = [c for c in ran if c not in where]
    multi = {c: v for c, v in where.items() if len(v) != 1}
    extra = sorted(set(where) - set(ran))
    doc = {"what": f"by-key coverage of {pool} {q}: every ran cell of the final run table and the routed root that holds its sim-arm run", "note": note,
           "run_table": f"results/hygiene/{rt_p.name} @ {head(rt_p)}", "roots": roots,
           "counts": {"ran_cells": len(ran), "in_roots": sum(c in where for c in ran), "staged": sum(where[c][0]["staged"] for c in ran if c in where),
                      "staging_excluded": sum(not where[c][0]["staged"] for c in ran if c in where), "missing": len(missing), "in_several_roots": len(multi),
                      "in_roots_not_ran": len(extra)},
           "missing": missing, "in_several_roots": multi, "in_roots_not_ran": extra, "cells": {c: where.get(c) for c in ran}}
    Path(out).open("x").write(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc["counts"]))
    if missing or multi or extra:
        raise SystemExit("coverage FAILED")


if __name__ == "__main__":
    main()
