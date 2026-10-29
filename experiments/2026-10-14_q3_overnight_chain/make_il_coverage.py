"""By-key coverage of an interleaved (run3 + sim2) family against its registered task file, from committed files: every task-file cell has
exactly one run3 (attack) row and one sim2 row in the committed manifests, each staged or staging-excluded per its staging record; status-row
keys appear in no manifest row at all (donor rows included); honest donor rows only for task-file cells. Keys only; fails loudly.

  python3 make_il_coverage.py <name_prefix_q, e.g. ilold_oldsel> <task file "commit:path"> <status rows file "commit:path" or -> <out.json> [worlds, default run3,sim2]
(a sim-only family, e.g. ilnm_s4rgap, passes sim2: it has no run3 manifest)
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REL = "experiments/2026-10-14_q3_overnight_chain"
WORLD_ARM = {"run3": "attack", "sim2": "sim_attack@september"}


def git(*a) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout


def main():
    pq, task_ref, status_ref, out = sys.argv[1:5]
    want = sys.argv[5].split(",") if len(sys.argv) > 5 else list(WORLD_ARM)
    assert want and set(want) <= set(WORLD_ARM), want
    tf = json.loads(git("show", task_ref))
    cells = [c["key"] if isinstance(c, dict) else c for c in (tf["cells"] if isinstance(tf, dict) else tf)]
    assert len(cells) == len(set(cells)), "duplicate task-file cells"
    status_keys = [r["key"] for r in json.loads(git("show", status_ref))["rows"]] if status_ref != "-" else []
    status_cells = {k.split("|", 1)[1] for k in status_keys}
    problems, worlds = [], {}
    for w, arm in ((w, WORLD_ARM[w]) for w in want):
        t = f"{pq}_{w}"
        man_rel = f"config/manifest_{t}.json"
        man = json.loads(git("show", f"HEAD:{REL}/{man_rel}"))
        rows = [f for f in man["files"] if f["arm_dir"] == arm and f.get("world") == w]
        donors = sorted({f["task"] for f in man["files"] if f["arm_dir"] in ("honest_a", "honest_b")})
        recs = sorted(p for p in git("ls-files", "results").split() if p.startswith(f"results/chain_stage_{t}_2026-"))
        if len(recs) != 1:
            problems.append(f"{t}: {len(recs)} staging records")
            continue
        rec = json.loads(git("show", f"HEAD:{REL}/{recs[0]}"))
        rec = rec[next(k for k in rec if k not in ("git", "stamp"))]
        by_path = {f["path"]: f["task"] for f in rows}
        excl = {}
        for k in rec["c_gate"]["excluded"]:
            p = k.split("#", 1)[0]
            if p not in by_path:
                problems.append({"world": w, "exclusion_not_a_manifest_row": p})
            else:
                excl[by_path[p]] = rec["c_gate"]["excluded"][k]
        tasks = [f["task"] for f in rows]
        if len(tasks) != len(set(tasks)) or set(tasks) != set(cells):
            problems.append({"world": w, "missing": sorted(set(cells) - set(tasks)), "extra": sorted(set(tasks) - set(cells)), "duplicates": len(tasks) - len(set(tasks))})
        if rec["c_gate"].get("manifest_keys") != len(rows) or rec["c_gate"].get("staged") != len(rows) - len(excl):
            problems.append({"world": w, "c_gate": {k: rec["c_gate"].get(k) for k in ("manifest_keys", "staged")}, "rows": len(rows), "excluded": len(excl)})
        leaked = sorted(f["task"] for f in man["files"] if f["task"] in status_cells)
        if leaked:
            problems.append({"world": w, "status_cells_in_manifest_rows": leaked})
        if set(donors) - set(cells):
            problems.append({"world": w, "donors_outside_task_file": sorted(set(donors) - set(cells))})
        worlds[w] = {"manifest": f"{man_rel} @ {git('log', '-1', '--format=%h', '--', man_rel).strip()}", "staging_record": f"{recs[0]} @ {git('log', '-1', '--format=%h', '--', recs[0]).strip()}",
                     "rows": len(rows), "staged": len(rows) - len(excl), "staging_excluded": excl, "donor_cells": len(donors)}
    doc = {"what": f"by-key coverage of the interleaved family {pq} against its registered task file", "task_file": task_ref, "status_rows": status_ref, "worlds_checked": want,
           "cells": len(cells), "status_keys": status_keys, "worlds": worlds, "problems": problems}
    Path(out).open("x").write(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"cells": len(cells), **{w: {k: v[k] for k in ("rows", "staged")} | {"excluded": sorted(v["staging_excluded"])} for w, v in worlds.items()}, "status_keys": len(status_keys), "problems": len(problems)}))
    if problems:
        raise SystemExit("coverage FAILED")


if __name__ == "__main__":
    main()
