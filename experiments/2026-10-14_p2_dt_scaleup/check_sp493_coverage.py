"""Amendment 45: DT's final S_paper493 coverage by key (keys and counts only; no score is read). Lists every DT sp493 folder on the work Volume
(/scaleup/sp493_*/ with manifest_dt.json; /simh/sp493_g*/dt/*/), drops folders of SUPERSEDED staged sets (the job's list at HEAD), and counts
folders, runs and distinct (world, cell) joins per family, the cells covered per world against the 492-cell list, and repeated joins (must be 0).
Usage: python3 -I check_sp493_coverage.py <repo> <scratch dir> <out json>"""
import ast, collections, json, re, subprocess, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
REPO, SC, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
M, V = "/Users/aswinsurya/.local/bin/modal", "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
HERE = "experiments/2026-10-14_p2_dt_scaleup"
src = subprocess.run(["git", "-C", REPO, "show", f"HEAD:{HERE}/run_dt_q4b.py"], capture_output=True, check=True, text=True).stdout
g, tree = {"re": re}, ast.parse(src)
want = {"SET_WORLD", "SP493", "SP493_SET", "SP493_POOLS", "SP493_MIN_TRY", "SP493_STRATUM", "SP493_WORLDS", "SP493_WORLD_MAP", "SP493_SIMH_WORLD", "SP493_SUPERSEDED",
        "SP493_NAMES", "SP493_NAMES_ALL", "SP493_G12_REMAINDER_OK", "SP493_TAIL_POOLS", "SP493_GROUP", "sp493_fam"}
exec(compile(ast.Module(body=[n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in want) or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in want for t in n.targets))], type_ignores=[]), "j", "exec"), g)
ls = lambda p: [x.strip("/").split("/")[-1] for x in subprocess.run([M, "volume", "ls", V, p], capture_output=True, check=True, text=True).stdout.split()]  # noqa: E731
paths = [f"scaleup/{n}" for n in ls("scaleup") if n.startswith("sp493_")] + [f"simh/{k}/dt/{n}" for k in ls("simh") if k.startswith("sp493_g") for n in ls(f"simh/{k}/dt")]
cells = {c["key"]: c for c in json.loads(subprocess.run(["git", "-C", REPO, "show", f"HEAD:{HERE}/config/sp493_cells_v1.json"], capture_output=True, check=True).stdout)["cells"]}
SC.mkdir(parents=True, exist_ok=True)
fam_of, sup, folders, joins_of = {}, [], {}, collections.defaultdict(list)
for p in sorted(paths):
    fo = p.split("/")[-1]
    if fo.split("__")[0] in g["SP493_SUPERSEDED"]:
        sup.append(fo)
        continue
    if "manifest_dt.json" not in ls(p):
        folders[fo] = {"error": "no manifest_dt.json"}
        continue
    dst = SC / f"{fo}.manifest_dt.json"
    subprocess.run([M, "volume", "get", "--force", V, f"{p}/manifest_dt.json", str(dst)], check=True, capture_output=True)
    m = json.loads(dst.read_text())
    f_ = g["sp493_fam"](fo)
    fam = {"real_main": "real", "real_run2": "real", "sim_attack": "g1/g2 sim" if f_[1] in ("g1", "g2") else "tail sim", "sim_honest": "sim-honest"}[f_[0]]
    folders[fo] = {"family": fam, "pool": f_[1], "runs": len(m["runs"]), "world": list(m["d9_arm_to_world"].values())[0], "rows": m["rows"]}
    for j in m["runs"]:
        joins_of[j].append(fo)
fams = collections.Counter(v.get("family") for v in folders.values())
per_fam = {f: {"folders": n, "runs": sum(v["runs"] for v in folders.values() if v.get("family") == f)} for f, n in sorted(fams.items(), key=str)}
dups = {j: f for j, f in joins_of.items() if len(f) > 1}
by_world = collections.defaultdict(set)
for j in joins_of:
    by_world[j.split("|")[0]].add(j.split("|")[1])
cov = {w: {"cells": len(c), "in_sp493_list": len(c & set(cells)), "t196_task_keys": len(c - set(cells))} for w, c in sorted(by_world.items())}
tail = {j.split("|")[1] for j, f in joins_of.items() if any(folders[x].get("family") == "tail sim" for x in f)}
g12 = {j.split("|")[1] for j, f in joins_of.items() if any(folders[x].get("family") == "g1/g2 sim" for x in f)}
out = {"what": "DT S_paper493 coverage by key (Amendment 45): folders, runs, joins; superseded folders excluded; no score read",
       "written_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S"),
       "git": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
       "folders_counted": sum(1 for v in folders.values() if "family" in v), "folders_without_manifest": sorted(k for k, v in folders.items() if "error" in v),
       "superseded_folders_excluded": sup, "per_family": per_fam, "joins": len(joins_of), "repeated_joins": dups, "cells_by_world": cov,
       "sim_cells": {"g1_g2": len(g12), "tails": len(tail), "union": len(g12 | tail), "overlap": len(g12 & tail)}, "folders": folders}
out["pass"] = not dups and not out["folders_without_manifest"]
OUT.open("x").write(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "folders"}, indent=0))
