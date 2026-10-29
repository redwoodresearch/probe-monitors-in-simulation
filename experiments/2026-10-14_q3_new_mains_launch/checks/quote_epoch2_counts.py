"""Second real-honest epoch quote, part 1 (d9 for Aswin via 0d, 10-15 ~11:05; QUOTE ONLY): current pooled real honest runs by origin
and arm, from records only (counts and keys, no outcome read):
  - T196 (S_old) and +311 (S_new): the honest calibration pools in 09's FINAL X_err record @ 3ce6ea55e (primary scope; pools B and AuB);
  - S_newmains (nm360): the main-target honest manifests in the chain's registry_S_newmains.json (aswin/q3-overnight-chain), deduplicated
    by record path, minus keys the stager excluded (chain_stage_nm360_*.json c_gate.excluded) and minus 2026-10-14_newmains_s4 rows
    of cells outside the S4 source map (s4_source_map_4r_2026-10-15.json @ b68ab0a53: those cells are 4r's);
  - the 1e-4 real honest set (T196 x 5 replicates, honest_b; branch aswin/q3-q35-realhonest-1e4 @ 0c57a4c8e ACTIVATIONS line), reported
    separately (not in FINAL's primary pools).
  python3 quote_epoch2_counts.py <xerr final json> <out json>"""
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

final_p, dest = Path(sys.argv[1]), Path(sys.argv[2])
CH = "experiments/2026-10-14_q3_overnight_chain"
REF = "origin/aswin/q3-overnight-chain"
show = lambda ref, p: subprocess.run(["git", "show", f"{ref}:{p}"], capture_output=True, text=True, check=True, timeout=120).stdout
fin = json.loads(final_p.read_text())
cr = fin["result"]["calibration_records"]["primary"]
pools = {}
for k, v in cr.items():
    for side in ("old", "new"):
        pools.setdefault((side, v[f"pool_{side}"]), set()).add(v[f"n_pool_{side}"])
pool_n = {f"{s}_{p}": sorted(n) for (s, p), n in pools.items()}
auB = cr[next(k for k in cr if k.endswith("|AuB|AuB|AuB"))]["schemes"]["union"]  # Codex r57: distinct tasks from the A-or-B pool keys
pool_tasks = {side: sorted({x.split("|", 1)[1] for x in auB[f"pool_keys_{side}"]}) for side in ("old", "new")}
reg = json.loads(show(REF, f"{CH}/results/hygiene/registry_S_newmains.json"))
excl = set()
for n in subprocess.run(["git", "ls-tree", "--name-only", f"{REF}:{CH}/results"], capture_output=True, text=True, check=True).stdout.split():
    if n.startswith("chain_stage_nm360_"):
        for part, v in json.loads(show(REF, f"{CH}/results/{n}")).items():
            if isinstance(v, dict):
                excl |= {k.split("#")[0] for k in (v.get("c_gate") or {}).get("excluded", {})}
s4map = json.loads(show("b68ab0a53", f"{CH}/results/s4_source_map/s4_source_map_4r_2026-10-15.json"))
outside = {o["record"] for o in s4map["outside_source_map"]}
rows = {}
for name, e in reg["entries"].items():
    if not re.fullmatch(r"nm360_s\d[r]?_main(_s\d)?", e["target"]):
        continue
    for hm in e["honest_manifests"]:
        path, commit = [x.strip() for x in hm["manifest"].split("@")]
        for f in json.loads(show(commit, path))["files"]:
            if f["world"] in ("honest_a", "honest_b"):
                rows[f["path"]] = {"world": f["world"], "launch": f["launch"], "task": f["task"], "target": e["target"],
                                   "excluded": f["path"] in excl, "outside_s4_map": f["path"] in outside}
cnt = collections.Counter((r["launch"], r["world"], "excluded" if r["excluded"] else ("outside_s4_map" if r["outside_s4_map"] else "pooled"))
                          for r in rows.values())
nm = collections.Counter(r["world"] for r in rows.values() if not r["excluded"] and not r["outside_s4_map"])
nm_tasks = sorted({r["task"] for r in rows.values() if not r["excluded"] and not r["outside_s4_map"]})
tasks_nm = len(nm_tasks)
old_ab, old_b = max(pool_n["old_AuB"]), max(pool_n["old_B"])
new_ab, new_b = max(pool_n["new_AuB"]), max(pool_n["new_B"])
doc = {"sources": {"final": "09's X_err FINAL record @ 3ce6ea55e, experiments/2026-10-14_xerr_scaleup_09/results/xerr_scaleup_final_2026-10-15_013255.json",
                   "registry": f"{REF}:{CH}/results/hygiene/registry_S_newmains.json", "s4_map": "s4_source_map_4r_2026-10-15.json @ b68ab0a53",
                   "rh1e4": "aswin/q3-q35-realhonest-1e4 @ 0c57a4c8e: rh1e4_extract_v1 (980 counted / 963 staged real honest, honest_b)"},
       "final_primary_pools": pool_n,
       "pooled_real_honest": {
           "S_old_T196_cohort": {"cells": fin["result"]["cohort"]["old_v6"], "tasks_with_pooled_honest": len(pool_tasks["old"]),
                                 "honest_a": old_ab - old_b, "honest_b": old_b, "total": old_ab},
           "S_new_plus311": {"cells": fin["result"]["cohort"]["S_new_attempted_all_three"], "tasks_with_pooled_honest": len(pool_tasks["new"]),
                             "honest_a": new_ab - new_b, "honest_b": new_b, "total": new_ab},
           "S_newmains_nm360": {"tasks": tasks_nm, "honest_a": nm["honest_a"], "honest_b": nm["honest_b"], "total": nm["honest_a"] + nm["honest_b"]}},
       "task_lists": {"S_old_T196": pool_tasks["old"], "S_new_plus311": pool_tasks["new"], "S_newmains_nm360": nm_tasks},
       "rh1e4_separate": {"counted": 980, "staged": 963, "arm": "honest_b", "tasks": 196, "replicates": 5, "in_final_primary_pool": False},
       "nm360_by_launch": {f"{l}|{w}|{s}": n for (l, w, s), n in sorted(cnt.items())}}
p = doc["pooled_real_honest"]
doc["totals"] = {"final_plus_nm360": sum(v["total"] for v in p.values()), "with_rh1e4_staged": sum(v["total"] for v in p.values()) + 963}
dest.write_text(json.dumps(doc, indent=1) + "\n")
print(json.dumps(doc["pooled_real_honest"]), json.dumps(doc["totals"]))
