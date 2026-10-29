"""DT sim-honest coverage per committed record with DT rows (results/ and results/superseded/): mapped roots (record meta) vs the DT sim-honest
files the record read (meta simh_files DT entries; for S_newmains, DT_SIMH_NM of the code at the record's git commit)."""
import ast, glob, json, subprocess, sys
REPO = sys.argv[1]
g = lambda c, p: subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True, cwd=REPO).stdout
MAPDIR = "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/simh_mapping"
def roots(f, c):
    return {r["arm"].removesuffix("__sim_honest") for r in json.loads(g(c, f"{MAPDIR}/{f}.json"))["roots"]}
def nm_dt_roots(commit):
    src = g(commit, "experiments/2026-10-15_fpr_sweep_507_prep/run_peraction311.py").decode()
    t = ast.parse(src); names = {}
    for node in t.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", "") == "DT_SIMH_NM":
            names = ast.literal_eval(node.value)
    out = set()
    for c, n, _ in names:
        out |= {f.removesuffix("__sim_honest") for f in json.loads(g(c, f"experiments/2026-10-14_p2_dt_scaleup/results/{n}.json"))["folders"]}
    return out, len(names)
res = []
for f in sorted(glob.glob("results/*.json") + glob.glob("results/superseded/*.json")):
    d = json.load(open(f)); rows = d.get("rows") or {}
    dtl = [k for k in rows if str(k).startswith("DT")] if isinstance(rows, dict) else []
    if not dtl: continue
    m = d.get("meta", {})
    # S_new mapped roots
    mapped = set()
    if m.get("v2"):
        for s_ in m["v2"]["sol_mappings"].values():
            mapped |= roots(s_.split(" @ ")[0], s_.split(" @ ")[1].split(" ")[0])
    elif isinstance(m.get("sol_mappings"), str):
        dct, c = m["sol_mappings"].rsplit(" @ ", 1)
        for fn in ast.literal_eval(dct).values():
            mapped |= roots(fn, c)
    read = {e["root"] for k, v in (m.get("simh_files") or {}).items() if k.startswith("DT") for e in v}
    if not read:  # all-tasks: the S_new DT list is in p2 manifest dt_records
        for k, v in (m.get("p2_manifest_sha256") or {}).items() if isinstance(m.get("p2_manifest_sha256"), dict) else []:
            pass
    r_ = {"record": f, "git": m.get("git", "")[:9], "dt_rows": len(dtl), "S_new_mapped": len(mapped), "S_new_dt_read": len(read),
          "S_new_missing": sorted(mapped - read) if read else "DT S_new files not in meta.simh_files"}
    if m.get("alltasks"):
        nmm = set()
        for s_ in m["alltasks"]["nm360_mappings"].values():
            nmm |= roots(s_.split(" @ ")[0], s_.split(" @ ")[1].split(" ")[0])
        nr, nrec = nm_dt_roots(m["git"])
        r_ |= {"S_newmains_mapped": len(nmm), "S_newmains_dt_records": nrec, "S_newmains_missing": sorted(nmm - nr)}
    lab = json.dumps(m.get("alltasks", {}).get("label", "")) + json.dumps(d.get("label", "")) + json.dumps(m.get("pooled_label", "")) + json.dumps(m.get("row_label", ""))
    r_["label_says"] = [s for s in ("s1 only (partial)", "partial") if s in lab]
    res.append(r_)
print(json.dumps(res, indent=1))
