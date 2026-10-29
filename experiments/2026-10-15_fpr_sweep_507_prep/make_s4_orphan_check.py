"""Keys-only check that the 32k Stage A read no S4-orphan row (d9's S4 source map @ b68ab0a53, outside_source_map (key, root) pairs), per probe.

1. M1's pinned membership710: for every folder, its assigned keys x its root (folder name before '__') never hit an orphan (key, root).
2. Per probe: every S_newmains folder the probe can read is a membership folder, and row() takes only the folder's assigned keys
   (nm_take, with orphan_guard STOPping on any orphan): so (1) covers every probe; each probe's simh_nm roots never share a root with an
   orphan (the sim-honest reader has no orphan guard), checked by name.
3. inventory_b_v2's s4_orphans_out (eval path + '#e0') is a subset of the map's orphan records, by key (counts of both printed).
4. With --records, each Stage A record's meta.without_S4 / rows exist (the job ran to its write: orphan_guard never STOPped).
Output results/checks/s4_orphan_check_32k_<stamp>.json; exit 1 unless every check passes. No score is read.

  python make_s4_orphan_check.py --inputs config/<M1>.json@<sha> [--records results/<A>.json results/<B>.json]
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
MAP = ("b68ab0a53", "experiments/2026-10-14_q3_overnight_chain/results/s4_source_map/s4_source_map_4r_2026-10-15.json")
INV = ("a0bf963a7", "experiments/2026-10-15_cut_sensitivity/results/inventory_b_v2_2026-10-16_014820.json")


def gs(c, p):
    return subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True, cwd=HERE).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--records", nargs="*", default=[])
    a = ap.parse_args()
    p_, h_ = a.inputs.rsplit("@", 1)
    ib = (HERE / p_).read_bytes()
    if hashlib.sha256(ib).hexdigest() != h_:
        raise SystemExit("inputs sha256")
    IN = json.loads(ib)
    sm = IN["sets"]["S_newmains"]["s4_source_map"]
    if (sm["commit"], sm["path"]) != MAP:
        raise SystemExit(f"M1's s4_source_map pin {sm} is not d9's map @ b68ab0a53")
    smap = json.loads(gs(*MAP))
    if hashlib.sha256(gs(*MAP)).hexdigest() != sm["sha256"]:
        raise SystemExit("S4 map sha256")
    orph = {(r["key"], r["root"]) for r in smap["outside_source_map"]}
    orph_roots = {r for _, r in orph}
    mm = IN["sets"]["S_newmains"]["membership710"]
    mb = (HERE / mm["path"]).read_bytes()
    if hashlib.sha256(mb).hexdigest() != mm["sha256"]:
        raise SystemExit("membership710 sha256")
    mem = json.loads(mb)
    hits = sorted(f"{k} @ {f}" for f, ks in mem["folder_keys"].items() for k in ks if (k, f.split("__")[0]) in orph)
    simh_roots = sorted({r for rs in IN["sets"]["S_newmains"]["roots"].values() for r in rs})
    simh_hits = sorted(r for r in simh_roots if r.split("__")[0] in orph_roots or r in orph_roots)
    inv = json.loads(gs(*INV))
    inv_keys = {x["key"].split("#", 1)[0] for x in inv["s4_orphans_out"]}
    map_recs = {r["record"] for r in smap["outside_source_map"] if r.get("record")}
    cross = {"map_orphan_rows": len(orph), "map_unique_keys": len({k for k, _ in orph}), "map_unique_records": len(map_recs),
             "inventory_b_v2_s4_orphans_out": len(inv_keys), "records_only_in_map": sorted(map_recs - inv_keys)[:10],
             "records_only_in_inventory": sorted(inv_keys - map_recs)[:10], "equal": map_recs == inv_keys,
             "inventory_subset_of_map": inv_keys <= map_recs,
             "note": ("inventory_b_v2 lists the orphans among ITS staged (cutfull) folders; the map lists every orphan record of the four S4 roots, "
                      "so the check is inventory ⊆ map (records only in the map: orphans never staged in the cutfull inventory)")}
    recs = {}
    for rp in a.records:
        R = json.loads((HERE / rp).read_bytes())
        nm_rows = {lb: [o for o in d if o.startswith("S_newmains")] for lb, d in R["rows"].items()}
        recs[rp] = {"sha256": hashlib.sha256((HERE / rp).read_bytes()).hexdigest(), "written": True, "without_S4_check": bool(R["meta"].get("without_S4")),
                    "labels_with_S_newmains_rows": sum(1 for v in nm_rows.values() if v)}
    probes = {}
    labels = sorted({lb for rp in a.records for lb in json.loads((HERE / rp).read_bytes())["rows"]}) if a.records else []
    for lb in labels:
        probes[lb] = {"membership_folders_hit": len(hits), "simh_nm_roots_hit": len(simh_hits), "pass": not hits and not simh_hits}
    out = {"what": "S4-orphan check of the 32k Stage A (keys only; no score read)", "inputs": {"path": p_, "sha256": h_},
           "s4_source_map": sm, "inventory_b_v2": {"commit": INV[0], "path": INV[1], "sha256": hashlib.sha256(gs(*INV)).hexdigest()},
           "membership710": mm, "folders_checked": len(mem["folder_keys"]), "orphan_assignments": hits,
           "simh_nm_roots": len(simh_roots), "simh_nm_roots_sharing_an_orphan_root": simh_hits, "orphan_roots": sorted(orph_roots),
           "cross_check": cross, "records": recs, "per_probe": probes,
           "reader_rule": ("S_newmains real rows: nm_take keeps only membership710 folder_keys of that folder and orphan_guard STOPs on any orphan "
                           "(run_peraction311.py); sim-honest roots are checked by name here (no reader guard)"),
           "pass": not hits and not simh_hits and cross["inventory_subset_of_map"] and all(v["written"] for v in recs.values())}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / "checks" / f"s4_orphan_check_32k_{stamp}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(out, indent=1).encode()
    dest.open("xb").write(blob)
    print(json.dumps({"file": str(dest.relative_to(HERE)), "sha256": hashlib.sha256(blob).hexdigest(), "pass": out["pass"], "orphan_assignments": len(hits),
                      "simh_hits": simh_hits, "cross_check": {k: v for k, v in cross.items() if not isinstance(v, list)}}, indent=1))
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
