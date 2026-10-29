"""09's ruling (via 2e, relayed by 4f, 10-15 ~17:3x PT): d9's S4 source map (4r part @ b68ab0a53) is authoritative for every arm and pool
(honest_a, honest_b, run1, run2, sim). For 4r-owned cells only nm360_s4r_* rows count; the map's outside_source_map rows (nm360_s4_main[_s1]
and nm360_s4_run2[_s1] roots) are ORPHANS: status rows only, never in a pool or estimate. By key, no compute, keys and counts only:
  (a) in which DT folders orphan rows were scored (per-folder DT outputs written before the map), and that no other DT folder holds a
      4r-owned cell's real or sim-attack row outside nm360_s4r_*;
  (b) which DT records read across folders, and whether any pools or estimates (DT computes none: every arm job writes per-folder scores).
Usage: python3 -I check_s4_orphans.py <wt> <out json> <manifest dir> [<manifest dir> ...] (later dirs win per folder name)"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

WT, OUT, DIRS = Path(sys.argv[1]), Path(sys.argv[2]), [Path(x) for x in sys.argv[3:]]
MAP = ("b68ab0a53", "experiments/2026-10-14_q3_overnight_chain/results/s4_source_map/s4_source_map_4r_2026-10-15.json")
WORLDS = {"honest_a", "honest_b", "run1", "run2", "sim"}
raw = subprocess.run(["git", "-C", str(WT), "show", f"{MAP[0]}:{MAP[1]}"], capture_output=True, check=True).stdout
smap = json.loads(raw)
assert smap["part"] == "4r" and all(smap["checks"].values()) and len(smap["outside_source_map"]) == 115
own4r = {c["cell"] for c in smap["cells"] if c["owner"] == "4r"}
orph = {(x["root"], x["world"], x["cell"]): x["staging"] for x in smap["outside_source_map"]}
man = {}
for d in DIRS:
    for f in d.glob("*.json"):
        if "__" in f.stem:
            man[f.stem] = f
found, stray, s4r_bad = {}, {}, {}
for fo, f in sorted(man.items()):
    m = json.loads(f.read_text())
    root = fo.split("__")[0]
    for k in m.get("runs", {}):
        w, c = k.split("|", 1)
        if (root, w, c) in orph:
            found.setdefault(fo, []).append(k)
        elif c in own4r and w in WORLDS and not fo.startswith("nm360_s4r_"):
            stray.setdefault(fo, []).append(k)
        if fo.startswith("nm360_s4r_") and c not in own4r:
            s4r_bad.setdefault(fo, []).append(k)
staged = {k for k, s in orph.items() if s == "staged"}
found_keys = {(fo.split("__")[0], *k.split("|", 1)) for fo, ks in found.items() for k in ks}
res = WT / "experiments/2026-10-14_p2_dt_scaleup/results"
s4_folders = sorted(fo for fo in man if fo.startswith("nm360_s4_"))
cross = {}
for f in sorted(res.glob("*.json")):
    t = f.read_text()
    hits = [fo for fo in s4_folders if fo in t]
    if hits and not f.name.startswith("dt_q4b_arm_nm360_s4_"):
        d = json.loads(t)
        cross[f.name] = {"mentions": hits, "label": str(d.get("label") or d.get("what") or "")[:160]}
out = {"what": "nm360 S4 orphan rows (d9's source map, 4r @ b68ab0a53) in DT folders and DT records, by key; keys and counts only; PRELIMINARY",
       "written_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S"),
       "git": subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
       "ruling": __doc__.split("Usage")[0].strip(),
       "source_map": {"commit": MAP[0], "path": MAP[1], "sha256": hashlib.sha256(raw).hexdigest(), "counts": smap["counts"], "4r_cells": len(own4r)},
       "manifests_read": {fo: hashlib.sha256(f.read_bytes()).hexdigest() for fo, f in sorted(man.items())},
       "orphan_rows_in_DT_folders": {fo: sorted(ks) for fo, ks in found.items()},
       "orphan_rows_in_DT_folders_count": len(found_keys),
       "orphan_rows_in_DT_equal_map_staged_orphans": found_keys == staged,
       "map_staged_orphans_not_in_any_DT_folder": sorted("|".join(k) for k in staged - found_keys),
       "4r_cell_rows_outside_nm360_s4r_not_in_map": {fo: sorted(ks) for fo, ks in stray.items()},
       "non_4r_cells_in_nm360_s4r_folders": {fo: sorted(ks) for fo, ks in s4r_bad.items()},
       "DT_records_that_mention_nm360_s4_folders": cross,
       "pooled_or_estimate_records": "none: DT computes no pool, threshold, rate, catch or estimate. Each arm job writes per-folder scores "
                                     "(sealed); the records listed above are integrity or accounting checks (fold check, sidecars, call counts, "
                                     "the H_A check). The orphan rows sit in the per-folder outputs of the folders listed, scored before the map; "
                                     "readers must exclude them by the map."}
OUT.write_text(json.dumps(out, indent=1))
print("orphan rows in DT:", out["orphan_rows_in_DT_folders_count"], {fo: len(ks) for fo, ks in found.items()}, "| equal to the map's staged orphans:",
      out["orphan_rows_in_DT_equal_map_staged_orphans"], "| missing:", out["map_staged_orphans_not_in_any_DT_folder"])
print("4r rows outside s4r not in map:", {fo: len(ks) for fo, ks in stray.items()}, "| non-4r in s4r:", {fo: len(ks) for fo, ks in s4r_bad.items()})
print("records mentioning nm360_s4 folders:", {k: v["mentions"][:3] for k, v in cross.items()})
