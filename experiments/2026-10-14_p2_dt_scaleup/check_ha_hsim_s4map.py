"""Note to dt_ha_vs_hsim_check_2026-10-15_172250.json (append-only; no compute; keys and counts only): nm360 H_A vs H_sim with d9's S4 source map
applied (4r part @ b68ab0a53: 61 cells owned by 4r, disjoint from s4d's 19; newmains_s4 keys of non-s4d cells are orphans, not used).
Rule: an H_sim cell owned by 4r takes nm360_s4r_main_v1__honest_a only; every other cell takes its single non-4r DT honest_a folder.
Correction: DT's 17:2x message said the 14 S4r cells "have NO DT honest_a"; the record's field H_sim_cells_without_one_DT_honest_a_source
listed cells without exactly ONE source, and each of those 14 has TWO (nm360_s4_main[_s1]_v1__honest_a and nm360_s4r_main_v1__honest_a).
Usage: python3 -I check_ha_hsim_s4map.py <simh manifests dir> <honest_a manifests dir> <earlier record> <wt> <out json>"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SIMH, HA, PREV, WT, OUT = (Path(x) for x in sys.argv[1:6])
MAP = ("b68ab0a53", "experiments/2026-10-14_q3_overnight_chain/results/s4_source_map/s4_source_map_4r_2026-10-15.json")
S4R = "nm360_s4r_main_v1__honest_a"
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
raw = subprocess.run(["git", "-C", str(WT), "show", f"{MAP[0]}:{MAP[1]}"], capture_output=True, check=True).stdout
smap = json.loads(raw)
assert smap["part"] == "4r" and all(smap["checks"].values()) and len(smap["cells"]) == 61, "source map checks"
own4r = {c["cell"] for c in smap["cells"] if c["owner"] == "4r"}
hsim = {}
for f in sorted(SIMH.glob("simh_nm360_*__sim_honest.json")):
    for k in json.loads(f.read_text())["runs"]:
        hsim.setdefault(k.split("|", 1)[1], []).append(f.stem)
ha_of = {}
for f in sorted(HA.glob("nm360_*__honest_a.json")):
    for k in json.loads(f.read_text())["runs"]:
        ha_of.setdefault(k.split("|", 1)[1], []).append(f.stem)
src, left = {}, []
for c in sorted(hsim):
    s = ha_of.get(c, [])
    src[c] = [S4R] if c in own4r and S4R in s else [g for g in s if g != S4R] if c not in own4r else []
    if len(src[c]) != 1:
        left.append({"cell": c, "dt_honest_a_folders": s, "owned_by_4r": c in own4r})
orphans = {c: [g for g in ha_of[c] if g != S4R] for c in sorted(hsim) if c in own4r and len(ha_of.get(c, [])) > 1}
s4r_cells_not_4r = sorted(c for c, s in ha_of.items() if S4R in s and c not in own4r)
out = {"what": "nm360 H_A vs H_sim with d9's S4 source map applied; a note to the earlier check (append-only); keys and counts only; PRELIMINARY",
       "written_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S"),
       "git": subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
       "note_to": {"record": PREV.name, "sha256": sha(PREV.read_bytes())},
       "source_map": {"commit": MAP[0], "path": MAP[1], "sha256": sha(raw), "4r_cells": len(own4r), "checks": smap["checks"]},
       "rule_and_correction": __doc__.split("Usage")[0].strip(),
       "H_sim_cells": len(hsim), "H_A_cells": sum(1 for c in hsim if len(src[c]) == 1), "H_A_equals_H_sim": not left,
       "H_sim_cells_left_without_one_source": left,
       "H_sim_cells_owned_by_4r": sum(1 for c in hsim if c in own4r),
       "4r_cells_whose_newmains_s4_honest_a_is_an_orphan_not_used": orphans,
       "cells_in_s4r_honest_a_folder_not_owned_by_4r": s4r_cells_not_4r,
       "H_A_source_counts": {g: sum(1 for c in hsim if src[c] == [g]) for g in sorted({g for v in ha_of.values() for g in v})}}
OUT.write_text(json.dumps(out, indent=1))
print("nm360 | H_sim", out["H_sim_cells"], "| H_A", out["H_A_cells"], "| equal", out["H_A_equals_H_sim"], "| left", [x["cell"] for x in left],
      "| 4r-owned in H_sim", out["H_sim_cells_owned_by_4r"], "| orphans", len(orphans), "| s4r folder cells not 4r", s4r_cells_not_4r)
