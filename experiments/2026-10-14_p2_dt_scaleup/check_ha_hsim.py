"""DT's H_A vs H_sim, by key (no compute, keys and counts only). For each pool (+311: simh311 vs su311 honest_a; nm360: simh_nm360 vs nm360
honest_a):
  H_sim = the union of run keys over every DT sim-honest folder of the pool (all snapshots, finals and _rlr roots; a key scored once per
          batch, 25h section 11, so this is after supersession; staging-excluded keys have no rows and are not in it);
  DT honest_a = every DT honest_a folder of the pool (manifest_dt.json fetched read-only from the work Volume);
  H_A source per H_sim cell = the 25f folder for the cells 09 moved there (t309, t325; t389 too, but t389 is not in H_sim), else the
          batch-level main honest_a folder(s) holding that cell.
Writes a JSON record. Usage: python3 -I check_ha_hsim.py <simh manifests dir> <honest_a manifests dir> <results dir> <wt> <out json>"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SIMH, HA, RES, WT, OUT = (Path(x) for x in sys.argv[1:6])
HA25F = {"t389_model_registry": ("su311_q1_main_ha25f_v1__honest_a", "d160ffdc956ffc89180043b3575b002d4b0f1cc3f7600082f38decc81f28dae0"),
         "t309_slack_clone": ("su311_q2_main_ha25f_v1__honest_a", "69c964700620933d0fbfa771fc83068d208a3e20c48dfad47d443a2ec88f5028"),
         "t325_model_registry": ("su311_q4_main_ha25f_v1__honest_a", "e1cdf8d1594bac78b6e64377557bf63d008f2af0248912f21c7bd60c523b7247")}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
out = {"what": "DT's H_A (honest_a source per cell) against H_sim (sim-honest cells scored by DT), by key; keys and counts only; PRELIMINARY",
       "written_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S"),
       "git": subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
       "definitions": __doc__.split("Writes")[0].strip(), "pools": {}}
for pool, simh_pre, ha_pre in (("+311 (25f)", "simh311_", "su311_"), ("nm360", "simh_nm360_", "nm360_")):
    sf = sorted(f for f in SIMH.glob(f"{simh_pre}*__sim_honest.json"))
    hsim, by_folder = {}, {}
    for f in sf:
        ks = [k.split("|", 1)[1] for k in json.loads(f.read_text())["runs"]]
        by_folder[f.stem] = {"runs": len(ks), "manifest_sha256": sha(f)}
        for c in ks:
            hsim.setdefault(c, []).append(f.stem)
    dup = {c: v for c, v in hsim.items() if len(v) > 1}
    hf = sorted(f for f in HA.glob(f"{ha_pre}*__honest_a.json"))
    ha_of, ha_folders = {}, {}
    for f in hf:
        ks = [k.split("|", 1)[1] for k in json.loads(f.read_text())["runs"]]
        ha_folders[f.stem] = {"runs": len(ks), "manifest_sha256": sha(f), "25f": "ha25f" in f.stem}
        for c in ks:
            ha_of.setdefault(c, []).append(f.stem)
    src, missing = {}, []
    for c in sorted(hsim):
        if c in HA25F:
            src[c] = [HA25F[c][0]] if HA25F[c][0] in ha_of.get(c, []) else []
        else:
            src[c] = [g for g in ha_of.get(c, []) if "ha25f" not in g]
        if len(src[c]) != 1:
            missing.append({"cell": c, "sources": src[c]})
    ha_cells = set(ha_of)
    rec_sha = {}
    for c, (fo, want) in HA25F.items():
        r = sorted(RES.glob(f"dt_q4b_arm_{fo.split('_v1')[0]}_*.json"))
        if r and pool.startswith("+311"):
            pk = json.loads(r[-1].read_text())["preflight"]["key"]
            rec_sha[c] = {"record": r[-1].name, "eval_sha256_in_record": pk.get(f"honest_a|{c}", {}).get("eval_sha256"), "equals_09": pk.get(f"honest_a|{c}", {}).get("eval_sha256") == want,
                          "also_in_batch_level_folders": [g for g in ha_of.get(c, []) if "ha25f" not in g], "in_H_sim": c in hsim,
                          "in_DT_H_A": c in hsim and src.get(c) == [fo]}
    out["pools"][pool] = {
        "sim_honest_folders": by_folder, "honest_a_folders": ha_folders,
        "H_sim_cells": len(hsim), "H_sim_keys_in_two_folders": dup,
        "H_A_cells": sum(1 for c in hsim if len(src[c]) == 1), "H_A_equals_H_sim": not missing and not dup,
        "H_sim_cells_without_one_DT_honest_a_source": missing,
        "honest_a_cells_outside_H_sim": sorted(ha_cells - set(hsim)),
        "H_A_source_counts": {g: sum(1 for c in hsim if src[c] == [g]) for g in sorted(ha_folders)},
        "25f_cells": rec_sha}
OUT.write_text(json.dumps(out, indent=1))
for p, v in out["pools"].items():
    print(p, "| H_sim", v["H_sim_cells"], "| H_A", v["H_A_cells"], "| equal", v["H_A_equals_H_sim"], "| dup", len(v["H_sim_keys_in_two_folders"]),
          "| H_sim w/o single source", [m["cell"] for m in v["H_sim_cells_without_one_DT_honest_a_source"]],
          "| honest_a outside H_sim", len(v["honest_a_cells_outside_H_sim"]))
    for c, x in v["25f_cells"].items():
        print("   ", c, x)
