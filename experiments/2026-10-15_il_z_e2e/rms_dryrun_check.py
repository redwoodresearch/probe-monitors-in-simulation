"""RMS stage B dry run (no Modal compute): for each RMS config built from INPUTS_FINAL, check by metadata listing (`modal volume ls`) that every folder,
root and manifest file exists; write the launch plan and SPEND estimate.

  python3 experiments/2026-10-15_il_z_e2e/rms_dryrun_check.py config/inputs_rms_S_new_<tag>.json config/inputs_rms_S_newmains_<tag>.json
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
MODAL, VOL = "/Users/aswinsurya/.local/bin/modal", "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
listing = {}


def ls(d):
    if d not in listing:
        r = subprocess.run([MODAL, "volume", "ls", VOL, d], capture_output=True, text=True)
        listing[d] = set() if r.returncode else {x.strip().split("/")[-1] for x in r.stdout.split("\n") if x.strip()}
    return listing[d]


plan = {"label": "RMS stage B dry run; no Modal compute; PRELIMINARY", "configs": {}, "missing": []}
for cp in sys.argv[1:]:
    c = json.loads((HERE / cp).read_text())
    need = []
    for fol, v in c["folders"].items():
        for sub in ("sol", "p2"):
            d, f = v[sub]["path"].replace("/work/", "").rsplit("/", 1)
            need.append((d, f))
    for r in c["H_sim"]["roots"]:
        need.append((r["sol"]["path"].strip("/").rstrip("/"), next(iter(r["sol"]["files"]))))
        need.append((r["p2"]["path"], r["p2"]["manifest"]["name"] if "manifest" in r["p2"] else next(iter(r["p2"]["files"]))))
    miss = [f"{d}/{f}" for d, f in need if f not in ls(d)]
    plan["configs"][cp] = {"sha256": hashlib.sha256((HERE / cp).read_bytes()).hexdigest(), "set": c["set"], "cells": len(c["cells"]), "folders": len(c["folders"]),
                           "roots": len(c["H_sim"]["roots"]), "files_checked": len(need), "missing": miss, "side_pins": c["sources"].get("side_pins")}
    plan["missing"] += miss
plan["launch"] = ["run_z.py --mode rms --rms config/inputs_rms_S_old.json (S_old; unchanged inputs)",
                  "run_rms_hits.py --config <S_new final> (Sol (a)/c_asis under §1)", "run_rms_hits.py --config <S_newmains final> (FIBS, harmfulness, refusal and high-stakes rows)",
                  "run_rms_hits.py --config <S_newmains final> --deviation-mode (Sol (a)/c_asis on S_newmains via the 25h §1 extension, gate passed @ "
                  "c6023b805; label: 'fold −1-only pool: five-model mean with all-unit per-model moments (25h §1 extension; POST HOC)')",
                  "add-on rows (gap-fill, sel97): built when pinned in INPUTS_FINAL and unreadable until 09's notes"]
plan["spend_estimate"] = {"S_old": "~$0.5", "S_new + S_newmains + extension": "~$0.3", "total": "~$0.8 raw, ~$1.0 with the 30% buffer"}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
b = json.dumps(plan, indent=1).encode()
(HERE / "results" / f"rms_dryrun_{stamp}.json").open("xb").write(b)
print(f"rms_dryrun_{stamp}.json", hashlib.sha256(b).hexdigest(), "| missing", len(plan["missing"]))
