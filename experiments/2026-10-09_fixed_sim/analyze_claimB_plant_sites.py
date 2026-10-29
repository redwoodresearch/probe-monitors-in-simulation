"""The planted checks scored at the plant's own readout (PLAN.md "Probe step: planted-difference checks": "Scored per
run, and at the plant's own segment-end readout"), local, from per-output scores of A's chosen plant1 / plant2 probes:
positives are the planted copies' site outputs, negatives the same outputs of their twins (the sites from staging's
manifest; message index mapped to step and action with fixes.calls_of). Output-level AUROC with a 95% main-task
interval, and the share of pairs whose planted site scores above its twin's. Writes results/claimB_plant_sites_<stamp>.json.

  uv run python experiments/2026-10-09_fixed_sim/analyze_claimB_plant_sites.py <plant1 scores> <plant2 scores> <manifest_l1.json> <local v2 run>
"""

import gzip
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import fixes  # noqa: E402

from probe_shift.scoring import auroc_ci  # noqa: E402


def main(p1, p2, manifest, run_dir):
    man = json.loads(Path(manifest).read_text())
    msgs = {}
    for f in sorted((Path(run_dir) / "inputs").glob("shard_*.jsonl")):
        for r in map(json.loads, f.read_text().splitlines()):
            if r["arm"] == "sim-attack":
                msgs[r["key"]] = {c["msg"]: (c["step"], c["action"]) for c in fixes.calls_of(r["messages"])}
    res = {}
    for check, path, sites in (("plant1", p1, [(s["key"], [s["msg"]]) for s in man["plant1"]["sites"]]),
                               ("plant2", p2, [(s["key"], s["msgs"]) for s in man["plant2"]["sites"]])):
        rows = json.loads(gzip.decompress(Path(path).read_bytes()))["levels"]["level1"]["rows"]
        score = {(r["key"], r["step"], r["action"]): (r["score"], r["main_task"]) for r in rows}
        y, s, task, wins, missing = [], [], [], [], 0
        for key, ms in sites:
            for m in ms:
                sa = msgs[key][m]
                twin, planted = score.get((key, *sa)), score.get((f"{key}__{check}", *sa))
                if twin is None or planted is None:  # the site's readout fell outside the cut in this copy
                    missing += 1
                    continue
                y += [0, 1]
                s += [twin[0], planted[0]]
                task += [twin[1], twin[1]]
                wins.append(planted[0] > twin[0])
        res[check] = auroc_ci(np.array(y, bool), np.array(s), np.array(task), 1000, 0) | {
            "site_pairs": len(wins), "planted_above_twin": round(float(np.mean(wins)), 4), "sites_outside_cut": missing}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"claimB_plant_sites_{stamp}.json"
    with open(path, "x") as f:
        f.write(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(path)


if __name__ == "__main__":
    main(*sys.argv[1:5])
