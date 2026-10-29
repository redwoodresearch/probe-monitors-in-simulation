"""$0 count (Aswin's question via 0d, 10-11 about 23:50 PT): which attack cells already have attack material (an endorsed
side task with its September attack guidance) and were never scored by any claim B probe, and aren't sealed? Reads
metadata only: splits/exclusions_v4.json (cell keys, environments, main tasks; no text) and the la18 cohort's
metadata result on aswin/claimb-weaksim-control-plan (the whole file is parsed; only its cell counts and environment
names are used).

  uv run --no-project --python 3.12 python experiments/2026-10-11_claimB_halves_confirm/count_attack_material.py
"""

import collections
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
REPO = HERE.parents[1]
LA18 = "origin/aswin/claimb-weaksim-control-plan:experiments/2026-10-11_claimB_weaksim_control/results/la18_cohort_2026-10-11_211045.json"


def main():
    raw = (REPO / "splits/exclusions_v4.json").read_bytes()
    d = json.loads(raw)
    paired = set(d["paired_attack_cells"]["cells"])
    att = [s for s in d["samples"] if s["arm"] in ("real-attack-e1", "real-attack-e2", "sim-attack")]
    cells = {s["cell"]: (s["environment"], s["main_task"]) for s in att}
    unpaired = sorted(c for c in cells if c not in paired)
    scored_tasks = {cells[c][1] for c in paired}
    unpaired_tasks = {cells[c][1] for c in unpaired}
    fresh_tasks = sorted(unpaired_tasks - scored_tasks)
    arms_unpaired = collections.Counter(s["arm"] for s in att if s["cell"] in unpaired)
    la18 = json.loads(subprocess.run(["git", "show", LA18], capture_output=True, text=True, check=True, cwd=REPO).stdout)
    shared = set(la18["cells"]["environments_shared_with_claimB"])
    per_env = la18["cells_per_environment"]["real_R0"]
    out = {
        "label": "attack material never scored by a claim B probe (metadata counts only)",
        "public10": {"attack_cells_with_runs": len(cells), "paired_cells_scored_by_claimB": len(paired), "unpaired_cells": len(unpaired),
                     "unpaired_by_environment": dict(sorted(collections.Counter(cells[c][0] for c in unpaired).items())),
                     "attack_samples_on_unpaired_cells_by_arm": dict(arms_unpaired),
                     "main_tasks_scored": len(scored_tasks), "unpaired_main_tasks": len(unpaired_tasks),
                     "unpaired_main_tasks_never_scored": len(fresh_tasks)},
        "la18": {"cells": la18["cells"]["in_all_three_arms"], "environments_shared_with_claimB": sorted(shared),
                 "cells_in_shared_environments": sum(v for e, v in per_env.items() if e in shared),
                 "cells_in_other_environments": sum(v for e, v in per_env.items() if e not in shared),
                 "note": "being scored tonight by the frozen step A probe (I1, the weak-simulator positive control)"},
        "inputs": {"splits/exclusions_v4.json": hashlib.sha256(raw).hexdigest()},
        "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"attack_material_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("public10", "la18")}, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
