"""H1 swap test (amendment 4 note; pricing only, Aswin about 14:55 PT via 0d: "price the new simulated runs needed to
reach its precision target"): how many new cells (a fixed-sim run, a real e1 run and a real e2 run on the same task)
bring gate 2, the projected SE of S = SE(mean G) / mean G over triple-matched identical bash calls, to <= 0.15 with
80% probability. Written before it runs; $0; reads amendment 3's cells.json (keys and stored scores only).

Model: new cells resemble the 473 test cells: each new cell is drawn with replacement from them, bringing its own
triples (zero for most cells) with their stored natural gaps G = s(sim) - s(e1) at L63. The new cells are pooled with
the existing 102 triples, and gate 2 is recomputed exactly as the runner does (boot: 2,000 main-task draws, a new cell's
copies count as new tasks, since new runs would use untouched main tasks). The test environments share no tasks with
the dev pilot (amendment 3: overlap 0), so no test cell is excluded and the zero-yield padding is exact.
400 draws per size, default_rng(20261011). Also reports the triple yield per cell.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy \\
      python experiments/2026-10-11_claimB_m2_new_runs/analyze_h1_triples_needed.py
"""

import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np

HERE = Path(__file__).parent
CELLS, PACK, BAR = "claimB/h1_test_swap_v1/cells.json", "claimB/level1/packed_39f842231a46/runs.L63.json", 0.15
TEST_ENVS = {"clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"}
NEW = (0, 50, 100, 150, 200, 300, 400, 600)


def boot_se_mean(rows, rng, draws=2000):
    tasks = sorted({m for m, _ in rows})
    by = defaultdict(list)
    for m, d in rows:
        by[m].append(d)
    sums, ns = np.array([sum(by[m]) for m in tasks]), np.array([len(by[m]) for m in tasks])
    w = rng.multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), size=draws)
    b = (w @ sums) / (w @ ns)
    return float(sums.sum() / ns.sum()), float(b.std())


def main():
    vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
    raw = b"".join(vol.read_file(CELLS))
    plan = json.loads(raw)
    arms = defaultdict(set)
    for r in json.loads(b"".join(vol.read_file(PACK))):
        if r["environment"] in TEST_ENVS:
            arms[r["cell"]].add(r["arm"])
    TEST_CELLS = sum(1 for a in arms.values() if {"sim-attack-fixed", "real-attack-e1", "real-attack-e2"} <= a)
    triples = plan["triples"]
    rows = [(t["main_task"], t["stored"]["sim"]["63"] - t["stored"]["e1"]["63"]) for t in triples]
    by_cell = defaultdict(list)
    for t in triples:
        by_cell[t["cell"]].append(t["stored"]["sim"]["63"] - t["stored"]["e1"]["63"])
    cells = list(by_cell.values()) + [[] for _ in range(TEST_CELLS - len(by_cell))]
    rng = np.random.default_rng(20261011)
    m0, se0 = boot_se_mean(rows, rng)
    out = {"label": "H1 gate 2: new cells needed for projected SE of S <= 0.15 (pricing only)", "existing": {
        "triples": len(rows), "main_tasks": len({m for m, _ in rows}), "cells_with_triples": len(by_cell), "test_cells": TEST_CELLS,
        "triples_per_cell": len(rows) / TEST_CELLS, "G": m0, "projected_se": se0 / m0 if m0 > 0 else float("inf"), "recorded_gate2": plan["gates"]["gate2_projected_se_of_S"]},
        "by_new_cells": {}}
    for n in NEW:
        proj, added = [], []
        for _ in range(400):
            new = [cells[i] for i in rng.integers(0, len(cells), n)]
            extra = [(f"new{j}", d) for j, ds in enumerate(new) for d in ds]
            m, se = boot_se_mean(rows + extra, rng)
            proj.append(se / m if m > 0 else np.inf)
            added.append(len(extra))
        proj = np.array(proj)
        out["by_new_cells"][str(n)] = {"p_gate2_pass": float((proj <= BAR).mean()), "projected_se_median": float(np.median(proj)),
                                       "new_triples_median": float(np.median(added))}
    out |= {"inputs": {"cells_json": hashlib.sha256(raw).hexdigest()},
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"h1_triples_needed_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out["existing"], indent=1))
    for n, r in out["by_new_cells"].items():
        print(n, r)
    print("->", dst)


if __name__ == "__main__":
    main()
