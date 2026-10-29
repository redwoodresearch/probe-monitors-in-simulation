"""Layer-0 control (PLAN.md), part 2: step B's grid copy (experiments/2026-10-10_claimB_qwen35_reader/run_claimB_grid.py,
the 857cb3cc code), deployed unchanged, run on each layer-0 pack: level 1, layer 0, readout output_end, its C grid,
folds_v2. Launch-only: the config is built exactly as that file's main() builds it (same code tag), one grid cell per
pack is spawned on the deployed app, and the summary is that file's own summary(); per-(comparison, C) records are on
the Volume as they finish. The result is written here, in that file's result format, one file per reading.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with pyyaml modal deploy \\
      experiments/2026-10-10_claimB_qwen35_reader/run_claimB_grid.py
  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with pyyaml \\
      python experiments/2026-10-11_claimB_layer0_control/run_layer0_grid.py spawn E=<digest> B0=<digest>
  ... run_layer0_grid.py summarize E=<digest> B0=<digest>
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
GRID = ROOT / "experiments/2026-10-10_claimB_qwen35_reader/run_claimB_grid.py"
sys.path.insert(0, str(GRID.parent))
# Constants only; the code that runs is the deployed copy of that file.
import run_claimB_grid as G


def cfg_for(digest: str) -> dict:
    folds_bytes = (ROOT / "splits" / "folds_v2.json").read_bytes()
    folds = json.loads(folds_bytes)
    code = hashlib.sha256(GRID.read_bytes() + (ROOT / "src/probe_shift/torch_probe.py").read_bytes()
                          + (ROOT / "src/probe_shift/q1_eval.py").read_bytes() + folds_bytes).hexdigest()[:12]
    return {"packs": {"level1": digest}, "seed": G.SEED, "bootstrap": G.BOOT, "tag": code,
            "main_task_to_fold": folds["task_folds"]["main_task_to_fold"], "environment_to_fold": folds["environment_folds"]["environment_to_fold"]}


def main():
    mode, packs = sys.argv[1], dict(a.split("=") for a in sys.argv[2:])
    if mode == "spawn":
        f = modal.Function.from_name(G.APP, "grid")
        calls = {r: f.spawn(cfg_for(d), "level1", 0, G.C_GRID, ["output_end"]).object_id for r, d in packs.items()}
        print(json.dumps({"tag": cfg_for(next(iter(packs.values())))["tag"], "calls": calls}))
        return
    s = modal.Function.from_name(G.APP, "summary")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    for reading, digest in packs.items():
        cfg = cfg_for(digest)
        res = s.remote(cfg, ["level1"], [0], G.C_GRID, ["output_end"])
        missing = [c for c in G.LEVEL_COMPS["level1"] if c not in res.get("level1", {})]
        if missing:
            raise SystemExit(f"{reading}: cells still missing for {missing}")
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        dst = HERE / "results" / f"layer0_grid_{reading}_{stamp}.json"
        meta = {"reading": reading, "git": git, "code_tag": cfg["tag"], "packs": cfg["packs"], "levels": ["level1"], "layers": [0],
                "C_grid": G.C_GRID, "readouts": ["output_end"], "test_envs": G.TEST_ENVS, "grid_file": str(GRID.relative_to(ROOT))}
        with open(dst, "x") as fh:
            fh.write(json.dumps({"meta": meta, "levels": res}, default=str) + "\n")
        for c, x in res["level1"].items():
            b = x["by_readout"]["output_end"]["best"]
            for sub in ("test_envs", "all_envs"):
                if sub in b:
                    print(f"EXPLORATORY {reading} level1 {c} {sub}: {b[sub]['test']} | null p95 {b[sub]['null'].get('null_p95')} p {b[sub]['null'].get('p')}")
        print("->", dst)


if __name__ == "__main__":
    main()
