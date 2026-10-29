"""Launch run_grid_plants.py's grid on the DEPLOYED app, so the calls survive the laptop's client dropping (`modal run
--detach` keeps only the last triggered function alive; the 18:27 PT outage showed it). Builds the same cfg as
run_grid_plants.main (code tag from the same three files, the same folds) and spawns one grid call per (level, layer);
finished records are skipped by grid() itself. Run `modal deploy` on run_grid_plants.py first; summarize afterwards
with `run_grid_plants.py::main --packs ... --summarize-only`.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' python \
      experiments/2026-10-10_q1_planted_checks/spawn_grid_plants.py level0=<d0>,level7=<d7>
"""

import hashlib
import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from run_grid_plants import APP, BOOT, C_GRID, LAYERS, READOUTS, SEED  # noqa: E402


def main(packs: str):
    folds = json.loads((ROOT / "splits" / "folds_v2.json").read_text())
    code = hashlib.sha256((HERE / "run_grid_plants.py").read_bytes() + (ROOT / "src/probe_shift/torch_probe.py").read_bytes()
                          + (ROOT / "src/probe_shift/q1_eval.py").read_bytes()).hexdigest()[:12]
    cfg = {"packs": dict(x.split("=") for x in packs.split(",")), "seed": SEED, "bootstrap": BOOT, "tag": code,
           "main_task_to_fold": folds["task_folds"]["main_task_to_fold"], "environment_to_fold": folds["environment_folds"]["environment_to_fold"]}
    grid = modal.Function.from_name(APP, "grid")
    calls = [grid.spawn(cfg, lv, l, C_GRID, READOUTS) for lv in ("level0", "level7") for l in LAYERS]
    print(json.dumps({"code_tag": code, "spawned": [c.object_id for c in calls]}))


if __name__ == "__main__":
    main(sys.argv[1])
