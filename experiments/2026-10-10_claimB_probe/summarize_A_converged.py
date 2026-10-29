"""Probe step A re-summarized with converged fits only (the fixed-sim plan's amendment 3): run_claimB_grid.summarize at
this commit, which masks unconverged (layer, C, fold) entries before the choice, over the records the A grid wrote
(code tag e402bcc321a8, a1821be0; ap-l8l3lTAt26K3D4RRG44h10, stopped at the $85 stop with 422 of 432 records). Planted
check 2 uses the largest complete grid, as the simulator-expert session's summary did (layers 19-58, C 1e-6 to 10).
Modal CPU, a few cents.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with pyyaml \
      modal run experiments/2026-10-10_claimB_probe/summarize_A_converged.py
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
if modal.is_local():
    sys.path.insert(0, str(HERE))
else:
    sys.path.insert(0, "/root")
import run_claimB_grid as G  # noqa: E402

TAG, PACKS = "e402bcc321a8", {"level0": "10ac44f21b93", "level1": "39f842231a46"}
PLAN = {lv: {c: ([19, 32, 41, 44, 47, 52, 58], [C for C in G.C_GRID if C != 100.0]) if c == "plant2" else (G.LAYERS, G.C_GRID)
             for c in G.LEVEL_COMPS[lv]} for lv in PACKS}

app = modal.App("probe-shift_aswin-surya_sub-claimb-probe_summary_kill-2026-11-10")
image = G.image.add_local_file(HERE / "run_claimB_grid.py", "/root/run_claimB_grid.py")


@app.function(image=image, cpu=8, memory=32768, volumes={"/acts": G.vol}, timeout=1200, retries=0)
def summarize_all(cfg: dict) -> dict:
    G.vol.reload()
    out = {}
    for lv, comps in PLAN.items():
        for c, (L, Cs) in comps.items():
            missing = [(l, C) for l in L for C in Cs if not G.done_path(cfg, lv, c, l, C).exists()]
            assert not missing, f"{lv} {c}: missing records {missing}"
            out.setdefault(lv, {})[c] = G.summarize(cfg, lv, c, L, Cs) | {"layers": L, "C_grid": Cs}
    return out


@app.local_entrypoint()
def main():
    folds = json.loads((G.ROOT / "splits" / "folds_v2.json").read_text())
    cfg = {"packs": PACKS, "seed": G.SEED, "bootstrap": G.BOOT, "tag": TAG,
           "main_task_to_fold": folds["task_folds"]["main_task_to_fold"], "environment_to_fold": folds["environment_folds"]["environment_to_fold"]}
    res = summarize_all.remote(cfg)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dst = HERE / "results" / f"claimB_A_converged_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps({"meta": {"git": git, "records_tag": TAG, "packs": PACKS, "test_envs": G.TEST_ENVS, "converged_only": True,
                                      "plant2_grid": "layers 19-58, C 1e-6..10 (10 records missing)"}, "levels": res}, indent=1, default=str))
    for lv, by in res.items():
        for c, s in by.items():
            print(f"{lv} {c} converged-only: {s['convergence']['converged_only']}")
            for name in ("best_layer_and_C", "earliest_good_layer_and_C"):
                for sub, v in s[name].items():
                    print(f"  {name} {sub}: {v['test']} | null {v['null']}" if isinstance(v, dict) else f"  {name}: {v}")
    print("->", dst)
