"""Probe step A's summary (PLAN.md amendment 2), on Modal CPU: d9's `summarize` from the pinned commit a1821be0
(experiments/2026-10-10_claimB_probe/run_claimB_grid.py, imported unchanged) over the grid records the stopped run wrote.

The grid (ap-l8l3lTAt26K3D4RRG44h10) was stopped at 02:10 PT at the $85 stop with 422 of 432 (comparison, layer, C)
records written. Every main comparison and planted check 1 are complete (8 layers x 9 C). Planted check 2 lacks 10
records (layer 63 at 8 of 9 C values; layers 19 and 32 at C = 100), so it is summarized on the largest complete grid:
layers 19, 32, 41, 44, 47, 52, 58 and C 1e-6 to 10. Same selection, null and scoring otherwise.

  cd ~/probe-shift-claimb && PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with pyyaml \
      modal run ~/probe-shift-fixed-sim/experiments/2026-10-09_fixed_sim/run_claimB_summary_modal.py
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

CLAIMB = Path.home() / "probe-shift-claimb"  # the worktree at d9's pinned commit a1821be0
GRID_FILE = CLAIMB / "experiments/2026-10-10_claimB_probe/run_claimB_grid.py"
OUT = Path(__file__).parent / "results"
PINNED = "a1821be0"
if modal.is_local():
    sys.path.insert(0, str(GRID_FILE.parent))
else:
    sys.path.insert(0, "/root")
import run_claimB_grid as G  # noqa: E402  (d9's module, unchanged)

PACKS = {"level0": "10ac44f21b93", "level1": "39f842231a46"}
PLANT2_LAYERS = [19, 32, 41, 44, 47, 52, 58]
PLANT2_C = [c for c in G.C_GRID if c != 100.0]
PLAN = {"level0": {c: (G.LAYERS, G.C_GRID) for c in G.LEVEL_COMPS["level0"]},
        "level1": {c: (PLANT2_LAYERS, PLANT2_C) if c == "plant2" else (G.LAYERS, G.C_GRID) for c in G.LEVEL_COMPS["level1"]}}

app = modal.App("probe-shift_aswin-surya_sub-claimb-probe_summary_kill-2026-11-10")
image = G.image.add_local_file(GRID_FILE, "/root/run_claimB_grid.py")


@app.function(image=image, cpu=8, memory=32768, volumes={"/acts": G.vol}, timeout=1200, retries=0)
def summarize_all(cfg: dict, plan: dict) -> dict:
    G.vol.reload()
    out = {}
    for lv, comps in plan.items():
        for c, (L, Cs) in comps.items():
            missing = [(l, C) for l in L for C in Cs if not G.done_path(cfg, lv, c, l, C).exists()]
            assert not missing, f"{lv} {c}: missing records {missing}"
            out.setdefault(lv, {})[c] = G.summarize(cfg, lv, c, L, Cs) | {"layers": L, "C_grid": Cs}
    return out


@app.local_entrypoint()
def main():
    import subprocess

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CLAIMB, capture_output=True, text=True, check=True).stdout.strip()
    assert head.startswith(PINNED), f"the claimb worktree is at {head}, not {PINNED}"
    root = CLAIMB
    folds = json.loads((root / "splits" / "folds_v2.json").read_text())
    tag = hashlib.sha256(GRID_FILE.read_bytes() + (root / "src/probe_shift/torch_probe.py").read_bytes()
                         + (root / "src/probe_shift/q1_eval.py").read_bytes()).hexdigest()[:12]
    cfg = {"packs": PACKS, "seed": G.SEED, "bootstrap": G.BOOT, "tag": tag,
           "main_task_to_fold": folds["task_folds"]["main_task_to_fold"], "environment_to_fold": folds["environment_folds"]["environment_to_fold"]}
    res = summarize_all.remote(cfg, PLAN)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = OUT / f"claimB_probe_summary_{stamp}.json"
    with open(path, "x") as f:
        f.write(json.dumps({"meta": {"d9_code": head, "code_tag": tag, "packs": PACKS, "test_envs": G.TEST_ENVS,
                                     "grid_app": "ap-l8l3lTAt26K3D4RRG44h10",
                                     "note": "grid stopped at the $85 stop; planted check 2 summarized on layers 19-58 and C 1e-6..10 (10 records missing)"},
                            "levels": res}, indent=1, default=str))
    for lv, by in res.items():
        for c, s in by.items():
            for name in ("best_layer_and_C", "earliest_good_layer_and_C"):
                for sub, v in s[name].items():
                    print(f"{lv} {c} {name} {sub}: {v['test']} | null p95 {v['null']['null_p95']} p {v['null']['p']}")
            print(f"{lv} {c} convergence: {s['convergence']['real']} real, {s['convergence']['shuffled']} shuffled, failing {len(s['convergence']['failing'])}")
    print(path)
