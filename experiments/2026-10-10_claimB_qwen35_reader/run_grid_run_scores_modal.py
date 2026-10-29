"""Per-run test scores of claim B grids' chosen probes, for the 1%-false-alarm table (analyze_tpr_at_fpr.py), on Modal CPU.
No fitting: for each published grid result, each (level, comparison) and each selector listed below, the per-fold choice
saved in the result file (`...best.choices`: {fold: [readout, layer, C]}) picks the cell; that cell's saved
`test_run_score` (column 0, the real labels) is taken for the fold's runs, exactly as the grid's `summarize` assembles
them. The assembled scores must reproduce the published test-environment AUROC and interval (sc.auc_ci) exactly, or
nothing is written. Writes run key, arm, environment, main task, cell, fold, label and score (no text) to the Volume
first, then data/grid_run_scores_<stamp>.json.gz.

Grids: step B (d9's 857cb3cc, 4 readouts), the on-policy dev/test swap and follow-up (b), all on the grid copy in this
folder (`done_path` and `setup` unchanged from d9's).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with pyyaml modal run --detach \
      experiments/2026-10-10_claimB_qwen35_reader/run_grid_run_scores_modal.py
"""

import gzip
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
GRID_FILE = HERE / "run_claimB_grid.py"
REPO = HERE.parents[1] if modal.is_local() else HERE
sys.path.insert(0, str(HERE) if modal.is_local() else "/root")
import run_claimB_grid as G  # noqa: E402  (the grid copy in this folder)

GRIDS = {  # name: (result file, folds file used by that grid)
    "stepB": ("experiments/2026-10-09_fixed_sim/results/claimB_b_grid_2026-10-10_050016_d9-857cb3cc.json", "splits/folds_v2.json"),
    "swap": ("experiments/2026-10-10_claimB_onpolicy/results/claimB_onpolicy_swap_grid_2026-10-10_213914.json",
             "experiments/2026-10-10_claimB_onpolicy/folds_swap_dev_test.json"),
    "b": ("experiments/2026-10-10_claimB_qwen35_reader/results/claimB_b_grid_2026-10-10_221638.json", "splits/folds_v2.json"),
    # the small-model baseline (Llama-3.2-1B, pack 091e169bff86; the grid copy wrote it under this folder's name)
    "llama": ("experiments/2026-10-10_claimB_qwen35_reader/results/claimB_b_grid_2026-10-10_234721.json", "splits/folds_v2.json"),
}
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

app = modal.App("probe-shift_aswin-surya_sub-claimb_grid-run-scores_kill-2026-11-10")
image = G.image.add_local_file(GRID_FILE, "/root/run_claimB_grid.py")


@app.function(image=image, cpu=8, memory=32768, volumes={"/acts": G.vol}, timeout=1800, retries=0)
def dump(jobs: list[dict], stamp: str) -> list[dict]:
    import numpy as np

    G.vol.reload()
    out = []
    for j in jobs:
        cfg, level, comp, layer0 = j["cfg"], j["level"], j["comparison"], j["layer0"]
        _, runs, _, sc, Y, env = G.setup(cfg, level, comp, layer0)
        cells = {}
        s = np.full(len(runs), np.nan)
        for f, (rd, l, C) in j["choices"].items():
            key = (rd, l, C)
            if key not in cells:
                cells[key] = np.load(G.done_path(cfg, level, comp, rd, l, C))
            m = env == int(f)
            s[m] = cells[key]["test_run_score"][m, 0]
        r_in = np.flatnonzero(sc.runs_in)
        test = r_in[np.isin([runs[i]["environment"] for i in r_in], G.TEST_ENVS)]
        if not np.isfinite(s[r_in]).all():
            raise ValueError(f"{j['name']}: a compared run has no score")
        got = sc.auc_ci(Y[test, 0], s[test], sc.main[test])
        if got != j["published"]:
            raise ValueError(f"{j['name']}: reassembled {got}, published {j['published']}; nothing written")
        rec = [{k: runs[i].get(k) for k in ("key", "arm", "environment", "main_task", "cell")}
               | {"fold": int(env[i]), "label": int(Y[i, 0]), "score": float(s[i])} for i in r_in]
        out.append({k: j[k] for k in ("name", "grid", "level", "comparison", "selector", "choices", "published")} | {"reproduced": got, "runs": rec})
    p = Path("/acts/claimB/grid_run_scores") / f"grid_run_scores_{stamp}.json.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p.with_suffix(".tmp"), "wt") as fh:
        json.dump({"script_sha256": SCRIPT_SHA, "results": out}, fh)
    p.with_suffix(".tmp").replace(p)
    G.vol.commit()
    return out


@app.local_entrypoint()
def main():
    import subprocess

    jobs = []
    for grid, (res_file, folds_file) in GRIDS.items():
        res = json.loads((REPO / res_file).read_text())
        folds = json.loads((REPO / folds_file).read_text())
        meta = res["meta"]
        cfg = {"packs": meta["packs"], "seed": G.SEED, "bootstrap": G.BOOT, "tag": meta["code_tag"],
               "main_task_to_fold": folds["task_folds"]["main_task_to_fold"], "environment_to_fold": folds["environment_folds"]["environment_to_fold"]}
        for level, comps in res["levels"].items():
            for comp, v in comps.items():
                sels = [(f"by_readout.{rd}.best", x["best"]) for rd, x in v["by_readout"].items()]
                if grid == "stepB":
                    sels.append(("all_readouts.best", v["all_readouts"]["best"]))
                for sel, x in sels:
                    if "invalid" in x:
                        continue
                    jobs.append({"name": f"{grid}|{level}|{comp}|{sel}", "grid": grid, "level": level, "comparison": comp, "selector": sel,
                                 "cfg": cfg, "layer0": meta["layers"][0], "choices": x["choices"], "published": x["test_envs"]["test"]})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    print(f"{len(jobs)} dumps; script sha256 {SCRIPT_SHA[:12]}", flush=True)
    out = dump.remote(jobs, stamp)
    bad = [o["name"] for o in out if o["reproduced"] != o["published"]]
    for o in out:
        print(f"{o['name']}: published {o['published']} | reproduced {o['reproduced']} | runs {len(o['runs'])}", flush=True)
    if bad:
        raise SystemExit(f"not reproduced, nothing written locally: {bad}")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dst = HERE / "data" / f"grid_run_scores_{stamp}.json.gz"
    dst.parent.mkdir(exist_ok=True)
    with gzip.open(dst, "xt") as fh:
        json.dump({"git": git, "script_sha256": SCRIPT_SHA, "grids": GRIDS, "results": out}, fh)
    print("->", dst)
