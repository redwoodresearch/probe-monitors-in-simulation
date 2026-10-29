"""Small-model baseline, the per-environment check (0d's relay of Aswin, 22:41 PT): fixed sim vs real e1 AUROC per test
environment for Llama-3.2-1B and for Qwen3-32B (step B's output_end grid, step A's readout), from the grids' own run
scores (run_grid_run_scores_modal.py dump). Local, $0.

  uv run --no-project --python 3.13 --with numpy --with scikit-learn python experiments/2026-10-10_claimB_small_lm/analyze_small_lm_per_env.py
"""
import gzip
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

DUMP = Path(__file__).parents[1] / "2026-10-10_claimB_qwen35_reader/data/grid_run_scores_2026-10-10_235051.json.gz"
TEST = ["clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"]
d = json.loads(gzip.decompress(DUMP.read_bytes()))
pick = {"Llama-3.2-1B": "llama|level1|fixed_vs_e1|by_readout.output_end.best",
        "Qwen3-32B (step B output_end)": "stepB|level1|fixed_vs_e1|by_readout.output_end.best"}
res = {}
for name, key in pick.items():
    runs = next(g for g in d["results"] if g["name"] == key)["runs"]
    res[name] = {}
    for env in TEST:
        r = [x for x in runs if x["environment"] == env]
        y, s = np.array([x["label"] for x in r]), np.array([x["score"] for x in r])
        res[name][env] = {"auroc": round(float(roc_auc_score(y, s)), 3), "runs": len(r)}
print(json.dumps(res, indent=1))
