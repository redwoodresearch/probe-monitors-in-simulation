"""POST HOC (after H3's result, h3_small_2026-10-11_155930.json: planted check 1 failed for step A's pipeline at L58/L63,
AUROC 0.794 vs the 0.833 floor): is that the four fixes, or the layer set? H3's linear pipeline selects over L58 and
L63 only; H2's floors (0.833 / 0.910) were set against A0, which selects over all 8 layers. From H2's saved UNEVENED
planted fits (plant1_k1/A0.npz, plant2_k1/A0.npz; real-label column), redo A0's selection restricted to L58/L63 and read
the planted AUROCs, beside A0's own (all 8 layers). Numbers only; $0 (laptop, reads saved score arrays).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with scikit-learn --with scipy \\
      --with pyyaml python experiments/2026-10-11_claimB_h2_probe_best_found/analyze_POSTHOC_plant1_layers.py results/h2_summary_2026-10-11_132926.json
"""

import hashlib
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import analyze_h2_search as A  # noqa: E402

summary_path = HERE / sys.argv[1]
summary = json.loads(summary_path.read_text())
vol = modal.Volume.from_name(A.H.VOL, version=2)
read = lambda p: b"".join(vol.read_file(p))  # noqa: E731
folds_file = json.loads((A.CLAIMB / "splits/folds_v2.json").read_text())
cfg = {"packs": {"level1": "39f842231a46"}, "seed": 0, "bootstrap": 1000, "tag": "h2",
       "main_task_to_fold": folds_file["task_folds"]["main_task_to_fold"], "environment_to_fold": folds_file["environment_folds"]["environment_to_fold"]}
runs = json.loads(read(f"{A.H.PACK_A}/runs.L63.json"))
env = np.array([cfg["environment_to_fold"][m["environment"]] for m in runs])
folds = sorted({cfg["environment_to_fold"][e] for e in A.H.TEST_ENVS})
out = {"label": "POST HOC: H2's unevened planted checks with A0's selection restricted to L58/L63 (H3's layer set)", "summary": summary_path.name, "rows": {}, "inputs": {}}
for comp in ("plant1", "plant2"):
    sc, Y = A.labels(cfg, runs, comp)
    test = sc.runs_in & np.isin(env, folds)
    raw = read(f"{A.H.OUT}/{comp}_k1/A0.npz")
    if hashlib.sha256(raw).hexdigest() != summary["inputs"][f"{comp}/A0.npz"]:
        raise SystemExit(f"{comp}_k1/A0.npz changed since the summary")
    out["inputs"][f"{comp}_k1/A0.npz"] = hashlib.sha256(raw).hexdigest()
    z = np.load(io.BytesIO(raw))
    subs = json.loads(str(z["subs"]))
    keep = np.array([s["layer"] in (58, 63) for s in subs])
    y = Y[test, 0]
    full = A.select(z["val_auroc"], z["test_run_score"], env, folds)[test, 0]
    two = A.select(z["val_auroc"][keep], z["test_run_score"][keep], env, folds)[test, 0]
    out["rows"][comp] = {"A0_all_8_layers": float(roc_auc_score(y, full)), "A0_L58_L63_only": float(roc_auc_score(y, two)),
                         "floor": A.PLANT_MIN[comp], "runs": int(test.sum())}
out |= {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
dst = HERE / "results" / f"plant1_layers_POSTHOC_{stamp}.json"
with open(dst, "x") as fh:
    fh.write(json.dumps(out, indent=1))
print(json.dumps(out["rows"], indent=1))
print("->", dst)
