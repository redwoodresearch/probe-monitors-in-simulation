"""POST HOC (after h2_summary_2026-10-11_132926.json): which configuration and sub-configuration the best-found
selection picked per test fold on the real-label column, for each comparison. Checks that the e2-vs-e1 D of exactly 0
means A0 was picked in every fold, not a bug. Same files, eligibility and selection rule as analyze_h2_search.py.
Numbers and keys only.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with scikit-learn --with scipy \\
      --with pyyaml python experiments/2026-10-11_claimB_h2_probe_best_found/analyze_POSTHOC_h2_selected.py results/h2_summary_<stamp>.json
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

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import analyze_h2_search as A  # noqa: E402

summary_path = HERE / sys.argv[1]
summary = json.loads(summary_path.read_text())
vol = modal.Volume.from_name(A.H.VOL, version=2)
folds_file = json.loads((A.CLAIMB / "splits/folds_v2.json").read_text())
folds = sorted({folds_file["environment_folds"]["environment_to_fold"][e] for e in A.H.TEST_ENVS})
out = {"label": "POST HOC: best-found selection per test fold, real-label column", "summary": summary_path.name,
       "eligible": summary["eligible"], "picked": {}, "inputs": {}}
for comp in A.COMPS:
    folder = f"{comp}_k1" if comp.startswith("plant") else comp
    V, names = [], []
    for name in summary["eligible"]:
        raw = b"".join(vol.read_file(f"{A.H.OUT}/{folder}/{name}.npz"))
        out["inputs"][f"{folder}/{name}.npz"] = hashlib.sha256(raw).hexdigest()
        if out["inputs"][f"{folder}/{name}.npz"] != summary["inputs"][f"{comp}/{name}.npz"]:
            raise SystemExit(f"{folder}/{name}.npz changed since the summary")
        z = np.load(io.BytesIO(raw))
        V.append(z["val_auroc"][:, :, 0])
        names += [f"{name}:{json.dumps(s)}" for s in A.H.subconfigs(name)]
    V = np.concatenate(V)
    best = V.argmax(axis=0)
    out["picked"][comp] = {str(f): {"config": names[b], "val_auroc": float(V[b, i]),
                                    "best_A0_val_auroc": float(max(V[j, i] for j, n in enumerate(names) if n.startswith("A0:")))}
                           for i, (f, b) in enumerate(zip(folds, best))}
out |= {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
dst = HERE / "results" / f"h2_selected_POSTHOC_{stamp}.json"
with open(dst, "x") as fh:
    fh.write(json.dumps(out, indent=1))
for comp, p in out["picked"].items():
    print(comp, {f: (v["config"].split(":")[0], round(v["val_auroc"], 4), round(v["best_A0_val_auroc"], 4)) for f, v in p.items()})
print("->", dst)
