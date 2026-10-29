"""POST HOC ($0; Codex's cheapest check on the catch-at-1% lead, after Aswin's 13:45 request): does the best found's
catch-at-1% lead over A0 (fixed sim vs e1) survive dropping each test environment, and how does it look per
environment? Same saved test scores and selection as h2_summary (eligible configurations, real-label column); the 1%
table's two-way cross-fit threshold on each subset. Numbers and keys only.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with scikit-learn --with scipy \\
      --with pyyaml python experiments/2026-10-11_claimB_h2_probe_best_found/analyze_POSTHOC_h2_catch_by_env.py results/h2_summary_<stamp>.json
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
read = lambda p: b"".join(vol.read_file(p))  # noqa: E731
folds_file = json.loads((A.CLAIMB / "splits/folds_v2.json").read_text())
cfg = {"packs": {"level1": "39f842231a46"}, "seed": 0, "bootstrap": 1000, "tag": "h2",
       "main_task_to_fold": folds_file["task_folds"]["main_task_to_fold"], "environment_to_fold": folds_file["environment_folds"]["environment_to_fold"]}
runs = json.loads(read(f"{A.H.PACK_A}/runs.L63.json"))
env = np.array([cfg["environment_to_fold"][m["environment"]] for m in runs])
envname = np.array([m["environment"] for m in runs])
folds = sorted({cfg["environment_to_fold"][e] for e in A.H.TEST_ENVS})
comp = "fixed_vs_e1"
sc, Y = A.labels(cfg, runs, comp)
test = sc.runs_in & np.isin(env, folds)
per = {}
for name in summary["eligible"]:
    raw = read(f"{A.H.OUT}/{comp}/{name}.npz")
    if hashlib.sha256(raw).hexdigest() != summary["inputs"][f"{comp}/{name}.npz"]:
        raise SystemExit(f"{comp}/{name}.npz changed since the summary")
    z = np.load(io.BytesIO(raw))
    per[name] = (z["val_auroc"], z["test_run_score"])
cat = lambda cfgs: A.select(np.concatenate([per[n][0] for n in cfgs]), np.concatenate([per[n][1] for n in cfgs]), env, folds)  # noqa: E731
best, a0 = cat(summary["eligible"])[:, 0], cat(["A0"])[:, 0]
y, task = Y[:, 0], sc.main


def row(mask):
    out = {"sim_runs": int((mask & y).sum()), "real_runs": int((mask & ~y).sum())}
    for nm, s in (("best", best), ("A0", a0)):
        c = A.catch(y[mask], s[mask], task[mask])
        out[nm] = {pct: {"caught": c[pct]["caught"], "positives": c[pct]["positives"], "false_alarms": c[pct]["false_alarms"],
                         "negatives": c[pct]["negatives"]} for pct in A.T.TARGETS} if c else None
    if out["best"] and out["A0"]:
        out["gap_1pct"] = out["best"][1]["caught"] / out["best"][1]["positives"] - out["A0"][1]["caught"] / out["A0"][1]["positives"]
    return out


res = {"label": "POST HOC: catch at 1% (and 5%), best found vs A0, per test environment and leaving each out", "summary": summary_path.name,
       "all": row(test), "leave_one_out": {e: row(test & (envname != e)) for e in A.H.TEST_ENVS},
       "per_environment": {e: row(test & (envname == e)) for e in A.H.TEST_ENVS}}
res |= {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
dst = HERE / "results" / f"h2_catch_by_env_POSTHOC_{stamp}.json"
with open(dst, "x") as fh:
    fh.write(json.dumps(res, indent=1))
for k in ("all",):
    print(k, round(res[k]["gap_1pct"], 4))
for part in ("leave_one_out", "per_environment"):
    for e, r in res[part].items():
        f = lambda x: f"{x[1]['caught']}/{x[1]['positives']} FA {x[1]['false_alarms']}/{x[1]['negatives']}" if x else "n/a"  # noqa: E731
        print(part, e, "best", f(r["best"]), "| A0", f(r["A0"]), "| gap", round(r.get("gap_1pct", float("nan")), 4))
print("->", dst)
