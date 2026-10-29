"""POST HOC (Aswin about 13:45 PT 10-11, via 0d): the shuffled-label null for H2's catch at 1% (and 5%), fixed sim vs
e1, from the saved shuffle columns ($0, no refit). Two versions of the search, each redone per shuffle column with the
same selection rule (best sub-configuration per fold and column on validation AUROC):
- "eligible": the configurations that passed their planted checks (A0, M2, L3), as h2_summary's best_p used;
- "all fitted": all 11 fitted configurations (L2 skipped), planted checks ignored, i.e. the whole search.
For each: the observed catch, the null's median, p95 and max, p, and the same for the paired gap best - A0 (A0 also
reselected per column). Numbers and keys only.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with scikit-learn --with scipy \\
      --with pyyaml python experiments/2026-10-11_claimB_h2_probe_best_found/analyze_POSTHOC_h2_catch_null.py results/h2_summary_<stamp>.json
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
folds = sorted({cfg["environment_to_fold"][e] for e in A.H.TEST_ENVS})
names = [n for n in sum(A.H.FAMILIES.values(), []) if n not in A.SKIPPED]
comp = "fixed_vs_e1"
sc, Y = A.labels(cfg, runs, comp)
test = sc.runs_in & np.isin(env, folds)
y, task = Y[test], sc.main[test]
per = {}
for name in names:
    raw = read(f"{A.H.OUT}/{comp}/{name}.npz")
    if hashlib.sha256(raw).hexdigest() != summary["inputs"][f"{comp}/{name}.npz"]:
        raise SystemExit(f"{comp}/{name}.npz changed since the summary")
    z = np.load(io.BytesIO(raw))
    per[name] = (z["val_auroc"], z["test_run_score"])


def best(cfgs):
    return A.select(np.concatenate([per[n][0] for n in cfgs]), np.concatenate([per[n][1] for n in cfgs]), env, folds)[test]


def share(k, s, pct):
    c = A.catch(y[:, k], s[:, k], task)
    return c[pct]["caught"] / c[pct]["positives"] if c else np.nan


a0 = best(["A0"])
out = {"label": "POST HOC: shuffled-label null for H2's catch at 1% and 5% (fixed sim vs e1), whole selection redone per column",
       "summary": summary_path.name, "shuffles": A.H.K - 1, "versions": {}}
for version, cfgs in (("eligible", summary["eligible"]), ("all fitted", names)):
    b = best(cfgs)
    V = np.concatenate([per[n][0] for n in cfgs])
    labels = [n for n in cfgs for _ in range(per[n][0].shape[0])]
    row = {"configurations": cfgs, "picked_column0": [labels[i] for i in V[:, :, 0].argmax(0)]}
    for pct in A.T.TARGETS:
        obs_b, obs_a = share(0, b, pct), share(0, a0, pct)
        nb = np.array([share(k, b, pct) for k in range(1, A.H.K)])
        ng = nb - np.array([share(k, a0, pct) for k in range(1, A.H.K)])
        ok, okg = np.isfinite(nb), np.isfinite(ng)
        row[f"catch_{pct}pct"] = {"best": obs_b, "A0": obs_a, "null_median": float(np.median(nb[ok])), "null_p95": float(np.percentile(nb[ok], 95)),
                                  "null_max": float(nb[ok].max()), "p": float((1 + (nb[ok] >= obs_b).sum()) / (1 + ok.sum())), "null_n": int(ok.sum()),
                                  "gap": obs_b - obs_a, "gap_null_median": float(np.median(ng[okg])), "gap_null_p95": float(np.percentile(ng[okg], 95)),
                                  "gap_null_max": float(ng[okg].max()), "gap_p": float((1 + (ng[okg] >= obs_b - obs_a).sum()) / (1 + okg.sum()))}
    out["versions"][version] = row
out |= {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
dst = HERE / "results" / f"h2_catch_null_POSTHOC_{stamp}.json"
with open(dst, "x") as fh:
    fh.write(json.dumps(out, indent=1))
print(json.dumps(out["versions"], indent=1))
print("->", dst)
