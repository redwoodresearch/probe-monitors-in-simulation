"""Post hoc ($0): Δc against a pooled real reference, since the direction flips with the reference run (orchestrator).

The pre-registered Δc uses e1 as the reference (P(sim > e1) - P(e2 > e1)); against e2 it changes sign for B. Pooled,
both real runs count symmetrically: dc_pooled = mean over cells of [c(sim, e1) + c(sim, e2)] / 2 - 0.5, where c is 1
if the first run scores higher, 0.5 on ties. The real-vs-real term [c(e1, e2) + c(e2, e1)] / 2 is 0.5 by
construction, and dc_pooled equals the mean of the two one-reference Δc's. It is 0 when sim is exchangeable with the
real runs. Diagnostic null: per main task, the three runs' labels permuted at random (2,000 draws, seed 20261012).
Planted scale: e1 with the far eval line cannot use its own twin e1 as a reference, so its pooled-style value uses e2
only, P(e1 far > e2) - 0.5. Same cells (all four runs scored) and task bootstrap (2,000 draws, seed 20261011) as
analyze_planted_power.py.

  uv run --no-project --with pandas python experiments/2026-10-11_eval_awareness_real_vs_sim/analyze_pooled_reference.py --scores results/claimb_scores_pilot_<stamp>.json results/claimb_scores_rest_<stamp>.json
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
B = 2000
git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT)
st = git("status", "--porcelain", "--untracked-files=no")
if st.returncode or st.stdout.strip() or git("ls-files", "--error-unmatch", __file__).returncode:
    raise SystemExit("commit first: the result must record the commit of the code that made it (review thread on #596)")
ap = argparse.ArgumentParser()
ap.add_argument("--scores", nargs="+", required=True)
a = ap.parse_args()
frames, protos = [], set()
for f in a.scores:
    s = json.loads((HERE / f).read_text())
    raw = (ROOT / ".run_cache" / "eval_awareness_real_vs_sim" / s["rows"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != s["rows_sha256"]:
        raise ValueError(f"{s['rows']} does not match its recorded sha256")
    protos.add(s["protocol_sha256"])
    frames.append(pd.DataFrame(json.loads(raw)))
if len(protos) != 1:
    raise ValueError("different scoring protocols")
df = pd.concat(frames)
df["rv"] = df["arm"] + "/" + df["version"]
cols = ["score_B", "score_A", "score_old"]
runs = df.groupby(["cell", "rv"])[cols].mean().unstack("rv")
need = ["e1/original", "e1/far", "e2/original", "sim/original"]
runs = runs.dropna(subset=[(c, v) for c in cols for v in need])
task = df.drop_duplicates("cell").set_index("cell").loc[runs.index, "main_task"].to_numpy()
tasks = sorted(set(task))
t = np.searchsorted(tasks, task)
W = np.stack([np.bincount(d, minlength=len(tasks)) for d in np.random.default_rng(20261011).integers(0, len(tasks), (B, len(tasks)))])
cnt = W @ np.bincount(t, None, len(tasks))
bm = lambda x: (W @ np.bincount(t, x, len(tasks))) / cnt
ci = lambda v: [round(float(q), 4) for q in np.percentile(v, [2.5, 97.5])]
c = lambda x, y: (x > y) + 0.5 * (x == y)
perm = np.random.default_rng(20261012).permuted(np.tile(np.arange(3), (B, len(tasks), 1)), axis=2)[:, t]  # per task, per draw
out = {"post hoc": "pooled real reference, asked for after the main reading", "scores": a.scores, "cells": int(len(runs)),
       "main tasks": len(tasks),
       "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()}
for col in cols:
    e1, far, e2, sim = (runs[(col, v)].to_numpy() for v in need)
    pooled = (c(sim, e1) + c(sim, e2)) / 2 - 0.5
    one = {"e1": c(sim, e1) - c(e2, e1), "e2": c(sim, e2) - c(e1, e2)}
    if not np.allclose(pooled, (one["e1"] + one["e2"]) / 2):
        raise ValueError("the pooled Δc is not the mean of the two one-reference Δc's")
    X = np.stack([sim, e1, e2], 1)
    rows = np.arange(len(X))[:, None]
    null = np.array([np.mean((c(Y[:, 0], Y[:, 1]) + c(Y[:, 0], Y[:, 2])) / 2 - 0.5) for Y in (X[rows, p] for p in perm)])
    planted = c(far, e2) - 0.5
    out[col.replace("score_", "")] = {
        "dc_pooled = P(sim > a real run) - 0.5": round(float(pooled.mean()), 4), "dc_pooled 95%": ci(bm(pooled)),
        "dc_pooled null 95% band (labels permuted within each task; diagnostic)": ci(null),
        "dc against e1 (pre-registered)": round(float(one["e1"].mean()), 4), "dc against e2": round(float(one["e2"].mean()), 4),
        "planted, pooled-style = P(e1 far > e2) - 0.5": round(float(planted.mean()), 4), "planted 95%": ci(bm(planted)),
        "planted minus dc_pooled 95%": ci(bm(planted) - bm(pooled))}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = HERE / "results" / f"pooled_reference_{stamp}.json"
path.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
print(json.dumps(out, indent=1))
print(f"-> {path}")
