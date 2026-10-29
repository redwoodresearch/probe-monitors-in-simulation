"""Post hoc ($0): would this design have seen a sim effect the size of the far eval line? A positive control.

Added after the main reading, not pre-registered. Treat e1 with the far eval line as a pseudo-sim arm against the other
real run e2: dc_planted = P(e1 far > e2) - P(e1 > e2), and R_planted = mean|e1 far - e2| / mean|e1 - e2|, over cells
whose e1, e1 far and e2 all have scored calls. Same task bootstrap as analyze_real_vs_sim.py (2,000 draws, seed
20261011). The comparison with the observed result uses the same real reference arm, e2, for both: P(e1 far > e2) -
P(sim > e2). (The first version compared against the observed dc, which uses e1; that mixes in any e1/e2 asymmetry.)

  uv run --no-project --with pandas python experiments/2026-10-11_eval_awareness_real_vs_sim/analyze_planted_power.py --scores results/claimb_scores_pilot_<stamp>.json results/claimb_scores_rest_<stamp>.json
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
out = {"post hoc": "positive control for the main reading, added after it", "scores": a.scores, "cells": int(len(runs)),
       "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()}
for col in cols:
    e1, far, e2, sim = (runs[(col, v)].to_numpy() for v in need)
    dcp = bm(c(far, e2)) - bm(c(e1, e2))
    dco = (bm(c(sim, e1)) - bm(c(e2, e1)))
    dco2 = bm(c(sim, e2)) - bm(c(e1, e2))  # the observed contrast against the same real arm as the planted one (e2)
    Rp = bm(np.abs(far - e2)) / bm(np.abs(e1 - e2))
    out[col.replace("score_", "")] = {"dc_planted = P(e1 far > e2) - P(e1 > e2)": round(float(c(far, e2).mean() - c(e1, e2).mean()), 4), "dc_planted 95%": ci(dcp),
                                     "observed dc (sim vs e1, minus e2 vs e1), same cells": round(float(c(sim, e1).mean() - c(e2, e1).mean()), 4), "observed dc 95%": ci(dco),
                                     "observed dc against e2 = P(sim > e2) - P(e1 > e2)": round(float(c(sim, e2).mean() - c(e1, e2).mean()), 4),
                                     "observed dc against e2 95%": ci(dco2),
                                     "dc_planted minus observed dc, both against e2 = P(e1 far > e2) - P(sim > e2), 95%": ci(dcp - dco2),
                                     "dc_planted minus observed dc (mixed reference arms; superseded) 95%": ci(dcp - dco),
                                     "R_planted = mean|e1 far - e2| / mean|e1 - e2|": round(float(np.abs(far - e2).mean() / np.abs(e1 - e2).mean()), 4), "R_planted 95%": ci(Rp)}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = HERE / "results" / f"planted_power_{stamp}.json"
path.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
print(json.dumps(out, indent=1))
print(f"-> {path}")
