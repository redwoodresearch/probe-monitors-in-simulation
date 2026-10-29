"""Stage 5 of the concept probe: real against fixed-sim, as README.md freezes it ($0, stored rows only).

Per readout (content C decides; turn start T beside it). Run score = mean over each run's own scored calls.
- Over cells whose e1, e2 and sim all have scored calls (at least 60, and mean|e2 - e1| above 0):
  - deciding: R = mean|sim - e1| / mean|e2 - e1|. Above 1.2 (whole interval) = "separates fixed-sim from real beyond
    real-vs-real"; within [1/1.2, 1.2] = "no more than real-vs-real"; otherwise "inconclusive";
  - secondary: dc = P(sim > e1) - P(e2 > e1), ties 0.5, against +-0.05 (direction only);
  - diagnostic: a label-swap null for P(sim > e1), one fair coin per main task.
Percentile 95% intervals from 2,000 draws of main tasks with their cells (seed 20261011), shared across readouts;
decisions use unrounded bounds.

  uv run --no-project --with pandas python experiments/2026-10-11_fake_env_concept_probe/analyze_concept_real.py --scores results/concept_scores_real_<stamp>.json --gate-analysis results/concept_gate_analysis_<stamp>.json [--extension results/concept_scores_extend_<stamp>.json]

With --extension (README "Extension", post hoc): stage 5's cells plus the extension's new ones, after the overlap cells
the extension scored again reproduce stage 5's rows (same calls and skips, scores within 0.01 logits); the same
statistics, written to concept_real_analysis_extension_<stamp>.json. The pre-registered reading stays stage 5's alone.
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
B, R_MARGIN, C_MARGIN, MIN_CELLS = 2000, 1.2, 0.05, 60
READOUTS = ("C", "T")
CALL = ["cell", "step", "action"]
TOL = 0.01  # README "Extension": the overlap reproduction tolerance, in logits

ap = argparse.ArgumentParser()
ap.add_argument("--scores", nargs="+", required=True)
ap.add_argument("--gate-analysis", required=True)
ap.add_argument("--extension", default="")
a = ap.parse_args()
gate = json.loads((HERE / a.gate_analysis).read_text())
if not (gate.get("GATE passes") and gate.get("stage 4 may run")):
    raise ValueError("the prerequisites or the gate did not pass: stage 5 is not analyzed")
REF = {"C": gate["stage 3 (C)"]["reference (median held-out pair change)"], "T": gate["stage 3 (T)"]["reference (median held-out pair change)"]}
parts, frames = [], []
for f in a.scores:
    s = json.loads((HERE / f).read_text())
    raw = (ROOT / ".run_cache" / "fake_env_concept_probe" / s["rows"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != s["rows_sha256"]:
        raise ValueError(f"{s['rows']} does not match its recorded sha256")
    parts.append(s)
    frames.append(pd.DataFrame(json.loads(raw)))
if len({p["protocol_sha256"] for p in parts} | {gate["gate protocol_sha256"]}) != 1:  # the same protocol as the gate
    raise ValueError("the score files used different scoring protocols (code, reader, probes, packs, cut or line)")
scored = [c for p in parts for c in p["cells"]]
if len(set(scored)) != len(scored):
    raise ValueError("a cell was scored in two stages")
repro = None
if a.extension:  # README "Extension": the overlap must reproduce stage 5 before any new cell is read
    x = json.loads((HERE / a.extension).read_text())
    raw = (ROOT / ".run_cache" / "fake_env_concept_probe" / x["rows"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != x["rows_sha256"] or x["stage"] != "extend" or x["after"] not in a.scores:
        raise ValueError("the extension rows do not match their sha256, or the file is not an extension of these scores")
    if any(x[k] != parts[0][k] for k in ("probes", "packs", "manifest_sha256", "max_tokens", "seed")):
        raise ValueError("the extension used other probes, packs, cut or seed")
    xf = pd.DataFrame(json.loads(raw))
    over = sorted(set(x["cells"]) & set(scored))
    if len(over) != x["overlap cells"] or len(x["cells"]) - len(over) != x["cells requested"]:
        raise ValueError(f"{len(over)} overlap cells and {len(x['cells']) - len(over)} new ones, not as requested")
    key = CALL + ["arm", "version"]
    old = pd.concat(frames)
    m = old[old["cell"].isin(over)].merge(xf[xf["cell"].isin(over)], on=key, how="outer", suffixes=("", "_x"), indicator=True,
                                         validate="one_to_one")  # a duplicated call row raises
    sk = lambda ps: sorted((d["cell"], d["arm"], d["version"]) for p in ps for d in p["skipped"] if d["cell"] in over)
    gap = float(np.abs(m[["score_C", "score_T"]].to_numpy() - m[["score_C_x", "score_T_x"]].to_numpy()).max())
    repro = {"overlap cells": over, "call rows": int(len(m)), "max abs score difference (logits)": round(gap, 6), "tolerance": TOL}
    if (m["_merge"] != "both").any() or sk(parts) != sk([x]) or not gap <= TOL:
        raise ValueError(f"the overlap does not reproduce stage 5: {repro}: stop and report")
    frames.append(xf[~xf["cell"].isin(over)])
    parts.append(x | {"cells": [c for c in x["cells"] if c not in over], "skipped": [d for d in x["skipped"] if d["cell"] not in over]})
    scored += parts[-1]["cells"]
df = pd.concat(frames)
cols = [f"score_{r}" for r in READOUTS]
if df.duplicated(CALL + ["arm", "version"]).any() or not np.isfinite(df[cols].to_numpy()).all():
    raise ValueError("duplicate call rows or non-finite scores")
skips = pd.DataFrame([x for p in parts for x in p["skipped"]], columns=["cell", "arm", "version", "environment", "reason"])
task_of = df.drop_duplicates("cell").set_index("cell")["main_task"]
df["rv"] = df["arm"] + "/" + df["version"]
runs = df.groupby(["cell", "rv"])[cols].mean().unstack("rv")
trio = ["e1/original", "e2/original", "sim/original"]
main = runs.dropna(subset=[(c, v) for c in cols for v in trio]).index if set(trio) <= set(runs.columns.get_level_values(1)) else pd.Index([])
def boot_setup(cells):
    task = task_of.loc[cells].to_numpy()
    tasks = sorted(set(task))
    t = np.searchsorted(tasks, task)
    W = np.stack([np.bincount(d, minlength=len(tasks)) for d in np.random.default_rng(20261011).integers(0, len(tasks), (B, len(tasks)))])
    return t, tasks, W


def booter(t, tasks, W):
    cnt = W @ np.bincount(t, None, len(tasks))
    return lambda x: (W @ np.bincount(t, x, len(tasks))) / cnt


ci = lambda v: [round(float(q), 4) for q in np.percentile(v, [2.5, 97.5])]
t_m, tasks_m, W_m = boot_setup(main)
bm = booter(t_m, tasks_m, W_m)
coins = np.random.default_rng([20261012]).integers(0, 2, (B, len(tasks_m))).astype(bool)[:, t_m]  # one coin per main task
n_cells = len(scored)
if len(main) < MIN_CELLS:
    raise ValueError(f"{len(main)} complete cells, fewer than {MIN_CELLS}: stop and report")
out = {"reading": ("post hoc extension after an inconclusive pre-registered result (README 'Extension'); the "
                   "pre-registered reading is stage 5's alone" if a.extension else
                   "README 'Planted check' and 'Statistic and reading', committed before running"),
       "scores": a.scores + ([a.extension] if a.extension else []), **({"overlap reproduction": repro} if a.extension else {}),
       "cells scored": n_cells, "cells in the main comparison": int(len(main)),
       "main tasks (main comparison)": len(tasks_m),
       "skipped run-versions": skips.groupby(["arm", "version", "reason"]).size().rename("n").reset_index().to_dict("records"),
       "skipped run-versions by environment": skips.groupby("environment", dropna=False).size().to_dict(),
       "share of cells dropped from the main comparison (all stages)": round(1 - len(main) / n_cells, 4),
       "bootstrap": {"resamples": B, "seed": 20261011, "unit": "main task"}, "margins": {"R": R_MARGIN, "dc": C_MARGIN},
       "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()}
for r in READOUTS:
    col = f"score_{r}"
    e1, e2, sim = (runs.loc[main, (col, v)].to_numpy() for v in trio)
    if np.abs(e2 - e1).mean() <= 0:
        raise ValueError(f"{r}: mean|e2 - e1| is 0: stop and report")
    den = bm(np.abs(e2 - e1))
    if (den <= 0).any():
        raise ValueError(f"{r}: some bootstrap draws have mean|e2 - e1| = 0, so R's interval is undefined: stop and report")
    Rb = bm(np.abs(sim - e1)) / den
    R = float(np.abs(sim - e1).mean() / np.abs(e2 - e1).mean())
    r_lo, r_hi = np.percentile(Rb, [2.5, 97.5])
    verdict = ("separates fixed-sim from real beyond real-vs-real" if r_lo > R_MARGIN
               else "no more than real-vs-real (less than real-vs-real)" if r_hi < 1 / R_MARGIN
               else "no more than real-vs-real" if r_hi <= R_MARGIN else "inconclusive")  # README "Retry (a)": the revised rule
    c = lambda x, y: (x > y) + 0.5 * (x == y)
    cs, cr = c(sim, e1), c(e2, e1)
    dcb = bm(cs) - bm(cr)
    d_lo, d_hi = np.percentile(dcb, [2.5, 97.5])
    null = np.array([np.where(sw, c(e1, sim), cs).mean() for sw in coins])
    out[r] = {"R = mean|sim - e1| / mean|e2 - e1|": round(R, 4), "R 95%": [round(float(r_lo), 4), round(float(r_hi), 4)],
              "mean|sim - e1| (logits)": round(float(np.abs(sim - e1).mean()), 4), "mean|e2 - e1| (logits)": round(float(np.abs(e2 - e1).mean()), 4),
              "verdict" + (" (deciding)" if r == "C" else ""): verdict,
              "dc = P(sim > e1) - P(e2 > e1)": round(float(cs.mean() - cr.mean()), 4), "dc 95%": [round(float(d_lo), 4), round(float(d_hi), 4)],
              "direction": ("sim higher beyond real-vs-real" if d_lo > C_MARGIN else "sim lower beyond real-vs-real" if d_hi < -C_MARGIN
                            else "direction not established"),
              "P(sim > e1)": round(float(cs.mean()), 4), "P(e2 > e1)": round(float(cr.mean()), 4),
              "P(sim > e1) task-swap null 95% band (diagnostic)": ci(null),
              "shift sim - e1 (logits)": round(float((sim - e1).mean()), 4), "shift sim - e1 95%": ci(bm(sim - e1)),
              "shift sim - e1 (fraction of reference)": round(float((sim - e1).mean() / REF[r]), 4),
              "shift e2 - e1 (logits)": round(float((e2 - e1).mean()), 4), "shift e2 - e1 95%": ci(bm(e2 - e1)),
              "shift e2 - e1 (fraction of reference)": round(float((e2 - e1).mean() / REF[r]), 4)}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = HERE / "results" / f"concept_real_analysis_{'extension_' if a.extension else ''}{stamp}.json"
path.parent.mkdir(exist_ok=True)
path.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
print(json.dumps(out, indent=1))
print(f"-> {path}")
