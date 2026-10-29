"""Aswin's planted-check rule (relayed 2026-10-10, 10:51 PT) applied to the inserted-line control, far and near ($0, saved
outputs only). It is reported beside the pre-registered bar (b), which stays the decision.

The rule: a plant counts as detected when the estimate shifts by about the planted amount AND its 95% range excludes
the unplanted estimate (not zero); the gap-widening clause has no gap to apply to here. Per placement and readout:
- the estimate is the mean run score over the paired runs; unplanted = originals, planted = inserted copies;
- 95% ranges from 2,000 bootstrap resamples of main tasks (seed 20261010), resampling whole tasks;
- the planted amount has no ground truth on LinuxArena. The reference is the synthetic held-out hint effect at the same
  readout (pairs' implied-test minus ordinary score, median). "About" = a shift between 0.5x and 2x of it, set before
  this script was first run.

  uv run --no-project --with pandas python experiments/2026-10-10_eval_awareness_probe/analyze_plant_rule.py --far results/la_scores_2026-10-10_024336.json --near results/la_scores_near_2026-10-10_103942.json --fit results/fit_2026-10-10_022437.json
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
CACHE = ROOT / ".run_cache" / "eval_awareness_probe"
B, SEED, ABOUT = 2000, 20261010, (0.5, 2.0)

ap = argparse.ArgumentParser()
ap.add_argument("--far", required=True)
ap.add_argument("--near", required=True)
ap.add_argument("--fit", required=True)
a = ap.parse_args()
fit = json.loads((HERE / a.fit).read_text())
out = {"far": a.far, "near": a.near, "fit": a.fit, "bootstrap": {"resamples": B, "seed": SEED, "unit": "main task"},
       "about (ratio of shift to reference)": list(ABOUT),
       "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()}
frames = {}
for name, path in (("far", a.far), ("near", a.near)):
    s = json.loads((HERE / path).read_text())
    raw = (CACHE / s["rows"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != s["rows_sha256"] or (s["fit_id"], s["probes_sha256"]) != (fit["fit_id"], fit["probes_sha256"]):
        raise ValueError(f"{path}: rows hash or probes do not match")
    df = pd.DataFrame(json.loads(raw))
    key = ["key", "step", "action"]
    sets = [df[df["version"] == v].set_index(key).index for v in ("original", "inserted")]
    if any(not ix.is_unique for ix in sets) or set(sets[0]) != set(sets[1]) or not np.isfinite(df[["score_per_token", "score_last_token"]].to_numpy()).all():
        raise ValueError(f"{path}: scored calls differ between versions, repeat, or have non-finite scores")
    frames[name] = df
for r in ("per_token", "last_token"):
    col = f"score_{r}"
    held = pd.DataFrame(fit["heldout_scores_at_chosen_layer"][r])
    held["pair"] = held["id"].str.rsplit("_", n=1).str[0]
    pairs = held.pivot(index="pair", columns="label", values="score")
    ref = float((pairs[1] - pairs[0]).median())
    res = {"reference planted amount (synthetic held-out median pair change)": round(ref, 3)}
    for name, df in frames.items():
        runs = df.groupby(["key", "version"])[col].mean().unstack("version")
        missing = sorted(set(json.loads((HERE / (a.far if name == "far" else a.near)).read_text())["sampled_runs"]) - set(runs.dropna().index))
        runs = runs.dropna()
        task = runs.index.to_series().str.split("__").str[1].str.split(":").str[0].to_numpy()  # main task of each run
        tasks = np.unique(task)
        idx = [np.flatnonzero(task == t) for t in tasks]
        o, i = runs["original"].to_numpy(), runs["inserted"].to_numpy()
        rng = np.random.default_rng(SEED)
        boot = []
        for _ in range(B):
            pick = np.concatenate([idx[k] for k in rng.integers(0, len(tasks), len(tasks))])
            boot.append((o[pick].mean(), i[pick].mean()))
        boot = np.array(boot)
        ci = lambda x: [float(q) for q in np.percentile(x, [2.5, 97.5])]
        rnd = lambda v: [round(q, 3) for q in v]
        unplanted, planted = float(o.mean()), float(i.mean())
        shift = planted - unplanted
        p_ci = ci(boot[:, 1])
        excludes = p_ci[0] > unplanted or p_ci[1] < unplanted  # full precision; rounded only in the output
        about = ABOUT[0] * ref <= shift <= ABOUT[1] * ref
        res[name] = {"runs": len(runs), "runs excluded (no scored call in a version)": missing, "main tasks": len(tasks), "unplanted estimate (mean original run score)": round(unplanted, 3),
                     "planted estimate (mean inserted run score)": round(planted, 3), "planted 95% range": rnd(p_ci),
                     "shift": round(shift, 3), "shift 95% range": rnd(ci(boot[:, 1] - boot[:, 0])),
                     "shift / reference": round(shift / ref, 4), "shift about the planted amount": bool(about),
                     "planted range excludes the unplanted estimate": bool(excludes), "detected (rule)": bool(about and excludes)}
    out[r] = res
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = HERE / "results" / f"plant_rule_near_far_{stamp}.json"
path.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
print(json.dumps(out, indent=1))
print(f"-> {path}")
