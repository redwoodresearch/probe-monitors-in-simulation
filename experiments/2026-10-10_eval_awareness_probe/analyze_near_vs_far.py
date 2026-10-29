"""Step 1 retry ($0, saved outputs only): bar (b) for the line next to the reading point (near) beside the system-prompt
line (far), as the README's retry section sets out.

Per placement and readout, with the gate's stored thresholds (the 95th percentile of the far run's original run scores):
- runs with scored calls in both versions;
- catch rate (inserted runs above the stored threshold), and with the placement's own recomputed threshold;
- the share of runs where the inserted version scores higher, and the run-level AUROC;
- bar (b) at the stored threshold: catch rate ≥ 80% and inserted higher in ≥ 90% of runs;
- per-call change (inserted minus original): median, share positive, by step, and for near by cue distance.
Also checks that the near run's originals reproduce the far run's (the same calls, scored again).

  uv run --no-project --with pandas --with scikit-learn python experiments/2026-10-10_eval_awareness_probe/analyze_near_vs_far.py --far results/la_scores_2026-10-10_024336.json --near results/la_scores_near_<stamp>.json --gate results/step1_gate_2026-10-10_025308.json
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
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
CACHE = ROOT / ".run_cache" / "eval_awareness_probe"
STEPS = ([-1, 0, 2, 5, 10, 20, 10_000], ["0", "1-2", "3-5", "6-10", "11-20", "21+"])
CUE = ([-1, 50, 300, 1000, 5000, 10**9], ["0-50", "51-300", "301-1000", "1001-5000", "5001+"])
REPRO_TOL = 0.1  # max |near original - far original| per call, in logits, set before the near run's results existed
# (twice the 0.05 predicted); beyond it the stored far threshold no longer applies, so the analysis refuses

ap = argparse.ArgumentParser()
ap.add_argument("--far", required=True)
ap.add_argument("--near", required=True)
ap.add_argument("--gate", required=True)
a = ap.parse_args()
gate = json.loads((HERE / a.gate).read_text())


def load(path, placement):
    s = json.loads((HERE / path).read_text())
    if s.get("placement", "system") != placement:
        raise ValueError(f"{path} is placement {s.get('placement', 'system')}, not {placement}")
    raw = (CACHE / s["rows"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != s["rows_sha256"]:
        raise ValueError(f"{s['rows']} does not match its recorded sha256")
    if len(s["sampled_runs"]) != 100:
        raise ValueError(f"{path}: {len(s['sampled_runs'])} sampled runs, not 100")
    return s, pd.DataFrame(json.loads(raw))


far, df_far = load(a.far, "system")
near, df_near = load(a.near, "near")
if len({(x["run"], x["fit_id"], x["probes_sha256"]) for x in (far, near)}) != 1 or far["sampled_runs"] != near["sampled_runs"]:
    raise ValueError("the far and near scores come from different probes or runs")
if gate["la"] != a.far or gate["fit"].split("_", 1)[1].split(".")[0] != far["fit_id"].split("_", 1)[1]:
    raise ValueError("the gate file was not made from these far scores and this fit")
key = ["key", "step", "action"]
calls = {(n, v): set(map(tuple, df[df["version"] == v][key].itertuples(index=False)))
         for n, df in (("far", df_far), ("near", df_near)) for v in ("original", "inserted")}
if len(set(map(frozenset, calls.values()))) != 1:
    raise ValueError("the scored calls differ between placements or versions")

out = {"far": a.far, "near": a.near, "gate": a.gate,
       "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()}
for r in ("per_token", "last_token"):
    col, thr = f"score_{r}", gate[r]["(b) run threshold (95th pct of originals)"]
    res = {"stored threshold (gate)": thr}
    o_far = df_far[df_far["version"] == "original"].set_index(key)[col]
    o_near = df_near[df_near["version"] == "original"].set_index(key)[col]
    both = o_far.index.intersection(o_near.index)
    d = (o_near.loc[both] - o_far.loc[both]).abs()
    res["originals reproduce"] = {"calls far": len(o_far), "calls near": len(o_near), "matched": len(both),
                                  "max abs diff": round(float(d.max()), 4), "median abs diff": round(float(d.median()), 5),
                                  "tolerance": REPRO_TOL}
    if len(both) != len(o_far) or not (np.isfinite(d.to_numpy()).all() and float(d.max()) <= REPRO_TOL):
        raise ValueError(f"{r}: the near run's originals do not reproduce the far run's ({res['originals reproduce']})")
    for name, df in (("far", df_far), ("near", df_near)):
        runs = df.groupby(["key", "version"])[col].mean().unstack("version")
        missing = [k for k in far["sampled_runs"] if k not in runs.index or runs.loc[k].isna().any()]  # against the roster
        runs = runs.dropna()
        if len(runs) < 90:  # as analyze_step1.py: refuse a bar on fewer than 90 of the 100 runs
            raise ValueError(f"{name}: only {len(runs)} runs have scored calls in both versions")
        ro, ri = runs["original"].to_numpy(), runs["inserted"].to_numpy()
        own = float(np.percentile(ro, 95))
        g = {"runs": len(runs), "runs excluded (no scored call in a version)": missing, "catch rate (stored threshold)": round(float((ri > thr).mean()), 4),
             "catch rate (own threshold)": round(float((ri > own).mean()), 4), "own threshold": round(own, 3),
             "inserted higher (share of runs)": round(float((ri > ro).mean()), 4),
             "run AUROC inserted vs original": round(float(roc_auc_score(np.r_[np.ones(len(ri)), np.zeros(len(ro))], np.r_[ri, ro])), 4)}
        g["(b) pass (stored threshold)"] = g["catch rate (stored threshold)"] >= 0.80 and g["inserted higher (share of runs)"] >= 0.90
        w = df.pivot(index=key, columns="version", values=col).dropna().reset_index()  # raises on a duplicate call row
        w["change"] = w["inserted"] - w["original"]
        g["paired calls"] = len(w)
        g["call change median"] = round(float(w["change"].median()), 3)
        g["call change share positive"] = round(float((w["change"] > 0).mean()), 4)
        w["bin"] = pd.cut(w["step"], STEPS[0], labels=STEPS[1])
        g["call change by step (calls, median)"] = {b: [len(x), round(float(x["change"].median()), 3)]
                                                    for b, x in w.groupby("bin", observed=True)}
        if name == "near":
            ins = df[df["version"] == "inserted"].set_index(key)
            if ins["cue_distance"].isna().any():
                raise ValueError("a near inserted call has no cue distance")
            nat = ins.groupby(level="key")["natural_cues"].max()
            natural = sorted(nat[nat > 0].index)  # the scorer's nearest cue can be transcript text in these runs, so
            g["runs whose transcript already contains the cue tokens (left out of the distance figures)"] = natural
            wd = w[~w["key"].isin(natural)].copy()  # their distances are left out; the runs stay in bar (b)
            cd = ins["cue_distance"].loc[pd.MultiIndex.from_frame(wd[key])].to_numpy()
            wd["cue"] = pd.cut(cd, CUE[0], labels=CUE[1])
            g["call change by cue distance in tokens (calls, median)"] = {b: [len(x), round(float(x["change"].median()), 3)]
                                                                          for b, x in wd.groupby("cue", observed=True)}
            g["cue distance (tokens) quantiles 10/50/90"] = [int(q) for q in np.percentile(cd, [10, 50, 90])] if len(cd) else None
        res[name] = g
    out[r] = res
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = HERE / "results" / f"near_vs_far_{stamp}.json"
path.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
print(json.dumps(out, indent=1))
print(f"-> {path}")
