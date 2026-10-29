"""Distance check, as the README's "Distance check" sets out ($0, stored score rows only).

Per placement (far, near, checker) and readout: each scored call's paired shift (inserted or plant minus original or
control, same run, step and action) against x = log2(distance + 1), distance in tokens to the call's span start (near
and checker: tokens from the nearest inserted line; far: the call's position in the original render). Reports quintile
means, pooled and within-run slopes per doubling, the top-over-bottom quintile ratio, and a planted-slope sensitivity
check, all as fractions of the synthetic reference, with 95% intervals from 2,000 main-task bootstrap draws. Quintiles
split tied distances by call order (run key, step, action). Verdicts compare unrounded interval ends with 0.

  uv run --no-project --with pandas python experiments/2026-10-10_eval_awareness_probe/analyze_distance.py --far results/la_scores_2026-10-10_024336.json --near results/la_scores_near_2026-10-10_103942.json --checker results/la_scores_checker_2026-10-10_155537.json --draw results/checker_pilot_draw.json
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
B, SEED, Q = 2000, 20261010, 5
REF = {"far": {"per_token": 8.159, "last_token": 3.994}, "near": {"per_token": 8.159, "last_token": 3.994}}
CALL = ["key", "step", "action"]

ap = argparse.ArgumentParser()
for a_ in ("far", "near", "checker", "draw"):
    ap.add_argument(f"--{a_}", required=True)
a = ap.parse_args()
draw = json.loads((HERE / a.draw).read_text())
REF["checker"] = {r: draw["reference: synthetic checker_header median pair change"][r]["median change"] for r in ("per_token", "last_token")}


def rows_of(path):
    s = json.loads((HERE / path).read_text())
    raw = (ROOT / ".run_cache" / "eval_awareness_probe" / s["rows"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != s["rows_sha256"]:
        raise ValueError(f"{s['rows']} does not match its recorded sha256")
    df = pd.DataFrame(json.loads(raw))
    if not np.isfinite(df[["score_per_token", "score_last_token"]].to_numpy()).all():
        raise ValueError(f"{path}: non-finite scores")
    return s, df


def paired(df, lo, hi):
    df = df[df["version"].isin([lo, hi])]
    if df.duplicated(CALL + ["version"]).any():
        raise ValueError("duplicate (call, version) rows")
    w = df.pivot(index=CALL, columns="version", values=["score_per_token", "score_last_token"]).sort_index()
    UNPAIRED[f"{lo}/{hi}"] = int(w.isna().any(axis=1).sum())
    w = w.dropna()
    return pd.DataFrame({r: w[(f"score_{r}", hi)] - w[(f"score_{r}", lo)] for r in ("per_token", "last_token")})


UNPAIRED = {}


sf, far = rows_of(a.far)
sn, near = rows_of(a.near)
sc, chk = rows_of(a.checker)
if len({x["probes_sha256"] for x in (sf, sn, sc)}) != 1:
    raise ValueError("the placements were scored with different probes")
pos = near[near["version"] == "original"].set_index(CALL)["span_start"]
cue = near[near["version"] == "inserted"].set_index(CALL)["cue_distance"]
if not (pos.index.is_unique and cue.index.is_unique) or cue.isna().any():
    raise ValueError("near rows: duplicate calls or a missing cue distance")
d_far, d_near, d_chk = paired(far, "original", "inserted"), paired(near, "original", "inserted"), paired(chk, "control", "plant")
if set(d_far.index) != set(pos.index):
    raise ValueError("far's paired calls are not near's original calls")
matched = d_chk.index.isin(cue.index)
first_span = pos.groupby(level="key").min()
out = {"reading": "README 'Distance check', committed before computing (5eb021fa)",
       "files": {"far": a.far, "near": a.near, "checker": a.checker, "draw": a.draw},
       "bootstrap": {"resamples": B, "seed": SEED, "unit": "main task"}, "references": REF,
       "far offset bound: first span start per run (min/median/max)": [int(first_span.min()), int(first_span.median()), int(first_span.max())],
       "calls in one version only (dropped)": UNPAIRED, "checker calls without a near match (dropped)": int((~matched).sum()),
       "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()}


def stats(y, dist, x, run, task, ref, rng):
    """Point estimates and main-task bootstrap draws, from per-task sums so that each draw is one matrix product."""
    q = pd.qcut(pd.Series(x).rank(method="first"), Q, labels=False).to_numpy()
    xd = x - pd.Series(x).groupby(run).transform("mean").to_numpy()
    yd = y - pd.Series(y).groupby(run).transform("mean").to_numpy()
    tasks, t = np.unique(task, return_inverse=True)
    cols = [np.ones_like(y), x, y, x * y, x * x, xd * yd, xd * xd] + [(q == k) * 1.0 for k in range(Q)] + [(q == k) * y for k in range(Q)]
    S = np.stack([np.bincount(t, c, len(tasks)) for c in cols], 1)  # tasks x sums
    W = np.vstack([np.ones(len(tasks)), rng.multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), B)])
    n, sx, sy, sxy, sxx, wxy, wxx = (W @ S[:, :7]).T
    qn, qy = W @ S[:, 7:7 + Q], W @ S[:, 7 + Q:]
    est = {"pooled": (sxy - sx * sy / n) / (sxx - sx * sx / n), "within": wxy / wxx}
    est |= {f"q{k + 1}": qy[:, k] / qn[:, k] for k in range(Q)}
    est["ratio"] = est[f"q{Q}"] / est["q1"]
    raw = {k: np.percentile(v[1:], [2.5, 97.5]) for k, v in est.items()}
    ci = lambda v: [round(float(z / ref), 5) for z in np.percentile(v[1:], [2.5, 97.5])]
    res = {"within-run slope per doubling": round(float(est["within"][0] / ref), 5), "within-run 95%": ci(est["within"]),
           "within-run slope (logits)": round(float(est["within"][0]), 5),
           "pooled slope per doubling": round(float(est["pooled"][0] / ref), 5), "pooled 95%": ci(est["pooled"]),
           "quintiles": [{"distance range": [int(dist[q == k].min()), int(dist[q == k].max())],
                          "calls": int((q == k).sum()), "mean": round(float(est[f"q{k + 1}"][0] / ref), 5),
                          "95%": ci(est[f"q{k + 1}"])} for k in range(Q)]}
    q1_lo = np.percentile(est["q1"][1:], 2.5)
    res["top / bottom quintile"] = ({"ratio": round(float(est["ratio"][0]), 4),
                                     "95%": [round(float(z), 4) for z in np.percentile(est["ratio"][1:], [2.5, 97.5])]}
                                    if q1_lo > 0 else "not reported: the bottom quintile's interval is not above 0")
    return res, est["within"][0], xd, raw["within"]


for name, d, dist in (("far", d_far, pos.loc[d_far.index]), ("near", d_near, cue.loc[d_near.index]),
                      ("checker", d_chk[matched], cue.loc[d_chk.index[matched]])):
    keys = d.index.get_level_values("key").to_numpy()
    task = np.array([k.split("__")[1].split(":")[0] for k in keys])
    dist = dist.to_numpy()
    x = np.log2(dist.astype(float) + 1)
    res = {"calls": len(d), "runs": len(set(keys)), "main tasks": len(set(task)),
           "distance (min/median/max)": [int(dist.min()), int(np.median(dist)), int(dist.max())]}
    for r in ("per_token", "last_token"):
        y, ref = d[r].to_numpy(), REF[name][r]
        obs, b_w, xd, (lo, hi) = stats(y, dist, x, keys, task, ref, np.random.default_rng(SEED))
        plant = -abs(y.mean()) / (np.percentile(x, 90) - np.percentile(x, 10))
        pl, _, _, (_, pl_hi) = stats(y - b_w * xd + plant * xd, dist, x, keys, task, ref, np.random.default_rng(SEED))
        caught = pl_hi < 0
        obs["mean shift"] = round(float(y.mean() / ref), 5)
        obs["planted check"] = {"planted slope per doubling": round(float(plant / ref), 5),
                                "estimate": pl["within-run slope per doubling"], "95%": pl["within-run 95%"], "caught": bool(caught)}
        obs["reading"] = ("decays" if hi < 0 else "rises" if lo > 0 else
                          "no detectable trend" if caught else "underpowered for a decay of that size")
        res[r] = obs
    out[name] = res
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = HERE / "results" / f"distance_{stamp}.json"
path.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
print(json.dumps(out, indent=1))
print(f"-> {path}")
