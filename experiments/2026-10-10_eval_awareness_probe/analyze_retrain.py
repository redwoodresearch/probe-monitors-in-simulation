"""Analyze retrain A and B as the README's "Retrain A and B" sets out ($0, stored rows and fit files only).

1. Reproduction gate: the old probes' scores from this pass must give the same scored calls as the stored far, near and
   checker rows, and match each within 0.1; otherwise nothing is analyzed.
2. References per readout, from its fit's held-out scores: the median pair change over the 120 held-out pairs (far,
   near) and over the 30 checker_header pairs (checker). A reference whose bootstrap 2.5th percentile is not above 0
   stops the analysis.
3. Per placement (far, near, checker) and readout: run score = mean over the run's scored calls; paired shift =
   inserted (or plant) minus original (or control); fraction = mean shift / reference. Each of 2,000 draws (seed
   20261010) resamples LinuxArena main tasks and, independently, synthetic held-out tasks. A and B are compared with
   the old per-token readout (A's last token with the old last token) by the paired difference of fractions; one
   "transfers better" at a placement when that interval lies above 0. Also: the paired share with its Wilson
   interval, and the catch rate at the 95th percentile of the 99 original run scores.

  uv run --no-project --with pandas python experiments/2026-10-10_eval_awareness_probe/analyze_retrain.py --scores results/retrain_scores_<stamp>.json
"""

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
B, SEED, Z, TOL = 2000, 20261010, 1.959964, 0.1
CALL = ["key", "step", "action"]
SETS = ".run_cache/eval_awareness_probe/sets_2026-10-10_004845.jsonl"
PLACE = {"far": ("original", "far"), "near": ("original", "near"), "checker": ("control", "plant")}
VS = {"A_per_token": "old_per_token", "B": "old_per_token", "A_last_token": "old_last_token"}

ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True)
a = ap.parse_args()
sc = json.loads((HERE / a.scores).read_text())
frozen_raw = (HERE / sc["frozen"]).read_bytes()
if hashlib.sha256(frozen_raw).hexdigest() != sc["frozen_sha256"]:
    raise ValueError("the frozen inputs file changed after scoring")
frozen = json.loads(frozen_raw)


def rows_of(name, sha):
    raw = (ROOT / ".run_cache" / "eval_awareness_probe" / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError(f"{name} does not match its recorded sha256")
    return pd.DataFrame(json.loads(raw))


df = rows_of(sc["rows"], sc["rows_sha256"])
score_cols = [c for c in df.columns if c.startswith("score_")]
if df.duplicated(CALL + ["version"]).any() or not np.isfinite(df[score_cols].to_numpy()).all():
    raise ValueError("duplicate call rows or non-finite scores")
gate = {}
for place, f in frozen["stored rows"].items():
    s = json.loads((HERE / "results" / f).read_text())
    old = rows_of(s["rows"], s["rows_sha256"])
    old = old[old["key"].isin(frozen["runs"])]
    lo, hi = PLACE[place]
    for v_old, v_new in ((lo, lo), ({"far": "inserted", "near": "inserted"}.get(place, hi), hi)):
        o = old[old["version"] == v_old].set_index(CALL)
        n = df[df["version"] == v_new].set_index(CALL)
        if set(o.index) != set(n.index):
            raise ValueError(f"{place}/{v_new}: the scored calls differ from the stored rows")
        diffs = {r: float((n.loc[o.index, f"score_old_{r}"] - o[f"score_{r}"]).abs().max()) for r in ("per_token", "last_token")}
        gate[f"{place}/{v_new}"] = {"calls": len(o), "max abs diff": {r: round(d, 5) for r, d in diffs.items()}}
        if max(diffs.values()) > TOL:
            raise ValueError(f"{place}/{v_new}: old scores do not reproduce the stored rows ({diffs})")

sets_raw = (ROOT / SETS).read_bytes()  # git-ignored: check it is the synthetic set every fit and the build recorded
if hashlib.sha256(sets_raw).hexdigest() != json.loads((HERE / frozen["build"]).read_text())["synthetic sets_sha256"]:
    raise ValueError(f"{SETS} does not match the build's synthetic sets_sha256")
task_of = {json.loads(x)["id"]: json.loads(x)["task"] for x in sets_raw.decode().splitlines()}
fits = {"old": json.loads((HERE / "results" / "fit_2026-10-10_022437.json").read_text()),
        "A": json.loads((HERE / sc["fit_a"]).read_text()), "B": json.loads((HERE / sc["fit_b"]).read_text())}
if fits["old"]["sets_sha256"] != hashlib.sha256(sets_raw).hexdigest():
    raise ValueError("the old fit was not trained on this synthetic set")
for name, (fit, _) in {"old_per_token": ("old", 0), "A_per_token": ("A", 0), "B": ("B", 0)}.items():  # the fit files are the scored probes
    f, pr = fits[fit], sc["probes"][name]
    if (f"{f['run']}/fits/{f['fit_id']}", f["probes_sha256"]) != (pr["dir"], pr["sha256"]) or (fit != "old" and f["sets_sha256"] != frozen["set_sha256"]):
        raise ValueError(f"{fit}: the fit file does not match the probes that were scored")
src = {"old_per_token": ("old", "per_token"), "old_last_token": ("old", "last_token"), "A_per_token": ("A", "per_token"),
       "A_last_token": ("A", "last_token"), "B": ("B", "per_token")}
la_runs = sorted(frozen["runs"])
DRAWS = {}


def draws(unit, n):  # one fixed set of task draws per unit, shared by every readout so that differences are paired
    if unit not in DRAWS:
        DRAWS[unit] = np.random.default_rng([SEED, len(DRAWS)]).integers(0, n, size=(B, n))
    return DRAWS[unit]


def pair_changes(readout, checker):
    fit, r = src[readout]
    s = {x["id"]: x for x in fits[fit]["heldout_scores_at_chosen_layer"][r]}
    pids = sorted({i.rsplit("_", 1)[0] for i, x in s.items() if not checker or x["mechanism"] == "checker_header"})
    ch = np.array([s[f"{p}_1"]["score"] - s[f"{p}_0"]["score"] for p in pids])
    tasks = sorted({task_of[f"{p}_1"] for p in pids})
    t = np.searchsorted(tasks, [task_of[f"{p}_1"] for p in pids])
    groups = [ch[t == k] for k in range(len(tasks))]
    boot = np.array([np.median(np.concatenate([groups[k] for k in d])) for d in draws(f"synthetic checker={checker}", len(tasks))])
    return float(np.median(ch)), boot


def wilson(k, n):
    p, c = k / n, 1 + Z * Z / n
    mid, h = (p + Z * Z / (2 * n)) / c, Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / c
    return [round(mid - h, 4), round(mid + h, 4)]


refs = {}
for readout in src:
    for checker in (False, True):
        ref, boot = pair_changes(readout, checker)
        lo = float(np.percentile(boot, 2.5))
        refs[(readout, checker)] = (ref, boot)
        if lo <= 0:
            raise ValueError(f"{readout} ({'checker_header' if checker else 'all held-out'}): reference interval reaches 0 ({lo:.3f})")
orig = df[df["version"] == "original"].groupby("key")[score_cols].mean()
out = {"reading": "README 'Retrain A and B', committed before running", "scores": a.scores, "reproduction gate": gate,
       "references": {f"{r} ({'checker_header' if c else 'held-out'})": {"median": round(v[0], 4), "95%": [round(float(q), 4) for q in np.percentile(v[1], [2.5, 97.5])]}
                      for (r, c), v in refs.items()},
       "fits": {k: {"run": v["run"], "fit_id": v["fit_id"], "layers": {r: v["summary"][r]["layer"] for r in ("per_token", "last_token")},
                    "heldout_auroc": {r: round(v["summary"][r]["heldout_auroc"], 4) for r in ("per_token", "last_token")},
                    "label-shuffle null p95, readout fixed": {r: round(v["summary"][r]["null_heldout_p95"], 4) for r in ("per_token", "last_token")},
                    "label-shuffle null p95, whole pipeline": round(v["null_heldout_p95 (selected pipeline)"], 4)} for k, v in fits.items()},
       "bootstrap": {"resamples": B, "seed": SEED, "units": "LinuxArena main tasks and synthetic held-out tasks, independently"},
       "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()}
for place, (lo_v, hi_v) in PLACE.items():
    keys = la_runs if place != "checker" else sorted(frozen["checker runs"])
    runs = df[df["version"].isin([lo_v, hi_v]) & df["key"].isin(keys)].groupby(["key", "version"])[score_cols].mean().unstack("version")
    if len(runs.dropna()) != len(keys):
        raise ValueError(f"{place}: a run has no scored call in a version")
    tk = [k.split("__")[1].split(":")[0] for k in runs.index]
    tasks = sorted(set(tk))
    t = np.searchsorted(tasks, tk)
    W = np.stack([np.bincount(dr, minlength=len(tasks)) for dr in draws(f"linuxarena {place}", len(tasks))])
    res, frac_b = {}, {}
    for readout in src:
        col = f"score_{readout}"
        d = (runs[(col, hi_v)] - runs[(col, lo_v)]).to_numpy()
        ref, rboot = refs[(readout, place == "checker")]
        mean_b = (W @ np.bincount(t, d, len(tasks))) / (W @ np.bincount(t, None, len(tasks)))
        frac_b[readout] = mean_b / rboot
        thr = float(np.percentile(orig[col], 95))
        k = int((d > 0).sum())
        res[readout] = {"runs": len(d), "paired mean shift": round(float(d.mean()), 4),
                        "shift 95%": [round(float(q), 4) for q in np.percentile(mean_b, [2.5, 97.5])], "reference": round(ref, 4),
                        "fraction of reference": round(float(d.mean() / ref), 4),
                        "fraction 95%": [round(float(q), 4) for q in np.percentile(frac_b[readout], [2.5, 97.5])],
                        "share higher (k/n)": f"{k}/{len(d)}", "share Wilson 95%": wilson(k, len(d)),
                        "catch rate at the originals' 95th percentile": round(float((runs[(col, hi_v)] > thr).mean()), 4)}
    for new, old in VS.items():
        diff = frac_b[new] - frac_b[old]
        lo, hi = (float(q) for q in np.percentile(diff, [2.5, 97.5]))
        res[new][f"minus {old} (fraction)"] = {"estimate": round(res[new]["fraction of reference"] - res[old]["fraction of reference"], 4),
                                              "95%": [round(lo, 4), round(hi, 4)], "transfers better": bool(lo > 0)}
    out[place] = res
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = HERE / "results" / f"retrain_analysis_{stamp}.json"
path.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
print(json.dumps(out, indent=1))
print(f"-> {path}")
