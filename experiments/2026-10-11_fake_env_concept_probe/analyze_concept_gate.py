"""Stage 3 prerequisites and the stage 4 gate, as README.md freezes them ($0, stored files only).

Prerequisites (from the fit files): the content readout's held-out AUROC exceeds its whole-pipeline null p95, and its
reference (median held-out pair change) has a task-bootstrap 2.5th percentile above 0.
Gate (from the gate scores): matched calls per run, the (step, action) calls scored in all five versions; run score =
mean over them. At the far placement, on the content readout, the mean over runs of plant minus neutral must have a
95% percentile interval above 0 (2,000 draws resampling the eligible runs, seed 20261011), with at least 15 of the 20
runs having a matched call. Turn start, near, and neutral minus original are reported beside it. References in logits.

  uv run --no-project --with pandas python experiments/2026-10-11_fake_env_concept_probe/analyze_concept_gate.py --fit-c ../2026-10-10_eval_awareness_probe/results/fit_<C>.json --fit-t ../2026-10-10_eval_awareness_probe/results/fit_<T>.json --build results/concept_sets_v2_<stamp>.json --null-unit task [--gate results/concept_scores_gate_<stamp>.json]
(the retry's settings; the first run used the default pair null and the v1 build)
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
B, SEED, MIN_RUNS = 2000, 20261011, 15
VERSIONS = ["original", "far_plant", "near_plant", "far_neutral", "near_neutral"]

ap = argparse.ArgumentParser()
ap.add_argument("--fit-c", required=True)
ap.add_argument("--fit-t", required=True)
ap.add_argument("--gate")
ap.add_argument("--build", default="")  # the build result the fits must come from (default: the newest)
ap.add_argument("--null-unit", default="pair")  # the retry requires "task" (README "Retry (a)")
a = ap.parse_args()
fits = {"C": json.loads((HERE / a.fit_c).read_text()), "T": json.loads((HERE / a.fit_t).read_text())}
for k, fit in fits.items():  # each fit must be its own readout's run
    if not fit["run"].startswith(f"concept{k}_"):
        raise ValueError(f"--fit-{k.lower()} is not a concept{k} fit")
builds = [json.loads(p.read_text()) for p in sorted((HERE / "results").glob("concept_sets_*.json"))]
match = [b for b in builds if b["sha256"] == fits["C"]["sets_sha256"]]  # the build the fits came from, not the newest
build = json.loads((HERE / a.build).read_text()) if a.build else (match[0] if match else None)  # reruns share the hash
if build is None:
    raise ValueError("no build result matches the fits' sets_sha256; pass --build")
for k, fit in fits.items():  # README "Retry (a)": the fits' settings must match the frozen ones
    if fit.get("null_unit", "pair") != a.null_unit:
        raise ValueError(f"{k}: the fit's null unit is {fit.get('null_unit', 'pair')}, not {a.null_unit}")
    if (fit["C"], fit["shuffles"], fit["seed"]) != (0.1, 50, 20261010):
        raise ValueError(f"{k}: fit settings {(fit['C'], fit['shuffles'], fit['seed'])} differ from the frozen (0.1, 50, 20261010)")
if fits["C"]["sets_sha256"] != fits["T"]["sets_sha256"]:
    raise ValueError("the content and turn-start fits were trained on different sets")
sets_raw = (ROOT / ".run_cache" / "fake_env_concept_probe" / build["file"]).read_bytes()
if hashlib.sha256(sets_raw).hexdigest() != build["sha256"] or any(f["sets_sha256"] != build["sha256"] for f in fits.values()):
    raise ValueError("the fits or the local set file are not the recorded build")
task_of = {json.loads(x)["id"]: json.loads(x)["task"] for x in sets_raw.decode().splitlines()}
out = {"reading": "README stages 3 and 4, committed before running", "fits": {k: a.fit_c if k == "C" else a.fit_t for k in fits},
       "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()}
ref = {}
for k, fit in fits.items():
    s = {x["id"]: x for x in fit["heldout_scores_at_chosen_layer"]["per_token"]}
    pids = sorted({i.rsplit("_", 1)[0] for i in s})
    ch = np.array([s[f"{p}_1"]["score"] - s[f"{p}_0"]["score"] for p in pids])
    tasks = sorted({task_of[f"{p}_1"] for p in pids})
    groups = [ch[np.searchsorted(tasks, [task_of[f"{p}_1"] for p in pids]) == t] for t in range(len(tasks))]
    boot = np.array([np.median(np.concatenate([groups[t] for t in d])) for d in np.random.default_rng(SEED).integers(0, len(tasks), (B, len(tasks)))])
    ref[k] = float(np.median(ch))
    summ = fit["summary"]["per_token"]
    out[f"stage 3 ({k})"] = {"layer": summ["layer"], "held-out AUROC": round(summ["heldout_auroc"], 4),
                             "whole-pipeline null p95": round(fit["null_heldout_p95 (selected pipeline)"], 4),
                             "reference (median held-out pair change)": round(ref[k], 4),
                             "reference 95%": [round(float(q), 4) for q in np.percentile(boot, [2.5, 97.5])],
                             "AUROC above the null": bool(summ["heldout_auroc"] > fit["null_heldout_p95 (selected pipeline)"]),
                             "reference interval above 0": bool(np.percentile(boot, 2.5) > 0)}
out["stage 4 may run"] = bool(out["stage 3 (C)"]["AUROC above the null"] and out["stage 3 (C)"]["reference interval above 0"])
if a.gate:
    g = json.loads((HERE / a.gate).read_text())
    raw = (ROOT / ".run_cache" / "fake_env_concept_probe" / g["rows"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != g["rows_sha256"] or g["stage"] != "gate":
        raise ValueError("the gate rows do not match their recorded sha256, or the file is not a gate stage")
    if any(g["probes"][k]["sha256"] != fits[k]["probes_sha256"] for k in ("C", "T")):
        raise ValueError("the gate was scored with other probes")
    if not out["stage 4 may run"]:
        raise ValueError("the stage 3 prerequisites failed: the gate is not analyzed")
    out["gate protocol_sha256"] = g["protocol_sha256"]
    df = pd.DataFrame(json.loads(raw))
    if not np.isfinite(df[["score_C", "score_T"]].to_numpy()).all():
        raise ValueError("non-finite scores")
    if df.duplicated(["cell", "version", "step", "action"]).any():
        raise ValueError("duplicate call rows")
    df["call"] = list(zip(df["step"], df["action"]))
    keep = []
    for cell, d in df.groupby("cell"):  # matched calls: scored in all five versions
        common = set.intersection(*[set(d.loc[d["version"] == v, "call"]) for v in VERSIONS]) if set(d["version"]) == set(VERSIONS) else set()
        keep.append(d[d["call"].isin(common)])
    m = pd.concat(keep)
    runs = m.groupby(["cell", "version"])[["score_C", "score_T"]].mean().unstack("version")
    n = len(runs)
    out["stage 4"] = {"runs with a matched call": n, "of": len(g["cells"]), "minimum": MIN_RUNS,
                      "skipped run-versions": g["skipped"]}
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, max(n, 1), (B, max(n, 1)))
    for k, name in (("C", "content"), ("T", "turn start")):
        res = {}
        for place in ("far", "near"):
            for lab, (x, y) in (("plant minus neutral", (f"{place}_plant", f"{place}_neutral")), ("plant minus original", (f"{place}_plant", "original")),
                                ("neutral minus original", (f"{place}_neutral", "original"))):
                d = (runs[(f"score_{k}", x)] - runs[(f"score_{k}", y)]).to_numpy() if n else np.array([])
                bt = d[idx].mean(1) if n else np.array([np.nan])
                res[f"{place}: {lab}"] = {"logits": round(float(d.mean()), 4) if n else None, "95%": [round(float(q), 4) for q in np.percentile(bt, [2.5, 97.5])],
                                          "fraction of reference": round(float(d.mean() / ref[k]), 4) if n else None,
                                          "runs above 0": f"{int((d > 0).sum())}/{n}"}
        out[f"stage 4 ({name})"] = res
    dg = (runs[("score_C", "far_plant")] - runs[("score_C", "far_neutral")]).to_numpy() if n else np.array([])
    lo = float(np.percentile(dg[idx].mean(1), 2.5)) if n else float("nan")  # unrounded, as the decision uses
    out["GATE passes"] = bool(out["stage 4 may run"] and n >= MIN_RUNS and lo > 0)
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = HERE / "results" / f"concept_gate_{'analysis' if a.gate else 'prereq'}_{stamp}.json"
path.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
print(json.dumps(out, indent=1))
print(f"-> {path}")
