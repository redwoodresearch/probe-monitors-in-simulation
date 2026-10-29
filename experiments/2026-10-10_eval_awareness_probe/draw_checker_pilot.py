"""Draw the checker-header pilot's 20 runs and fix its reference ($0, saved outputs only; README "Revised positive
control").

- Runs: of the 100 sampled real honest runs, the ones with scored calls in both versions of the far run; grouped by
  main task (the key's "<task>:<hash>" part, before the colon); 20 tasks drawn at random (seed 20261010), then one run per
  task, drawn the same way. The first 3 drawn runs are also scored unedited, as the reproduction check.
- Reference planted amount: the synthetic held-out checker_header pairs' median change (implied-test minus ordinary),
  per readout, from the fit's stored held-out scores.

  uv run --no-project python experiments/2026-10-10_eval_awareness_probe/draw_checker_pilot.py --far results/la_scores_2026-10-10_024336.json --gate results/step1_gate_2026-10-10_025308.json --fit results/fit_2026-10-10_022437.json
"""

import argparse
import hashlib
import json
import random
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
SEED, N = 20261010, 20

ap = argparse.ArgumentParser()
ap.add_argument("--far", required=True)
ap.add_argument("--gate", required=True)
ap.add_argument("--fit", required=True)
a = ap.parse_args()
far, gate, fit = (json.loads((HERE / p).read_text()) for p in (a.far, a.gate, a.fit))
if gate["la"] != a.far or gate["fit"] != a.fit or (fit["run"], fit["fit_id"], fit["probes_sha256"]) != (far["run"], far["fit_id"], far["probes_sha256"]):
    raise ValueError("the gate, fit and far files do not belong together")
excluded = set(gate["per_token"]["(b) runs without a scored call in both versions (excluded)"])
runs = sorted(k for k in far["sampled_runs"] if k not in excluded)
task = lambda k: k.split("__")[1].split(":")[0]
by_task = {}
for k in runs:
    by_task.setdefault(task(k), []).append(k)
rng = random.Random(SEED)
tasks = rng.sample(sorted(by_task), N)
keys = [rng.choice(by_task[t]) for t in tasks]
ref = {}
for r in ("per_token", "last_token"):
    s = {x["id"]: x for x in fit["heldout_scores_at_chosen_layer"][r]}
    ch = [s[f"{p}_1"]["score"] - s[f"{p}_0"]["score"] for p in sorted({i.rsplit("_", 1)[0] for i, x in s.items() if x["mechanism"] == "checker_header"})]
    ref[r] = {"pairs": len(ch), "median change": round(statistics.median(ch), 3)}
out = {"seed": SEED, "runs drawn": N, "eligible runs": len(runs), "eligible tasks": len(by_task), "keys": keys,
       "reproduction keys (scored unedited too)": keys[:3], "reference: synthetic checker_header median pair change": ref,
       "far": a.far, "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()}
path = HERE / "results" / "checker_pilot_draw.json"
if path.exists():  # the draw is fixed: a rerun only checks that it reproduces it, and records its inputs' provenance
    old = json.loads(path.read_text())
    same = all(old[k] == out[k] for k in ("seed", "keys", "reproduction keys (scored unedited too)", "reference: synthetic checker_header median pair change"))
    if not same:
        raise ValueError(f"{path} exists and this run's draw differs from it")
    prov = {"draw": path.name, "draw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "reproduces the draw": True,
            "inputs": {k: {"path": v, "sha256": hashlib.sha256((HERE / v).read_bytes()).hexdigest()} for k, v in
                       (("far", a.far), ("gate", a.gate), ("fit", a.fit))},
            "checked": "gate['la'] == far, gate['fit'] == fit, and fit and far share run, fit_id and probes_sha256", "git": out["git"]}
    path = path.with_name("checker_pilot_draw_provenance.json")
    if path.exists():
        raise FileExistsError(f"{path} exists")
    path.write_text(json.dumps(prov, indent=1) + "\n")
    out = prov
else:
    path.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "keys"}, indent=1))
print(f"-> {path}")
