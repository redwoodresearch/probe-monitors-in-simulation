"""Freeze the retrain's scoring inputs before launch ($0, local): run keys, checker keys, staged shard hashes, old probes.

README "Retrain A and B": the 99 evaluation runs (step 1's sample less the run without a scored call in both
versions), the checker draw's 20 runs, the staged input shards' sha256s (from the build, which hashed them where they
live), and the old fit. run_score_retrain.py refuses if any of these differ.

  uv run --no-project python experiments/2026-10-10_eval_awareness_probe/freeze_retrain_inputs.py --build results/build_retrain_<stamp>.json
"""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--build", required=True)
a = ap.parse_args()
step1 = json.loads((HERE / "results" / "la_scores_2026-10-10_024336.json").read_text())
gate = json.loads((HERE / "results" / "step1_gate_2026-10-10_025308.json").read_text())
draw_raw = (HERE / "results" / "checker_pilot_draw.json").read_bytes()
draw = json.loads(draw_raw)
build = json.loads((HERE / a.build).read_text())
excluded = gate["per_token"]["(b) runs without a scored call in both versions (excluded)"]
runs = sorted(k for k in step1["sampled_runs"] if k not in excluded)
if len(runs) != 99 or not set(draw["keys"]) <= set(runs) or len(draw["keys"]) != 20:
    raise ValueError("expected 99 runs that include the checker draw's 20")
out = {"runs": runs, "checker runs": sorted(draw["keys"]), "draw_sha256": hashlib.sha256(draw_raw).hexdigest(),
       "staged shards sha256": build["staged shards sha256"], "build": a.build, "set_sha256": build["set_sha256"],
       "old fit": {"run": step1["run"], "fit_id": step1["fit_id"], "probes_sha256": step1["probes_sha256"]},
       "stored rows": {"far": "la_scores_2026-10-10_024336.json", "near": "la_scores_near_2026-10-10_103942.json",
                       "checker": "la_scores_checker_2026-10-10_155537.json"}}
path = HERE / "results" / "retrain_frozen_inputs.json"
if path.exists():
    raise ValueError(f"{path.name} exists: results/ is append-only")
path.write_text(json.dumps(out, indent=1) + "\n")
print(f"{len(runs)} runs, {len(draw['keys'])} checker runs, {len(out['staged shards sha256'])} shards -> {path}")
