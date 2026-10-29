"""Relaunch run_retrain_qwen.py's grid on the DEPLOYED app, so the calls survive the laptop's client dropping (`modal run
--detach` keeps only the last triggered function alive; the 18:27 PT outage stopped the first launch). Computes the same
tag as run_retrain_qwen.main (the script's bytes and the attempted run-1 list) and spawns one grid call per (layer,
readout); finished records are skipped by grid() itself. Run `modal deploy` on run_retrain_qwen.py first.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' python experiments/2026-10-08_q3_onpolicy_figures/spawn_retrain_qwen.py <labels> <expected tag>
"""

import hashlib
import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q3onpol_retrainqwen_kill-2026-11-10"
LAYERS, KINDS = [39, 51, 63], ["output_end", "call_end"]


def main(labels: str, expected: str):
    attempted = []
    for f in labels.split(","):  # as run_retrain_qwen.main
        if "second_attack" in Path(f).name:
            continue
        attempted += [f"run1|{r['task']}" for r in map(json.loads, Path(f).read_text().splitlines()) if r["arm"] == "attack" and r.get("label") == "attempted"]
    if len(attempted) != len(set(attempted)):
        raise ValueError("a run-1 task is labelled twice")
    tag = hashlib.sha256((HERE / "run_retrain_qwen.py").read_bytes() + json.dumps(sorted(attempted)).encode()).hexdigest()[:12]
    if tag != expected:
        raise SystemExit(f"tag {tag} differs from the first launch's {expected}: the script or labels changed")
    grid = modal.Function.from_name(APP, "grid")
    calls = [grid.spawn(tag, l, k, attempted, False) for l in LAYERS for k in KINDS]
    print(json.dumps({"tag": tag, "spawned": [c.object_id for c in calls]}))


if __name__ == "__main__":
    main(*sys.argv[1:3])
