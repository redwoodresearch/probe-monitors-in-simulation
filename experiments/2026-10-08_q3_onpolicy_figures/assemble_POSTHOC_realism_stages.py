"""Assemble a realism job's output from its saved stage files when the job stopped before writing result.json (the
command-token job, stopped after shuffle 5 on 0d's instruction, 13:41:57 PT; the null is the shuffles done). Reads, from the score
Volume: stage_realism.json, stage_control.json, every stage_null_<i>.json present, and alignment.json (written by
run_realism_probe.py --alignment-for). Writes data/realism_<label>_assembled_<stamp>.json in run_realism_probe.py's
output format, for analyze_POSTHOC_realism_196.py. $0; small JSON only.

  uv run --no-project --python 3.14 --with 'modal>=1.4.3' --with pyyaml python assemble_POSTHOC_realism_stages.py \\
      --key <job key> --app-id <app id> --config score_scaleup196.yaml --launch-git <the job's launch commit>
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import run_score_onpolicy as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--key", required=True)
ap.add_argument("--app-id", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--launch-git", required=True)
a = ap.parse_args()
v = modal.Volume.from_name(R.SCORE, version=2)
names = sorted(e.path.split("/")[-1] for e in v.listdir(a.key))
read = lambda n: json.loads(b"".join(v.read_file(f"{a.key}/{n}")))  # noqa: E731
if "result.json" in names:
    raise SystemExit("the job finished: use its own output")
st = read("stage_realism.json")
nulls = sorted((read(n) for n in names if n.startswith("stage_null_")), key=lambda x: x["i"])
if [x["i"] for x in nulls] != list(range(1, len(nulls) + 1)):
    raise SystemExit(f"the saved shuffles are not 1..n: {[x['i'] for x in nulls]}")
al = read("alignment.json")
cfg = yaml.safe_load((R.HERE / "config" / a.config).read_text())
out = st | {"control": read("stage_control.json")["control"], "nulls": nulls, "folds": al["folds"], "seconds": None,
            "git": a.launch_git, "app": "probe-shift_aswin-surya_sub-q3onpol_realism_kill-2026-11-08", "app_id": a.app_id, "key": a.key,
            "config": cfg, "folds_map": R.frozen_folds(R.HERE.parents[1] / cfg["pilot_tasks"]), "wall_seconds": None,
            "assembled": {"stage_files": names, "n_nulls": len(nulls), "alignment_seconds": al["seconds"],
                          "note": "the job stopped before writing result.json (the command job: stopped by app ID after shuffle 5 on 0d's instruction, 13:41:57 PT); the null is the shuffles saved"}}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = R.HERE / "data" / f"realism_{cfg['label']}_assembled_{stamp}.json"
path.parent.mkdir(exist_ok=True)
with path.open("x") as fh:
    fh.write(json.dumps(out))
print(json.dumps(out["assembled"]))
print("->", path)
