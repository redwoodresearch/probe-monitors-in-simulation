"""Launcher for the by-layer run (RULE_POSTHOC_catch_by_layer.md; SPEND row 71893daa). It runs
run_score_onpolicy.score_layers in an ephemeral app, detached, so the run survives the laptop. The deployed app of the
same name is not touched. A frozen copy of watch_layers.py starts on the app ID at once. The checkpoint key is
main()'s for this config, with "_layers"; the results file is written as main() writes it.

  uv run --no-project --python 3.14 --with 'modal>=1.4.3' --with pyyaml --with numpy python run_layers.py \\
      --config score_scaleup196_layers.yaml --watcher <frozen watch_layers.py> --prior-usd <billed by earlier runs>
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import run_score_onpolicy as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--watcher", required=True)
ap.add_argument("--prior-usd", type=float, required=True)
ap.add_argument("--cap-usd", type=float, default=2.30)  # the $2.50 stop less billing lag
ap.add_argument("--max-minutes", type=float, default=23)  # the 1,380 s timeout plus start-up
a = ap.parse_args()
cfg = yaml.safe_load((R.HERE / "config" / a.config).read_text())
if not cfg.get("all_layers") or cfg.get("mode") or cfg.get("null_test") or cfg.get("label_rule"):
    raise SystemExit("run_layers takes a plain scoring config with all_layers")
folds = R.frozen_folds(R.HERE.parents[1] / cfg["pilot_tasks"])
git = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=R.HERE).stdout.strip(),
       "dirty": bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=R.HERE).stdout.strip())}
if git["dirty"]:
    raise SystemExit("the repo has uncommitted tracked changes")
t0 = time.time()
extractions = {r: json.loads(b"".join(R.acts_vol.read_file(f"{r}/complete.json"))) for r in cfg["extractions"]}
ident = {"config": cfg, "folds": folds, "extractions": extractions, "scorer_sha256": hashlib.sha256((R.HERE / "run_score_onpolicy.py").read_bytes()).hexdigest()}
key = f"{cfg['label']}_{hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]}_layers"
with modal.enable_output(), R.app.run(detach=True):
    app_id = R.app.app_id
    wlog = Path(a.watcher).with_name(f"watch_{key}.log")
    # its own uv environment: the launcher's temporary one is removed when the launcher exits (the 20-task run's watchdog lost it)
    subprocess.Popen(["uv", "run", "-q", "--isolated", "--no-project", "--python", "3.14", "--with", "modal==1.6.1", "python", a.watcher,
                      app_id, str(a.prior_usd), str(a.cap_usd), str(a.max_minutes)],
                     stdout=open(wlog, "a"), stderr=subprocess.STDOUT, start_new_session=True)
    print(f"LAUNCHED app {app_id} key {key} at {datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(timespec='seconds')}; watchdog log {wlog}", flush=True)
    p = R.score_layers.remote(cfg, folds, key)
out = {"label": cfg["label"], "config": cfg, "amendments": cfg["amendments"], "git": git, "app": R.APP, "app_id": app_id, "checkpoint_folder": key,
       "scorer_sha256": ident["scorer_sha256"], "extractions": extractions, "folds": folds, "summary": p["summary"],
       "wall_seconds": round(time.time() - t0), "load_seconds": p["load_seconds"], "draws": p["draws"], "meta": p["meta"]}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = R.HERE / "results" / f"onpolicy_2x2_{cfg['label']}_draws0-0_{stamp}.json"
with path.open("x") as fh:
    fh.write(json.dumps(out))
print(json.dumps({k: out[k] for k in ("label", "app_id", "summary", "wall_seconds", "load_seconds", "checkpoint_folder")}, indent=1))
print("->", path)
