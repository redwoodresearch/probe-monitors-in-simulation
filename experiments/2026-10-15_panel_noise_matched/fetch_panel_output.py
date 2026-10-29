"""Re-materialise a PANEL run's results/ files from its Volume copy (detached runs whose client dropped; Aswin 10-16 via 2e). Finds the ONE file
xfer/panel_out/<kind>_<config>_<stamp>_<app id>.json on the Volume for the app id, downloads it (`modal volume get`), checks that the local script is the
launch's (sha256), and writes results/ with that script's own writer, as its entrypoint would after the run (exclusive create: an existing record is never
overwritten).

  PYTHONPATH=src python3 experiments/2026-10-15_panel_noise_matched/fetch_panel_output.py <app id>
"""

import hashlib
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from probe_shift.panel_out import OUT_DIR, load_out

HERE = Path(__file__).resolve().parent
MODAL = str(Path(sys.executable).parent / "modal")
MODULES = {"pooled": "run_pooled", "plant_rows": "run_plant_rows", "tool": "run_tool"}

app_id = sys.argv[1]
sys.path.insert(0, str(HERE))
vol = importlib.import_module("run_pooled").VOL
ls = json.loads(subprocess.run([MODAL, "volume", "ls", vol, OUT_DIR, "--json"], capture_output=True, check=True, text=True).stdout)
hits = [e["filename"] for e in ls if e.get("type") == "file" and e["filename"].endswith(f"_{app_id}.json")]
if len(hits) != 1:
    raise SystemExit(f"STOP: {len(hits)} Volume copies for {app_id} under {OUT_DIR} (need exactly one)")
with tempfile.TemporaryDirectory() as td:
    dst = Path(td) / "out.json"
    subprocess.run([MODAL, "volume", "get", vol, hits[0], str(dst)], capture_output=True, check=True)
    payload = load_out(dst)
ctx = payload["context"]
exp_name = f"{ctx.get('kind')}_{ctx.get('cfg_name')}_{ctx.get('stamp')}_{app_id}"  # every filename-derived field must agree (Codex detach r1)
if ctx.get("app_id") != app_id or ctx.get("kind") not in MODULES or ctx.get("out_name") != exp_name or Path(hits[0]).name != f"{exp_name}.json":
    raise SystemExit(f"STOP: {hits[0]} context {ctx.get('app_id')} / {ctx.get('kind')} / {ctx.get('cfg_name')} / {ctx.get('stamp')} / {ctx.get('out_name')} does not match")
if "exception" in payload:  # the remote body raised (detach gap, 2e): nothing to materialise; show the saved traceback
    print(payload["exception"])
    for k_ in ("returncode", "stdout", "stderr"):  # a tool run's saved output (Codex runner r2)
        if payload.get(k_) is not None:
            print(f"--- {k_} ---\n{payload[k_]}")
    raise SystemExit(f"STOP: {hits[0]} records an exception in the remote body (no results to write)")
mod = importlib.import_module(MODULES[ctx["kind"]])
if hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest() != ctx["script_sha256"]:
    raise SystemExit(f"STOP: {Path(mod.__file__).name} is not the launch's script (launch git {ctx['git']}); check that commit out and re-run")
print(json.dumps({"app_id": app_id, "volume_copy": hits[0], "kind": ctx["kind"], "config": ctx["cfg_name"], "stamp": ctx["stamp"], "launch_git": ctx["git"]}))
mod.write_outputs(ctx["cfg_name"], payload["cfg"], payload["out"], ctx)
