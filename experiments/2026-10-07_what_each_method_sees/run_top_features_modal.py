"""Run analyze_top_features.py on Modal CPU, one 16-core container per text level, instead of on the laptop.

Uploads the three inputs (joined scores, the staging's outputs.jsonl.gz, the folds file) to a Volume, captures
provenance at launch, runs `analyze_top_features.main` per level, keeps the outputs on the Volume under /results/ and
copies them to results/ when the local entrypoint is still attached.

  uv run modal run --detach experiments/2026-10-07_what_each_method_sees/run_top_features_modal.py::main
  uv run modal run experiments/2026-10-07_what_each_method_sees/run_top_features_modal.py::fetch --stamp <stamp>
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
NAME = "probe-shift_aswin-surya_sub-what-sees_top-feats_kill-2026-11-06"
TIMEOUT_S = 2400
app = modal.App(NAME)
vol = modal.Volume.from_name(NAME, create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.13")
         .uv_pip_install("numpy", "scipy", "scikit-learn==1.9.1", "joblib==1.6.0", "pyyaml")
         .env({"JOBLIB_TEMP_FOLDER": "/tmp"})  # joblib's shared-memory default (/dev/shm) is small in containers
         .add_local_python_source("probe_shift")
         .add_local_file(HERE / "analyze_top_features.py", "/root/analyze_top_features.py")
         .add_local_file(HERE / "config/what_sees.yaml", "/root/config/what_sees.yaml"))


@app.function(image=image, volumes={"/data": vol}, cpu=16, memory=65536, timeout=TIMEOUT_S)
def level_job(level: int, stamp: str) -> dict:
    sys.path.insert(0, "/root")
    import analyze_top_features as A

    vol.reload()
    paths = A.main(["--config", "/root/config/what_sees.yaml", "--levels", str(level), "--stamp", stamp,
                    "--joined", f"/data/inputs/{stamp}/joined.npz", "--staging-outputs", f"/data/inputs/{stamp}/outputs.jsonl.gz",
                    "--folds", f"/data/inputs/{stamp}/folds.json", "--provenance", f"/data/inputs/{stamp}/provenance.json",
                    "--out-dir", f"/data/results/{stamp}"])
    vol.commit()
    return {p.name: p.read_bytes() for p in paths}


@app.local_entrypoint()
def main(levels: str = "0,7"):
    import yaml

    from probe_shift.provenance import capture

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    cfg = yaml.safe_load((HERE / "config/what_sees.yaml").read_text())
    prov = capture(ROOT, [HERE / "analyze_top_features.py", HERE / "run_top_features_modal.py", HERE / "config/what_sees.yaml"])
    tmp = HERE / ".run_cache" / f"provenance_top_features_{stamp}.json"
    tmp.write_text(json.dumps(prov | {"launched": stamp, "modal_app": NAME}))
    with vol.batch_upload() as batch:
        batch.put_file(str(HERE / ".run_cache/joined.npz"), f"/inputs/{stamp}/joined.npz")
        batch.put_file(str(Path(cfg["staging_dir"]).expanduser() / "outputs.jsonl.gz"), f"/inputs/{stamp}/outputs.jsonl.gz")
        batch.put_file(str(ROOT / "splits" / cfg["folds_file"]), f"/inputs/{stamp}/folds.json")
        batch.put_file(str(tmp), f"/inputs/{stamp}/provenance.json")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"top_features_launch_{stamp}.json").write_text(json.dumps(prov | {"stamp": stamp, "modal_app": NAME}, indent=1))
    print(f"stamp {stamp}; outputs also on the Volume under /results/{stamp}/", flush=True)
    for files in level_job.map([int(x) for x in levels.split(",")], kwargs={"stamp": stamp}):
        for name, data in files.items():
            (HERE / "results" / name).write_bytes(data)
            print(f"wrote results/{name}", flush=True)


@app.local_entrypoint()
def fetch(stamp: str):
    """Copy a detached run's outputs from the Volume to results/ (names checked against the expected pattern)."""
    for entry in vol.listdir(f"/results/{stamp}"):
        name = Path(entry.path).name
        assert name.startswith("top_features_L") and name.endswith((".json", ".md")), f"unexpected file {entry.path}"
        (HERE / "results" / name).write_bytes(b"".join(vol.read_file(entry.path)))
        print(f"wrote results/{name}")
