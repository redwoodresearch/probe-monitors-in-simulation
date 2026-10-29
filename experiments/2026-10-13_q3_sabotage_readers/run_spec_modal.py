"""B3: make_grid_spec.py on Modal CPU (Aswin, 17:24 PT 10-13: no compute on the laptop). Inputs from the B3 Volume: the pinned
row files (b3_inputs/), the votes and split (b3_inputs2/), Qwen3.5's metas (q35_metas/) and the staged folders; the stage
records come from results/ here. The spec is written to the Volume (grid_specs_v1/) and into results/.

  uv run --no-project --python 3.12 --with "modal==1.6.1" modal run run_spec_modal.py --reader qwen3 --stage <stage record> [--repair-stage <record>]
"""

import subprocess
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-q3sabr_spec_kill-2026-11-13"
HERE = Path(__file__).resolve().parent
app = modal.App(APP)
b3 = modal.Volume.from_name("probe-shift_aswin-surya_sub-q3sabr_acts_kill-2026-11-13", version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy").env({"B3_STAGE_ROOT": "/v", "B3_RESULTS": "/tmp/results"})
         .add_local_file(HERE / "make_grid_spec.py", "/root/b3/make_grid_spec.py"))


@app.function(image=image, cpu=4, memory=16384, timeout=900, retries=0, volumes={"/v": b3})
def spec(reader: str, stage: bytes, repair_stage: bytes, commit: str) -> dict:
    import os
    import sys

    sys.path.insert(0, "/root/b3")
    os.makedirs("/tmp/results")
    os.environ["B3_GIT_COMMIT"] = commit
    Path("/tmp/stage.json").write_bytes(stage)
    if repair_stage:
        Path("/tmp/repair_stage.json").write_bytes(repair_stage)
    import make_grid_spec

    make_grid_spec.main(reader, "/v/b3_inputs", "/v/b3_inputs2/scaleup_panel_votes_k116.csv", "/v/b3_inputs2/split_v1.json", "/v/q35_metas",
                        "/tmp/stage.json", "/tmp/repair_stage.json" if repair_stage else "")
    files = {p.name: p.read_bytes() for p in Path("/tmp/results").iterdir()}
    os.makedirs("/v/grid_specs_v1", exist_ok=True)
    for n, b in files.items():
        Path(f"/v/grid_specs_v1/{n}").write_bytes(b)
    b3.commit()
    return files


@app.local_entrypoint()
def main(reader: str, stage: str, repair_stage: str = ""):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--", "make_grid_spec.py", Path(__file__).name):
        raise SystemExit("commit make_grid_spec.py and this script before they run")
    files = spec.remote(reader, Path(stage).read_bytes(), Path(repair_stage).read_bytes() if repair_stage else b"", git("rev-parse", "HEAD"))
    for n, b in files.items():
        with (HERE / "results" / n).open("xb") as fh:
            fh.write(b)
        print("->", HERE / "results" / n)
