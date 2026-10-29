"""POST-HOC: the 196-task planted checks, run unchanged on the sim-trained monitor's cells ($0, local, stored scores only).

analyze_scaleup_plants.py (unchanged since 951ff7f7, the version that wrote scaleup_plants_2026-10-10_164614.json) reads
the real-trained cells by name. This wrapper gives it a copy of the draw-0 per-call file whose cells are relabelled:
  "real->real" <- "sim->real"   (sim-trained model, real-world calls)
  "real->sim"  <- "sim->sim"    (sim-trained model, simulated calls)
There is no sim->run2 cell, so there are no run-2 control plants. Everything else in the file (meta, folds, seed) is
unchanged. The script's own plant sizes, 2,000 resamples and detection rule apply. Its "(b) threshold set in simulation"
then reads the sim-trained model's own calibration scores, i.e. its usual threshold. The script writes into a temporary
folder; this wrapper saves its output verbatim under "script_output", with provenance.

  uv run --no-project --with numpy python -I run_plants_simtrained_POSTHOC.py \
      ../2026-10-08_q3_onpolicy_figures/analyze_scaleup_plants.py <onpolicy_2x2_scaleup196_draws0-0_2026-10-10_141824.json>
"""
import hashlib
import importlib.util
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

INPUT_SHA = "c2656658baaf3aa150f83f4d105e7572bfd854f98032831621419e1f8e8f36bf"  # s3_manifest_scaleup196_v1.json
SCRIPT_SHA = "d454cc89ad4fc6ecfa77d40fb286caae0c7886c93dfe017fcebca3ac356009d9"  # analyze_scaleup_plants.py at 951ff7f7
RELABEL = {"real->real": "sim->real", "real->sim": "sim->sim"}
DRAWS = 2000  # the as-run resamples (scaleup_plants_2026-10-10_164614.json "resamples")


def main(script_path, input_path):
    raw = Path(input_path).read_bytes()
    if (got := hashlib.sha256(raw).hexdigest()) != INPUT_SHA:
        raise SystemExit(f"{input_path}: sha256 {got} is not the manifest's {INPUT_SHA}")
    r = json.loads(raw)
    if len(r["draws"]) != 1 or r["draws"][0]["draw"] != 0:
        raise SystemExit("expected the observed draw 0 only")
    cells = r["draws"][0]["cells"]
    if "sim->run2" in cells:
        raise SystemExit("the file has a sim->run2 cell: add the run-2 control plants")
    r["draws"][0]["cells"] = {new: cells[old] for new, old in RELABEL.items()}

    if (got := hashlib.sha256(Path(script_path).read_bytes()).hexdigest()) != SCRIPT_SHA:
        raise SystemExit(f"{script_path}: sha256 {got} is not the as-run script's {SCRIPT_SHA}")
    spec = importlib.util.spec_from_file_location("plants", script_path)
    plants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plants)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        relabelled = tmp / (Path(input_path).stem + "__simtrained_relabelled.json")
        relabelled.write_text(json.dumps(r))
        (tmp / "results").mkdir()
        plants.HERE = tmp  # the script writes to HERE/results; keep that copy out of its own folder
        plants.main(str(relabelled), DRAWS)
        (produced,) = (tmp / "results").glob("scaleup_plants_*.json")
        script_output = json.loads(produced.read_text())

    out = {"label": "POST-HOC: planted checks on the sim-trained monitor (approved as a free check, 2026-10-12)",
           "script": "experiments/2026-10-08_q3_onpolicy_figures/analyze_scaleup_plants.py (unchanged)",
           "script_sha256": SCRIPT_SHA,
           "input": Path(input_path).name, "input_sha256": INPUT_SHA,
           "relabelled_cells": RELABEL, "resamples": DRAWS,
           "reading": "'real' in the script output = the sim-trained model on real calls (sim->real); 'sim' = sim->sim",
           "script_output": script_output}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = Path(__file__).parent / "results" / f"plants_simtrained_POSTHOC_{stamp}.json"
    with dst.open("x") as fh:
        json.dump(out, fh, indent=1)
    print("->", dst)


if __name__ == "__main__":
    main(*sys.argv[1:3])
