"""The yardstick's planted check (RULE.md): analyze_scaleup_plants.py, unchanged (sha256 pinned to its 951ff7f7
version), on this rescoring's per-call draw 0, with the sim-trained cells relabelled as the cells it reads:
  "real->real" <- "sim->real", "real->sim" <- "sim->sim", "real->run2" <- "sim->run2".
Its "control plant: run 2 catches down 10 points" is the yardstick's plant. The script writes into a temporary
folder; its output is saved verbatim under "script_output", with provenance.

  python -I run_plants_yardstick.py ../2026-10-08_q3_onpolicy_figures/analyze_scaleup_plants.py <new draws0-0 per-call json>
"""
import hashlib
import importlib.util
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SCRIPT_SHA = "d454cc89ad4fc6ecfa77d40fb286caae0c7886c93dfe017fcebca3ac356009d9"  # analyze_scaleup_plants.py at 951ff7f7
RELABEL = {"real->real": "sim->real", "real->sim": "sim->sim", "real->run2": "sim->run2"}
DRAWS = 2000


def main(script_path, input_path):
    if (got := hashlib.sha256(Path(script_path).read_bytes()).hexdigest()) != SCRIPT_SHA:
        raise SystemExit(f"{script_path}: sha256 {got} is not the as-run script's {SCRIPT_SHA}")
    raw = Path(input_path).read_bytes()
    r = json.loads(raw)
    if [d["draw"] for d in r["draws"]] != [0]:
        raise SystemExit("expected the observed draw 0 only")
    cells = r["draws"][0]["cells"]
    r["draws"][0]["cells"] = {new: cells[old] for new, old in RELABEL.items()}  # KeyError if a cell is missing
    for old in RELABEL.values():  # the plant script zips these lists: unequal lengths would truncate silently
        pc = cells[old]["per_call"]
        if len({len(pc[k]) for k in ("row", "score", "flag_frozen", "flag_calibrated")}) != 1:
            raise SystemExit(f"{old}: per-call lists differ in length")

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

    out = {"label": "POST HOC: planted checks on the sim-trained monitor, with run 2 (RULE.md)",
           "script": "experiments/2026-10-08_q3_onpolicy_figures/analyze_scaleup_plants.py (unchanged)",
           "script_sha256": SCRIPT_SHA, "input": Path(input_path).name, "input_sha256": hashlib.sha256(raw).hexdigest(),
           "relabelled_cells": RELABEL, "resamples": DRAWS,
           "reading": "'real' = sim->real (run 1), 'sim' = sim->sim, 'run2' = sim->run2, all scored by the sim-trained model",
           "script_output": script_output}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = Path(__file__).parent / "results" / f"plants_yardstick_{stamp}.json"
    with dst.open("x") as fh:
        json.dump(out, fh, indent=1)
    print("->", dst)


if __name__ == "__main__":
    main(*sys.argv[1:3])
