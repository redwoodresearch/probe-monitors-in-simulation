"""The 196-task scoring's simulated-trained row, for the 2x2 figure (an export of computed outputs; no new scoring). $0,
local. The sim-trained monitor's cells sim -> real and sim -> sim at the 1% target, exactly as
analyze_q22_levels_POSTHOC.py exports the real-trained row: catch at the frozen training-world (simulated) threshold,
its achieved false-alarm rate, and AUROC; observed = draw 0, interval = the 2.5-97.5 percentiles over refit draws
1-500, the same draws as the reported gap.

  uv run --no-project --with numpy python analyze_sim_trained_levels_POSTHOC.py <draws 0-500 file>
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
CELLS = {"sim-trained on real runs (sim → real)": "sim->real", "sim-trained on simulated runs (sim → sim)": "sim->sim"}


def main(draws_path: str):
    rd = Path(draws_path).read_bytes()
    D = json.loads(rd)
    dr = {d["draw"]: d for d in D["draws"]}
    if sorted(dr) != list(range(501)):
        raise SystemExit("expected draws 0-500")
    out = {"label": "export of computed outputs; the 196-task scoring's sim-trained row at the 1% target",
           "inputs": {Path(draws_path).name: hashlib.sha256(rd).hexdigest()}, "1% target (refit draws 1-500)": {}}
    for name, cell in CELLS.items():
        out["1% target (refit draws 1-500)"][name] = {}
        for metric in ("catch_frozen", "false_alarm_frozen", "auroc"):
            vals = [dr[b]["cells"][cell]["values"][metric] for b in range(1, 501) if dr[b]["cells"][cell]["values"].get(metric) is not None]
            out["1% target (refit draws 1-500)"][name][metric] = {"observed": dr[0]["cells"][cell]["values"][metric],
                                                                   "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], "draws": len(vals)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"sim_trained_levels_POSTHOC_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out))
    print(json.dumps(out, indent=1))
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("draws_path")
    main(ap.parse_args().draws_path)
