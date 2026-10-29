"""The meaning control's analysis (PLAN.md Amendment 2, the secondary probe), on score_frozen.py's output. Local, free.

Per cell, line (simulation = the tell, neutral, production) and position (early, mid): shift = run score(copy) - run
score(unplanted), raw logits, positive = toward simulated.
- Each line's mean shift at each position, with a 95% interval from 1,000 main-task resamples (seed 0; a cell's weight is
  how often its main task was drawn).
- The paired differences simulation - production, simulation - neutral and neutral - production, with the same resamples.
- Rule (committed before running): at a position, "reads meaning" only if the simulation - production interval's lower
  bound is > 0 AND the mean shifts are ordered production <= neutral <= simulation; otherwise "not shown to read meaning".
  Overall: both positions "reads meaning"; neither "not shown"; otherwise "mixed". Each line's shift is reported on its
  own. An ordering shows sensitivity to these lines, which word association could also produce; it is not proof of
  meaning.

  python3 experiments/2026-10-10_q1_planted_checks/analyze_meaning.py results/frozen_scores_<stamp>.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
N_BOOT, SEED = 1000, 0
LINES = {"simulation": "tell", "neutral": "neutral", "production": "production"}
PAIRS = [("simulation", "production"), ("simulation", "neutral"), ("neutral", "production")]


def main(path: str):
    d = json.loads(Path(path).read_text())
    if not (d["gate"]["passed"] and d["gate"].get("all_fits_converged")) or d["cells"] is None:
        raise SystemExit(f"the frozen probe did not reproduce: {d['gate']}")
    cells = d["cells"]
    tasks = sorted({c["main_task"] for c in cells})
    ti = np.array([tasks.index(c["main_task"]) for c in cells])
    rng = np.random.default_rng(SEED)
    W = np.stack([np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks))[ti]
                  for _ in range(N_BOOT)]).astype(float)  # [resample, cell]: how often the cell's main task was drawn

    def summary(v):
        v = np.asarray(v, float)
        boots = (W * v).sum(1) / W.sum(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"mean": round(float(v.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "lo_exact": float(lo)}

    out = {"input": Path(path).name, "git": d["git"], "gate": d["gate"], "cells": len(cells), "main_tasks": len(tasks), "positions": {}}
    readings = []
    for pos in ("early", "mid"):
        sh = {ln: [c["scores"][f"{k}-{pos}"] - c["scores"]["unplanted"] for c in cells] for ln, k in LINES.items()}
        r = {"shift": {ln: summary(v) for ln, v in sh.items()},
             "difference": {f"{a} - {b}": summary(np.subtract(sh[a], sh[b])) for a, b in PAIRS}}
        m = {ln: float(np.mean(sh[ln])) for ln in LINES}  # unrounded for the ordering
        ordered = m["production"] <= m["neutral"] <= m["simulation"]
        lo = r["difference"]["simulation - production"]["lo_exact"]
        r["rule"] = {"simulation_minus_production_lower_bound_above_0": bool(lo > 0), "ordered_production_neutral_simulation": bool(ordered),
                     "reading": "reads meaning" if lo > 0 and ordered else "not shown to read meaning"}
        readings.append(r["rule"]["reading"])
        out["positions"][pos] = r
    out["overall"] = ("reads meaning" if all(x == "reads meaning" for x in readings) else
                      "not shown to read meaning" if not any(x == "reads meaning" for x in readings) else "mixed")
    out["caveat"] = "An ordering shows sensitivity to these lines, which word association could also produce; it is not proof of meaning."
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"meaning_verdicts_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({"overall": out["overall"], "positions": {p: {"shift": {k: v["mean"] for k, v in r["shift"].items()},
                                                                       "sim-prod": r["difference"]["simulation - production"]["ci95"],
                                                                       "reading": r["rule"]["reading"]} for p, r in out["positions"].items()},
                      "file": dst.name}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
