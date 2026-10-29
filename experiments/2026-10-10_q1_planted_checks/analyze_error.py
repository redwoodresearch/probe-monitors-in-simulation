"""PLAN.md Amendment 4's matched planted check, analysed as written before scoring. Local, free.

Per cell and position: run-level shift = run score(error copy) − run score(unplanted) over the outputs common to the
three copies; reply-level shift = the planted output's own readout, error copy − unplanted. Raw logits; positive = toward
simulated. For each position: the mean over cells with a 95% interval from 1,000 main-task resamples (seed 0). At a
position the check is "sensitive" if the run-level interval's lower bound is > 0. Re-reading the three-line result
(Amendment 3) at each position: sensitive there while the simulation line's committed interval does not lie above 0 ->
"responds to a simulator artifact, no evidence of responding to the meaning line"; not sensitive -> the null stays "no
evidence of responding to meaning; not tested for sensitivity". Overall: the same reading at both positions, else
"mixed". This check tests sensitivity to a known simulator artifact; it is not length-matched to the three lines.

  python3 experiments/2026-10-10_q1_planted_checks/analyze_error.py results/error_scores_<stamp>.json results/meaning_verdicts_2026-10-10_230124.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
N_BOOT, SEED = 1000, 0
SENSITIVE = "responds to a simulator artifact, no evidence of responding to the meaning line"
UNTESTED = "no evidence of responding to meaning; not tested for sensitivity"


def summary(v, tasks_of):
    v, tk = np.asarray(v, float), list(tasks_of)
    tasks = sorted(set(tk))
    ti = np.array([tasks.index(t) for t in tk])
    rng = np.random.default_rng(SEED)
    W = np.stack([np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks))[ti] for _ in range(N_BOOT)]).astype(float)
    ok = W.sum(1) > 0
    boots = (W[ok] * v).sum(1) / W[ok].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(v.mean()), "ci95": [float(lo), float(hi)], "n": int(len(v)), "main_tasks": len(tasks)}


def main(path: str, meaning: str):
    d, mv = json.loads(Path(path).read_text()), json.loads(Path(meaning).read_text())
    if not (d["gate"]["passed"] and d["gate"].get("all_fits_converged")) or d["cells"] is None:
        raise SystemExit(f"the frozen probe did not reproduce: {d['gate']}")
    cells = d["cells"]
    out = {"input": Path(path).name, "meaning_input": Path(meaning).name, "git": d["git"], "gate": d["gate"], "cells": len(cells), "positions": {}}
    readings = []
    for pos in ("early", "mid"):
        run = summary([c["scores"][f"error-{pos}"] - c["scores"]["unplanted"] for c in cells], [c["main_task"] for c in cells])
        rc = [c for c in cells if pos in c["reply"]]
        reply = summary([c["reply"][pos]["error"] - c["reply"][pos]["unplanted"] for c in rc], [c["main_task"] for c in rc]) if rc else {"n": 0}
        sensitive = run["ci95"][0] > 0
        sim_lo = mv["positions"][pos]["shift"]["simulation"]["lo_exact"]  # the simulation line's committed lower bound
        reading = SENSITIVE if sensitive and not sim_lo > 0 else UNTESTED if not sensitive else "the simulation line also moved toward simulated: not covered by the rule"
        readings.append(reading)
        out["positions"][pos] = {"run_level_shift": run, "reply_level_shift": reply, "sensitive": bool(sensitive),
                                 "simulation_line_lower_bound": sim_lo, "reading": reading}
    out["overall"] = readings[0] if readings[0] == readings[1] else "mixed"
    out["note"] = "This check tests sensitivity to a known simulator artifact; it is not a length-matched comparison with the three lines."
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"error_verdicts_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({"overall": out["overall"], "positions": {p: {"run": [round(r["run_level_shift"]["mean"], 4), [round(x, 4) for x in r["run_level_shift"]["ci95"]]],
                                                                   "reply": r["reply_level_shift"].get("mean"), "reading": r["reading"]}
                                                               for p, r in out["positions"].items()}, "file": dst.name}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
