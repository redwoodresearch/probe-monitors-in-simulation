"""Q2.2's per-arm catch levels, for the figure (an export of computed outputs; no new scoring). $0, local.

- **1% target, refit intervals.** Each arm's level from the 196-task scoring's 500 refit draws: real → real (real catch),
  real → sim (simulated catch) and real → run 2. Reported: catch at the frozen training-world threshold, its achieved
  false-alarm rate, and AUROC. The observed value is draw 0; the interval is the 2.5–97.5 percentiles over draws
  1–500. These are the same draws as the reported gap (+0.46 points [−2.32, +2.50]).
- **5% target, fixed-score intervals (secondary).** From the per-call observed draw:
  - per fold model k, q2's rule at 5% on the real-trained probe's training-world (real) read-only calls of calibration
    fold k + 1;
  - catch and false alarms on each arm's test-fold calls;
  - the scorer's 2,000 task resamples, with the threshold re-estimated and scores fixed.
  These intervals do not refit, so they are narrower than refit ones would be.

  uv run --no-project --with numpy python analyze_q22_levels_POSTHOC.py <draws 0-500 file> <per-call draw-0 file>
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from analyze_scaleup_plants import task_weights, threshold

HERE = Path(__file__).parent
CELLS = {"real catch (real → real)": "real->real", "simulated catch (real → sim)": "real->sim", "run 2 catch (real → run 2)": "real->run2"}


def main(draws_path: str, percall_path: str):
    rd, rp = Path(draws_path).read_bytes(), Path(percall_path).read_bytes()
    D, P = json.loads(rd), json.loads(rp)
    out = {"label": "export of computed outputs; Q2.2 levels behind the reported gap",
           "inputs": {Path(draws_path).name: hashlib.sha256(rd).hexdigest(), Path(percall_path).name: hashlib.sha256(rp).hexdigest()},
           "1% target (refit draws 1-500)": {}, "5% target (fixed scores, 2,000 task resamples)": {}}
    dr = {d["draw"]: d for d in D["draws"]}
    for name, cell in CELLS.items():
        out["1% target (refit draws 1-500)"][name] = {}
        for metric in ("catch_frozen", "false_alarm_frozen", "auroc"):
            vals = [dr[b]["cells"][cell]["values"][metric] for b in range(1, 501) if dr[b]["cells"][cell]["values"].get(metric) is not None]
            out["1% target (refit draws 1-500)"][name][metric] = {"observed": dr[0]["cells"][cell]["values"][metric],
                                                                   "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], "draws": len(vals)}
    meta, cells, folds, seed = P["meta"], P["draws"][0]["cells"], P["folds"], P["config"]["seed"]
    tasks = sorted({m["task"] for m in meta})
    ti = {t: i for i, t in enumerate(tasks)}
    W = {k: task_weights([folds[t] for t in tasks], k, seed) for k in range(2001)}
    cal = cells["real->real"]["per_call"]["calibration"]
    cs = [(ti[meta[i]["task"]], s, k) for i, s, k in zip(cal["row"], cal["score"], cal["model_fold"]) if meta[i]["world"] == "real" and meta[i]["y"] == 0]
    c_t, c_s, c_k = (np.array(x) for x in zip(*cs))
    ev = {}
    for name, cell in CELLS.items():
        world = cell.split(">")[1]
        p = cells[cell]["per_call"]
        rows = [(ti[meta[i]["task"]], folds[meta[i]["task"]], s, meta[i]["y"]) for i, s in zip(p["row"], p["score"]) if meta[i]["world"] == world]
        ev[name] = tuple(np.array(x) for x in zip(*rows))

    def levels(w):
        th = {}
        for f in range(5):
            m = (c_k == f) & (w[c_t] > 0)
            if not m.any():
                return None
            th[f] = threshold(c_s[m], w[c_t[m]], fpr=0.05)
        res = {}
        for name, (t, f, s, y) in ev.items():
            fl, ww = s > np.array([th[x] for x in f]), w[t]
            res[name] = {"catch": float((ww * fl)[y == 1].sum() / ww[y == 1].sum()), "false_alarm": float((ww * fl)[y == 0].sum() / ww[y == 0].sum())}
        return res

    vals = [levels(W[k]) for k in range(2001)]
    ok = [v for v in vals[1:] if v is not None]
    for name in CELLS:
        out["5% target (fixed scores, 2,000 task resamples)"][name] = {
            m: {"observed": vals[0][name][m], "ci95": [float(np.percentile([v[name][m] for v in ok], 2.5)), float(np.percentile([v[name][m] for v in ok], 97.5))]}
            for m in ("catch", "false_alarm")} | {"failed": 2000 - len(ok)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"q22_levels_POSTHOC_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    for part in ("1% target (refit draws 1-500)", "5% target (fixed scores, 2,000 task resamples)"):
        print("==", part)
        for name, d in out[part].items():
            print(" ", name, {m: f"{v['observed']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]" for m, v in d.items() if isinstance(v, dict)})
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("draws_path")
    ap.add_argument("percall_path")
    a = ap.parse_args()
    main(a.draws_path, a.percall_path)
