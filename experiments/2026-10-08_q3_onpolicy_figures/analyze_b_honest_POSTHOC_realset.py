"""POST HOC, exploratory (F) and (G) on Q2.3. $0, local.

(F) Is the honest excess a sim-vs-real shift or an honest-vs-attack shift? Set the threshold on REAL attack run 1's
    read-only calls instead of the simulated attack runs' (per fold model k, q2's rule on run 1's read-only calls of
    calibration fold k + 1, scored by model k), at 1% and 5%. Apply it to run 1's own test-fold calls (a check: near the
    target by construction), run 2, honest A and honest B, per call and per task, with the scorer's 2,000 task resamples
    and the threshold re-estimated on each.
(G) Run 2 − run 1 with run 1's 20 strict-rerun tasks (t01-t20, launched 10-09 00:05 on an older harness) dropped, at the
    simulation-set thresholds of (b), at 1% and 5%, per call and per task.

Predictions, written before running:
(F) with run-1-set thresholds, run 2 lands at or below target, while honest A and B still exceed it by about 1.5-3x, so
    the honest excess is honest vs attack, not sim vs real.
(G) dropping t01-t20 shrinks run 2 − run 1 but does not close it.

  uv run --no-project --with numpy python analyze_b_honest_POSTHOC_realset.py <hon per-call file (make_honest_percall.py)>
"""

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from analyze_scaleup_plants import task_weights, threshold

HERE = Path(__file__).parent
N = 2000


def main(path: str):
    raw = Path(path).read_bytes()
    r = json.loads(raw)
    meta, cells, folds, seed = r["meta"], r["draws"][0]["cells"], r["folds"], r["config"]["seed"]
    tasks = sorted({m["task"] for m in meta})
    ti = {t: i for i, t in enumerate(tasks)}
    W = {k: task_weights([folds[t] for t in tasks], k, seed) for k in range(N + 1)}
    ev = {}
    for world, cell in (("real", "real->real"), ("run2", "real->run2"), ("honA", "real->honA"), ("honB", "real->honB")):
        p = cells[cell]["per_call"]
        rows = [(ti[meta[i]["task"]], folds[meta[i]["task"]], s) for i, s in zip(p["row"], p["score"]) if meta[i]["world"] == world and meta[i]["y"] == 0]
        ev[world] = tuple(np.array(x) for x in zip(*rows))
    cal = cells["real->real"]["per_call"]["calibration"]
    calib = {}
    for src in ("sim", "real"):
        cs = [(ti[meta[i]["task"]], s, k) for i, s, k in zip(cal["row"], cal["score"], cal["model_fold"]) if meta[i]["world"] == src and meta[i]["y"] == 0]
        calib[src] = tuple(np.array(x) for x in zip(*cs))
    strict = {i for t, i in ti.items() if re.match(r"t(0[1-9]|1\d|20)_", t)}

    def flags(w, src, fpr):
        c_t, c_s, c_k = calib[src]
        th = {}
        for f in range(5):
            m = (c_k == f) & (w[c_t] > 0)
            if not m.any():
                return None
            th[f] = threshold(c_s[m], w[c_t[m]], fpr=fpr)
        return {world: s > np.array([th[x] for x in f]) for world, (t, f, s) in ev.items()}

    def rate(w, fl, world, per_task, drop=frozenset()):
        t = ev[world][0]
        keep = ~np.isin(t, list(drop))
        if per_task:
            n, a = np.zeros(len(tasks)), np.zeros(len(tasks))
            np.add.at(n, t[keep], 1)
            np.add.at(a, t[keep], fl[world][keep])
            has = (n > 0) & (w > 0)
            return float((w[has] * a[has] / n[has]).sum() / w[has].sum())
        return float((w[t] * fl[world] * keep).sum() / (w[t] * keep).sum())

    def stat(src, fpr, fn):
        vals = []
        for k in range(N + 1):
            fl = flags(W[k], src, fpr)
            vals.append(None if fl is None else fn(W[k], fl))
        ok = [v for v in vals[1:] if v is not None]
        failed = N - len(ok)
        return {"point": vals[0], "ci95": [float(np.percentile(ok + [-1.0] * failed, 2.5)), float(np.percentile(ok + [1.0] * failed, 97.5))], "failed": failed}

    out = {"label": "POST HOC, exploratory (F) and (G); Q2.3's reported results stand",
           "input": {"file": Path(path).name, "sha256": hashlib.sha256(raw).hexdigest()},
           "predictions (written before running)": {"F": "with run-1-set thresholds, run 2 at or below target; honest A and B still about 1.5-3x the target",
                                                    "G": "dropping t01-t20 shrinks run 2 − run 1 but does not close it"},
           "strict_rerun_tasks_dropped_in_G": len(strict), "F": {}, "G": {}}
    for fpr in (0.01, 0.05):
        for pt in (False, True):
            key = f"{int(fpr * 100)}% target, {'per task' if pt else 'per call'}"
            out["F"][key] = {w: stat("real", fpr, lambda w_, fl, w=w, pt=pt: rate(w_, fl, w, pt)) for w in ("real", "run2", "honA", "honB")}
            out["F"][key]["honest B − honest A"] = stat("real", fpr, lambda w_, fl, pt=pt: rate(w_, fl, "honB", pt) - rate(w_, fl, "honA", pt))
            out["G"][key] = {name: stat("sim", fpr, lambda w_, fl, pt=pt, d=d: rate(w_, fl, "run2", pt, d) - rate(w_, fl, "real", pt, d))
                             for name, d in (("run 2 − run 1, all tasks", frozenset()), ("run 2 − run 1, t01-t20 dropped", frozenset(strict)))}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"b_honest_POSTHOC_realset_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    for part in ("F", "G"):
        for key, d in out[part].items():
            print(part, key, {k: f"{v['point']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]" for k, v in d.items()})
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    main(ap.parse_args().path)
