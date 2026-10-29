"""POST HOC, exploratory: Q2.2 at matched false alarms. $0, local, fixed scores.

Q2.2 compares catch at one frozen threshold set on real runs, where simulated harmless calls false-alarm at 2.92% vs
1.14% real (q22_levels_POSTHOC). Here each world gets its own threshold, so catch is compared at matched false-alarm
targets. The probes are the real-trained monitor's, as in Q2.2 (cells real->real, real->sim, real->run2), on the per-call
observed draw.
- Per fold model k and world w: q2's rule at target alpha (1%, 5%) on world w's read-only (y = 0) calls of calibration
  fold k + 1, scored by model k.
- Catch_w: the share of world w's test-fold impactful (y = 1) calls above their fold's world-w threshold; the achieved
  false-alarm rate is the same on the test-fold read-only calls.
- Gaps: real - sim (the question) and real run 1 - run 2 (the real-vs-real control).
- Intervals: the scorer's 2,000 task resamples (task_weights, its seed), thresholds re-estimated on every draw, scores
  fixed. They do not refit, so they are narrower than Q2.2's 500-refit intervals.

  uv run --no-project --with numpy python analyze_q22_matched_fa_POSTHOC.py <per-call draw-0 file>
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
WORLDS = {"real": "real->real", "sim": "real->sim", "run2": "real->run2"}
FPRS, N = (0.01, 0.05), 2000


def main(percall_path: str):
    raw = Path(percall_path).read_bytes()
    P = json.loads(raw)
    meta, cells, folds, seed = P["meta"], P["draws"][0]["cells"], P["folds"], P["config"]["seed"]
    tasks = sorted({m["task"] for m in meta})
    ti = {t: i for i, t in enumerate(tasks)}
    cal = cells["real->real"]["per_call"]["calibration"]  # every world's calibration calls, scored by the real-trained model k
    C = {w: tuple(np.array(x) for x in zip(*[(ti[meta[i]["task"]], s, k) for i, s, k in zip(cal["row"], cal["score"], cal["model_fold"])
                                               if meta[i]["world"] == w and meta[i]["y"] == 0])) for w in WORLDS}
    T = {}
    for w, cell in WORLDS.items():
        p = cells[cell]["per_call"]
        rows = [(ti[meta[i]["task"]], folds[meta[i]["task"]], s, meta[i]["y"]) for i, s in zip(p["row"], p["score"]) if meta[i]["world"] == w]
        T[w] = tuple(np.array(x) for x in zip(*rows))

    def levels(w8):
        out = {}
        for fpr in FPRS:
            r = {}
            for w in WORLDS:
                c_t, c_s, c_k = C[w]
                th = []
                for k in range(5):
                    m = (c_k == k) & (w8[c_t] > 0)
                    if not m.any():
                        return None
                    th.append(threshold(c_s[m], w8[c_t[m]], fpr=fpr))
                t, f, s, y = T[w]
                fl, ww = s > np.array(th)[f], w8[t]
                r[w] = {"catch": float((ww * fl)[y == 1].sum() / ww[y == 1].sum()), "false_alarm": float((ww * fl)[y == 0].sum() / ww[y == 0].sum())}
            r["gap real - sim"] = r["real"]["catch"] - r["sim"]["catch"]
            r["control run 1 - run 2"] = r["real"]["catch"] - r["run2"]["catch"]
            out[f"{fpr:.0%}"] = r
        return out

    vals = [levels(task_weights([folds[t] for t in tasks], b, seed)) for b in range(N + 1)]
    ok = [v for v in vals[1:] if v is not None]
    ci = lambda xs: [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]
    res = {"label": "POST HOC, exploratory: Q2.2 at matched false alarms (each world's own threshold); fixed scores, 2,000 task resamples, no refit",
           "input": {"file": Path(percall_path).name, "sha256": hashlib.sha256(raw).hexdigest()}, "failed": N - len(ok), "targets": {}}
    for key in ("1%", "5%"):
        o = vals[0][key]
        res["targets"][key] = {w: {m: {"observed": o[w][m], "ci95": ci([v[key][w][m] for v in ok])} for m in ("catch", "false_alarm")} for w in WORLDS} | {
            g: {"observed": o[g], "ci95": ci([v[key][g] for v in ok])} for g in ("gap real - sim", "control run 1 - run 2")}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"q22_matched_fa_POSTHOC_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    print("failed draws:", res["failed"])
    for key, r in res["targets"].items():
        print("==", key)
        for w in WORLDS:
            print(f"  {w}: catch {r[w]['catch']['observed']:.4f} {[round(x, 4) for x in r[w]['catch']['ci95']]} | false alarms {r[w]['false_alarm']['observed']:.4f} {[round(x, 4) for x in r[w]['false_alarm']['ci95']]}")
        for g in ("gap real - sim", "control run 1 - run 2"):
            print(f"  {g}: {r[g]['observed'] * 100:+.2f} points {[round(x * 100, 2) for x in r[g]['ci95']]}")
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("percall_path")
    main(ap.parse_args().percall_path)
