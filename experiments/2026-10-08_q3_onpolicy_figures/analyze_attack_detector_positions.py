"""The attack detector at matched absolute positions: windows of tool outputs 1-10, 11-20, 21-30, 31-40 and 41-50 (by
turn, then call index). A run enters a window only if it reaches the window's end; its score is the window's mean. $0,
local. Prediction first: predictions_attack_detector_positions_2026-10-09.md. Worlds, errored-run drops and the joint
task bootstrap (2,000, seed 0) are analyze_attack_detector.py's. Null per window and world (added after review): the
fixed scores, attack and honest labels permuted among each task's runs in the window, 1,000 draws (seed [0, 9]);
p = (1 + #{null >= observed}) / 1,001. It tests separation within a window, not the gap.

  uv run --no-project --python 3.14 --with numpy --with scikit-learn python analyze_attack_detector_positions.py \
      results/attack_detector_per_output_<stamp>.json --run-dir <rerun folder> [--cell L51_output_end]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from analyze_attack_detector import GAPS, WORLDS, aucs, ci, errored_runs

HERE = Path(__file__).parent
WINDOWS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]


def main(path: str, run_dir: str, cell: str):
    R = json.loads(Path(path).read_text())
    bad = set(errored_runs(Path(run_dir).expanduser()))
    kind = cell.split("_", 1)[1]
    per_run = {}
    for r, s in zip(R["rows"], R["scores"][cell]):
        if r["kind"] == kind and s is not None:
            per_run.setdefault(f"{r['arm']}|{r['task']}", []).append((r["turn"], r["call_index"], s))
    seq = {k: [x[2] for x in sorted(v)] for k, v in per_run.items() if k not in bad}
    rng = np.random.default_rng(0)
    tasks = sorted({k.split("|")[1] for k in seq})
    draws = [np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks)) for _ in range(2000)]
    out = {"input": Path(path).name, "cell": cell, "errored_runs_dropped": sorted(bad), "windows": {}}
    for lo, hi in WINDOWS:
        runs = sorted(k for k, v in seq.items() if len(v) >= hi)
        arm = np.array([r.split("|")[0] for r in runs])
        ti = np.searchsorted(tasks, [r.split("|")[1] for r in runs])
        s = np.array([np.mean(seq[r][lo:hi]) for r in runs])

        def world(wname, w=None):
            a, h = WORLDS[wname]
            m = np.isin(arm, [a, h])
            return aucs((arm[m] == a).astype(int), s[m], None if w is None else w[ti[m]])[0]
        res = {}
        for wname, (a, h) in WORLDS.items():
            au = world(wname)
            res[wname] = {"auroc": None if np.isnan(au) else round(float(au), 4), "ci95": ci([world(wname, d) for d in draws]),
                          "runs": {a: int((arm == a).sum()), h: int((arm == h).sum())}}
        rng_n = np.random.default_rng([0, 9])
        for wname in ("real", "september", "real (honest_a replicate)"):
            a_, h_ = WORLDS[wname]
            m = np.flatnonzero(np.isin(arm, [a_, h_]))
            y0, s0, t0 = (arm[m] == a_).astype(int), s[m], ti[m]
            obs = aucs(y0, s0)[0]
            null = []
            for _ in range(1000):
                yy = y0.copy()
                for t in np.unique(t0):
                    i = np.flatnonzero(t0 == t)
                    yy[i] = rng_n.permutation(yy[i])
                null.append(aucs(yy, s0)[0])
            null = np.array(null)
            res[wname]["null"] = {"null_mean": round(float(np.nanmean(null)), 4), "p": round(float((1 + (null >= obs).sum()) / 1001), 4)}
        for g, (w1, w0) in GAPS.items():
            v = world(w1) - world(w0)
            res[f"gap {g}"] = {"value": None if np.isnan(v) else round(float(v), 4), "ci95": ci([world(w1, d) - world(w0, d) for d in draws])}
        out["windows"][f"outputs {lo + 1}-{hi}"] = res
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"attack_detector_positions_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    for w, r in out["windows"].items():
        print(f"{cell} | {w} | real {r['real']['auroc']} p {r['real']['null']['p']} | sept {r['september']['auroc']} p {r['september']['null']['p']}"
              f" | honest_a {r['real (honest_a replicate)']['auroc']} p {r['real (honest_a replicate)']['null']['p']} | gap sept-real {r['gap september - real']}")
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cell", default="L51_output_end")
    a = ap.parse_args()
    main(a.path, a.run_dir, a.cell)
