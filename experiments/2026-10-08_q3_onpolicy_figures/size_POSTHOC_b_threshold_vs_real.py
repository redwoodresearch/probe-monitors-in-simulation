"""Sizing calculation for more real Qwen honest runs (POST HOC; not a result): split check (b)'s interval, over real run 1
+ A + B + C at the threshold set in simulation, into its two sources. 1,000 within-fold task resamples (seed
[20261011, 92, b]): both resampled; the real side only (threshold fixed at the point); the threshold only (real calls fixed,
the simulated calibration resampled). More real runs shrink only the real side. $0, local.

  uv run --no-project --with numpy python size_POSTHOC_b_threshold_vs_real.py <main per-call> <honest_scores.json> <honest_scores_C.json>
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from analyze_scaleup_plants import threshold

HERE = Path(__file__).parent


def main(main_path, honest_path, c_path):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True, cwd=HERE).stdout:
        raise SystemExit("the repo has uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    raws = {p: Path(p).read_bytes() for p in (main_path, honest_path, c_path)}
    M, H, C = (json.loads(raws[p]) for p in (main_path, honest_path, c_path))
    meta, folds = M["meta"], M["folds"]
    tasks = sorted({m["task"] for m in meta}); ti = {t: i for i, t in enumerate(tasks)}; fold_of = np.array([folds[t] for t in tasks])
    cal = M["draws"][0]["cells"]["real->real"]["per_call"]["calibration"]
    cs = [(ti[meta[i]["task"]], s, k) for i, s, k in zip(cal["row"], cal["score"], cal["model_fold"]) if meta[i]["world"] == "sim" and meta[i]["y"] == 0]
    c_t, c_s, c_k = (np.array(x) for x in zip(*cs))

    def ths(w):
        return np.array([threshold(c_s[(c_k == k) & (w[c_t] > 0)], w[c_t[(c_k == k) & (w[c_t] > 0)]]) for k in range(5)])

    pc = M["draws"][0]["cells"]["real->real"]["per_call"]
    R = [(ti[meta[i]["task"]], folds[meta[i]["task"]], s) for i, s in zip(pc["row"], pc["score"]) if meta[i]["world"] == "real" and meta[i]["y"] == 0]
    for src, w in ((H, "honA"), (H, "honB"), (C, "honC")):
        R += [(ti[x["task"]], x["fold"], x["s"][x["fold"]]) for x in src["rows"] if x["world"] == w and x["y"] == 0 and x["task"] in ti]
    t, f, s = (np.array(x) for x in zip(*R))
    one = np.ones(len(tasks)); th0 = ths(one)

    def rate(th, w):
        return (w[t] * (s > th[f])).sum() / w[t].sum()

    both, real_only, th_only = [], [], []
    for b in range(1, 1001):
        rng, w = np.random.default_rng([20261011, 92, b]), np.zeros(len(tasks))
        for k in range(5):
            idx = np.flatnonzero(fold_of == k)
            np.add.at(w, rng.choice(idx, len(idx), replace=True), 1)
        th = ths(w)
        both.append(rate(th, w)); real_only.append(rate(th0, w)); th_only.append(rate(th, one))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out = {"label": "POST HOC sizing calculation: (b)'s interval split into threshold vs real-run variation", "git": git,
           "inputs": {Path(p).name: hashlib.sha256(b).hexdigest() for p, b in raws.items()}, "read_only_calls": int(len(t)),
           "point": rate(th0, one), "both_resampled": ci(both), "real_side_only": ci(real_only), "threshold_only": ci(th_only)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"size_b_threshold_vs_real_POSTHOC_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out))
    print({k: (round(v * 100, 2) if isinstance(v, float) else [round(x * 100, 2) for x in v] if isinstance(v, list) else v) for k, v in out.items() if k not in ("inputs", "label", "git")})
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("main_path", "honest_path", "c_path"):
        ap.add_argument(a)
    a = ap.parse_args()
    main(a.main_path, a.honest_path, a.c_path)
