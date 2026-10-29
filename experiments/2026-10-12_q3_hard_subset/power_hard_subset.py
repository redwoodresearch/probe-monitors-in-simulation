"""Margin X for the hard-subset comparison, from a power check on SYNTHETIC outcomes (no probe score is read). $0, local.

Reads only the TF-IDF per-call draw-0 file (its meta, folds and frozen flags, which give the hard calls per task and
world; the probe file's meta and folds are identical, count_hard_subset.py). For each scenario (assumed probe catch p on
hard calls, within-task correlation rho) it makes S synthetic datasets with NO true gap: every task and world draws its
own catch probability from Beta(mean p, ICC rho) (rho = 0: p itself), independently across worlds, and its caught count
is Binomial(hard calls, that probability). Each dataset gets the analysis' interval: the scorer's own within-fold task
resamples (task_weights, seed [0, b], b = 1..B), the 2.5-97.5 percentiles of sim - real catch. need = max(|lo|, |hi|).
X80 = the 80th percentile of need over datasets (the smallest margin the interval sits inside with 80% chance).
X = the largest X80 over the scenarios, rounded up to the next half point. Thresholds are not modelled (fixed), so the
real analysis' threshold re-estimation can add width; the real-vs-sim gap shares one threshold per draw, which damps it.

  uv run --no-project --with numpy python experiments/2026-10-12_q3_hard_subset/power_hard_subset.py <tfidf draw-0 file>
"""
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
SHA = "112088a04a7e2cf376dae486ead8dd94cf86f64a0ac189377a4da3c6bdaf770f"  # manifest_textcatch_v1.json @ 251924f8
CELLS = {"real": "real->real", "sim": "real->sim", "run2": "real->run2"}
P_LEVELS = {"probe overall catch, real->real draw 0 (scaleup196 headline)": 0.8915766738660907, "maximum variance": 0.5}
RHOS = (0.0, 0.2)
S, B, SEED = 200, 1000, 20261012


def task_weights(fold_of_task, b, seed=0):
    """analyze_scaleup_plants.task_weights (run_score_onpolicy.task_weights), copied."""
    f = np.asarray(fold_of_task)
    if b == 0:
        return np.ones(len(f))
    rng, w = np.random.default_rng([seed, b]), np.zeros(len(f))
    for k in range(5):
        idx = np.flatnonzero(f == k)
        np.add.at(w, rng.choice(idx, len(idx), replace=True), 1)
    return w


def main(tfidf_p):
    if (got := hashlib.sha256(Path(tfidf_p).read_bytes()).hexdigest()) != SHA:
        raise SystemExit(f"TF-IDF file sha256 {got} differs from the manifest")
    T = json.loads(Path(tfidf_p).read_text())
    meta, folds = T["meta"], T["folds"]
    tasks = sorted({m["task"] for m in meta})
    ti = {t: i for i, t in enumerate(tasks)}
    n = {}
    for w, c in CELLS.items():
        p = T["draws"][0]["cells"][c]["per_call"]
        hard = [ti[meta[r]["task"]] for r, fl in zip(p["row"], p["flag_frozen"]) if meta[r]["y"] == 1 and not fl]
        n[w] = np.bincount(hard, minlength=len(tasks)).astype(float)
    W = np.stack([task_weights([folds[t] for t in tasks], b) for b in range(1, B + 1)])
    rng = np.random.default_rng(SEED)
    scen = []
    for pname, p in P_LEVELS.items():
        for rho in RHOS:
            need, need_y, in10 = [], [], []
            for _ in range(S):
                c = {}
                for w in n:
                    q = np.full(len(tasks), p) if rho == 0 else rng.beta(p * (1 / rho - 1), (1 - p) * (1 / rho - 1), len(tasks))
                    c[w] = rng.binomial(n[w].astype(int), q).astype(float)
                rate = {w: (W @ c[w]) / (W @ n[w]) for w in n}
                for key, (a, b_) in (("main", ("sim", "real")), ("yard", ("run2", "real"))):
                    lo, hi = np.percentile(rate[a] - rate[b_], [2.5, 97.5])
                    (need if key == "main" else need_y).append(max(abs(lo), abs(hi)))
                    if key == "main":
                        in10.append(-0.10 < lo and hi < 0.10)
            scen.append({"assumed catch": pname, "p": p, "rho": rho,
                         "X80 main (points)": round(100 * float(np.percentile(need, 80)), 3),
                         "X80 yardstick (points, descriptive)": round(100 * float(np.percentile(need_y, 80)), 3),
                         "P(main interval inside +-10)": round(float(np.mean(in10)), 3)})
    x = math.ceil(max(s["X80 main (points)"] for s in scen) * 2 - 1e-9) / 2
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(Path(__file__).resolve())], capture_output=True, text=True, cwd=HERE).stdout.strip()
    out = {"what": "margin X from synthetic outcomes (no probe score read)", "git": head, "script_uncommitted_changes": bool(dirty),
           "tfidf sha256": got, "hard calls": {w: int(v.sum()) for w, v in n.items()}, "tasks": len(tasks),
           "S datasets": S, "B resamples": B, "seed": SEED, "scenarios": scen, "X (points)": x}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"power_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(sys.argv[1])
