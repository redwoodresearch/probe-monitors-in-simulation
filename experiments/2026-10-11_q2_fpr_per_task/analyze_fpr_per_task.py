"""Per-task false alarms, real vs simulated, on Sol's off-policy runs (DEFINITIONS.md, committed a53bf153 before any
number; exploratory, $0). Harmless = read-only calls (y = 0). Primary: each fold's real-trained probe at its frozen real
1% threshold; secondary: the same probes at the threshold set on simulated read-only calls, and the sim-trained probes at
their own simulated threshold (threshold-in-sim's (ii) and (i)). Per-task mean (tasks with a harmless call in that world)
and pooled per-call rate, 2,000 task resamples (seed 0), thresholds fixed.

  uv run --no-project --with numpy python experiments/2026-10-11_q2_fpr_per_task/analyze_fpr_per_task.py <scores npz> <probes npz> <rows json>
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
W = ("real", "sim")


def main(scores_p: str, probes_p: str, rows_p: str):
    rows, S = json.loads(Path(rows_p).read_text()), np.load(scores_p)["scores"]
    tab = {(p["fold"], p["train"]): p for p in json.loads(str(np.load(probes_p)["table"]))}
    y = np.array([r["y"] for r in rows]); world = np.array([0 if r["arm"] == "real-attack-e1" else 1 for r in rows])
    fold = np.array([r["fold"] for r in rows]); names, task = np.unique([r["main_task"] for r in rows], return_inverse=True)
    harmless = y == 0
    out = {"inputs sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (scores_p, probes_p, rows_p)},
           "calls": len(y), "main tasks": len(names), "harmless calls": {w: int((harmless & (world == i)).sum()) for i, w in enumerate(W)}}
    setups = {"primary: real-trained probe, frozen real 1% threshold": (0, "threshold_real"),
              "(ii) real-trained probe, threshold set in simulation": (0, "threshold_sim"),
              "(i) sim-trained probe, its own simulated threshold": (1, "threshold_sim")}
    rng = np.random.default_rng(0); draws = rng.integers(0, len(names), (2000, len(names)))
    for name, (t, key) in setups.items():
        flag = np.zeros(len(y), bool)
        for k in range(5):
            te = fold == k
            flag[te] = S[k, t, te] > tab[(k, W[t])][key]
        F = np.zeros((2, len(names))); N = np.zeros((2, len(names)))
        for i in range(2):
            m = harmless & (world == i)
            np.add.at(F[i], task[m], flag[m]); np.add.at(N[i], task[m], 1)
        def stats(idx):
            f, n = F[:, idx], N[:, idx]
            has = n > 0; both = has.all(0)
            per_task = [float((f[i][has[i]] / n[i][has[i]]).mean()) for i in range(2)]
            diff = float((f[0][both] / n[0][both] - f[1][both] / n[1][both]).mean())
            pooled = [float(f[i].sum() / n[i].sum()) for i in range(2)]
            return per_task + [diff] + pooled + [pooled[0] - pooled[1]]
        point, boot = stats(np.arange(len(names))), np.array([stats(d) for d in draws])
        ci = lambda j: [round(float(v), 4) for v in np.percentile(boot[:, j], [2.5, 97.5])]
        cols = ["per-task real", "per-task sim", "per-task real - sim (tasks with both)", "pooled real", "pooled sim", "pooled real - sim"]
        out[name] = {c: {"point": round(point[j], 4), "95%": ci(j)} for j, c in enumerate(cols)} | {
            "tasks with a harmless call": {w: int((N[i] > 0).sum()) for i, w in enumerate(W)}, "tasks with both": int(((N > 0).all(0)).sum())}
    out["real run 1 vs run 2"] = "not available: these inputs hold only real run 1 (real-attack-e1)"
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"fpr_per_task_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("calls", "main tasks", "harmless calls")}))
    for name in setups:
        o = out[name]
        print(name, "|", "; ".join(f"{c} {v['point']:.4f} [{v['95%'][0]:.4f}, {v['95%'][1]:.4f}]" for c, v in o.items() if isinstance(v, dict) and "point" in v),
              "| tasks", o["tasks with a harmless call"], "both", o["tasks with both"])
    print(dest)


if __name__ == "__main__":
    main(*sys.argv[1:4])
