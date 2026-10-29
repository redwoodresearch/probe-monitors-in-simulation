"""Power check for PLAN.md amendment 8 (ask Qwen3.5 on its own acting runs), before any of its data exists ($0, laptop;
numbers only). Same stand-in, worlds and rule for m as analyze_power.py (Q3.3's claim-B run scores, rows checked against
its s3_manifest_v1.json; null worlds shuffle the three arm labels within each cell; shift worlds add d times the SD of
e2 - e1 to the simulated run), for three designs at N_TASKS main tasks:
- **honest, against run 2 only.** Phase 2's simulated runs were built from honest_a (the 2026-10-11 source-run audit), so
  the simulated run is compared with honest_b alone: Δc = mean c(sim, b) - 0.5, R = mean |sim - b| / mean |a - b|.
  The yardstick is P(a > b) - 0.5, as before. One cell per task.
- **attack, the standard rule** (both real attack runs are clean comparators): as analyze_power.py's part 1.
- **honest and attack pooled:** two cells per task, the honest one scored as above and the attack one by the standard
  rule. Δc and the yardstick are means over cells, R is the summed numerators over the summed denominators, and
  bootstrap draws resample tasks. Stand-in tasks need two complete cells (215 of 501), so worlds overlap more.
Both verdict rules are reported: "within" (yardstick, Δc within ±m, R within [1/1.2, 1.2]) and "within2" (R dropped
from "within"; R still decides "beyond"). Rule for m, set before the run: per design, in the no-shift worlds (d = 0),
the smallest margin in the grid whose P(within2) is at least 0.80 with MC SE2 at most 0.005, for both stand-ins
(8,000 worlds). If no margin qualifies, that design gets no m: the plan says its "within" reading has under 80% power
at any margin up to 0.20, and the design is not proposed for launch. P(within) and P(beyond) at d = 0.25 are reported
beside it.

  uv run --no-project --with numpy python experiments/2026-10-11_is_this_a_test/analyze_power_acting_own.py <pilot rows> <rest rows> <n tasks>
"""

import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
Q33 = HERE.parent / "2026-10-11_eval_awareness_real_vs_sim"
WORLDS, DRAWS, NULL_DRAWS, R_MARGIN = 8000, 2000, 200, 1.2
MARGINS = (0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.14, 0.16, 0.18, 0.20)
DESIGNS = ("honest, against run 2 only", "attack, the standard rule", "honest and attack pooled")
c = lambda a, b: (a > b) + 0.5 * (a == b)


def load(paths):
    man = json.loads((Q33 / "results" / "s3_manifest_v1.json").read_text())["run_cache"]
    rows = []
    for p in paths:
        raw = Path(p).read_bytes()
        if hashlib.sha256(raw).hexdigest() != man[Path(p).name]["sha256"]:
            raise ValueError(f"{p} is not Q3.3's archived rows")
        rows += json.loads(raw)
    runs = {}
    for r in rows:
        if r["version"] == "original":
            runs.setdefault((r["cell"], r["main_task"]), {}).setdefault(r["arm"], []).append([r["score_A"], r["score_B"]])
    complete = sorted(k for k, v in runs.items() if set(v) >= {"e1", "e2", "sim"})
    by_task = {}
    for cell, task in complete:
        by_task.setdefault(task, []).append(cell)
    S = {k: np.array([np.mean(runs[k][a], axis=0) for a in ("e1", "e2", "sim")]) for k in complete}
    return S, by_task, len(complete)


def parts(Y, kind):
    """Per cell: Δc contribution, yardstick contribution, R numerator, R denominator. Columns: e1 (a), e2 (b), sim."""
    if kind == "honest":
        return c(Y[:, 2], Y[:, 1]) - 0.5, c(Y[:, 0], Y[:, 1]) - 0.5, np.abs(Y[:, 2] - Y[:, 1]), np.abs(Y[:, 1] - Y[:, 0])
    return ((c(Y[:, 2], Y[:, 0]) + c(Y[:, 2], Y[:, 1])) / 2 - 0.5, c(Y[:, 0], Y[:, 1]) - 0.5,
            (np.abs(Y[:, 2] - Y[:, 0]) + np.abs(Y[:, 2] - Y[:, 1])) / 2, np.abs(Y[:, 1] - Y[:, 0]))


def rule(Ys, kinds, n_tasks, W, rng):
    """Ys: one (n_tasks, 3) array per cell slot, kinds: its scoring. Task-level sums, so draws resample tasks."""
    def stats(Zs):
        p = [parts(Z, k) for Z, k in zip(Zs, kinds)]
        return [sum(q[i] for q in p) for i in range(4)]  # per-task sums over the task's cells
    dc, yd, num, den = stats(Ys)
    k = len(Ys)
    nulls = [[np.take_along_axis(Z, rng.permuted(np.tile([0, 1, 2], (n_tasks, 1)), axis=1), 1) for Z in Ys] for _ in range(NULL_DRAWS)]
    ns = [stats(Z) for Z in nulls]
    db, yb, rb = W @ dc / (n_tasks * k), W @ yd / (n_tasks * k), (W @ num) / (W @ den)
    lo, hi = np.percentile(db, [2.5, 97.5])
    rlo, rhi = np.percentile(rb, [2.5, 97.5])
    r_pt = num.sum() / den.sum()
    dnull = [z[0].sum() / (n_tasks * k) for z in ns]
    rnull = [z[2].sum() / z[3].sum() for z in ns]
    return {"dc": dc.sum() / (n_tasks * k), "lo": lo, "hi": hi, "n_lo": np.percentile(dnull, 2.5), "n_hi": np.percentile(dnull, 97.5),
            "r_ok": 1 / R_MARGIN <= rlo and rhi <= R_MARGIN, "r_beyond": rlo > R_MARGIN and r_pt > np.percentile(rnull, 97.5),
            "yard": np.percentile(yb, 2.5) <= 0 <= np.percentile(yb, 97.5)}


def job(args):
    design, j, name, d, paths, n_tasks = args
    S, by_task, _ = load(paths)
    rng = np.random.default_rng(int(hashlib.sha256(f"20261012|{design}|{name}|{d}".encode()).hexdigest()[:8], 16))
    W = rng.multinomial(n_tasks, np.full(n_tasks, 1 / n_tasks), size=DRAWS).astype(float)
    pooled = design == "honest and attack pooled"
    pool = sorted(t for t, cs in by_task.items() if len(cs) >= 2) if pooled else sorted(by_task)
    kinds = ("honest", "attack") if pooled else ("honest",) if design.startswith("honest") else ("attack",)
    ws = []
    for _ in range(WORLDS):
        pick = rng.choice(len(pool), n_tasks, replace=False)
        cells = [rng.choice(len(by_task[pool[i]]), len(kinds), replace=False) for i in pick]
        Ys = []
        for s in range(len(kinds)):
            X = np.array([S[(by_task[pool[i]][cs[s]], pool[i])][:, j] for i, cs in zip(pick, cells)])
            Y = np.take_along_axis(X, rng.permuted(np.tile([0, 1, 2], (n_tasks, 1)), axis=1), 1)
            Y[:, 2] += d * float(np.std(X[:, 1] - X[:, 0]))
            Ys.append(Y)
        ws.append(rule(Ys, kinds, n_tasks, W, rng))
    row = {"mean dc": round(float(np.mean([w["dc"] for w in ws])), 4), "P(yardstick holds)": round(float(np.mean([w["yard"] for w in ws])), 4),
           "P(R within)": round(float(np.mean([w["r_ok"] for w in ws])), 4)}
    for m in MARGINS:
        within = np.mean([w["yard"] and w["r_ok"] and -m <= w["lo"] and w["hi"] <= m for w in ws])
        within2 = np.mean([w["yard"] and not w["r_beyond"] and -m <= w["lo"] and w["hi"] <= m for w in ws])
        beyond = np.mean([w["yard"] and ((w["lo"] > m and w["dc"] > w["n_hi"]) or (w["hi"] < -m and w["dc"] < w["n_lo"]) or w["r_beyond"]) for w in ws])
        row[f"m {m}"] = {"P(within)": round(float(within), 4), "MC SE": round(float(np.sqrt(within * (1 - within) / WORLDS)), 4),
                         "P(within2)": round(float(within2), 4), "MC SE2": round(float(np.sqrt(within2 * (1 - within2) / WORLDS)), 4),
                         "P(beyond)": round(float(beyond), 4)}
    return design, name, d, row


if __name__ == "__main__":
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT)
    st = git("status", "--porcelain", "--untracked-files=no")
    if st.returncode or st.stdout.strip() or git("ls-files", "--error-unmatch", __file__).returncode:
        raise SystemExit("commit first: the result must record the commit of the code that made it")
    paths, n_tasks = sys.argv[1:3], int(sys.argv[3])
    _, by_task, n_cells = load(paths)
    jobs = [(dz, j, nm, d, paths, n_tasks) for dz in DESIGNS for j, nm in ((0, "A content"), (1, "B turn start")) for d in (0.0, 0.25)]
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as ex:
        res = list(ex.map(job, jobs))
    out = {"git": git("rev-parse", "HEAD").stdout.strip(), "rows": [Path(p).name for p in paths], "complete cells": n_cells,
           "main tasks available": len(by_task), "tasks with two complete cells": sum(len(v) >= 2 for v in by_task.values()),
           "tasks per world": n_tasks, "worlds": WORLDS, "draws": DRAWS, "null draws": NULL_DRAWS,
           "stand-in": "Q3.3 claim-B run scores (A content, B turn start)"}
    for design, name, d, row in res:
        out.setdefault(f"{design}: {name}", {})[f"shift {d}"] = row
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"power_acting_own_{stamp}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if ":" not in k}, indent=1), f"\n-> {path}")
