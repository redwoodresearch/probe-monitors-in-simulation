"""Power check for the comparison rule of both parts (PLAN.md section 4), before any of this plan's data exists ($0,
laptop; numbers only).

Q3.3's claim-B run scores stand in for the new readouts (`../2026-10-11_eval_awareness_real_vs_sim/`, rows checked
against its s3_manifest_v1.json): run score = mean over the run's scored calls, per readout. The acting set-up has one
cell per main task (190 tasks), so each world draws 190 main tasks and one cell from each. "Null worlds" then shuffle
the three arm labels within each cell, so real and simulated truly match, and apply the whole rule with 2,000 main-task
bootstrap draws and a 200-draw shuffle null:
- the yardstick: the interval of P(e1 > e2) - 0.5 must include 0, or no reading is made;
- "within": Δc (mean over cells of [c(sim, e1) + c(sim, e2)] / 2 - 0.5) within ±m, and R (mean (|sim - e1| +
  |sim - e2|) / 2 over mean |e2 - e1|) within [1/1.2, 1.2] - one R for part 1, one per cross-fit fold (two halves of
  the tasks) for part 2;
- "beyond": Δc's lower end above m (or upper below -m) with the point beyond the shuffle null's 97.5th (2.5th)
  percentile, or an R's lower end above 1.2 with its point above that R's shuffle 97.5th percentile.
"Shift worlds" add d times the SD of e2 - e1 to the simulated run, for the chance of "beyond".
Second rule, after the first run (results/power_2026-10-11_193125.json at b12ae0fa) showed that the R band can't
reach 80% at 190 tasks (P(R within) 0.58 with one R; 0.003 per fold): "within2" drops R from the "within" verdict
(yardstick, Δc within ±m and no R "beyond"), and R keeps deciding "farther than two real runs". Both rules are reported.
Rule for m, set before the run: per part, the smallest margin in the grid whose P(within) is at least 0.80 for both
stand-ins, at a Monte Carlo SE of at most 0.005 (8,000 worlds); if none reaches 0.80, the highest-power margin, and the
plan says the "within" reading has under 80% power.

  uv run --no-project --with numpy python experiments/2026-10-11_is_this_a_test/analyze_power.py <pilot rows> <rest rows>
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
Q33 = HERE.parent / "2026-10-11_eval_awareness_real_vs_sim"
WORLDS, DRAWS, NULL_DRAWS, SEED, N_TASKS, R_MARGIN = 8000, 2000, 200, 20261011, 190, 1.2
MARGINS = (0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.14, 0.16, 0.18, 0.20)
git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT)
st = git("status", "--porcelain", "--untracked-files=no")
if st.returncode or st.stdout.strip() or git("ls-files", "--error-unmatch", __file__).returncode:
    raise SystemExit("commit first: the result must record the commit of the code that made it")
man = json.loads((Q33 / "results" / "s3_manifest_v1.json").read_text())["run_cache"]
rows = []
for p in sys.argv[1:3]:
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
tasks_all = sorted(by_task)
S_all = {k: np.array([np.mean(runs[k][a], axis=0) for a in ("e1", "e2", "sim")]) for k in complete}
rng = np.random.default_rng(SEED)
W = rng.multinomial(N_TASKS, np.full(N_TASKS, 1 / N_TASKS), size=DRAWS).astype(float)  # one cell per task: tasks = cells
c = lambda a, b: (a > b) + 0.5 * (a == b)
dc_of = lambda Y: (c(Y[:, 2], Y[:, 0]) + c(Y[:, 2], Y[:, 1])) / 2 - 0.5
fold = np.arange(N_TASKS) % 2  # part 2's two cross-fit halves


def rule(Y, folds):
    dc = dc_of(Y)
    num, den = (np.abs(Y[:, 2] - Y[:, 0]) + np.abs(Y[:, 2] - Y[:, 1])) / 2, np.abs(Y[:, 1] - Y[:, 0])
    perms = [rng.permuted(np.tile([0, 1, 2], (len(Y), 1)), axis=1) for _ in range(NULL_DRAWS)]
    nulls = [np.take_along_axis(Y, p, 1) for p in perms]
    r_ok, r_beyond = True, False
    for f in sorted(set(folds)):
        mk = (folds == f).astype(float)
        rb = (W @ (num * mk)) / (W @ (den * mk))
        r_pt = (num * mk).sum() / (den * mk).sum()
        r_null = [((lambda Z: ((np.abs(Z[:, 2] - Z[:, 0]) + np.abs(Z[:, 2] - Z[:, 1])) / 2 * mk).sum() / (np.abs(Z[:, 1] - Z[:, 0]) * mk).sum())(Z)) for Z in nulls]
        lo, hi = np.percentile(rb, [2.5, 97.5])
        r_ok = r_ok and 1 / R_MARGIN <= lo and hi <= R_MARGIN
        r_beyond = r_beyond or (lo > R_MARGIN and r_pt > np.percentile(r_null, 97.5))
    db = (W @ dc) / N_TASKS
    yb = (W @ (c(Y[:, 0], Y[:, 1]) - 0.5)) / N_TASKS
    dnull = [dc_of(Z).mean() for Z in nulls]
    return {"dc": dc.mean(), "lo": np.percentile(db, 2.5), "hi": np.percentile(db, 97.5), "n_lo": np.percentile(dnull, 2.5),
            "n_hi": np.percentile(dnull, 97.5), "r_ok": r_ok, "r_beyond": r_beyond,
            "yard": np.percentile(yb, 2.5) <= 0 <= np.percentile(yb, 97.5)}


out = {"git": git("rev-parse", "HEAD").stdout.strip(), "rows": [Path(p).name for p in sys.argv[1:3]], "complete cells": len(complete),
       "main tasks available": len(tasks_all), "tasks per world": N_TASKS, "worlds": WORLDS, "draws": DRAWS, "null draws": NULL_DRAWS,
       "seed": SEED, "stand-in": "Q3.3 claim-B run scores (A content, B turn start), one cell per task"}
for part, folds in (("part 1 (one R)", np.zeros(N_TASKS, int)), ("part 2 (R per fold)", fold)):
    for j, name in ((0, "A content"), (1, "B turn start")):
        res = {}
        for d in (0.0, 0.25):
            ws = []
            for _ in range(WORLDS):
                pick = rng.choice(len(tasks_all), N_TASKS, replace=False)
                X = np.array([S_all[(by_task[tasks_all[i]][rng.integers(len(by_task[tasks_all[i]]))], tasks_all[i])][:, j] for i in pick])
                Y = np.take_along_axis(X, rng.permuted(np.tile([0, 1, 2], (N_TASKS, 1)), axis=1), 1)
                Y[:, 2] += d * float(np.std(X[:, 1] - X[:, 0]))
                ws.append(rule(Y, folds))
            row = {"mean dc": round(float(np.mean([w["dc"] for w in ws])), 4), "P(yardstick holds)": round(float(np.mean([w["yard"] for w in ws])), 4),
                   "P(R within)": round(float(np.mean([w["r_ok"] for w in ws])), 4)}
            for m in MARGINS:
                within = np.mean([w["yard"] and w["r_ok"] and -m <= w["lo"] and w["hi"] <= m for w in ws])
                within2 = np.mean([w["yard"] and not w["r_beyond"] and -m <= w["lo"] and w["hi"] <= m for w in ws])
                beyond = np.mean([w["yard"] and ((w["lo"] > m and w["dc"] > w["n_hi"]) or (w["hi"] < -m and w["dc"] < w["n_lo"]) or w["r_beyond"]) for w in ws])
                row[f"m {m}"] = {"P(within)": round(float(within), 4), "MC SE": round(float(np.sqrt(within * (1 - within) / WORLDS)), 4),
                                 "P(within2)": round(float(within2), 4), "MC SE2": round(float(np.sqrt(within2 * (1 - within2) / WORLDS)), 4),
                                 "P(beyond)": round(float(beyond), 4)}
            res[f"shift {d}"] = row
        out[f"{part}: {name}"] = res
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = HERE / "results" / f"power_{stamp}.json"
path.parent.mkdir(exist_ok=True)
if path.exists():
    raise FileExistsError(path)
path.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1), f"\n-> {path}")
