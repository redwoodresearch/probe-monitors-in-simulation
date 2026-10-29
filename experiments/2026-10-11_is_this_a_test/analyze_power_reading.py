"""Power check for the reading set-up (Qwen3-32B on claim B's cells), for PLAN.md's comparison rule ($0, laptop).

Adapted from ../2026-10-11_tool_output_cue_probe/analyze_power.py (reviewed): all of claim B's complete cells (no
tasks held out, since this plan has none), task-clustered bootstrap and per-task arm permutation as before; part 1 one
R over all cells, part 2 one R per fold (fold B holds a task when int(sha256("20261011|" + task), 16) is odd).

Q3.3's claim-B run scores stand in for the new readouts (`../2026-10-11_eval_awareness_real_vs_sim/`, rows checked
against its s3_manifest_v1.json): run score = mean over the run's scored calls, per readout. "Null worlds" shuffle the
three arm labels with one permutation per main task (as the registered null), so the simulated run is exchangeable
with the real ones; each world gets the whole rule, with 2,000 main-task bootstrap draws and a 200-draw task-level
shuffle null:
- the yardstick: the interval of P(e1 > e2) - 0.5 must include 0, or no reading is made;
- "within" (full rule): Δc (mean over cells of [c(sim, e1) + c(sim, e2)] / 2 - 0.5) within ±m, and each R (mean
  (|sim - e1| + |sim - e2|) / 2 over mean |e2 - e1|) within [1/1.2, 1.2];
- "within2": the yardstick, Δc within ±m and no R "beyond";
- "beyond": Δc's lower end above m (or upper below -m) with the point beyond the shuffle null's 97.5th (2.5th)
  percentile, or an R's lower end above 1.2 with its point above that R's shuffle 97.5th percentile.
"Shift worlds" add d times the SD of e2 - e1 to the simulated run after the shuffle, for the chance of "beyond".
Rule for the margin, set before this 4,000-world run: m is the smallest margin in 0.05-0.10 whose P(within2) is at
least 0.80 for both stand-ins, at a Monte Carlo SE of at most 0.005. If none reaches 0.80, the margin with the highest
power is used and the plan says the "within" reading has under 80% power.
(This docstring was corrected after the run at c31ff141, whose docstring carried stale lines from the cue probe's
script: its fold and P-task wording, an 8,000-world count, P(within) in the margin rule, and the run path. The code is
unchanged.)

  uv run --no-project --with numpy python experiments/2026-10-11_is_this_a_test/analyze_power_reading.py <pilot rows> <rest rows>
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
WORLDS, DRAWS, NULL_DRAWS, SEED, MARGINS, R_MARGIN = 4000, 2000, 200, 20261011, (0.05, 0.06, 0.07, 0.08, 0.09, 0.10), 1.2
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
sets = json.loads((HERE.parent / "2026-10-11_la_format_sim_probe" / "results" / "build_split_2026-10-11_104954.json").read_text())["sets"]
p_tasks = set()
runs = {}
for r in rows:
    if r["version"] == "original" and r["main_task"] not in p_tasks:
        runs.setdefault((r["cell"], r["main_task"]), {}).setdefault(r["arm"], []).append([r["score_A"], r["score_B"]])
cells = sorted(k for k, v in runs.items() if set(v) >= {"e1", "e2", "sim"})
tasks = sorted({t for _, t in cells})
ti = np.array([tasks.index(t) for _, t in cells])
fold = np.array([int(hashlib.sha256(f"20261011|{t}".encode()).hexdigest(), 16) % 2 == 1 for _, t in cells])  # every task, by sha256
S = np.array([[np.mean(runs[k][a], axis=0) for a in ("e1", "e2", "sim")] for k in cells])  # cells x arms x readouts
rng = np.random.default_rng(SEED)
W = rng.multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), size=DRAWS).astype(float)  # task bootstrap weights
cnt = np.bincount(ti, minlength=len(tasks)).astype(float)
c = lambda a, b: (a > b) + 0.5 * (a == b)


def boot(per_cell):  # bootstrap means of a per-cell quantity, tasks resampled with their cells
    return (W @ np.bincount(ti, per_cell, len(tasks))) / (W @ cnt)


def permute(X):  # one arm permutation per main task, applied to all its cells
    perm = rng.permuted(np.tile([0, 1, 2], (len(tasks), 1)), axis=1)[ti]
    return np.take_along_axis(X, perm, axis=1)


def rule(Y, folds):
    e1, e2, sim = Y[:, 0], Y[:, 1], Y[:, 2]
    dc = (c(sim, e1) + c(sim, e2)) / 2 - 0.5
    rparts = lambda Z: ((np.abs(Z[:, 2] - Z[:, 0]) + np.abs(Z[:, 2] - Z[:, 1])) / 2, np.abs(Z[:, 1] - Z[:, 0]))
    num, den = rparts(Y)
    nulls = [permute(Y) for _ in range(NULL_DRAWS)]
    r_ok, r_above = True, False
    for f in sorted(set(folds.tolist())):  # one R per fold (part 2) or one over all cells (part 1)
        mk = (folds == f).astype(float)
        bn, bd = W @ np.bincount(ti, num * mk, len(tasks)), W @ np.bincount(ti, den * mk, len(tasks))
        r_lo, r_hi = np.percentile(bn / bd, [2.5, 97.5])
        r_pt = (num * mk).sum() / (den * mk).sum()
        r_null = [(lambda nd: (nd[0] * mk).sum() / (nd[1] * mk).sum())(rparts(Z)) for Z in nulls]
        r_ok = r_ok and 1 / R_MARGIN <= r_lo and r_hi <= R_MARGIN
        r_above = r_above or (r_lo > R_MARGIN and r_pt > np.percentile(r_null, 97.5))
    d_lo, d_hi = np.percentile(boot(dc), [2.5, 97.5])
    y_lo, y_hi = np.percentile(boot(c(e1, e2) - 0.5), [2.5, 97.5])
    nul = [((c(Z[:, 2], Z[:, 0]) + c(Z[:, 2], Z[:, 1])) / 2 - 0.5).mean() for Z in nulls]
    n_lo, n_hi = np.percentile(nul, [2.5, 97.5])
    return {"dc": dc.mean(), "dc_lo": d_lo, "dc_hi": d_hi, "n_lo": n_lo, "n_hi": n_hi, "R_within": r_ok, "R_above": r_above,
            "yard_ok": y_lo <= 0 <= y_hi}


out = {"git": git("rev-parse", "HEAD").stdout.strip(), "rows": [Path(p).name for p in sys.argv[1:3]], "cells": len(cells),
       "main tasks": len(tasks), "cells per fold (A, B)": [int((~fold).sum()), int(fold.sum())], "worlds": WORLDS, "draws": DRAWS,
       "null draws": NULL_DRAWS, "seed": SEED, "stand-in": "Q3.3 run scores (A content, B turn start), all complete claim-B cells"}
for part, folds in (("part 1 (one R)", np.zeros(len(cells), bool)), ("part 2 (R per fold)", fold)):
    for j, name in ((0, "A content"), (1, "B turn start")):
        X = S[:, :, j]
        sd_rr = float(np.std(X[:, 1] - X[:, 0]))
        res = {}
        for d in (0.0, 0.25):
            worlds = []
            for _ in range(WORLDS):
                Y = permute(X)
                Y[:, 2] += d * sd_rr
                worlds.append(rule(Y, folds))
            row = {"mean dc": round(float(np.mean([w["dc"] for w in worlds])), 4), "P(yardstick holds)": round(float(np.mean([w["yard_ok"] for w in worlds])), 4),
                   "P(R within)": round(float(np.mean([w["R_within"] for w in worlds])), 4)}
            for m in MARGINS:
                w2 = np.mean([w["yard_ok"] and not w["R_above"] and -m <= w["dc_lo"] and w["dc_hi"] <= m for w in worlds])
                w1 = np.mean([w["yard_ok"] and w["R_within"] and -m <= w["dc_lo"] and w["dc_hi"] <= m for w in worlds])
                b = np.mean([w["yard_ok"] and ((w["dc_lo"] > m and w["dc"] > w["n_hi"]) or (w["dc_hi"] < -m and w["dc"] < w["n_lo"]) or w["R_above"]) for w in worlds])
                row[f"m {m}"] = {"P(within)": round(float(w1), 4), "P(within2)": round(float(w2), 4),
                                 "MC SE2": round(float(np.sqrt(w2 * (1 - w2) / WORLDS)), 4), "P(beyond)": round(float(b), 4)}
            res[f"shift {d}"] = row
        out[f"{part}: {name}"] = res
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = HERE / "results" / f"power_{stamp}.json"
path.parent.mkdir(exist_ok=True)
if path.exists():
    raise FileExistsError(path)
path.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1), f"\n-> {path}")
