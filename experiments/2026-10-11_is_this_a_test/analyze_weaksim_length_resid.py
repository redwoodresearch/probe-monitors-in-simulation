"""Exploratory (PLAN.md amendment 7): does the readout see the weak simulator beyond transcript length? On the bolster's
rows: numbers and identifiers only, on the laptop ($0).

On real runs only, fit score = cubic in log(tokens read) + environment + cut (ordinary least squares), cross-fitted. Two
folds of cells by sha256("20261017|" + cell) parity: fit on one fold's real runs, predict both runs of the other
fold's cells. Residual = score - prediction, on part 1's log-odds score. A cell counts only if both its runs' log
lengths lie inside the range of the fitting fold's real runs, and its environment appears there.
Statistics over counted cells: the mean of resid_R0 - resid_real, and the rank version mean c(resid_R0, resid_real) -
0.5. Intervals: 2,000 cell bootstrap draws (seed 20261011), conditional on the fits, so descriptive.
Readings:
- "a positive gap after the cubic length adjustment": both lower ends above 0;
- "a negative gap after the cubic length adjustment": both upper ends below 0;
- "no gap detected after the cubic length adjustment": otherwise;
- "too few to tell": fewer than 30 counted cells.
Beside: the same on all cells (extrapolating); per fold, the design's rank, the fit's R², and the held-out real runs'
prediction error; and the counted cells per environment.

  uv run --no-project --with numpy python experiments/2026-10-11_is_this_a_test/analyze_weaksim_length_resid.py results/weaksim_ask_<stamp>.json
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
DRAWS, SEED, MIN_CELLS = 2000, 20261011, 30
SIM, REAL = ["simulated", "Simulated", " simulated", " Simulated"], ["real", "Real", " real", " Real"]
git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT)
st = git("status", "--porcelain", "--untracked-files=no")
if st.returncode or st.stdout.strip() or git("ls-files", "--error-unmatch", __file__).returncode:
    raise SystemExit("commit first: the result must record the commit of the code that made it")
res = json.loads((HERE / sys.argv[1]).read_text())
f = ROOT / ".run_cache" / "is_this_a_test" / res["rows"]["file"]
if hashlib.sha256(f.read_bytes()).hexdigest() != res["rows"]["sha256"]:
    raise ValueError("the rows file is not the collected one")
lse = np.logaddexp.reduce
V = {(r["cell"], r["arm"]): r | {"lo": float(lse([r["logprobs"][v] for v in SIM]) - lse([r["logprobs"][v] for v in REAL]))}
     for r in json.loads(f.read_text())}
cells = sorted({c for c, _ in V if (c, "real") in V and (c, "r0") in V})
fold = {c: int(hashlib.sha256(f"20261017|{c}".encode()).hexdigest(), 16) % 2 for c in cells}
envs = sorted({V[(c, "real")]["environment"] for c in cells})


def design(rows, mu, sd):
    x = (np.log([r["n_tokens"] for r in rows]) - mu) / sd
    E = np.array([[r["environment"] == e for e in envs[1:]] for r in rows], float)
    return np.column_stack([np.ones(len(rows)), x, x ** 2, x ** 3, E, [float(r["cut"]) for r in rows]])


resid, counted, r2 = {}, [], {}
for k in (0, 1):
    train = [V[(c, "real")] for c in cells if fold[c] == k]
    logn = np.log([r["n_tokens"] for r in train])
    mu, sd = logn.mean(), logn.std()
    X, y = design(train, mu, sd), np.array([r["lo"] for r in train])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    held = [V[(c, "real")] for c in cells if fold[c] != k]
    yh = np.array([r["lo"] for r in held])
    r2[k] = {"rank": int(np.linalg.matrix_rank(X)), "columns": X.shape[1], "R²": float(1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
             "held-out real RMSE": float(np.sqrt(((yh - design(held, mu, sd) @ beta) ** 2).mean())), "held-out real SD": float(yh.std())}
    seen_env = {r["environment"] for r in train}
    for c in (c for c in cells if fold[c] != k):
        for a in ("real", "r0"):
            resid[(c, a)] = float(V[(c, a)]["lo"] - (design([V[(c, a)]], mu, sd) @ beta)[0])
        if all(logn.min() <= np.log(V[(c, a)]["n_tokens"]) <= logn.max() for a in ("real", "r0")) and V[(c, "real")]["environment"] in seen_env:
            counted.append(c)
cmp = lambda a, b: (a > b) + 0.5 * (a == b)


def stats(cs):
    d = np.array([resid[(c, "r0")] - resid[(c, "real")] for c in cs])
    q = np.array([cmp(resid[(c, "r0")], resid[(c, "real")]) - 0.5 for c in cs])
    W = np.random.default_rng(SEED).multinomial(len(cs), np.full(len(cs), 1 / len(cs)), size=DRAWS)
    ci = lambda v: [float(x) for x in np.percentile(W @ v / len(cs), [2.5, 97.5])]
    return {"cells": len(cs), "mean residual gap (log-odds)": {"point": float(d.mean()), "ci": ci(d)},
            "rank version": {"point": float(q.mean()), "ci": ci(q)}}


main = stats(counted) if len(counted) >= MIN_CELLS else {"cells": len(counted)}
if len(counted) < MIN_CELLS:
    reading = "too few to tell"
elif main["mean residual gap (log-odds)"]["ci"][0] > 0 and main["rank version"]["ci"][0] > 0:
    reading = "a positive gap after the cubic length adjustment"
elif main["mean residual gap (log-odds)"]["ci"][1] < 0 and main["rank version"]["ci"][1] < 0:
    reading = "a negative gap after the cubic length adjustment"
else:
    reading = "no gap detected after the cubic length adjustment"
out = {"git": git("rev-parse", "HEAD").stdout.strip(), "input": sys.argv[1], "exploratory": True, "reading": reading, "counted cells": main,
       "beside": {"all cells (extrapolating)": stats(cells), "each fold's fit": r2,
                  "counted cells per environment": {e: sum(V[(c, "real")]["environment"] == e for c in counted) for e in envs}}}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = HERE / "results" / f"weaksim_length_resid_{stamp}.json"
if path.exists():
    raise FileExistsError(path)
path.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1), f"\n-> {path}")
