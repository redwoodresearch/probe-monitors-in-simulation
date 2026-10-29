"""POST HOC, descriptive: does Qwen's P("simulated") depend on how many tokens it read, within real runs only?
Spearman correlation of the log-odds score with tokens read (n_tokens): part 1's originals and the bolster's la18 runs.
Also the bolster's Δ_wk on cells where the R0 run is at least 50% or 75% as long as the real run. Decides nothing.

  uv run --no-project --with numpy python experiments/2026-10-11_is_this_a_test/analyze_weaksim_length_POSTHOC.py
"""
import json, hashlib, sys
import numpy as np
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
HERE = Path(__file__).parent
ROOT = HERE.parent.parent
C = str(ROOT / ".run_cache" / "is_this_a_test") + "/"
OUT = {}
SIM, REAL = ["simulated", "Simulated", " simulated", " Simulated"], ["real", "Real", " real", " Real"]
lse = np.logaddexp.reduce
score = lambda lp: lse([lp[v] for v in SIM]) - lse([lp[v] for v in REAL])
rank = lambda x: np.argsort(np.argsort(x)).astype(float)
sp = lambda a, b: float(np.corrcoef(rank(np.array(a)), rank(np.array(b)))[0, 1])
p1 = json.load(open(C + "part1_rows_2026-10-11_225143.json"))
for arm in ("e1", "e2", "sim"):
    r = [x for x in p1 if x["version"] == "original" and x["arm"] == arm]
    OUT[f"part 1 {arm}: spearman(score, tokens)"] = {"n": len(r), "rho": sp([score(x["logprobs"]) for x in r], [x["n_tokens"] for x in r])}
w = json.load(open(C + "weaksim_rows_2026-10-12_004822.json"))
for arm in ("real", "r0"):
    r = [x for x in w if x["arm"] == arm]
    OUT[f"la18 {arm}: spearman(score, tokens)"] = {"n": len(r), "rho": sp([score(x["logprobs"]) for x in r], [x["n_tokens"] for x in r])}
V = {(x["cell"], x["arm"]): x for x in w}
cells = sorted({x["cell"] for x in w})
c = lambda a, b: (a > b) + 0.5 * (a == b)
for frac in (0.5, 0.75):
    sel = [x for x in cells if V[(x, "r0")]["n_tokens"] >= frac * V[(x, "real")]["n_tokens"]]
    d = [c(score(V[(x, "r0")]["logprobs"]), score(V[(x, "real")]["logprobs"])) - 0.5 for x in sel]
    OUT[f"la18 delta_wk, R0 at least {frac:.0%} of real's length"] = {"n": len(sel), "delta_wk": float(np.mean(d)) if d else None}
git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
for f in ("part1_rows_2026-10-11_225143.json", "weaksim_rows_2026-10-12_004822.json"):
    OUT[f"input sha256 {f}"] = hashlib.sha256(open(C + f, "rb").read()).hexdigest()
OUT["git"] = git
path = HERE / "results" / f"weaksim_length_POSTHOC_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
if path.exists():
    raise FileExistsError(path)
path.write_text(json.dumps(OUT, indent=1) + "\n")
print(json.dumps(OUT, indent=1), f"\n-> {path}")
