"""E1 fresh confirmation: the margin m and the price, fixed before any generation (PLAN.md here). From stage 1's
sentence-level result: the per-run SD of D (from its primary 95% interval over 14 runs) and the usable share (14 primary
pairs of 60 eligible runs). From the new index: eligible runs, tasks and context tokens. SE = SD / sqrt(N) x sqrt(DE), with
N = eligible x 14/60 and DE = 1 + rho q sum k(k-1)/sum k (rho 0.5 used, 1 beside), for tasks with two runs.
m = (1.96 + z_0.80) x SE, rounded up to 0.05. P(clear) is computed at true D = 0 (interval below +m) and true D = m
(interval above 0). Price from stage 1's measured bill per context token ($2.09 for 60 runs at 13.3k mean context,
loads included), with 30%. Numbers only; $0.

  uv run --no-project python experiments/2026-10-11_q2_e1_confirm/power_check.py <index summary json>
"""
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
S1 = HERE.parent / "2026-10-11_q2_e1_sentences" / "results" / "e1_sentences_2026-10-11_155202.json"
S1_SHA = "9b7f79cb1770995a9fc7142d1aa8bee4a884971bc166e9b257e3e17e3d25b31e"
PRIMARY = "primary (section-aware disclosure, amendment 5)"
USABLE, RATE_PER_1K = 14 / 60, 2.09 / (60 * 13.3)  # stage 1: primary pairs per eligible run; $ per 1k context tokens, loads included
Z = NormalDist()


def main(summary_p):
    import hashlib
    if hashlib.sha256(S1.read_bytes()).hexdigest() != S1_SHA:
        raise SystemExit("stage 1's sentence result differs")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    me = Path(__file__).resolve()
    if (subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode):
        raise SystemExit("the script differs from HEAD: commit it before running")
    s1 = json.loads(S1.read_text())["readout"][PRIMARY]
    lo, hi = s1["95%"]; sd = (hi - lo) / (2 * 1.96) * math.sqrt(s1["runs"])
    summ = json.loads(Path(summary_p).read_text())
    elig = [r for r in summ["runs"] if "problem" not in r]
    per_task = {}
    for r in elig:
        per_task[r["task"]] = per_task.get(r["task"], 0) + 1
    n = len(elig) * USABLE
    kk = sum(k * (k - 1) for k in per_task.values()) / sum(per_task.values())
    de = {rho: 1 + rho * USABLE * kk for rho in (0.5, 1.0)}  # size-weighted, for tasks with two runs
    se = sd / math.sqrt(n) * math.sqrt(de[0.5])
    m = math.ceil((1.96 + Z.inv_cdf(0.8)) * se * 20 - 1e-9) / 20
    p0 = Z.cdf((m - 1.96 * se) / se)  # true 0: upper bound below m
    pm = 1 - Z.cdf((1.96 * se - m) / se)  # true m: lower bound above 0
    se1 = sd / math.sqrt(n) * math.sqrt(de[1.0])
    tokens = summ["context tokens (min, median, max, total)"][3]
    raw = tokens / 1000 * RATE_PER_1K
    out = {"git": head, "stage-1 sentence result sha256": S1_SHA, "per-run SD of D (from stage 1's primary interval)": sd,
           "index summary sha256": hashlib.sha256(Path(summary_p).read_bytes()).hexdigest(), "index sha256": summ["index sha256"],
           "eligible runs": len(elig), "eligible tasks": len(per_task), "projected usable pairs": n,
           "design effect (rho 0.5, used; rho 1, beside)": [de[0.5], de[1.0]], "projected SE": se,
           "margin m (honest SD)": m, "P(clear) if true D = 0": p0, "P(clear) if true D = m": pm,
           "beside: at rho 1, P(clear) if true D = 0 / m": [Z.cdf((m - 1.96 * se1) / se1), 1 - Z.cdf((1.96 * se1 - m) / se1)],
           "launch allowed (both >= 0.80)": bool(p0 >= 0.80 and pm >= 0.80),
           "price raw ($)": raw, "price with 30% ($)": raw * 1.3, "watcher stop ($, the buffered estimate, never below)": math.ceil(raw * 1.3)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"power_check_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(sys.argv[1])
