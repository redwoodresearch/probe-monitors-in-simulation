"""Zoo C re-sized after Aswin's picks (relayed by the orchestrator, just before 15:18 PT): the written DE rule (largest
Kish over the four counts), and new real runs NOT counted as matched real; they only add same-batch pairs. Reads only the
committed sizing result (counts and Kish), not the label file. With the matched and simulated counts fixed, DE can't go
below 1.2 x the matched runs' Kish, so real-vs-simulated's pass probability is fixed at its value there and bounds
P(agree). Expected counts for new pairs (today's per-run rates); DE at its floor, which can only overstate P. $0.

  uv run --no-project python experiments/2026-10-11_q2_new_real_runs/analyze_c_resize_no_matched.py results/c_sizing_2026-10-11_150659.json
"""
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
SHA = "6c23727988fb437d54247ebd767ddf40b9eed56ed84790c86870ea34b2da1761"
RULE, GOAL, AVAILABLE, OUTSIDE = "largest Kish (committed, as Amendment 1)", 0.80, 274, 101


def p_pass(n_a, n_b, z, de):
    se = math.sqrt(0.25 * de * (1 / n_a + 1 / n_b))
    return max(0.0, math.erf((0.10 - z * se) / se / math.sqrt(2)))


def main(path):
    p = HERE / path if not Path(path).is_absolute() else Path(path)
    if hashlib.sha256(p.read_bytes()).hexdigest() != SHA:
        raise SystemExit("sizing result differs from the committed one")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(Path(__file__).resolve())], cwd=HERE).returncode:
        raise SystemExit("the script differs from HEAD: commit it before running")
    d = json.loads(p.read_text()); n = d["gate counts today"]; kish = d["today"][RULE]["Kish"]
    de = max(1.25, 1.2 * kish["matched real"])  # the floor while matched real is unchanged
    p_rs = p_pass(n["matched real"], n["simulated"], 2.24, de)
    r1, r2 = n["run 1"] / 163, n["run 2"] / 163
    curve = {k: p_rs * p_pass(n["run 1"] + r1 * k, n["run 2"] + r2 * k, 1.96, de) for k in range(0, AVAILABLE + OUTSIDE + 1)}
    first = next((k for k, v in curve.items() if v >= GOAL), None)
    lo, hi = 0.0, 50.0  # yield per new run that 274 tasks would need, at the DE floor
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if p_rs * p_pass(n["run 1"] + m * AVAILABLE, n["run 2"] + m * AVAILABLE, 1.96, de) < GOAL else (lo, m)
    out = {"input sha256": SHA, "git": head, "DE floor": de, "P(real vs simulated passes), fixed": p_rs, "upper bound on P(agree)": p_rs,
           "P(real vs real) needed": GOAL / p_rs, "P(agree) upper bound by new tasks": {str(k): curve[k] for k in (0, 50, 100, 166, 233, 274, 375)},
           "first k reaching 0.80 at expected counts": first, "tasks available inside the simulated cohort": AVAILABLE,
           "positives per new run that 274 tasks would need": hi, "today's positives per run (run 1, run 2)": [r1, r2],
           "outcome": "too few even with spending" if (first is None or first > AVAILABLE) else f"buyable at {first} tasks"}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"c_resize_no_matched_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, separators=(",", ":")) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(sys.argv[1])
