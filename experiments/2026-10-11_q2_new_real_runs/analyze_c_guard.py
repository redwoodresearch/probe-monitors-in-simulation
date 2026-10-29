"""Zoo C with the batch guard (AMENDMENT_C_GUARD.md, committed before this script): P(clear answer) = the committed
sizing's P(clear) before the guard x the guard's pass probability, at the plan's 233 tasks and at 274, for the same-task guard (the
plan) and an all-original-runs guard (beside). Reads only committed numbers and run IDs; $0.

  uv run --no-project python experiments/2026-10-11_q2_new_real_runs/analyze_c_guard.py results/c_sizing_2026-10-11_150659.json <recal_2026-10-10_225753.json>
"""
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
SHA = {"c_sizing": "6c23727988fb437d54247ebd767ddf40b9eed56ed84790c86870ea34b2da1761",
       "recal": "c9ba6dfe3bd028027e162ae386dc5c4ec5343eff5825d63c4ed94534b2edb9e7"}
UNPAIRED_SHA = "b735a5e06016a59df327b1ad6b7b373209a8a1dfb891853d943cf9eb87414c3f"
RULE, MODELS, SIZES, GOAL = "largest Kish (committed, as Amendment 1)", ("pessimistic (own task's runs)", "pair donor (today's 163 pairs)"), (233, 274), 0.80
RUN_RAW, BUF, BATCH, FIT = 1.00, 1.30, 1.11, 3.00


def p_pass(n_a, n_b, de):
    se = math.sqrt(0.25 * de * (1 / n_a + 1 / n_b))
    return max(0.0, math.erf((0.10 - 1.96 * se) / se / math.sqrt(2)))


def main(sizing_p, recal_p):
    sp = HERE / sizing_p if not Path(sizing_p).is_absolute() else Path(sizing_p)
    got = {"c_sizing": hashlib.sha256(sp.read_bytes()).hexdigest(), "recal": hashlib.sha256(Path(recal_p).read_bytes()).hexdigest()}
    if got != SHA:
        raise SystemExit(f"inputs differ: {got}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(Path(__file__).resolve())], cwd=HERE).returncode:
        raise SystemExit("the script differs from HEAD: commit it before running")
    sz = json.loads(sp.read_text())
    rate = sz["matched positives per run"]
    batch = lambda run: "overlap800" if "|overlap800_" in run else "base200" if "|base200_" in run else "other"
    runs, sim = defaultdict(set), set()
    for r in json.loads(Path(recal_p).read_text())["rows"]:
        (runs[r["task"]].add(r["run"]) if r["world"] == 0 else sim.add(r["task"]))
    first2 = {t: sorted(v)[:2] for t, v in runs.items() if len(v) >= 2}
    same = {t for t, (a, b) in first2.items() if batch(a) == batch(b) == "overlap800"}
    unpaired = sorted(sim - same)
    if hashlib.sha256("\n".join(unpaired).encode()).hexdigest() != UNPAIRED_SHA:
        raise SystemExit("unpaired list differs")
    o800_unpaired = sum(batch(x) == "overlap800" for t in unpaired for x in runs[t])
    curve = sz["sizing"][RULE]
    exact = {233: curve["size"]["per model at that k"], 274: curve["at k = 274"]}  # the sizing's exact records at these sizes
    for k, recs in exact.items():
        if any(recs[mo]["k"] != k for mo in MODELS):
            raise SystemExit(f"no exact sizing record at k = {k}")
    out = {"inputs sha256": got, "git": head, "matched positives per run": rate, "original overlap800 runs on the 274 unpaired tasks": o800_unpaired, "sizes": {}}
    for k in SIZES:
        n_new = 2 * k * rate
        res = {"new positives (expected)": n_new, "price raw": 2 * k * RUN_RAW + BATCH + FIT, "price with 30%": (2 * k * RUN_RAW + BATCH + FIT) * BUF}
        for guard, n_orig in (("same tasks (the plan)", k / len(unpaired) * o800_unpaired * rate), ("all 800 original runs (beside)", 800 * rate)):
            res[guard] = {"original positives (expected)": n_orig}
            for mo in MODELS:
                de = exact[k][mo]["median DE (real-sim, real-real)"][1]
                pg = p_pass(n_new, n_orig, de); pc = curve["P(clear) per k (both models)"][mo][k - 1]
                res[guard][mo] = {"DE": de, "P(guard passes)": pg, "P(clear before batch guard)": pc, "P(clear)": pc * pg}
        out["sizes"][str(k)] = res
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"c_guard_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, separators=(",", ":")) + "\n")
    for k, res in out["sizes"].items():
        for guard in ("same tasks (the plan)", "all 800 original runs (beside)"):
            print(k, guard, {mo: (round(v["P(guard passes)"], 3), round(v["P(clear)"], 3)) for mo, v in res[guard].items() if mo in MODELS},
                  f"${res['price raw']:.0f} raw / ${res['price with 30%']:.0f}")
    print(dest)


if __name__ == "__main__":
    main(*sys.argv[1:3])
