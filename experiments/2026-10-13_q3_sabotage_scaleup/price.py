"""Price and expected counts for the scale-up panel run, by task-list prefix (IDs and numbers only).

Rates per packet-judge pair come from what stage 1 actually spent on the same kind of packets (sabotage only, every call labelled).
They are the sums Aswin's two re-read exports printed for the 120 stage-1 packets, relayed verbatim by 0d on 2026-10-13:
- simulated: 218 attempts, reported $22.5692, charged or reserved $28.0821;
- real runs 1 and 2: 424 attempts, $46.5336, $52.3334.
There are 600 packet-judge pairs, retries included. The rates the precision fork used, $0.0817 (both datasets' charged-or-reserved
average) and $0.1023 (the combined pilot), are shown beside them for comparison. Each new task is 3 runs × 5 judges = 15 pairs.
The call-count ratio scales the price for packet size: mean calls per run in the prefix over stage 1's 8,369 / 120.

  uv run --no-project --python 3.12 python price.py --runs <runs csv> --calls <calls_all csv> --out <dir>
"""

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

RUNS_SHA = "52a2e288b1cc684d8b17016adda88778fb5077388961ccd6aa3bdcf4c915accf"  # runs_2026-10-13_125944.csv
CALLS_SHA = "6dc132affb389e3dd66737d2d5db199c2bb5fe70e13a723a3c0af319e95d7595"  # calls_all_2026-10-13_125944.csv
PAIRS, REPORTED, CHARGED = 600, 22.5692 + 46.5336, 28.0821 + 52.3334
RATES = {"stage1_sabotage_reported": REPORTED / PAIRS, "stage1_sabotage_charged_or_reserved": CHARGED / PAIRS,
         "precision_fork_central": 0.0817, "precision_fork_high": 0.1023}
STAGE1_CALLS_PER_RUN = 8369 / 120  # results/extraction/preparation.json at b90cd42a
PREFIXES = {"pooled_reference_for_15": 48, "real1_vs_sim_for_15": 80, "all_held_out": 116}  # new tasks; item (a) placeholder: 64 / 96 minus the 16 labelled


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(runs: str, calls: str, out: str):
    if sha256(runs) != RUNS_SHA or sha256(calls) != CALLS_SHA:
        raise SystemExit("input sha256 mismatch")
    R = list(csv.DictReader(open(runs)))
    per_run = Counter()
    sc = Counter()
    for c in csv.DictReader(open(calls)):
        per_run[c["run_id"]] += 1
        sc[c["run_id"]] += c["state_changing"] == "1"
    res = {"rates_per_pair_usd": {k: round(v, 4) for k, v in RATES.items()}, "stage1_calls_per_run": round(STAGE1_CALLS_PER_RUN, 2), "prefixes": {}}
    for name, n in PREFIXES.items():
        rs = [r for r in R if r["side"] == "held_out" and int(r["rank"]) <= n]
        if len({r["task"] for r in rs}) != n or len(rs) != 3 * n:
            raise SystemExit(f"prefix {n}: expected {n} tasks and {3 * n} runs")
        calls_n = sum(per_run[r["run_id"]] for r in rs)
        ratio = calls_n / len(rs) / STAGE1_CALLS_PER_RUN
        pairs = 15 * n
        res["prefixes"][name] = {
            "new_tasks": n, "runs": len(rs), "pairs": pairs, "calls": calls_n,
            "state_changing_by_world": {w: sum(sc[r["run_id"]] for r in rs if r["world"] == w) for w in ("real", "run2", "sim")},
            "calls_by_world": {w: sum(per_run[r["run_id"]] for r in rs if r["world"] == w) for w in ("real", "run2", "sim")},
            "size_ratio": round(ratio, 3),
            "usd": {k: {"raw": round(pairs * v * ratio, 2), "with_30pct": round(1.3 * pairs * v * ratio, 2)} for k, v in RATES.items()}}
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()  # noqa: E731
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = {"written": stamp, "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "."))},
           "inputs_sha256": {"runs": RUNS_SHA, "calls": CALLS_SHA}} | res
    with (Path(out) / f"price_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("--runs", "--calls", "--out"):
        ap.add_argument(a, required=True)
    main(**vars(ap.parse_args()))
