"""POST HOC, descriptive ($0, numbers only; 0d's request): plan B's strict alarm with intervals, on the 200 test cells.

1. At the 1% target: catch(Sol) - catch(probe), each detector's threshold set exactly as analyze_sol_b.catch sets it
   (alarm above the (floor(0.01 n) + 1)-th highest real-e1 score), and redone inside every draw on the drawn negatives.
   Achieved false-alarm rates are reported with their own intervals.
2. Equal alarm: Sol's scores take only 4 values (0, 1/3, 2/3, 1), so its 1% operating point lands at 0 false alarms.
   Both detectors at 0 false alarms (alarm above each one's highest real-e1 score), and the difference.
Intervals: percentile 95% of 2,000 draws resampling main tasks with replacement, default_rng(20261013).

  uv run --no-project --python 3.12 --with numpy --with scipy python \\
      experiments/2026-10-11_claimB_judge_sweep/analyze_POSTHOC_alarm_ci.py <pick.json> <test calls.jsonl> <probe_scores.json.gz>
"""

import gzip
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import analyze_sol_b as A  # noqa: E402  (scores and catch, unchanged)


def zero_fa(pos, neg):
    t = max(neg)
    return int(sum(p > t for p in pos)), int(sum(n > t for n in neg))


def main(pick_path, calls_path, probe_path):
    setup = json.loads(Path(pick_path).read_text())["pick"]
    sel = json.loads(subprocess.run(["git", "show", "58ce4715:experiments/2026-10-11_claimB_judge/results/judge_run_selection.json"],
                                    capture_output=True, text=True, check=True, cwd=HERE).stdout)
    cells, key = sel["cells"], sel["run_keys"]
    s = A.scores([json.loads(line) for line in Path(calls_path).read_text().splitlines()], set(cells), setup, "main")
    rows = defaultdict(list)
    for r in json.loads(gzip.decompress(Path(probe_path).read_bytes()))["levels"]["level1"]["rows"]:
        rows[r["key"]].append(r["score"])
    probe = {k: float(np.mean(v)) for k, v in rows.items()}
    det = {"sol": lambda c, a: s.get((c, a)), "probe": lambda c, a: probe[key[f"{c}|{a}"]] if (c, a) in s else None}

    def measure(cs):
        out = {}
        for name, f in det.items():
            pos = [v for v in (f(c, A.SIM) for c in cs) if v is not None]
            neg = [v for v in (f(c, A.E1) for c in cs) if v is not None]
            k = A.catch(pos, neg, 0.01)
            z = zero_fa(pos, neg)
            out[name] = {"catch": k["caught"] / k["positives"], "fpr": k["fpr"], "caught": k["caught"], "false_alarms": k["false_alarms"],
                         "positives": k["positives"], "negatives": k["negatives"], "catch_0fa": z[0] / len(pos), "caught_0fa": z[0]}
        return out

    point = measure(cells)
    by = defaultdict(list)
    for c in cells:
        by[c.split(":")[0]].append(c)
    tasks, rng, draws = sorted(by), np.random.default_rng(20261013), []
    for _ in range(2000):
        draws.append(measure([c for t in rng.choice(len(tasks), len(tasks)) for c in by[tasks[t]]]))
    ci = lambda xs: [float(x) for x in np.percentile(xs, [2.5, 97.5])]  # noqa: E731
    d1 = [m["sol"]["catch"] - m["probe"]["catch"] for m in draws]
    d0 = [m["sol"]["catch_0fa"] - m["probe"]["catch_0fa"] for m in draws]
    out = {"label": "POST HOC descriptive: strict-alarm catch with task-clustered intervals (test, 200 cells)", "pick": setup,
           "sol_distinct_scores": sorted({round(v, 6) for v in s.values()}),
           "at_1pct_target": {"sol": point["sol"] | {"fpr_ci95": ci([m["sol"]["fpr"] for m in draws])},
                              "probe": point["probe"] | {"fpr_ci95": ci([m["probe"]["fpr"] for m in draws])},
                              "diff_sol_minus_probe": point["sol"]["catch"] - point["probe"]["catch"], "diff_ci95": ci(d1)},
           "at_0_false_alarms": {"sol_caught": point["sol"]["caught_0fa"], "probe_caught": point["probe"]["caught_0fa"],
                                 "diff_sol_minus_probe": point["sol"]["catch_0fa"] - point["probe"]["catch_0fa"], "diff_ci95": ci(d0)},
           "draws": len(draws), "seed": 20261013,
           "inputs": {Path(x).name: hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in (pick_path, calls_path, probe_path)},
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"alarm_ci_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("inputs", "git", "script_sha256")}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
