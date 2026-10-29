"""The judge's results (run_judge_fixed.py): the main run, the real-vs-real control, their gap, and the main run against
the paper's stored verdicts on the unfixed pairs (the 0.633 baseline), on matched pairs.

  - accuracy per run and per order (real shown first, real shown second), as stored by run_judge_fixed.py;
  - main minus control, with a 95% interval from one bootstrap over main tasks applied to both runs together (they
    share cells);
  - fixed minus unfixed on matched pairs: cells whose stored verdict is on the same real and simulated runs, the paired
    difference with a 95% interval from a bootstrap over main tasks.
Writes results/judge_analysis_<stamp>.json.

  uv run python experiments/2026-10-09_fixed_sim/analyze_judge_fixed.py results/judge_main_<stamp>.json results/judge_control_<stamp>.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
STORED = HERE.parent / "2026-10-06_q1_text_baselines" / "results" / "paper_judge_pairs_2026-10-06_130221.jsonl"
STORED_JUDGE, BOOT, SEED = "gpt-5.6-sol_medium", 1000, 0


def scored(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())["pairs"]
    return [r for r in rows if r["accuracy"] is not None]


def per_order(rows: list[dict]) -> dict:
    out = {}
    for slot in ("a", "b"):
        v = [float(r[f"real_in_{slot}_correct"]) for r in rows if r[f"real_in_{slot}_correct"] is not None]
        out[f"real_in_{slot}"] = {"n": len(v), "accuracy": float(np.mean(v)) if v else None}
    return out


def boot(groups_a, vals_a, groups_b, vals_b, fn) -> list[float]:
    """95% interval of fn(resampled a, resampled b), resampling main tasks (shared between a and b) with replacement."""
    tasks = sorted(set(groups_a) | set(groups_b))
    by_a = {t: [v for g, v in zip(groups_a, vals_a) if g == t] for t in tasks}
    by_b = {t: [v for g, v in zip(groups_b, vals_b) if g == t] for t in tasks}
    rng, stats = np.random.default_rng(SEED), []
    for _ in range(BOOT):
        pick = rng.choice(tasks, len(tasks))
        a = [v for t in pick for v in by_a[t]]
        b = [v for t in pick for v in by_b[t]]
        if a and b:
            stats.append(fn(a, b))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def main(main_file: str, control_file: str):
    m, c = scored(HERE / main_file), scored(HERE / control_file)
    mg, mv = [r["main_task"] for r in m], [r["accuracy"] for r in m]
    cg, cv = [r["main_task"] for r in c], [r["accuracy"] for r in c]
    gap = float(np.mean(mv) - np.mean(cv))
    stored = {r["cell"]: r for r in map(json.loads, STORED.read_text().splitlines()) if r["judge"] == STORED_JUDGE}
    matched = [(r, stored[r["cell"]]) for r in m if r["cell"] in stored
               and (stored[r["cell"]]["real_key"], stored[r["cell"]]["sim_key"]) == (r["real_key"], r["other_key"])]
    diffs = [r["accuracy"] - s["accuracy"] for r, s in matched]
    dg = [r["main_task"] for r, _ in matched]
    out = {
        "main": {"file": main_file, "pairs": len(m), "accuracy": float(np.mean(mv)), "per_order": per_order(m)},
        "control": {"file": control_file, "pairs": len(c), "accuracy": float(np.mean(cv)), "per_order": per_order(c)},
        "main_minus_control": {"gap": gap, "ci95_main_task_bootstrap": boot(mg, mv, cg, cv, lambda a, b: np.mean(a) - np.mean(b))},
        "fixed_vs_unfixed_matched": {
            "stored_file": str(STORED.relative_to(HERE.parent)), "stored_judge": STORED_JUDGE, "pairs": len(matched),
            "unmatched_main_pairs": len(m) - len(matched),
            "fixed_accuracy": float(np.mean([r["accuracy"] for r, _ in matched])) if matched else None,
            "unfixed_accuracy": float(np.mean([s["accuracy"] for _, s in matched])) if matched else None,
            "difference": float(np.mean(diffs)) if diffs else None,
            "ci95_main_task_bootstrap": boot(dg, diffs, dg, diffs, lambda a, b: np.mean(a)) if diffs else None},
    }
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"judge_analysis_{stamp}.json"
    with open(path, "x") as f:
        f.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
