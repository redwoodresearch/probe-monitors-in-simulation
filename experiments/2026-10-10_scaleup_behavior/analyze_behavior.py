"""Scale-up behaviour: real-vs-sim gaps next to the real-vs-real gap (two real attack runs), paired by task, with 95%
bootstrap intervals over tasks; then the pre-registered planted check (README.md). Reads extract_behavior.py's CSV.

  uv run --no-project --python 3.12 --with pandas --with numpy python experiments/2026-10-10_scaleup_behavior/analyze_behavior.py \
      results/per_run_scaleup.csv
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SEED, DRAWS = 20261010, 10_000
df = pd.read_csv(sys.argv[1])
df = df[~df.sample_error & (df.n_calls > 0)].copy()  # usable runs only: a sample error has no per-call shares
# The measures, exactly as #330's 06_analyze.py defines them, plus tokens
df["sh_edit"] = (df.n_create + df.n_str_replace + df.n_insert) / df.n_calls
df["sh_errfield"] = df.n_errfield / df.n_out
df["sh_failtext"] = df.n_failtext / df.n_out
df["reread_rate"] = df.rereads / df.edits.replace(0, np.nan)
MEASURES = ["sh_edit", "reread_rate", "sh_errfield", "sh_failtext", "n_steps", "tok_out"]

second = df.launch.str.endswith("_second_attack")
arms = {"run1": df[(df.arm == "attack") & ~second], "run2": df[(df.arm == "attack") & second],
        "sim": df[(df.arm == "sim_attack") & (df.simulator == "september")],
        "honA": df[df.arm == "honest_a"], "honB": df[df.arm == "honest_b"]}
for a, d in arms.items():
    assert d.task.is_unique, f"{a}: a task appears twice"
arms = {a: d.set_index("task") for a, d in arms.items()}
GAPS = [("run1", "sim"), ("run1", "run2"), ("run2", "sim"), ("honA", "honB")]


def gap(a: pd.DataFrame, b: pd.DataFrame, m: str, rng: np.random.Generator) -> tuple:
    d = (a[m] - b[m]).dropna()  # tasks where both runs are usable (and, for re-reads, both made an edit)
    x = d.to_numpy()
    boot = x[rng.integers(0, len(x), (DRAWS, len(x)))].mean(1)
    return x.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), len(x)


commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print(f"script commit {commit[:12]} | seed {SEED} | {DRAWS} bootstrap draws over tasks | usable runs per arm: "
      + ", ".join(f"{a} {len(d)}" for a, d in arms.items()))
print("\narm means (usable runs):")
print(pd.DataFrame({a: d[MEASURES].mean() for a, d in arms.items()}).round(3).to_string())
out = {"script_commit": commit, "seed": SEED, "draws": DRAWS, "runs": {a: len(d) for a, d in arms.items()}, "gaps": {}}
print("\ngaps, mean of per-task differences [95% interval over tasks] (n tasks):")
for m in MEASURES:
    row = []
    for a, b in GAPS:
        est, lo, hi, n = gap(arms[a], arms[b], m, np.random.default_rng(SEED))
        out["gaps"][f"{m}: {a} - {b}"] = [round(est, 4), round(lo, 4), round(hi, 4), n]
        row.append(f"{a}-{b} {est:+.4g} [{lo:+.4g}, {hi:+.4g}] ({n})")
    print(f"  {m:12s} " + " | ".join(row))

# Planted check (README): 200 seeded copies; inflate a seeded half of the tasks' runs in one arm.
PLANTS = {"n_steps": lambda v: v * 1.15, "sh_failtext": lambda v: v + 0.02}
print("\nplanted check, 200 seeded copies each (a seeded half of tasks inflated):")
out["planted"] = {}
for m, f in PLANTS.items():
    for a, b, target in (("run1", "sim", "sim"), ("run1", "run2", "run2")):
        flagged, detected, shifts = 0, 0, []
        base = gap(arms[a], arms[b], m, np.random.default_rng(SEED))[0]
        for s in range(200):
            rng = np.random.default_rng(SEED + s)
            copy = arms[target].copy()
            copy[m] = copy[m].astype(float)
            tasks = copy.index.to_numpy()
            hit = rng.choice(tasks, len(tasks) // 2, replace=False)
            copy.loc[hit, m] = f(copy.loc[hit, m])
            est, lo, hi, _ = gap(arms[a], copy, m, rng)
            flagged += not (lo <= 0 <= hi)  # the old rule: the interval excludes 0
            # Aswin's rule (10-10 10:51 PT): the gap shifts by about the planted amount AND its interval excludes the
            # unplanted estimate. Expected shift: the planted change in the target arm, averaged over the paired tasks.
            paired = (arms[a][m] - arms[target][m]).dropna().index
            want = -float((copy.loc[paired, m] - arms[target].loc[paired, m].astype(float)).mean())
            got = est - base
            detected += bool(want) and 0.5 <= got / want <= 1.5 and not (lo <= base <= hi)
            shifts.append(got)
        out["planted"][f"{m} into {target}"] = {"old_rule_excludes_0_of_200": flagged, "new_rule_detected_of_200": detected,
                                                 "mean_shift_of_gap": round(float(np.mean(shifts)), 4)}
        print(f"  {m:12s} planted into {target:4s}: new rule (shift about the plant AND interval excludes the unplanted "
              f"estimate) {detected}/200 | old rule (interval excludes 0) {flagged}/200 | gap moved {np.mean(shifts):+.4g} "
              f"on average (unplanted {base:+.4g})")
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
res = HERE / "results" / f"behavior_gaps_{stamp}.json"
res.write_text(json.dumps(out, indent=1) + "\n")
print(f"-> {res.relative_to(HERE.parents[1])}")
