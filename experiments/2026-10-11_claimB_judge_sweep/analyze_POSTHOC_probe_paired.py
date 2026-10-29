"""POST HOC, descriptive ($0, numbers only; 0d's request): the frozen probe's paired accuracy, the comparator for S3's
side-by-side pair accuracy. Per cell (one fixed-sim and one real-e1 run of the same task): does step A's probe (level 1,
the stored per-output scores, run mean) score the simulated run higher (ties 1/2)?

(a) The 200 test cells (the old judge's selection). (b) The 50 practice cells S3 used, only if each run's score came
from a fit that held its own environment out: the stored scores are leave-one-environment-out, so this is checked per
environment (one fold per environment, no environment in two folds) and reported, with the caveat that the other nine
folds' fits did train on these environments. Intervals: percentile 95% of 2,000 draws resampling main tasks,
default_rng(20261013). Practice run keys come from claim B's level-1 pack on the Volume (metadata only).

  uv run --no-project --python 3.12 --with numpy --with 'modal>=1.4.3' python \\
      experiments/2026-10-11_claimB_judge_sweep/analyze_POSTHOC_probe_paired.py <probe_scores.json.gz> <practice calls.jsonl>
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

import modal
import numpy as np

HERE = Path(__file__).parent
SIM, E1 = "sim-attack-fixed", "real-attack-e1"
VOL, PACK = "probe-shift_aswin-surya_sub-la-activations", "claimB/level1/packed_39f842231a46/runs.L19.json"


def paired(cells, score):
    acc = lambda cs: float(np.mean([(score[(c, SIM)] > score[(c, E1)]) + 0.5 * (score[(c, SIM)] == score[(c, E1)]) for c in cs]))  # noqa: E731
    by = defaultdict(list)
    for c in cells:
        by[c.split(":")[0]].append(c)
    tasks, rng = sorted(by), np.random.default_rng(20261013)
    draws = [acc([c for t in rng.choice(len(tasks), len(tasks)) for c in by[tasks[t]]]) for _ in range(2000)]
    return {"pairs": len(cells), "main_tasks": len(tasks), "sim_scored_higher": int(sum(score[(c, SIM)] > score[(c, E1)] for c in cells)),
            "paired_accuracy": acc(cells), "ci95": [float(x) for x in np.percentile(draws, [2.5, 97.5])]}


def main(probe_path, practice_path):
    rows = json.loads(gzip.decompress(Path(probe_path).read_bytes()))["levels"]["level1"]["rows"]
    by_key, meta, env_folds = defaultdict(list), {}, defaultdict(set)
    for r in rows:
        by_key[r["key"]].append(r["score"])
        meta[r["key"]] = (r["environment"], r["fold"])
        env_folds[r["environment"]].add(r["fold"])
    run = {k: float(np.mean(v)) for k, v in by_key.items()}
    sel = json.loads(subprocess.run(["git", "show", "58ce4715:experiments/2026-10-11_claimB_judge/results/judge_run_selection.json"],
                                    capture_output=True, text=True, check=True, cwd=HERE).stdout)
    test = {(c, a): run[sel["run_keys"][f"{c}|{a}"]] for c in sel["cells"] for a in (SIM, E1)}
    out = {"label": "POST HOC descriptive: the frozen probe's paired accuracy (sim scored above its same-task real e1 run)",
           "test": paired(sel["cells"], test)}
    prac = sorted({r["cell"] for r in map(json.loads, Path(practice_path).read_text().splitlines()) if r["kind"] == "main|S3"})
    pack = {(r["cell"], r["arm"]): r["key"] for r in json.loads(b"".join(modal.Volume.from_name(VOL, version=2).read_file(PACK)))}
    keys = {(c, a): pack[(c, a)] for c in prac for a in (SIM, E1)}
    e2f = json.loads((HERE.parents[1] / "splits/folds_v2.json").read_text())["environment_folds"]["environment_to_fold"]  # the frozen LOEO map
    folds_ok = all(env_folds[e] == {e2f[e]} for e in env_folds)  # every row of an environment carries that environment's own held-out fold
    envs = sorted({meta[k][0] for k in keys.values() if k in meta})
    missing = sum(k not in run for k in keys.values())
    out["practice_check"] = {"cells": len(prac), "environments": envs, "rows_match_frozen_loeo_folds": folds_ok, "practice_runs_without_probe_score": missing,
                             "note": "each run is scored by the fit that held its environment out; the other nine folds' fits trained on it"}
    if folds_ok and all(k in run for k in keys.values()):
        out["practice"] = paired(prac, {ca: run[k] for ca, k in keys.items()})
    out |= {"seed": 20261013, "inputs": {Path(x).name: hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in (probe_path, practice_path, HERE.parents[1] / "splits/folds_v2.json")},
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"probe_paired_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("inputs", "git", "script_sha256")}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
