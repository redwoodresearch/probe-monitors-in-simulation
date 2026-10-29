"""Exploratory, not pre-registered (0d's go, 17:39 PT 10-10; $0): step A's probe scores per test environment.

Reads step A's saved per-output scores (`run_claimB_output_scores_modal.py`, d9's code a1821be0; refit of A's chosen
probes, one per environment fold), copied from the la-activations Volume `claimB/level{0,1}/output_scores_A_*e402bcc321a8.jsonl.gz`
into data/step_a_output_scores_e402bcc321a8/. Run score = mean of its output_end scores (A's rule). Planted checks pair
each planted copy with its own twin only. Per comparison, 5 test environments:
  pooled            AUROC over all test-environment runs (A's headline form; each environment scored by its own probe);
  per_environment   AUROC within each environment, with a 1,000-draw main-task bootstrap interval (seeded);
  pooled_env_z      AUROC after z-scoring run scores within each environment (removes per-probe offset and scale);
  null              fixed-probe label-shuffle null for pooled and each environment: one coin per main task swaps the arm
                    labels of all its runs (arm counts kept), 1,000 seeded draws; p95 and p (review of #439).

  uv run --no-project --python 3.13 --with numpy --with scikit-learn python experiments/2026-10-09_fixed_sim/analyze_claimB_A_per_env.py
"""

import gzip
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
SRC = HERE / "data" / "step_a_output_scores_e402bcc321a8"
TEST = ["clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"]
FILES = {  # comparison: (file, positive arm, negative arm); planted checks pair copies with their twins
    "fixed_vs_e1|level0": ("level0/output_scores_A_e402bcc321a8.jsonl.gz", "sim-attack-fixed", "real-attack-e1"),
    "fixed_vs_e1|level1": ("level1/output_scores_A_e402bcc321a8.jsonl.gz", "sim-attack-fixed", "real-attack-e1"),
    "e2_vs_e1|level0": ("level0/output_scores_A_e2_vs_e1_e402bcc321a8.jsonl.gz", "real-attack-e2", "real-attack-e1"),
    "e2_vs_e1|level1": ("level1/output_scores_A_e2_vs_e1_e402bcc321a8.jsonl.gz", "real-attack-e2", "real-attack-e1"),
    "plant1|level1": ("level1/output_scores_A_plant1_e402bcc321a8.jsonl.gz", "sim-attack-fixed-plant1", "sim-attack-fixed"),
    "plant2|level1": ("level1/output_scores_A_plant2_e402bcc321a8.jsonl.gz", "sim-attack-fixed-plant2", "sim-attack-fixed"),
}


def runs_of(path: Path) -> dict:
    scores, meta = defaultdict(list), {}
    for line in gzip.open(path, "rt"):
        r = json.loads(line)
        scores[r["key"]].append(r["score"])
        m = (r["arm"], r["environment"], r["main_task"], r["fold"])
        if meta.setdefault(r["key"], m) != m:
            raise ValueError(f"{path.name}: run {r['key']} has inconsistent metadata")
    return {k: {"score": float(np.mean(v)), "arm": meta[k][0], "env": meta[k][1], "task": meta[k][2], "outputs": len(v)}
            for k, v in scores.items()}


def select(runs: dict, pos: str, neg: str) -> list:
    if "plant" not in pos:
        return [r | {"y": int(r["arm"] == pos)} for r in runs.values() if r["arm"] in (pos, neg) and r["env"] in TEST]
    out = []
    for k, r in runs.items():
        if r["arm"] == pos and r["env"] in TEST:
            twin = runs.get(k.rsplit("__", 1)[0])
            if twin is None or twin["arm"] != neg:
                raise ValueError(f"no twin for {k}")
            out += [r | {"y": 1}, twin | {"y": 0}]
    return out


def auroc(rows: list) -> float:
    return float(roc_auc_score([r["y"] for r in rows], [r["score"] for r in rows]))


def boot(rows: list, rng, draws: int = 1000) -> list:
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    tasks, vals = list(by_task), []
    for _ in range(draws):
        s = [r for t in rng.choice(tasks, len(tasks)) for r in by_task[t]]
        if 0 < sum(r["y"] for r in s) < len(s):
            vals.append(auroc(s))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def null(rows: list, rng, draws: int = 1000) -> dict:
    obs, tasks = auroc(rows), sorted({r["task"] for r in rows})
    ti = np.array([tasks.index(r["task"]) for r in rows])
    y, s = np.array([r["y"] for r in rows]), np.array([r["score"] for r in rows])
    vals = np.array([roc_auc_score(np.where(rng.integers(0, 2, len(tasks))[ti] == 1, 1 - y, y), s) for _ in range(draws)])
    return {"p95": round(float(np.percentile(vals, 95)), 4), "p": round(float((1 + (vals >= obs).sum()) / (1 + draws)), 4), "draws": draws}


def main():
    rng = np.random.default_rng(20261010)
    out = {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "label": "exploratory, not pre-registered",
           "inputs": {}, "rows": {}}
    for comp, (f, pos, neg) in FILES.items():
        out["inputs"][f] = hashlib.sha256((SRC / f).read_bytes()).hexdigest()
        rows = select(runs_of(SRC / f), pos, neg)
        z = []
        for e in TEST:
            s = np.array([r["score"] for r in rows if r["env"] == e])
            z += [r | {"score": (r["score"] - s.mean()) / s.std()} for r in rows if r["env"] == e]
        per = {e: {"auroc": round(auroc(re), 4), "ci95": boot(re, rng), "null": null(re, rng),
                   "runs": [sum(r["y"] for r in re), sum(1 - r["y"] for r in re)]}
               for e in TEST for re in [[r for r in rows if r["env"] == e]]}
        out["rows"][comp] = {"pooled": round(auroc(rows), 4), "pooled_ci95": boot(rows, rng), "pooled_null": null(rows, rng),
                             "pooled_env_z": round(auroc(z), 4),
                             "runs": [sum(r["y"] for r in rows), sum(1 - r["y"] for r in rows)], "per_environment": per}
        print(comp, "pooled", out["rows"][comp]["pooled"], "env-z", out["rows"][comp]["pooled_env_z"],
              {e: v["auroc"] for e, v in per.items()}, flush=True)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"claimB_A_per_env_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("->", HERE / "results" / f"claimB_A_per_env_{stamp}.json")


if __name__ == "__main__":
    main()
