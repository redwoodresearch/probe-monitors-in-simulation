"""PLAN.md amendment 4 (local; per-output scores only): the max-over-outputs run score beside the mean, for probe step
A's chosen probes (output_end, best validation layer and C per outer fold), on the 5 test environments.
  comparisons: fixed sim vs real e1 and real e2 vs real e1 (levels 0 and 1); planted checks 1 and 2 (level 1)
  run score:   max of the run's out-of-fold output scores (and the mean, which reproduces the headline)
  interval:    95%, 1,000 main-task resamples (probe_shift.scoring.auroc_ci)
  shuffle:     the probes fixed, 1,000 draws swapping the two runs' labels within each cell (main comparisons) or
               planted pair, with probability 1/2; 2.5-97.5% range and p95 reported (selection is not rerun)
Writes results/claimB_max_score_<stamp>.json.

  uv run python experiments/2026-10-09_fixed_sim/analyze_claimB_max_score.py <fixed> <e2> <plant1> <plant2>   (score .json.gz files)
"""

import gzip
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

from probe_shift.scoring import auroc_ci

HERE = Path(__file__).parent
TEST = set(json.loads((HERE / "split.json").read_text())["test"])
POS = {"fixed_vs_e1": "sim-attack-fixed", "e2_vs_e1": "real-attack-e2", "plant1": "sim-attack-fixed-plant1", "plant2": "sim-attack-fixed-plant2"}
DRAWS, SEED = 1000, 0


def unit_of(key: str, comparison: str) -> str:
    """The cell (main comparisons) or planted pair a run belongs to."""
    if comparison.startswith("plant"):
        return key.removesuffix("__plant1").removesuffix("__plant2")
    sample_id = key.rsplit("__", 2)[-2]
    return ":".join(sample_id.split(":")[:2])


def evaluate(rows, comparison):
    by = defaultdict(list)
    meta = {}
    for r in rows:
        if r["environment"] in TEST:
            by[r["key"]].append(r["score"])
            meta[r["key"]] = (r["arm"] == POS[comparison], r["main_task"], unit_of(r["key"], comparison))
    keys = sorted(by)
    y = np.array([meta[k][0] for k in keys])
    task = np.array([meta[k][1] for k in keys])
    unit = np.array([meta[k][2] for k in keys])
    out = {"runs": {"positive": int(y.sum()), "negative": int((~y).sum())}}
    units = sorted(set(unit))
    members = {u: np.flatnonzero(unit == u) for u in units}
    paired = [u for u in units if len(members[u]) == 2 and y[members[u]].sum() == 1]
    rng = np.random.default_rng(SEED)
    for name, agg in (("mean", np.mean), ("max", np.max)):
        s = np.array([agg(by[k]) for k in keys])
        res = auroc_ci(y, s, task, 1000, 0)
        null = []
        for _ in range(DRAWS):
            yp = y.copy()
            for u in paired:
                if rng.random() < 0.5:
                    i, j = members[u]
                    yp[i], yp[j] = yp[j], yp[i]
            null.append(roc_auc_score(yp, s))
        null = np.array(null)
        out[name] = res | {"shuffle_range": [round(float(np.percentile(null, 2.5)), 4), round(float(np.percentile(null, 97.5)), 4)],
                           "shuffle_p95": round(float(np.percentile(null, 95)), 4),
                           "p": round(float((1 + (null >= res["auroc"]).sum()) / (1 + DRAWS)), 4)}
    out["units_swapped"] = len(paired)
    return out


def main(fixed, e2, plant1, plant2):
    res = {"files": {}, "rows": {}}
    for comparison, path in (("fixed_vs_e1", fixed), ("e2_vs_e1", e2), ("plant1", plant1), ("plant2", plant2)):
        data = json.loads(gzip.decompress(Path(path).read_bytes()))
        res["files"][comparison] = Path(path).name
        for level, d in data["levels"].items():
            res["rows"][f"{level}|{comparison}"] = evaluate(d["rows"], comparison)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"claimB_max_score_{stamp}.json"
    with open(path, "x") as f:
        f.write(json.dumps(res, indent=1))
    for k, v in res["rows"].items():
        print(f"{k:22s} runs {v['runs']} | mean {v['mean']['auroc']:.3f} {v['mean']['ci95']} shuffle p95 {v['mean']['shuffle_p95']} "
              f"| max {v['max']['auroc']:.3f} {v['max']['ci95']} shuffle p95 {v['max']['shuffle_p95']} p {v['max']['p']}")
    print(path)


if __name__ == "__main__":
    main(*sys.argv[1:5])
