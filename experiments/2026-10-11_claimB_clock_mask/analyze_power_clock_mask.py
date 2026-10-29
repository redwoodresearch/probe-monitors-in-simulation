"""EXPLORATORY clock mask, power as a sensitivity ($0, laptop, stored scores only): how wide is the "does 0.82 move"
interval likely to be, and how often would each reading come out under a few assumed effects of the re-read?

Stand-in data: step A's per-output scores (level 1, fixed sim vs real e1, the 5 test environments; a path argument,
normally `experiments/2026-10-09_fixed_sim/data/claimB_output_scores_A_2026-10-10_022646.json.gz` in the main checkout,
sha256 recorded, S3 manifest v1). A run's score is the mean of its kept outputs' scores, as step A. Which outputs the
mask changes comes from count_clock_mask.py's result (per run, the changed (step, action)); only kept rows inside the
32k cut move.

Scenarios (sigma = the SD of output scores within each fold):
- exact: nothing changes (the interval collapses to 0; listed for completeness);
- zero-mean perturbation f: each changed output's score gets N(0, (f sigma)^2), sim and real alike, f 0.5 and 1.0.
  This is not a zero-AUROC-change scenario: symmetric noise can still change rankings (Codex);
- correlated f: every kept output from a run's first changed output onward shifts by one shared N(0, (f sigma)^2) draw
  (an edit can change every later readout), f 0.5;
- sim shift d: as zero-mean perturbation 0.5, plus simulated changed outputs lose d sigma (the mask removes a
  simulated-looking cue), d 0.5 and 1.0.
Per scenario, 40 simulated re-reads. Each gets Delta = AUROC(after) - AUROC(before), paired on the same runs, with 2,000
main-task bootstrap draws (seeded, numpy default_rng(20261011)). Readings, UNADJUSTED (no Holm, no yardstick; a
sensitivity conditional on the gates passing): "within 0.02" if the 90% interval lies inside [-0.02, 0.02]; "moves" if
the 95% interval excludes 0. Reported: the share of simulated re-reads giving each reading, and the median 95% half-width.
Outputs that enter or leave the 32k cut after masking are not modelled.

  PYTHONPATH=src uv run --no-project --python 3.12 --with numpy python \\
      experiments/2026-10-11_claimB_clock_mask/analyze_power_clock_mask.py <counts json> <step A scores json.gz>
"""

import gzip
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
TEST = {"clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"}
SCORES_SHA = None  # recorded, not pinned: manifest v1 holds the reference hash


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    _, start, counts = np.unique(allv[order], return_index=True, return_counts=True)
    ranks = np.empty_like(allv)
    ranks[order] = np.repeat(start + (counts + 1) / 2.0, counts)  # average ranks for ties
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    counts_path, scores_path = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    counts = json.loads(counts_path.read_text())
    rows = [r for r in json.loads(gzip.decompress(scores_path.read_bytes()))["levels"]["level1"]["rows"] if r["environment"] in TEST]
    assert rows, "no test-environment rows"
    sd = {f: float(np.std([r["score"] for r in rows if r["fold"] == f])) for f in {r["fold"] for r in rows}}
    runs = {}
    for r in rows:
        runs.setdefault(r["key"], {"arm": r["arm"], "task": r["main_task"], "out": []})["out"].append(r)
    missing = [k for k in runs if k not in counts["per_run"]]
    assert not missing, f"{len(missing)} scored runs not in the counts"
    keys = sorted(runs)
    for k in keys:
        runs[k]["out"].sort(key=lambda o: (o["step"], o["action"]))
    is_sim = np.array([runs[k]["arm"] == "sim-attack-fixed" for k in keys])
    tasks = sorted({runs[k]["task"] for k in keys})
    run_task = np.array([tasks.index(runs[k]["task"]) for k in keys])
    base = np.array([np.mean([o["score"] for o in runs[k]["out"]]) for k in keys])
    n_kept = np.array([len(runs[k]["out"]) for k in keys])
    direct, after_first, total_changed = [], [], 0  # per run: sigmas of changed kept outputs; of kept outputs from the first change on
    for k in keys:
        ch = {tuple(x) for x in counts["per_run"][k]["changed"]}
        total_changed += len(ch)
        outs = runs[k]["out"]
        direct.append(np.array([sd[o["fold"]] for o in outs if (o["step"], o["action"]) in ch]))
        first = min(((s, a) for s, a in ch), default=None)
        after_first.append(np.array([sd[o["fold"]] for o in outs if first is not None and (o["step"], o["action"]) >= first]))
    rng = np.random.default_rng(20261011)
    by_task = [np.where(run_task == i)[0] for i in range(len(tasks))]
    draws = [np.concatenate([by_task[i] for i in rng.integers(0, len(tasks), len(tasks))]) for _ in range(2000)]
    before_draws = np.array([auroc(base[ix][is_sim[ix]], base[ix][~is_sim[ix]]) for ix in draws])

    def reading(after):
        d = np.array([auroc(after[ix][is_sim[ix]], after[ix][~is_sim[ix]]) for ix in draws]) - before_draws
        point = auroc(after[is_sim], after[~is_sim]) - auroc(base[is_sim], base[~is_sim])
        ci95, ci90 = np.percentile(d, [2.5, 97.5]), np.percentile(d, [5, 95])
        return {"point": point, "within": bool(ci90[0] > -0.02 and ci90[1] < 0.02), "moves": bool(ci95[0] > 0 or ci95[1] < 0),
                "half95": float((ci95[1] - ci95[0]) / 2)}

    def simulate(kind, f, d):
        after = base.copy()
        for i in range(len(keys)):
            if kind == "correlated":
                sig = after_first[i]
                if len(sig):
                    after[i] = base[i] + (rng.normal(0, f) * sig).sum() / n_kept[i]
                continue
            sig = direct[i]
            if len(sig):
                move = rng.normal(0, f * sig) - (d * sig if is_sim[i] else 0.0)
                after[i] = base[i] + move.sum() / n_kept[i]
        return after

    out = {"runs": {"sim": int(is_sim.sum()), "real_e1": int((~is_sim).sum())}, "main_tasks": len(tasks),
           "auroc_before": auroc(base[is_sim], base[~is_sim]),
           "runs_with_a_kept_changed_output": {"sim": int(sum(len(x) > 0 for x, s in zip(direct, is_sim) if s)),
                                               "real_e1": int(sum(len(x) > 0 for x, s in zip(direct, is_sim) if not s))},
           "changed_outputs_in_counts": total_changed, "changed_outputs_kept": int(sum(len(x) for x in direct)),
           "scenarios": {"exact_no_change": reading(base.copy())}}
    for name, kind, f, d in (("zero_mean_0.5", "direct", 0.5, 0.0), ("zero_mean_1.0", "direct", 1.0, 0.0),
                             ("correlated_0.5", "correlated", 0.5, 0.0), ("sim_shift_0.5", "direct", 0.5, 0.5),
                             ("sim_shift_1.0", "direct", 0.5, 1.0)):
        res = [reading(simulate(kind, f, d)) for _ in range(40)]
        out["scenarios"][name] = {"kind": kind, "f": f, "d": d, "sims": len(res),
                                  "median_delta": float(np.median([r["point"] for r in res])),
                                  "share_within_0.02": float(np.mean([r["within"] for r in res])),
                                  "share_moves": float(np.mean([r["moves"] for r in res])),
                                  "median_half_width_95": float(np.median([r["half95"] for r in res]))}
        print(name, json.dumps(out["scenarios"][name]))
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out |= {"label": "EXPLORATORY clock mask: unadjusted sensitivity of the 'does 0.82 move' reading on step A's stored scores",
            "inputs": {"scores": hashlib.sha256(scores_path.read_bytes()).hexdigest(), "counts": hashlib.sha256(counts_path.read_bytes()).hexdigest()},
            "git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"power_clock_mask_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "scenarios"}, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
