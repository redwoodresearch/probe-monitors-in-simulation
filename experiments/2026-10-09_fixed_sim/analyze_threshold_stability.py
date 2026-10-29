"""Amendment 8 (PLAN.md, written before any number): is step A's catch at a 1% false-alarm target stable? Local, $0.

Step A's level-1 run scores (mean of each run's out-of-fold output scores). Scopes: all 10 environments and the 5 test
environments. 5-fold cross-fit by main task: fold k's threshold is the (floor(f*n)+1)-th highest real e1 score among the
other folds' tasks, applied to fold k's runs; counts pooled. Uncertainty: 2,000 main-task bootstrap draws, each redoing
the split and thresholds; plus 200 alternative seeded splits of the original data. Control: real e2 vs real e1.
Also writes the ROC points for plot_threshold_stability.py (log false-alarm axis).

  uv run --no-project --python 3.13 --with numpy --with scikit-learn python experiments/2026-10-09_fixed_sim/analyze_threshold_stability.py
"""

import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

import analyze_tpr_at_fpr as T

HERE = Path(__file__).parent
K, DRAWS, SPLITS, SEED, TARGETS = 5, 2000, 200, 20261011, (1, 5)


def kfold(y, s, task, members, counts, order, pct):
    """Pooled 5-fold cross-fit over the tasks in `order` (task index repeated counts[t] times). Returns caught, positives,
    false alarms, negatives and per-fold (reference runs, reference tasks, rank, threshold); None if a fold lacks refs."""
    folds = [order[i::K] for i in range(K)]
    caught = fa = npos = nneg = 0
    info = []
    for k in range(K):
        cal = np.concatenate([np.repeat(members[t], counts[t]) for j in range(K) if j != k for t in folds[j]] or [np.array([], int)])
        ev = np.concatenate([np.repeat(members[t], counts[t]) for t in folds[k]] or [np.array([], int)])
        ref = cal[~y[cal]]
        if not len(ref) or not len(ev):
            return None
        t_, rank, n = T.threshold(s[ref], pct)
        caught += int((s[ev][y[ev]] > t_).sum())
        fa += int((s[ev][~y[ev]] > t_).sum())
        npos += int(y[ev].sum())
        nneg += int((~y[ev]).sum())
        info.append({"reference_runs": n, "reference_tasks": int(len({int(x) for x in task[ref]})), "rank": rank, "threshold": t_})
    return caught, npos, fa, nneg, info


def evaluate(job):
    y, s, task_names = job["y"], job["s"], job["task"]
    tasks, task = np.unique(task_names, return_inverse=True)
    T_ = len(tasks)
    members = [np.flatnonzero(task == j) for j in range(T_)]
    rng = np.random.default_rng(SEED)
    ones = np.ones(T_, int)
    out = {k: v for k, v in job.items() if k not in ("y", "s", "task")}
    out |= {"positive_runs": int(y.sum()), "negative_runs": int((~y).sum()), "main_tasks": T_, "auroc": float(roc_auc_score(y, s))}
    point_order = rng.permutation(T_)
    splits = [rng.permutation(T_) for _ in range(SPLITS)]
    draws = []
    for _ in range(DRAWS):
        c = np.bincount(rng.choice(T_, T_), minlength=T_)
        draws.append((c, rng.permutation(np.flatnonzero(c))))
    for pct in TARGETS:
        c, npos, fa, nneg, info = kfold(y, s, task, members, ones, point_order, pct)
        alt = [kfold(y, s, task, members, ones, o, pct) for o in splits]
        alt_c = np.array([a[0] / a[1] for a in alt])
        boot_c, boot_f = [], []
        for cnt, order in draws:
            r = kfold(y, s, task, members, cnt, order, pct)
            if r is None or r[1] == 0 or r[3] == 0:
                continue
            boot_c.append(r[0] / r[1])
            boot_f.append(r[2] / r[3])
        out[f"at_{pct}pct"] = {"caught": c, "positives": npos, "caught_share": c / npos, "false_alarms": fa, "negatives": nneg,
                               "fpr": fa / nneg, "folds": info,
                               "caught_ci95": [float(x) for x in np.percentile(boot_c, [2.5, 97.5])],
                               "fpr_ci95": [float(x) for x in np.percentile(boot_f, [2.5, 97.5])], "draws_used": len(boot_c),
                               "alt_splits": {"n": SPLITS, "p5": float(np.percentile(alt_c, 5)), "median": float(np.median(alt_c)),
                                              "p95": float(np.percentile(alt_c, 95)), "min": float(alt_c.min()), "max": float(alt_c.max())}}
        a = out[f"at_{pct}pct"]["alt_splits"]
        out[f"at_{pct}pct"]["stable"] = bool(a["p5"] >= c / npos - 0.03 and a["p95"] <= c / npos + 0.03)
    fpr, tpr, _ = roc_curve(y, s)
    out["roc"] = {"fpr": [float(x) for x in fpr], "tpr": [float(x) for x in tpr]}
    return out


def main():
    a = {"A_fixed": T.load_gz(T.SRC["A_fixed"]), "A_e2": T.load_gz(T.SRC["A_e2"])}
    jobs = []
    for comp, src in (("fixed_vs_e1", "A_fixed"), ("e2_vs_e1", "A_e2")):
        runs = T.runs_from_outputs(a[src]["levels"]["level1"]["rows"], np.mean)
        for scope, envs in (("all 10", None), ("test 5", T.TEST)):
            y, s, t = T.select(runs, comp, envs)
            jobs.append({"id": f"step A|level1|{comp}|{scope}", "comparison": comp, "scope": scope, "y": y, "s": s, "task": t})
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(evaluate, jobs))
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    res = {"amendment": 8, "git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "inputs": {k: hashlib.sha256(T.SRC[k].read_bytes()).hexdigest() for k in ("A_fixed", "A_e2")},
           "caveat": "each environment is scored by its own held-out probe, so pooled thresholds mix score scales", "rows": rows}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"threshold_stability_{stamp}.json"
    with open(path, "x") as fh:
        fh.write(json.dumps(res, indent=1))
    for r in rows:
        print(f"{r['id']}: AUROC {r['auroc']:.3f}; runs {r['positive_runs']}/{r['negative_runs']}, tasks {r['main_tasks']}")
        for pct in TARGETS:
            x = r[f"at_{pct}pct"]
            al = x["alt_splits"]
            print(f"  {pct}%: caught {x['caught']}/{x['positives']} = {100 * x['caught_share']:.1f}% [{100 * x['caught_ci95'][0]:.1f}, {100 * x['caught_ci95'][1]:.1f}]"
                  f" | false alarms {x['false_alarms']}/{x['negatives']} = {100 * x['fpr']:.2f}% [{100 * x['fpr_ci95'][0]:.2f}, {100 * x['fpr_ci95'][1]:.2f}]"
                  f" | 200 splits p5-p95 {100 * al['p5']:.1f}-{100 * al['p95']:.1f}% (median {100 * al['median']:.1f}) -> {'stable' if x['stable'] else 'unstable'}")
            print("     folds: " + "; ".join(f"refs {f['reference_runs']} runs/{f['reference_tasks']} tasks, rank {f['rank']}" for f in x["folds"]))
    print("->", path)


if __name__ == "__main__":
    main()
