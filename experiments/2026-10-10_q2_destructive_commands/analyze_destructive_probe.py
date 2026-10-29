"""Destructive-command probe, step 3 (local, free): the numbers PLAN.md lists, from the job's full output.

Per world (real, simulated honest): positives, TPR and FPR at the frozen thresholds, AUROC (mean over test folds).
Real minus simulated TPR (and FPR) with a task bootstrap (2,000, seed 0). Selection-aware null: the same statistics on
the 20 shuffled-label draws (p = (1 + draws at least as extreme) / 21). Real-vs-real control: tasks with at least two
real honest runs, the first run against the second (run keys sorted), same scores and thresholds, on the real labels
and on every null draw. Planted check (Aswin's rule, probe_shift.planted): round(5% of simulated positives) caught
simulated positives turned missed (seed 0); a ladder of larger plants is added after the plan and labelled so. The
TF-IDF baseline, when given, gets the same per-world numbers and bootstrap.

  uv run --no-project python experiments/2026-10-10_q2_destructive_commands/analyze_destructive_probe.py <probe full output> [tfidf output]
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from probe_shift.planted import detected, widening

HERE = Path(__file__).parent
N_BOOT = 2000


def fold_auroc(y, s, mask, fold):
    a = []
    for k in range(5):
        m = mask & (fold == k)
        if len(set(y[m])) == 2:
            a.append(roc_auc_score(y[m], s[m]))
    return float(np.mean(a))


class Tasks:
    """Per-task counts, so a task bootstrap is a weighted sum (weights = how often each task was drawn)."""

    def __init__(self, task, seed=0):
        self.names, self.inv = np.unique(task, return_inverse=True)
        rng = np.random.default_rng(seed)
        n = len(self.names)
        self.W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(N_BOOT)])

    def count(self, mask):
        return np.bincount(self.inv[mask], minlength=len(self.names))

    def rate(self, num, den):
        """Point estimate and 95% bootstrap interval of sum(num) / sum(den) over tasks."""
        boot = (self.W @ num) / np.maximum(self.W @ den, 1)
        return float(num.sum() / den.sum()), boot

    def gap(self, a, b):
        (pa, ba), (pb, bb) = self.rate(*a), self.rate(*b)
        return {"point": pa - pb, "95%": [float(np.percentile(ba - bb, 2.5)), float(np.percentile(ba - bb, 97.5))], "a": pa, "b": pb}


def world_stats(y, caught, world, T):
    out = {}
    for w, name in ((0, "real"), (1, "simulated")):
        pos, neg = (world == w) & (y == 1), (world == w) & (y == 0)
        out[name] = {"positives": int(pos.sum()), "negatives": int(neg.sum()), "tasks with a positive": len(set(T.inv[pos])),
                     "TPR": float(caught[pos].mean()), "FPR": float(caught[neg].mean())}
    tp = lambda w: (T.count((world == w) & (y == 1) & caught), T.count((world == w) & (y == 1)))
    fp = lambda w: (T.count((world == w) & (y == 0) & caught), T.count((world == w) & (y == 0)))
    out["TPR real minus simulated"] = T.gap(tp(0), tp(1))
    out["FPR real minus simulated"] = T.gap(fp(0), fp(1))
    return out


def first_second(rows):
    """Real rows of tasks with two or more real runs: 1 for the first run (sorted run key), 2 for the second, else 0."""
    runs = {}
    for r in rows:
        if r["world"] == 0:
            runs.setdefault(r["task"], set()).add(r["run"])
    order = {run: i + 1 for rs in runs.values() if len(rs) >= 2 for i, run in enumerate(sorted(rs)[:2])}
    return np.array([order.get(r["run"], 0) if r["world"] == 0 else 0 for r in rows])


def main(probe_path: Path, tfidf_path: Path | None):
    full = json.loads(probe_path.read_text())
    rows, draws = full["rows"], full["draws"]
    y = np.array([r["y"] for r in rows]); world = np.array([r["world"] for r in rows]); fold = np.array([r["fold"] for r in rows])
    task = np.array([r["task"] for r in rows])
    T = Tasks(task)
    lab = lambda d: np.array([int(c) for c in d["labels"]]); cau = lambda d: np.array([c == "1" for c in d["caught"]])
    if (lab(draws[0]) != y).any():
        raise ValueError("draw 0 labels differ from the rows' labels")
    caught = cau(draws[0])
    out = {"probe output": str(probe_path), "probe output sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
           "picks (layer, C, threshold) per test fold": draws[0]["picks"],
           "selection fits not converged": full["selection fits not converged"], "refits not converged": full["refits not converged"]}
    out["probe"] = world_stats(y, caught, world, T) | {"AUROC real (fold mean)": draws[0]["AUROC real (fold mean)"],
                                                       "AUROC simulated (fold mean)": draws[0]["AUROC sim (fold mean)"]}
    obs, n_null = draws[0], len(draws) - 1
    null_gap = [d["TPR real minus sim"] for d in draws[1:]]
    out["null (within-run label shuffles, selection repeated)"] = {
        "TPR real minus simulated": {"values": null_gap, "p (two-sided)": (1 + sum(abs(g) >= abs(obs["TPR real minus sim"]) for g in null_gap)) / (n_null + 1)},
        "AUROC real": {"values": [d["AUROC real (fold mean)"] for d in draws[1:]],
                       "p": (1 + sum(d["AUROC real (fold mean)"] >= obs["AUROC real (fold mean)"] for d in draws[1:])) / (n_null + 1)},
        "AUROC simulated": {"values": [d["AUROC sim (fold mean)"] for d in draws[1:]],
                            "p": (1 + sum(d["AUROC sim (fold mean)"] >= obs["AUROC sim (fold mean)"] for d in draws[1:])) / (n_null + 1)}}

    # Real-vs-real control: first real run vs second real run of the same tasks
    fs = first_second(rows)
    def rvr(yy, cc):
        return T.gap((T.count((fs == 1) & (yy == 1) & cc), T.count((fs == 1) & (yy == 1))),
                     (T.count((fs == 2) & (yy == 1) & cc), T.count((fs == 2) & (yy == 1))))
    ctrl = rvr(y, caught)
    null_ctrl = [rvr(lab(d), cau(d))["point"] for d in draws[1:]]
    out["real-vs-real control (first vs second real run)"] = ctrl | {
        "tasks": len(set(task[fs == 1])), "positives first": int(((fs == 1) & (y == 1)).sum()), "positives second": int(((fs == 2) & (y == 1)).sum()),
        "null values": null_ctrl, "p (two-sided)": (1 + sum(abs(g) >= abs(ctrl["point"]) for g in null_ctrl)) / (n_null + 1)}

    # Planted check: caught simulated positives turned missed
    sim_pos = np.flatnonzero((world == 1) & (y == 1))
    sim_caught = np.flatnonzero((world == 1) & (y == 1) & caught)
    unplanted = out["probe"]["TPR real minus simulated"]["point"]
    plants = {}
    for frac, tag in ((0.05, "5% (frozen in the plan)"), (0.10, "10% (added after the plan)"), (0.20, "20% (added after the plan)")):
        n = round(frac * len(sim_pos))
        if n > len(sim_caught):
            plants[tag] = f"not possible: {len(sim_caught)} caught simulated positives"
            continue
        c = caught.copy()
        c[np.random.default_rng(0).choice(sim_caught, n, replace=False)] = False
        g = world_stats(y, c, world, T)["TPR real minus simulated"]
        shift = n / len(sim_pos)
        plants[tag] = {"calls turned missed": n, "planted shift": shift, "estimate": g["point"], "95%": g["95%"],
                       "detected": detected(g["point"], tuple(g["95%"]), unplanted, shift), "widening direction": widening(unplanted, shift)}
    out["planted check"] = plants

    if tfidf_path:
        tf = json.loads(tfidf_path.read_text())
        if tf["probe output sha256"] != out["probe output sha256"]:
            raise ValueError("the TF-IDF output was made from a different probe output")
        s = np.array(tf["scores"]); c = np.array([ch == "1" for ch in tf["caught"]])
        out["TF-IDF (raw text)"] = world_stats(y, c, world, T) | {"AUROC real (fold mean)": fold_auroc(y, s, world == 0, fold),
                                                                  "AUROC simulated (fold mean)": fold_auroc(y, s, world == 1, fold),
                                                                  "output": str(tfidf_path), "sha256": hashlib.sha256(tfidf_path.read_bytes()).hexdigest()}
    stamp = probe_path.stem.removeprefix("destructive_probe_")
    dest = HERE / "results" / f"destructive_analysis_{stamp}.json"
    with open(dest, "x") as fh:  # results/ is append-only: a rerun needs a new name
        fh.write(json.dumps(out, indent=1) + "\n")
    for name in ("probe", "TF-IDF (raw text)"):
        if name in out:
            o = out[name]
            print(f"{name}: real TPR {o['real']['TPR']:.3f} (FPR {o['real']['FPR']:.3f}, {o['real']['positives']} positives), "
                  f"simulated TPR {o['simulated']['TPR']:.3f} (FPR {o['simulated']['FPR']:.3f}, {o['simulated']['positives']} positives); "
                  f"TPR gap {o['TPR real minus simulated']['point']:+.3f} {np.round(o['TPR real minus simulated']['95%'], 3).tolist()}; "
                  f"AUROC real {o['AUROC real (fold mean)']:.3f}, simulated {o['AUROC simulated (fold mean)']:.3f}")
    n = out["null (within-run label shuffles, selection repeated)"]
    print(f"null: TPR gap p {n['TPR real minus simulated']['p (two-sided)']:.3f}, AUROC real p {n['AUROC real']['p']:.3f}, AUROC sim p {n['AUROC simulated']['p']:.3f}")
    c = out["real-vs-real control (first vs second real run)"]
    print(f"real-vs-real: {c['point']:+.3f} {np.round(c['95%'], 3).tolist()} over {c['tasks']} tasks, null p {c['p (two-sided)']:.3f}")
    for tag, p in plants.items():
        print(f"plant {tag}: {p if isinstance(p, str) else {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()}}")
    print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None)
