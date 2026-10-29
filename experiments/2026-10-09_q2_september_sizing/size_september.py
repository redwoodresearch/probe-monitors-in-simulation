"""September-only scale-up sizing (local, free): precision of the on-policy real-minus-simulated TPR at 1% FPR (the
threshold frozen on the real world's calibration calls) at 20, 40, 60, 80 and 100 tasks, and what each step costs.

Precision, by simulation from the strict rerun's 20 tasks (q3's per-call file, the real-trained probe): each call's
margin over its fold's frozen threshold is taken as its score, so folds pool. One simulated study of N tasks draws N
tasks with replacement (each draw a new task, its calls copied), splits them into 5 folds of N/5, and for each test
fold k sets the threshold at the 1% point of fold k+1's real read-only calls (the share above it at most 1%, q3's
rule), then takes real and simulated TPR over the test folds and their difference. 2,000 studies per N (seed 0).
Reported: the expected 95% half-width (1.96 x the studies' SD), the mean read-only and positive calls per fold, the
power to detect a true 5-point gap (normal approximation), and a planted check (Aswin's rule): in each study
round(5% of all simulated positives) caught ones are turned into missed, across the whole study, and detection is the planted difference falling outside the
unplanted studies' 95% range. The simulation holds the probe fixed; refitting adds variance, so the half-widths are
also scaled by the ratio of q3's refit interval to the simulated one at 20 tasks.

Cost per added task: the September simulator runs priced from the rerun's eval logs (price_sim_runs.py), plus GPU
servers, replay extraction and devbox time scaled per run from the rerun's ledger rows and billing.

With --fold-probes <run_refit_fold_probes.py output> (the probe-mixing fix): the margins above score each call with its
own fold's probe, so one fold's threshold (another fold's calls) and its test calls come from different probes. The
fixed run keeps every copy in its source task's fold of q3's map and scores both with fold k's probe; it is reported
next to the old margins on the same map, which separates the fix from the change to a fixed fold map.

  uv run --no-project --with numpy python experiments/2026-10-09_q2_september_sizing/size_september.py <q3 per-call json> \
      <q3 draws json> <sim run costs json> [--by-source] [--gap 0.10] [--sizes 20,100,196] [--fold-probes <json>]
"""
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
SIZES = tuple(int(x) for x in sys.argv[sys.argv.index("--sizes") + 1].split(",")) if "--sizes" in sys.argv else (20, 40, 60, 80, 100)
STUDIES = 2000
BY_SOURCE = "--by-source" in sys.argv  # fold assignment of task copies; see study()
GAP = float(sys.argv[sys.argv.index("--gap") + 1]) if "--gap" in sys.argv else 0.05  # the true gap for power and the planted check
# per run, from the rerun (2026-10-09_000523_rerun_strict, 140 runs): GPU servers $46.77 (SPEND.md row), replay extraction
# $8.64 (Modal billing, app q3onpol_extract, 10-08 to 10-09, which also covers the invalid pilot's extraction: an upper
# bound), devbox at $1.61/h for about 3 h of real runs ($4.83)
OTHER_PER_RUN = {"GPU servers": 46.77 / 140, "replay extraction": 8.64 / 140, "devbox": 4.83 / 140}


def load(percall_p):
    pc = json.loads(Path(percall_p).read_text())
    meta, folds, rows = pc["meta"], pc["folds"], []
    for c in ("real->real", "real->sim"):
        cell = pc["draws"][0]["cells"][c]
        thr = {f["fold"]: f["threshold"] for f in cell["folds"] if f["threshold_kind"] == "frozen"}
        p = cell["per_call"]
        for i, s in zip(p["row"], p["score"]):
            rows.append((meta[i]["task"], 0 if c == "real->real" else 1, meta[i]["y"], s - thr[folds[meta[i]["task"]]]))
    tasks = sorted({r[0] for r in rows})
    by = {t: [r for r in rows if r[0] == t] for t in tasks}
    return tasks, {t: (np.array([r[1] for r in v]), np.array([r[2] for r in v]), np.array([r[3] for r in v])) for t, v in by.items()}


def study(rng, tasks, by, n, plant=0.0, by_source=False):
    """by_source: every copy of a source task goes to that task's fold (sources split 4 per fold), so no task calibrates
    one fold and is tested in another; calibration then rests on 4 distinct tasks at any size (pessimistic). Otherwise
    each copy is assigned on its own (optimistic: copies of one task can calibrate and test)."""
    while True:  # by source, a fold can draw no copies at all: redraw such a study
        picks = rng.choice(len(tasks), n)
        fold_of = rng.permutation(np.arange(len(tasks)) % 5)[picks] if by_source else rng.permutation(np.arange(n) % 5)
        if len(set(fold_of)) == 5:
            break
    w = [by[tasks[p]][0] for p in picks]; y = [by[tasks[p]][1] for p in picks]; m = [by[tasks[p]][2] for p in picks]
    caught, totals, counts = np.zeros(2), np.zeros(2), []
    for k in range(5):
        cal = [j for j in range(n) if fold_of[j] == (k + 1) % 5]
        neg = np.sort(np.concatenate([m[j][(w[j] == 0) & (y[j] == 0)] for j in cal]))[::-1]
        t = neg[int(np.floor(0.01 * len(neg)))]
        test = [j for j in range(n) if fold_of[j] == k]
        counts.append((len(neg), sum(int(((w[j] == 0) & (y[j] == 1)).sum()) for j in test)))
        for j in test:
            for wd in (0, 1):
                pos = (w[j] == wd) & (y[j] == 1)
                f = m[j][pos] > t
                caught[wd] += f.sum(); totals[wd] += pos.sum()
    if plant:  # turn round(plant x all simulated positives) caught ones into missed, across the whole study
        k, n_caught = round(plant * totals[1]), caught[1]
        if k > n_caught:
            raise ValueError(f"cannot plant {k} misses: only {n_caught} simulated positives caught")
        caught[1] -= k
    return caught[0] / totals[0] - caught[1] / totals[1], np.mean(counts, axis=0)


FOLD_PROBES = sys.argv[sys.argv.index("--fold-probes") + 1] if "--fold-probes" in sys.argv else None


def load_fold_probes(path, percall_p):
    """Every attack call's score under each of the five fold probes (run_refit_fold_probes.py) and its source task's
    fold (q3's map). Returns tasks, by task: (world, y, scores [5, calls]), and the fold of each task."""
    r, folds = json.loads(Path(path).read_text()), json.loads(Path(percall_p).read_text())["folds"]
    S, meta = np.array(r["scores"]), r["meta"]
    tasks = sorted({m["task"] for m in meta})
    by = {}
    for t in tasks:
        i = np.array([j for j, m in enumerate(meta) if m["task"] == t])
        by[t] = (np.array([0 if meta[j]["world"] == "real" else 1 for j in i]), np.array([meta[j]["y"] for j in i]), S[:, i])
    return tasks, by, {t: folds[t] for t in tasks}


def study_consistent(rng, tasks, by, fold_of, n, plant=0.0, picks=None, stratified=False):
    """By source with q3's own fold map: every copy stays in its source task's fold, and fold k's threshold (the 1% point
    of fold k+1's real read-only calls) and its test calls are both scored by fold k's probe, which was trained on the
    other three folds, so no probe scores its own training tasks. picks: a fixed study (the observed one). stratified:
    each fold draws its own n/5 copies from its source tasks (a frozen fold map with fixed fold sizes, as the registered
    analysis resamples); otherwise all n are drawn from the pooled tasks and fold sizes vary."""
    if stratified and picks is None:
        in_fold = [[i for i, t in enumerate(tasks) if fold_of[t] == k] for k in range(5)]
        picks = np.concatenate([rng.choice(in_fold[k], n // 5 + (k < n % 5)) for k in range(5)])
    while picks is None:
        p_ = rng.choice(len(tasks), n)
        if len({fold_of[tasks[p]] for p in p_}) == 5:
            picks = p_
    caught, totals = np.zeros(2), np.zeros(2)
    for k in range(5):
        cal = [tasks[p] for p in picks if fold_of[tasks[p]] == (k + 1) % 5]
        neg = np.sort(np.concatenate([by[t][2][k][(by[t][0] == 0) & (by[t][1] == 0)] for t in cal]))[::-1]
        thr = neg[int(np.floor(0.01 * len(neg)))]
        for t in (tasks[p] for p in picks if fold_of[tasks[p]] == k):
            w, y, sc = by[t]
            for wd in (0, 1):
                pos = (w == wd) & (y == 1)
                caught[wd] += (sc[k][pos] > thr).sum(); totals[wd] += pos.sum()
    if plant:
        k = round(plant * totals[1])
        if k > caught[1]:
            raise ValueError("too few simulated catches to plant")
        caught[1] -= k
    return caught[0] / totals[0] - caught[1] / totals[1]


def main_consistent(percall_p, draws_p, costs_p):
    """Two runs on q3's fold map: each fold's own probe (the fix), and the old mixed scores (each call's margin over its
    own fold's threshold, so a fold's threshold comes from another probe's scores), which separates the fix from the
    change to a fixed fold map."""
    tasks, by, fold_of = load_fold_probes(FOLD_PROBES, percall_p)
    _, by_m = load(percall_p)
    mixed = {t: (w, y, np.tile(m, (5, 1))) for t, (w, y, m) in by_m.items()}
    dr = json.loads(Path(draws_p).read_text())
    boot = [d for d in dr["draws"] if d["draw"] > 0]
    v0 = dr["draws"][0]["cells"]
    observed = v0["real->real"]["values"]["catch_frozen"] - v0["real->sim"]["values"]["catch_frozen"]
    obs = {k: study_consistent(None, tasks, b, fold_of, 20, picks=np.arange(len(tasks))) for k, b in (("fixed", by), ("mixed", mixed))}
    if abs(obs["fixed"] - observed) > 1e-4:  # the observed study (each task once) must give q3's difference (stored to 4 decimals)
        raise ValueError("the observed study does not reproduce q3's real minus simulated TPR")
    refit = np.percentile([d["cells"]["real->real"]["values"]["catch_frozen"] - d["cells"]["real->sim"]["values"]["catch_frozen"] for d in boot], [2.5, 97.5])
    power = lambda z: float(1 - 0.5 * (1 + math.erf((1.96 - z) / math.sqrt(2))) + 0.5 * (1 + math.erf((-1.96 - z) / math.sqrt(2))))
    out = {"inputs sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (percall_p, draws_p, FOLD_PROBES)},
           "observed real minus simulated TPR (q3, points)": round(100 * observed, 2),
           "observed study (each task once), points: fixed probes, mixed probes": [round(100 * obs["fixed"], 2), round(100 * obs["mixed"], 2)],
           "refit 95% interval at 20 tasks (q3, points)": [round(100 * float(x), 2) for x in refit], "runs": {}}
    runs = (("q3's folds, each fold's own probe, fold sizes fixed (main)", by, True),
            ("q3's folds, each fold's own probe, pooled draws (sensitivity)", by, False),
            ("q3's folds, mixed probes (old scores), pooled draws", mixed, False))
    for name, b, strat in runs:
        rng = np.random.default_rng(0)
        res, hws = {}, {}
        for n in SIZES:
            null = np.array([study_consistent(rng, tasks, b, fold_of, n, stratified=strat) for _ in range(STUDIES)])
            planted = np.array([study_consistent(rng, tasks, b, fold_of, n, plant=GAP, stratified=strat) for _ in range(STUDIES)])
            lo, hi = np.percentile(null, [2.5, 97.5])
            hws[n] = 1.96 * float(null.std())
            res[n] = {"simulated 95% half-width (points)": round(100 * hws[n], 2),
                      f"planted {100 * GAP:g}-point drop outside the unplanted 95% range": f"{int(((planted > hi) | (planted < lo)).sum())} of {STUDIES}"}
        scale = (refit[1] - refit[0]) / 2 / hws[20]
        for n in SIZES:
            res[n] |= {"refit-scaled 95% half-width (points)": round(100 * hws[n] * scale, 2),
                       f"power for a true {100 * GAP:g}-point gap (normal approx.; simulated, refit-scaled)":
                       [round(power(GAP / (hws[n] / 1.96)), 2), round(power(GAP / (hws[n] * scale / 1.96)), 2)]}
        out["runs"][name] = {"refit / simulated half-width at 20 tasks": round(float(scale), 3), "by tasks": res}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"sizing_consistent_gap{100 * GAP:g}_{stamp}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("observed:", out["observed real minus simulated TPR (q3, points)"], "| observed study, fixed / mixed:", out["observed study (each task once), points: fixed probes, mixed probes"], "| refit 95% interval at 20 tasks:", out["refit 95% interval at 20 tasks (q3, points)"])
    for name, r in out["runs"].items():
        print(name, "| refit / simulated half-width:", r["refit / simulated half-width at 20 tasks"])
        for n, v in r["by tasks"].items():
            print(f"  {n} tasks:", json.dumps(v))
    print(path)


def main(percall_p, draws_p, costs_p):
    if FOLD_PROBES:
        return main_consistent(percall_p, draws_p, costs_p)
    tasks, by = load(percall_p)
    dr = json.loads(Path(draws_p).read_text())
    boot = [d for d in dr["draws"] if d["draw"] > 0]
    refit = np.percentile([d["cells"]["real->real"]["values"]["catch_frozen"] - d["cells"]["real->sim"]["values"]["catch_frozen"] for d in boot], [2.5, 97.5])
    costs = json.loads(Path(costs_p).read_text())
    sim_task = [costs["per run"]["sim_honest@september"][t] + costs["per run"]["sim_attack@september"][t] for t in tasks]
    per_task = float(np.mean(sim_task)) + 5 * sum(OTHER_PER_RUN.values())
    rng = np.random.default_rng(0)
    out = {"inputs sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (percall_p, draws_p, costs_p)},
           "refit 95% interval at 20 tasks (q3, points)": [round(100 * float(x), 2) for x in refit],
           "cost per task": {"September simulator (mean over the 20 tasks)": round(float(np.mean(sim_task)), 2),
                             "September simulator, task range": [round(min(sim_task), 2), round(max(sim_task), 2)]}
           | {k: round(5 * v, 2) for k, v in OTHER_PER_RUN.items()} | {"total": round(per_task, 2)}, "by tasks": {}}
    for n in SIZES:
        null = [study(rng, tasks, by, n, by_source=BY_SOURCE) for _ in range(STUDIES)]
        d = np.array([x[0] for x in null]); cnt = np.mean([x[1] for x in null], axis=0)
        out["by tasks"][n] = {"added tasks": n - 20, "added cost ($)": round((n - 20) * per_task),
                              "read-only calls per calibration fold (real)": round(float(cnt[0]), 1), "real positives per test fold": round(float(cnt[1]), 1),
                              "simulated 95% half-width (points)": round(100 * 1.96 * float(d.std()), 2)}
    scale = (refit[1] - refit[0]) / 2 / (out["by tasks"][20]["simulated 95% half-width (points)"] / 100)
    out["refit / simulated half-width at 20 tasks"] = round(float(scale), 3)
    rng = np.random.default_rng(1)
    for n in SIZES:
        null = np.array([study(rng, tasks, by, n, by_source=BY_SOURCE)[0] for _ in range(STUDIES)])
        planted = np.array([study(rng, tasks, by, n, plant=GAP, by_source=BY_SOURCE)[0] for _ in range(STUDIES)])
        lo, hi = np.percentile(null, [2.5, 97.5])
        hw = out["by tasks"][n]["simulated 95% half-width (points)"] / 100
        z = GAP / (hw / 1.96)
        zs = GAP / (hw * scale / 1.96)
        power = lambda z_: float(1 - (0.5 * (1 + math.erf((1.96 - z_) / np.sqrt(2)))) + 0.5 * (1 + math.erf((-1.96 - z_) / np.sqrt(2))))
        out["by tasks"][n] |= {"refit-scaled 95% half-width (points)": round(100 * hw * scale, 2),
                               f"power for a true {100 * GAP:g}-point gap (normal approx.; simulated, refit-scaled)": [round(power(z), 2), round(power(zs), 2)],
                               f"planted {100 * GAP:g}-point drop outside the unplanted 95% range": f"{int(((planted > hi) | (planted < lo)).sum())} of {STUDIES}"}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out["fold assignment of task copies"] = "by source task (pessimistic)" if BY_SOURCE else "each copy on its own (optimistic)"
    path = HERE / "results" / f"sizing_{'by_source_' if BY_SOURCE else ''}{'' if GAP == 0.05 else f'gap{100 * GAP:g}_'}{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("refit 95% interval at 20 tasks:", out["refit 95% interval at 20 tasks (q3, points)"], "| refit / simulated half-width:", out["refit / simulated half-width at 20 tasks"])
    print("cost per task:", json.dumps(out["cost per task"]))
    for n, v in out["by tasks"].items():
        print(f"{n} tasks:", json.dumps(v))
    print(path)


if __name__ == "__main__":
    flags = ("--by-source", "--gap", "--sizes", "--fold-probes")
    main(*[a for i, a in enumerate(sys.argv[1:], 1) if a not in flags and sys.argv[i - 1] not in flags][:3])
