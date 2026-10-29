"""TPR at a threshold frozen at 1% FPR, real vs simulated, off- and on-policy (local, free); the README has the design.

Off-policy: the refit's replicate-0 scores, rows and thresholds, with the refit's 500-resample intervals. On-policy:
q3's per-call file, its 2,000 refit draws and its label-shuffle draws (real->real only). ROC curves pool margins over
each fold's frozen threshold. Positive control: a planted 5-point TPR drop in either world, fixed-score task bootstrap
(on-policy within the 5 outer folds, as q3). Below 1% FPR: per-world thresholds at 0.2% and 0.5%, task bootstrap.

  uv run --no-project --with numpy python experiments/2026-10-09_q2_tpr_at_1pct/analyze_tpr_frozen.py <refit scores npz> \
      <refit rows json> <refit probes npz> <refit aggregate json> <q3 per-call json> <q3 draws json> <q3 null draws json>
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from probe_shift.planted import detected, widening

HERE = Path(__file__).parent
W = ("real", "sim")
GRID = np.logspace(-3, 0, 61)  # FPRs for the log-scale ROC curves


def roc_points(margin: np.ndarray, y: np.ndarray) -> list[list[float]]:
    """(FPR, TPR) at thresholds placed at each grid FPR's quantile of the negatives' margins, plus margin 0 (the frozen
    threshold) and the step just above it, so the frozen point lies on the curve; sorted by FPR, then TPR."""
    neg, pos = np.sort(margin[y == 0])[::-1], margin[y == 1]
    ts = [neg[int(np.floor(f * len(neg)))] if int(np.floor(f * len(neg))) < len(neg) else -np.inf for f in GRID] + [0.0]
    out = {(round(float((margin[y == 0] > t).mean()), 5), round(float((pos > t).mean()), 5)) for t in ts}
    return [list(x) for x in sorted(out)]


def task_draws(n: int, strata, n_draws: int, rng) -> np.ndarray:
    """[draw, task] resampled task indices: global, or within each stratum (q3's on-policy draws resample within the 5
    outer folds, so every fold keeps its tasks' count)."""
    if strata is None:
        return rng.integers(0, n, (n_draws, n))
    strata = np.asarray(strata)
    return np.concatenate([rng.choice(np.flatnonzero(strata == s), (n_draws, int((strata == s).sum()))) for s in np.unique(strata)], axis=1)


def task_boot_diff(caught: np.ndarray, world: np.ndarray, task: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Real minus simulated TPR (positives only) per bootstrap draw of tasks; draws: [draw, n_tasks] task indices."""
    n = task.max() + 1
    c = np.zeros((2, n)); t = np.zeros((2, n))
    np.add.at(c, (world, task), caught)
    np.add.at(t, (world, task), 1)
    cs, ts = c[:, draws].sum(-1), t[:, draws].sum(-1)
    return cs[0] / ts[0] - cs[1] / ts[1]


def plant(caught, world, task, seeds=200, gap=0.05, n_draws=2000, strata=None):
    """Detection rate of a planted gap, each direction: a random `gap` share of one world's positives (all positives,
    flipped only where caught) turned from caught to missed. Old rule: detected = the 95% interval of real minus
    simulated TPR excludes zero. Aswin's rule (2026-10-10, probe_shift.planted): the estimate moves by about the planted
    gap and its 95% interval excludes the unplanted estimate; both are reported, with which direction widens the gap.
    Scores and thresholds are held fixed: no refit."""
    rng = np.random.default_rng(0)
    draws = task_draws(task.max() + 1, strata, n_draws, rng)
    base = np.percentile(task_boot_diff(caught, world, task, draws), [2.5, 97.5])
    obs = task_boot_diff(caught, world, task, np.arange(task.max() + 1)[None])[0]
    out = {"planted gap (TPR points)": 100 * gap, "fixed-score interval, unplanted (real minus sim, points)": [round(100 * x, 2) for x in base],
           "observed real minus sim (points)": round(100 * float(obs), 2)}
    for w, name in ((1, "simulated lowered"), (0, "real lowered")):
        idx = np.flatnonzero(world == w)
        hits, hits_new, mids = 0, 0, []
        want = gap if w == 1 else -gap  # lowering the simulated world raises real minus simulated, and the reverse
        for s in range(seeds):
            c = caught.copy()
            c[np.random.default_rng([1, w, s]).choice(idx[c[idx] == 1], round(gap * len(idx)), replace=False)] = 0
            lo, hi = np.percentile(task_boot_diff(c, world, task, draws), [2.5, 97.5])
            hits += (lo > 0) or (hi < 0)
            mids.append(task_boot_diff(c, world, task, np.arange(task.max() + 1)[None])[0])
            hits_new += detected(mids[-1], (lo, hi), obs, want)
        out[name] = {"detected": f"{hits} of {seeds}", "median planted real minus sim (points)": round(100 * float(np.median(mids)), 2),
                     "detected, Aswin's rule (moves about the planted gap; interval excludes the unplanted estimate)": f"{hits_new} of {seeds}",
                     "this direction widens the gap": widening(obs, want)}
    return out


def low_fpr(margin, y, world, task, fprs=(0.002, 0.005), n_draws=2000, strata_of=None):
    """Below the 1% operating point (Aswin): per world, the threshold at FPR f among that world's negatives, and TPR
    there; real minus simulated with a task bootstrap that recomputes both thresholds (scores fixed, no refit), and
    how many negatives sit above each threshold."""
    names, task = np.unique(task, return_inverse=True)
    rng, out = np.random.default_rng(0), {}
    draws = task_draws(len(names), None if strata_of is None else [strata_of[t] for t in names], n_draws, rng)
    by = [np.flatnonzero(task == t) for t in range(task.max() + 1)]
    def at(idx, f):
        r = []
        for w in (0, 1):
            i = idx[world[idx] == w]
            neg = np.sort(margin[i][y[i] == 0])[::-1]
            k = int(np.floor(f * len(neg)))
            r.append(((margin[i][y[i] == 1] > neg[k]).mean(), k))
        return r
    for f in fprs:
        (tr, kr), (ts, ks) = at(np.arange(len(y)), f)
        d = []
        for row in draws:
            idx = np.concatenate([by[t] for t in row])
            (a, _), (b, _) = at(idx, f)
            d.append(a - b)
        out[f"FPR {100 * f:.1f}%"] = {"TPR real": round(float(tr), 4), "TPR sim": round(float(ts), 4), "negatives above the threshold (real, sim)": [kr, ks],
                                    "real minus sim (points)": [round(100 * float(tr - ts), 2)] + [round(100 * float(x), 2) for x in np.percentile(d, [2.5, 97.5])]}
    return out


def offpolicy(scores_p, rows_p, probes_p, agg_p):
    rows, z = json.loads(Path(rows_p).read_text()), np.load(scores_p)
    tab = {(p["fold"], p["train"]): p for p in json.loads(str(np.load(probes_p)["table"]))}
    y = np.array([r["y"] for r in rows]); world = np.array([0 if r["arm"] == "real-attack-e1" else 1 for r in rows])
    fold = np.array([r["fold"] for r in rows]); _, task = np.unique([r["main_task"] for r in rows], return_inverse=True)
    if not (z["labels"] == y).all():
        raise ValueError("rows and scores disagree on labels")
    agg = json.loads(Path(agg_p).read_text())["readers"]["qwen3.5"]
    out = {"cells": {}}
    for t in (0, 1):
        s = z["scores"][fold, t, np.arange(len(y))]
        m = s - np.array([tab[(f, W[t])][f"threshold_{W[t]}"] for f in fold])  # margin over the frozen threshold
        for w in (0, 1):
            k, a = f"{W[t]}->{W[w]}", agg[f"{W[t]}->{W[w]}"]
            sel = world == w
            tpr, fpr = (m[sel & (y == 1)] > 0).mean(), (m[sel & (y == 0)] > 0).mean()
            if abs(tpr - a["catch, frozen training-world threshold"]["value"]) > 1e-4 or abs(fpr - a["false-alarm rate, frozen"]["value"]) > 1e-4:
                raise ValueError(f"{k}: TPR {tpr} or FPR {fpr} does not reproduce the refit's")
            out["cells"][k] = {"TPR": round(float(tpr), 4), "TPR 95% (refit)": a["catch, frozen training-world threshold"]["ci95"],
                               "FPR": round(float(fpr), 4), "FPR 95% (refit)": a["false-alarm rate, frozen"]["ci95"],
                               "positives": int((sel & (y == 1)).sum()), "negatives": int((sel & (y == 0)).sum())}
            if t == 0:
                out.setdefault("ROC, real-trained probe (FPR, TPR)", {})[W[w]] = roc_points(m[sel], y[sel])
        if t == 0:
            out["below 1% FPR, real-trained probe"] = low_fpr(m, y, world, task)
            pos = y == 1
            out["positive control"] = plant((m[pos] > 0).astype(int), world[pos], np.unique(task[pos], return_inverse=True)[1])
    g = agg["domain gap, trained on real (real->real minus real->sim)"]
    out["real minus sim, real-trained (refit 95%)"] = {k: g[k] for k in ("catch, frozen training-world threshold", "false-alarm rate, frozen")}
    out["tasks"] = int(task.max() + 1)
    out["tasks per world"] = {W[w]: len(np.unique(task[world == w])) for w in (0, 1)}
    return out


def onpolicy(percall_p, draws_p, null_p):
    pc, dr = json.loads(Path(percall_p).read_text()), json.loads(Path(draws_p).read_text())
    nulls = [d for d in json.loads(Path(null_p).read_text())["draws"] if d["draw"] > 0]  # q3's label-shuffle draws
    meta, folds = pc["meta"], pc["folds"]
    out, low = {"cells": {}, "calibration read-only calls per fold (real, frozen)": None}, []
    obs = {c: dr["draws"][0]["cells"][c]["values"] for c in dr["draws"][0]["cells"]}
    boot = [d for d in dr["draws"] if d["draw"] > 0]
    ci = lambda c, k: [round(float(x), 4) for x in np.percentile([d["cells"][c]["values"][k] for d in boot], [2.5, 97.5])]
    for c in ("real->real", "real->sim", "sim->real", "sim->sim"):
        cell = pc["draws"][0]["cells"][c]; p = cell["per_call"]
        ys = np.array([meta[i]["y"] for i in p["row"]]); fl = np.array(p["flag_frozen"])
        tpr, fpr = fl[ys == 1].mean(), fl[ys == 0].mean()
        if abs(tpr - obs[c]["catch_frozen"]) > 1e-4 or abs(fpr - obs[c]["false_alarm_frozen"]) > 1e-4:
            raise ValueError(f"{c}: the per-call file does not reproduce the observed draw")
        out["cells"][c] = {"TPR": round(float(tpr), 4), "TPR 95% (q3's refit draws)": ci(c, "catch_frozen"), "FPR": round(float(fpr), 4),
                           "FPR 95% (q3's refit draws)": ci(c, "false_alarm_frozen"), "positives": int((ys == 1).sum()), "negatives": int((ys == 0).sum())}
        nv = [d["cells"][c]["values"]["catch_frozen"] for d in nulls if c in d["cells"]]  # q3 shuffled real->real only
        if nv:
            out["cells"][c]["TPR label-shuffle null (q3, mean / max / p)"] = [round(float(np.mean(nv)), 4), round(float(np.max(nv)), 4),
                                                                               round(float((1 + sum(v >= tpr for v in nv)) / (1 + len(nv))), 4)]
        if c.startswith("real->"):
            thr = {f["fold"]: f["threshold"] for f in cell["folds"] if f["threshold_kind"] == "frozen"}
            m = np.array(p["score"]) - np.array([thr[folds[meta[i]["task"]]] for i in p["row"]])
            if ((m > 0) != fl).any():
                raise ValueError(f"{c}: margins disagree with the frozen flags")
            out.setdefault("ROC, real-trained probe (FPR, TPR)", {})[c[6:]] = roc_points(m, ys)
            low.append((m, ys, np.array([0 if c == "real->real" else 1] * len(ys)), np.array([meta[i]["task"] for i in p["row"]])))
            out["calibration read-only calls per fold (real, frozen)"] = {f["fold"]: f["calibration_read_only_calls"] for f in cell["folds"] if f["threshold_kind"] == "frozen"}
    out["below 1% FPR, real-trained probe"] = low_fpr(*(np.concatenate(x) for x in zip(*low)), strata_of=folds)
    d = np.array([b["cells"]["real->real"]["values"]["catch_frozen"] - b["cells"]["real->sim"]["values"]["catch_frozen"] for b in boot])
    f = np.array([b["cells"]["real->real"]["values"]["false_alarm_frozen"] - b["cells"]["real->sim"]["values"]["false_alarm_frozen"] for b in boot])
    out["real minus sim, real-trained (q3's refit draws, 95%)"] = {
        "TPR": [round(obs["real->real"]["catch_frozen"] - obs["real->sim"]["catch_frozen"], 4)] + [round(float(x), 4) for x in np.percentile(d, [2.5, 97.5])],
        "FPR": [round(obs["real->real"]["false_alarm_frozen"] - obs["real->sim"]["false_alarm_frozen"], 4)] + [round(float(x), 4) for x in np.percentile(f, [2.5, 97.5])]}
    rows = [(meta[i]["world"], meta[i]["task"], fl_) for c in ("real->real", "real->sim") for i, fl_, y_ in
            zip(pc["draws"][0]["cells"][c]["per_call"]["row"], pc["draws"][0]["cells"][c]["per_call"]["flag_frozen"],
                [meta[i]["y"] for i in pc["draws"][0]["cells"][c]["per_call"]["row"]]) if y_ == 1]
    _, task = np.unique([r[1] for r in rows], return_inverse=True)
    names = np.unique([r[1] for r in rows])
    out["positive control"] = plant(np.array([int(r[2]) for r in rows]), np.array([0 if r[0] == "real" else 1 for r in rows]), task,
                                    strata=[folds[t] for t in names])
    out["positive control"]["resampling"] = "within the 5 outer folds, as q3's draws"
    out["tasks"] = int(task.max() + 1)
    return out


def main(a: list[str]):
    res = {"inputs sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in a},
           "off-policy (Qwen3.5-27B reading Sol's runs; refit)": offpolicy(*a[:4]), "on-policy (Qwen3.5-27B acting; September simulator)": onpolicy(*a[4:7])}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"tpr_frozen_{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    for k, v in res.items():
        if k == "inputs sha256":
            continue
        print(f"== {k} ({v['tasks']} tasks" + (f"; per world {v['tasks per world']}" if "tasks per world" in v else "") + ")")
        for c, x in v["cells"].items():
            print(f"  {c}: TPR {x['TPR']:.4f} {next(v_ for k_, v_ in x.items() if k_.startswith('TPR 95'))} | FPR {x['FPR']:.4f} "
                  f"{next(v_ for k_, v_ in x.items() if k_.startswith('FPR 95'))} | {x['positives']} positives, {x['negatives']} negatives"
                  + (f" | TPR null mean/max/p {x['TPR label-shuffle null (q3, mean / max / p)']}" if 'TPR label-shuffle null (q3, mean / max / p)' in x else ""))
        print("  real minus sim:", json.dumps(next(v_ for k_, v_ in v.items() if k_.startswith("real minus sim"))))
        print("  positive control:", json.dumps(v["positive control"]))
        print("  below 1% FPR:", json.dumps(v["below 1% FPR, real-trained probe"]))
        if v.get("calibration read-only calls per fold (real, frozen)"):
            print("  calibration read-only calls per fold:", v["calibration read-only calls per fold (real, frozen)"])
    print(out)


if __name__ == "__main__":
    main(sys.argv[1:8])
