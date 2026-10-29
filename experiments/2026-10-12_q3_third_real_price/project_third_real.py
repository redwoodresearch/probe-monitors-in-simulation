"""Power projection for a third real run (PRICE.md). Local CPU, stored scores only, $0.

Acting: real run 2 - real run 1 catch at 1% (frozen thresholds), the 196-task scale-up's draw-0 per-call file; unit = task.
Reading: #36's real e2 - e1 catch at 5% and 1% (output_end, L63, #36's frozen thresholds), the 939 cells; unit = cell,
intervals resample main tasks. For each subset size: R seeded subsets, B task resamples inside each (fixed fits), half-width
of the 95% percentile interval. kappa = stored full-data half-width / fixed-fit full-data half-width scales it up.
    uv run python experiments/2026-10-12_q3_third_real_price/project_third_real.py --draws0 <...draws0-0_2026-10-10_141824.json> \
        --scores36 <#36 scores_L63.npz>
"""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).parent
REPO = HERE.parent.parent
Q3 = REPO / "experiments/2026-10-06_q3_attack_detector/results"
RUNS36, METRICS36 = Q3 / "run_scores_2026-10-06_181548/runs.json", Q3 / "metrics_2026-10-06_181548__2026-10-06_182739.json"
FIX939 = REPO / "experiments/2026-10-08_q3_swap_cue/results/fix939_POSTHOC_2026-10-11_020519.json"
VERDICTS = REPO / "experiments/2026-10-08_q3_onpolicy_figures/results/scaleup196_verdicts_2026-10-10_164622.json"
SHA = {"draws0": "c2656658baaf3aa150f83f4d105e7572bfd854f98032831621419e1f8e8f36bf",
       "scores36": "60f8e0d4b7b45baf46c6cf96f61677c7a0e966a3698a99a913763a4d763e6f10",
       "runs36": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"}
SIZES = {"acting": (50, 100), "reading": (200, 400)}
R, B, B_FULL, SEED, XS, MC = 50, 1000, 2000, 20261012, (5.0, 10.0), 400_000
E1_IN_TRAINING = "real_e2_minus_e1 (e1 was in training)"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rule(s, alpha):  # #36's frozen-threshold rule (analyze_fix939.py, unit weights): the (k+1)-th largest, k = floor(alpha x n)
    o = np.sort(s)[::-1]
    return float(o[min(int(np.floor(alpha * len(s) + 1e-9)), len(s) - 1)])


def acting_units(path):
    d = json.loads(Path(path).read_text())
    meta, cells = d["meta"], d["draws"][0]["cells"]
    tasks = sorted({m["task"] for m in meta})
    ti = {t: j for j, t in enumerate(tasks)}
    kn = []
    for name in ("real->real", "real->run2"):
        pc = cells[name]["per_call"]
        rows, flag = np.array(pc["row"]), np.array(pc["flag_frozen"], bool)
        y, t = np.array([meta[r]["y"] for r in rows]), np.array([ti[meta[r]["task"]] for r in rows])
        k = np.bincount(t[y == 1], weights=flag[y == 1], minlength=len(tasks))
        n = np.bincount(t[y == 1], minlength=len(tasks)).astype(float)
        if abs(k.sum() / n.sum() - cells[name]["values"]["catch_frozen"]) > 1e-12:
            raise SystemExit(f"gate: {name} catch_frozen not reproduced")
        kn += [k, n]
    keep = (kn[1] > 0) & (kn[3] > 0)
    U = np.stack([a[keep] for a in kn], 1)  # per task: k1, n1, k2, n2
    return U, np.arange(len(U))  # one unit per task


def acting_gap(U, w):  # w: (draws, units) counts
    s = w @ U
    return 100 * (s[:, 2] / s[:, 3] - s[:, 0] / s[:, 1])


def reading_units(scores_path):
    runs = json.loads(RUNS36.read_text())
    arm, fold, rtask = (np.array([r[k] for r in runs]) for k in ("arm", "fold", "main_task"))
    Z = np.load(scores_path)
    nz, oz = Z["output_end__mean__task_nested"], Z["output_end__mean__task_outer"]
    hon = np.flatnonzero(arm == "real-honest")
    M = json.loads(METRICS36.read_text())["readouts"]["probe_output_end_mean"]["task"]["thresholds"]
    T = {}
    for a in ("0.05", "0.01"):
        T[a] = np.full((5, 5), np.nan)
        for f_ in range(5):
            for v_ in range(5):
                if v_ != f_:
                    idx = hon[(fold[hon] == v_) & ~np.isnan(nz[f_, v_, hon])]
                    T[a][f_, v_] = rule(nz[f_, v_, idx], float(a))
        if max(abs(T[a][int(k[0]), int(k[2])] - v["threshold"]) for k, v in M[a].items()) > 5e-6:
            raise SystemExit(f"gate: #36 thresholds at {a} not rebuilt")

    def valid(i):
        return np.isfinite(oz[i]) and np.isfinite(np.delete(nz[fold[i], :, i], fold[i])).all()

    arms = ("sim-attack", "real-attack-e1", "real-attack-e2")
    idx = {a: {r["cell"]: i for i, r in enumerate(runs) if r["arm"] == a} for a in arms}
    cells = sorted(c for c in set(idx[arms[0]]) & set(idx[arms[1]]) & set(idx[arms[2]]) if all(valid(idx[a][c]) for a in arms))
    f939 = json.loads(FIX939.read_text())["readouts"]["output_end"]
    cols = []
    for a in ("0.05", "0.01"):
        for e in arms[1:]:
            ii = np.array([idx[e][c] for c in cells])
            S, F = np.stack([nz[fold[i], :, i] for i in ii]), fold[ii]
            ok = np.arange(5)[None, :] != F[:, None]
            cols.append(np.where(ok, S > T[a][F], False).sum(1) / ok.sum(1))
        if abs(cols[-1].mean() - cols[-2].mean() - f939[a][E1_IN_TRAINING]["observed"]) > 1e-12:
            raise SystemExit(f"gate: e2 - e1 at {a} not reproduced")
    U = np.stack(cols, 1)  # per cell: e1@5%, e2@5%, e1@1%, e2@1%
    tasks = sorted(set(rtask[[idx[arms[0]][c] for c in cells]]))
    tid = {t: j for j, t in enumerate(tasks)}
    return U, np.array([tid[rtask[idx[arms[0]][c]]] for c in cells])


def reading_gap(col):
    def g(U, w):
        s = w @ np.c_[U[:, col:col + 2], np.ones(len(U))]
        return 100 * (s[:, 1] - s[:, 0]) / s[:, 2]
    return g


def boot(U, unit_task, gap, rng, nb):
    """Resample the subset's main tasks (with replacement); every unit of a drawn task enters with that task's count."""
    tasks, inv = np.unique(unit_task, return_inverse=True)
    draws = rng.multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), size=nb)  # (nb, tasks)
    g = gap(U, draws[:, inv].astype(float))
    lo, hi = np.percentile(g, [2.5, 97.5])
    return float((hi - lo) / 2), float(g.std(ddof=1))


def power(h, delta, X):
    s = h / 1.96
    return 0.0 if X <= h else float(norm.cdf((X - h - delta) / s) - norm.cdf((-X + h - delta) / s))


def both_pairs(h, delta, X, rng):
    """r3 - r1 and r3 - r2 both inside: bivariate normal, correlation 0.5 (exchangeable runs), true gaps delta and 0."""
    if X <= h:
        return 0.0
    s = h / 1.96
    z = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=MC) * s
    return float(((np.abs(z[:, 0] + delta) <= X - h) & (np.abs(z[:, 1]) <= X - h)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws0", required=True)
    ap.add_argument("--scores36", required=True)
    args = ap.parse_args()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    if dirty:
        raise SystemExit(f"refusing to run from a dirty tree:\n{dirty}")
    gates = {"draws0": sha(args.draws0) == SHA["draws0"], "scores36": sha(args.scores36) == SHA["scores36"], "runs36": sha(RUNS36) == SHA["runs36"]}
    if not all(gates.values()):
        raise SystemExit(f"input hashes differ: {gates}")
    v = json.loads(VERDICTS.read_text())["endpoints"]["catch_control_run2_minus_run1"]
    f939 = json.loads(FIX939.read_text())["readouts"]["output_end"]
    Ua, ta = acting_units(args.draws0)
    Ur, tr = reading_units(args.scores36)
    specs = {"acting catch at 1% (run 2 - run 1)": ("acting", Ua, ta, acting_gap, 100 * v["observed"], [100 * x for x in v["ci95_exact"]],
                                                   "scaleup196_verdicts catch_control_run2_minus_run1 (500 refits)"),
             "#36 catch at 5% (e2 - e1)": ("reading", Ur, tr, reading_gap(0), 100 * f939["0.05"][E1_IN_TRAINING]["observed"],
                                           [100 * x for x in f939["0.05"][E1_IN_TRAINING]["ci95"]], "fix939 (2,000 task draws, thresholds redrawn)"),
             "#36 catch at 1% (e2 - e1)": ("reading", Ur, tr, reading_gap(2), 100 * f939["0.01"][E1_IN_TRAINING]["observed"],
                                           [100 * x for x in f939["0.01"][E1_IN_TRAINING]["ci95"]], "fix939 (2,000 task draws, thresholds redrawn)")}
    out = {"label": "price only, not launched; Aswin decides. Planning projection from stored scores.",
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip(),
           "inputs": {"draws0": {"file": Path(args.draws0).name, "sha256": SHA["draws0"]}, "scores36": {"file": Path(args.scores36).name, "sha256": SHA["scores36"]},
                      "runs36": str(RUNS36.relative_to(REPO)), "metrics36": str(METRICS36.relative_to(REPO)), "fix939": str(FIX939.relative_to(REPO)),
                      "verdicts": str(VERDICTS.relative_to(REPO))},
           "settings": {"R": R, "B": B, "B_full": B_FULL, "seed": SEED, "X": XS, "mc": MC}, "gates": gates | {"reproduce_stored_observed": True},
           "units": {"acting_tasks": int(len(Ua)), "reading_cells": int(len(Ur)), "reading_tasks": int(len(set(tr)))}, "yardsticks": {}}
    for j, (name, (kind, U, ut, gap, obs, ci, src)) in enumerate(specs.items()):
        rng = np.random.default_rng(SEED + 100 * j)
        h_full, sd_full = boot(U, ut, gap, rng, B_FULL)
        h_stored = (ci[1] - ci[0]) / 2
        kappa, extra2 = h_stored / h_full, h_stored ** 2 - h_full ** 2
        rows = {}
        for n in SIZES[kind] + (len(U),):
            if n == len(U):
                hs, ntask = [h_full], [len(set(ut))]
            else:
                hs, ntask = [], []
                for _ in range(R):
                    sub = np.sort(rng.choice(len(U), n, replace=False))
                    hs.append(boot(U[sub], ut[sub], gap, rng, B)[0])
                    ntask.append(len(set(ut[sub])))
            h = float(np.mean(hs))
            hk, ha = kappa * h, float(np.sqrt(h ** 2 + max(extra2, 0.0)))
            prng = np.random.default_rng(SEED + 100 * j + n)
            rows[n] = {"tasks_mean": float(np.mean(ntask)), "half_width_fixed_mean": h, "half_width_fixed_p10_p90": [float(np.percentile(hs, 10)), float(np.percentile(hs, 90))],
                       "half_width_projected": hk, "half_width_additive": ha,
                       "P_within": {f"±{X:g}": {"true_0": power(hk, 0.0, X), "true_observed": power(hk, obs, X),
                                                "both_pairs_true_0": both_pairs(hk, 0.0, X, prng), "additive_true_0": power(ha, 0.0, X)} for X in XS}}
        ns = np.array(list(rows), float)
        beta = float(-np.polyfit(np.log(ns), np.log([rows[n]["half_width_fixed_mean"] for n in rows]), 1)[0])
        out["yardsticks"][name] = {"stored": {"observed": obs, "ci95": ci, "half_width": h_stored, "source": src},
                                   "fixed_full": {"half_width": h_full, "sd": sd_full}, "kappa": kappa, "beta_fixed": beta, "by_n": rows}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    p = HERE / f"results/third_real_power_{stamp}.json"
    p.write_text(json.dumps(out, indent=1))
    print(p.relative_to(REPO))
    print(json.dumps({k: {"kappa": y["kappa"], "beta": y["beta_fixed"], **{str(n): [round(r["half_width_fixed_mean"], 2), round(r["half_width_projected"], 2),
                                                                                     {x: round(q["true_0"], 3) for x, q in r["P_within"].items()}]
                                                                           for n, r in y["by_n"].items()}} for k, y in out["yardsticks"].items()}, indent=1))


if __name__ == "__main__":
    main()
