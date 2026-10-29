"""POST HOC, price only: sizing 2e's matched-false-alarm comparison (SIZING.md). $0, local, fixed scores.

Gate: reproduce 2e's q22_matched_fa_POSTHOC_2026-10-11_033516.json exactly (its levels() logic, copied; task_weights and
threshold from analyze_scaleup_plants.py). Then: the task-subsampling scaling check, the pooled-runs check, the targets,
and the price. Writes results/sizing_<stamp>.json (exclusive create).

  uv run --no-project --with numpy python -I \\
      experiments/2026-10-12_q3_matchedfa_sizing/sizing_matched_fa.py <draw-0 per-call file> <2e result json>
"""

import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "experiments/2026-10-08_q3_onpolicy_figures"))
from analyze_scaleup_plants import task_weights, threshold  # noqa: E402

INPUT_SHA = "c2656658baaf3aa150f83f4d105e7572bfd854f98032831621419e1f8e8f36bf"
WORLDS = {"real": "real->real", "sim": "real->sim", "run2": "real->run2"}
FPRS, N = (0.01, 0.05), 2000
FRACS, SUBSETS, SUB_DRAWS = (0.5, 0.75), 20, 400
TARGETS = {"(i) W/2 <= 10": 20.0, "(ii) W/2 <= 4.2": 8.4,
           "(i') 80%: true 0": 10 / (0.5 + 1.2816 / 3.92), "(ii') 80%: true +5.8": 4.2 / (0.5 + 0.8416 / 3.92)}
RATES = {"sim_api_per_sim_run": 16.48, "gpu_per_run": 0.28, "devbox_per_run": 0.022, "extraction_per_run": 0.043,
         "fixed_per_launch": 15.0, "scoring_at_191": 2.0, "refit_grid_at_191": 116.87}
SENSITIVITY = {"sim_api_per_sim_run": 19.76, "gpu_per_run": 0.50, "devbox_per_run": 0.05, "extraction_per_run": 0.043,
               "fixed_per_launch": 30.0, "scoring_at_191": 2.0, "refit_grid_at_191": 116.87}


def load(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUT_SHA:
        raise SystemExit("input sha256 differs from the manifest")
    P = json.loads(raw)
    meta, cells, folds, seed = P["meta"], P["draws"][0]["cells"], P["folds"], P["config"]["seed"]
    tasks = sorted({m["task"] for m in meta})
    ti = {t: i for i, t in enumerate(tasks)}
    cal = cells["real->real"]["per_call"]["calibration"]
    C = {w: tuple(np.array(x) for x in zip(*[(ti[meta[i]["task"]], s, k) for i, s, k in zip(cal["row"], cal["score"], cal["model_fold"])
                                               if meta[i]["world"] == w and meta[i]["y"] == 0])) for w in WORLDS}
    T = {}
    for w, cell in WORLDS.items():
        p = cells[cell]["per_call"]
        rows = [(ti[meta[i]["task"]], folds[meta[i]["task"]], s, meta[i]["y"]) for i, s in zip(p["row"], p["score"]) if meta[i]["world"] == w]
        T[w] = tuple(np.array(x) for x in zip(*rows))
    return tasks, folds, seed, C, T


def levels(w8, C, T, worlds=WORLDS):
    """2e's levels(), copied: per fold model k, the world's own threshold on its calibration read-only calls."""
    out = {}
    for fpr in FPRS:
        r = {}
        for w in worlds:
            c_t, c_s, c_k = C[w]
            th = []
            for k in range(5):
                m = (c_k == k) & (w8[c_t] > 0)
                if not m.any():
                    return None
                th.append(threshold(c_s[m], w8[c_t[m]], fpr=fpr))
            t, f, s, y = T[w]
            fl, ww = s > np.array(th)[f], w8[t]
            r[w] = {"catch": float((ww * fl)[y == 1].sum() / ww[y == 1].sum()), "false_alarm": float((ww * fl)[y == 0].sum() / ww[y == 0].sum())}
        r["gap real - sim"] = r["real"]["catch"] - r["sim"]["catch"]
        r["control run 1 - run 2"] = r["real"]["catch"] - r["run2"]["catch"]
        out[f"{fpr:.0%}"] = r
    return out


def ci(xs):
    return [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]


def gate(tasks, folds, seed, C, T, stored):
    fold_list = [folds[t] for t in tasks]
    vals = [levels(task_weights(fold_list, b, seed), C, T) for b in range(N + 1)]
    ok = [v for v in vals[1:] if v is not None]
    worst = 0.0
    if N - len(ok) != stored["failed"] or vals[0] is None:
        return float("inf"), vals, ok

    def diff(mine, theirs):
        if not (math.isfinite(mine) and math.isfinite(theirs)):
            return float("inf")
        return abs(mine - theirs)
    for key in ("1%", "5%"):
        o, st = vals[0][key], stored["targets"][key]
        pairs = []
        for w in WORLDS:
            for m in ("catch", "false_alarm"):
                pairs.append((o[w][m], st[w][m]["observed"]))
                pairs += list(zip(ci([v[key][w][m] for v in ok]), st[w][m]["ci95"]))
        for g in ("gap real - sim", "control run 1 - run 2"):
            pairs.append((o[g], st[g]["observed"]))
            pairs += list(zip(ci([v[key][g] for v in ok]), st[g]["ci95"]))
        for a, b in pairs:
            worst = max(worst, diff(a, b))
    return worst, vals, ok


def sub_weights(fold_list, keep, b, rng_seed):
    """Resample within folds among the kept tasks only (others weight 0)."""
    f = np.asarray(fold_list)
    rng, w = np.random.default_rng([rng_seed, b]), np.zeros(len(f))
    for k in range(5):
        idx = np.flatnonzero((f == k) & keep)
        np.add.at(w, rng.choice(idx, len(idx), replace=True), 1)
    return w


def seed_of(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], "big")


def width(ok, key, g):
    lo, hi = ci([v[key][g] for v in ok])
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise SystemExit(f"nonfinite interval for {key} {g}")
    return 100 * (hi - lo), 100 * lo, 100 * hi


def main(percall, stored_path):
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"refusing to run from a dirty tree:\n{dirty}")
    stored = json.loads(Path(stored_path).read_text())
    tasks, folds, seed, C, T = load(percall)
    n0 = len(tasks)
    worst, vals, ok = gate(tasks, folds, seed, C, T, stored)
    out = {"label": "POST HOC, price only (not launched; Aswin decides): sizing 2e's matched-FA comparison", "git": head,
           "inputs": {"percall_sha256": INPUT_SHA, "stored": Path(stored_path).name,
                      "stored_sha256": hashlib.sha256(Path(stored_path).read_bytes()).hexdigest()},
           "n_tasks": n0, "gate": {"worst_abs_diff": worst, "passed": worst <= 1e-12, "failed_draws": N - len(ok)}}
    if not out["gate"]["passed"]:
        print(json.dumps(out, indent=1))
        raise SystemExit("gate failed: 2e's result is not reproduced")
    full = {}
    for key in ("1%", "5%"):
        for g in ("gap real - sim", "control run 1 - run 2"):
            W, lo, hi = width(ok, key, g)
            p = 100 * vals[0][key][g]
            full[f"{key} {g}"] = {"point": p, "lo": lo, "hi": hi, "W": W, "U": hi - p, "L": p - lo}
    out["full"] = full
    # task-scaling check
    fold_list = [folds[t] for t in tasks]
    f = np.asarray(fold_list)
    scal = {}
    for frac in FRACS:
        Ws, Wc, ns, nf, detail = [], [], [], 0, []
        for r in range(SUBSETS):
            rng = np.random.default_rng(seed_of("mfa-sizing", frac, r))
            keep = np.zeros(n0, bool)
            for k in range(5):
                idx = np.flatnonzero(f == k)
                keep[rng.choice(idx, int(round(frac * len(idx))), replace=False)] = True
            sv = [levels(sub_weights(fold_list, keep, b, seed_of("mfa-sizing-draws", frac, r)), C, T) for b in range(1, SUB_DRAWS + 1)]
            sok = [v for v in sv if v is not None]
            nf += len(sv) - len(sok)
            Ws.append(width(sok, "1%", "gap real - sim")[0])
            Wc.append(width(sok, "1%", "control run 1 - run 2")[0])
            ns.append(int(keep.sum()))
            detail.append({"r": r, "seed": seed_of("mfa-sizing", frac, r), "n": int(keep.sum()), "tasks": [tasks[i] for i in np.flatnonzero(keep)],
                           "W_gap": Ws[-1], "W_control": Wc[-1], "failed_draws": len(sv) - len(sok)})
        scal[f"{frac}"] = {"n": int(np.mean(ns)), "W_gap_mean": float(np.mean(Ws)), "W_gap_sd": float(np.std(Ws)),
                           "W_control_mean": float(np.mean(Wc)), "failed_draws_total": nf, "per_subset": detail}
    pts_n = [scal[f"{fr}"]["n"] for fr in FRACS] + [n0]
    pts_W = [scal[f"{fr}"]["W_gap_mean"] for fr in FRACS] + [full["1% gap real - sim"]["W"]]
    pts_Wc = [scal[f"{fr}"]["W_control_mean"] for fr in FRACS] + [full["1% control run 1 - run 2"]["W"]]

    def slope(ws):
        if not all(math.isfinite(x) and x > 0 for x in ws):
            return None
        b = float(-np.polyfit(np.log(pts_n), np.log(ws), 1)[0])
        return b if math.isfinite(b) else None
    beta, beta_c = slope(pts_W), slope(pts_Wc)
    beta_ok = beta is not None and beta >= 0.05  # below this the extrapolation is meaningless (flat or rising widths)
    out["scaling"] = {"subsets": scal, "points_n": pts_n, "points_W": pts_W, "beta_measured": beta, "beta_measured_usable": beta_ok, "beta_control": beta_c,
                      "note": "beta is a noisy descriptive slope over three points; it does not by itself separate calibration noise from other sources"}
    # pooled-runs check: real = run 1 U run 2
    Cp = dict(C)
    Tp = dict(T)
    Cp["real"] = tuple(np.concatenate([C["real"][i], C["run2"][i]]) for i in range(3))
    Tp["real"] = tuple(np.concatenate([T["real"][i], T["run2"][i]]) for i in range(4))
    pv = [levels(task_weights(fold_list, b, seed), Cp, Tp) for b in range(N + 1)]
    pok = [v for v in pv[1:] if v is not None]
    Wp, lop, hip = width(pok, "1%", "gap real - sim")
    out["pooled_runs"] = {"note": "real world = run 1 U run 2 (two runs per task); a descriptive pooling sensitivity check, not a variance decomposition",
                          "failed_draws": N - len(pok), "point": 100 * pv[0]["1%"]["gap real - sim"],
                          "lo": lop, "hi": hip, "W": Wp, "W_ratio_vs_run1_only": Wp / full["1% gap real - sim"]["W"]}
    # targets
    W0, U0, L0 = full["1% gap real - sim"]["W"], full["1% gap real - sim"]["U"], full["1% gap real - sim"]["L"]
    def need(ratio, b):
        if b is None:
            return None
        if ratio <= 1:
            return 0
        try:
            k = n0 * ratio ** (1 / b)
        except OverflowError:
            return None
        return math.ceil(k) - n0 if math.isfinite(k) else None
    bm = beta if beta_ok else None
    tg = {}
    for name, Wt in TARGETS.items():
        tg[name] = {"W_target": Wt, "met_now": W0 <= Wt, "added_tasks_beta_measured": need(W0 / Wt, bm), "added_tasks_beta_0.5": need(W0 / Wt, 0.5)}
    tg["arm (ii) U <= 4.2"] = {"U_now": U0, "added_tasks_beta_measured": need(U0 / 4.2, bm), "added_tasks_beta_0.5": need(U0 / 4.2, 0.5)}
    tg["arm (i) longer arm <= 10"] = {"longer_arm_now": max(U0, L0), "added_tasks_beta_measured": need(max(U0, L0) / 10, bm),
                                      "added_tasks_beta_0.5": need(max(U0, L0) / 10, 0.5)}
    out["targets"] = tg
    # price
    def per_task_of(R):
        return {"with_run2": R["sim_api_per_sim_run"] + 3 * (R["gpu_per_run"] + R["devbox_per_run"] + R["extraction_per_run"]),
                "without_run2": R["sim_api_per_sim_run"] + 2 * (R["gpu_per_run"] + R["devbox_per_run"] + R["extraction_per_run"])}
    out["rates"], out["sensitivity_rates"] = RATES, SENSITIVITY
    out["per_task"], out["per_task_sensitivity"] = per_task_of(RATES), per_task_of(SENSITIVITY)
    price = {}
    for scen, R in (("primary", RATES), ("sensitivity", SENSITIVITY)):
        pt_ = per_task_of(R)
        for name, tt in tg.items():
            for bkey in ("added_tasks_beta_measured", "added_tasks_beta_0.5"):
                k = tt[bkey]
                for v, pt in pt_.items():
                    if k is None:
                        price[f"{scen} | {name} | {bkey} | {v}"] = {"added_tasks": None, "raw": None, "plus30": None}
                        continue
                    raw = 0.0 if k == 0 else k * pt + R["fixed_per_launch"] + R["scoring_at_191"] * (n0 + k) / n0
                    price[f"{scen} | {name} | {bkey} | {v}"] = {"added_tasks": k, "raw": round(raw, 2), "plus30": round(1.3 * raw, 2),
                                                                 "optional_refit_raw": round(R["refit_grid_at_191"] * (n0 + k) / n0, 2) if k else 0.0}
    out["price"] = price
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"sizing_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("gate", "full", "scaling", "pooled_runs", "targets", "per_task")}, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
