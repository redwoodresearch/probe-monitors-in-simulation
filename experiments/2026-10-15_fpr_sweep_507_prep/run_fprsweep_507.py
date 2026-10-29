"""Unit-level FPR sweep on the 507-task scale-up cohort (RULE.md in this folder): PREPARED, gated on the FINAL 507 seal; NOT run.

The sweep core is part (b)'s (experiments/2026-10-14_fpr_sweep_unit/run_fprsweep_unit.py, closures inside run_remote) lifted to
module level verbatim, plus the scale-up generalization (strata, unit-level 21b fold -1 ensemble, 21e borrowed folds,
union or stratum-specific thresholds, main-level weights), panel 3 B's joint threshold bootstrap, and the score-level shift.

Modes (Modal CPU, app probe-shift_aswin-surya_sub-fpr507prep_kill-2026-11-14):
  selftest  synthetic stub data only, built in the container from a fixed seed, written in the scorers' per-row layout and read
            back through the scale-up reader: brute-force checks of every transformation, threshold, catch and statistic.
  repro     GR: on part (b)'s pinned corrected-B inputs, (i) the verbatim core and (ii) the generalized scale-up path (one
            stratum, union pool, cells as units, A24's draws) each reproduce fprsweep_unit_2026-10-15_003643.json exactly, and
            (iii) the joint bootstrap reproduces panel3_2026-10-15_021909.json's B rows for Sol (a) and FIBS exactly.
  scaleup   REFUSED unless --seal-lifted <path>@<sha256> names a committed decision record with that sha, the config is not
            the template, and it pins 09's answers (answers_09). Scale-up inputs are never opened before those checks.

  PYTHONPATH=src <modal 1.6.1> run --detach experiments/2026-10-15_fpr_sweep_507_prep/run_fprsweep_507.py --mode selftest|repro
"""

import hashlib
import io
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
UNIT = HERE.parent / "2026-10-14_fpr_sweep_unit"
APP = "probe-shift_aswin-surya_sub-fpr507prep_kill-2026-11-14"
SOLVOL, P2VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13"
D, SEED = 10000, [20261014, 1]
GRID = sorted(set([10 ** (-4 + 0.125 * i) for i in range(25)]) | {0.05})  # part (b)'s 26 points
WORLDS = ("run1", "run2", "sim", "honest_a", "honest_b", "honest_c")
A3, HW = ("run1", "run2", "sim"), ("honest_a", "honest_b", "honest_c")
REC_B = ("results/fprsweep_unit_2026-10-15_003643.json", "d979413913b988ff6f29fbc0a5910aa01d3bb048387bc41b1a848e039ebba7d2")
REC_P3 = ("results/panel3_2026-10-15_021909.json", "4c1987804f642ce2211e8efe422029c8f130830e5c07fddaba5e3c732956a09b")
ANSWERS_09 = "6a0ae013a7e778c8c840fa315999c920a7a2bf94d21961da6f6770cc3c40b2ee"  # fpr507_answers_09_2026-10-15.md @ 4c61a67ad (RULE.md Revision 1)
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
sol_vol = modal.Volume.from_name(SOLVOL, version=2)
p2_vol = modal.Volume.from_name(P2VOL, version=2)


# ================================================================ part (b)'s core, verbatim arithmetic (lifted closures)
def fdiff(x, y):
    import math

    if x is None or y is None:
        return 0.0 if x is None and y is None else float("inf")
    if isinstance(x, (bool, str)) or isinstance(y, (bool, str)):
        return 0.0 if x == y else float("inf")
    x, y = float(x), float(y)
    return abs(x - y) if math.isfinite(x) and math.isfinite(y) else (0.0 if x == y else float("inf"))


def flat(x, pre=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, f"{pre}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flat(v, f"{pre}[{i}]")
    else:
        yield pre, x


def leaf_eq(x, y):
    """Bit-exact leaf equality: floats by float.hex (signed zero kept), everything else by type-aware equality."""
    if isinstance(x, float) or isinstance(y, float):
        return isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) and not isinstance(y, bool) and float(x).hex() == float(y).hex()
    return type(x) is type(y) and x == y


def compare(mine, ref):
    """Every leaf of ref vs mine after a JSON round trip (int fold keys become strings, as in the records):
    (values, max |diff| (diagnostic only), symmetric key differences, leaves not bit-identical)."""
    a, b = dict(flat(json.loads(json.dumps(mine, default=float)))), dict(flat(ref))
    worst = max((fdiff(a.get(k), v) if k in a else float("inf") for k, v in b.items()), default=0.0)
    unequal = sum(1 for k, v in b.items() if k not in a or not leaf_eq(a[k], v))
    return len(b), worst, sorted(set(a) ^ set(b))[:5], unequal


def draws_a24(univ, env_of):  # A24's draws(): draw-major, strata sorted, rng.choice per stratum
    import numpy as np

    st = np.array([env_of[t] for t in univ])
    rng, W_ = np.random.default_rng(SEED), np.zeros((D + 1, len(univ)))
    W_[0] = 1
    groups = [np.flatnonzero(st == k) for k in sorted(set(st.tolist()))]
    for b in range(1, D + 1):
        for idx in groups:
            np.add.at(W_[b], rng.choice(idx, len(idx), replace=True), 1)
    return W_


def draws_main(units, key_of, seed=None):
    """09's draw clarification items 2-3: sorted units (main IDs), strata = key tuples in lexicographic order, ONE fresh
    default_rng([20261014, 1]) (or `seed`: S_paper493's own stream [20261015, 39], the all-origins union stream [20261015, 40]; Append 39 am.1
    §5); per stratum multinomial(n_s, uniform, size=D) for draws 1..D; draw 0 all ones."""
    import numpy as np

    units = sorted(units)
    rng, W_ = np.random.default_rng(SEED if seed is None else list(seed)), np.zeros((D + 1, len(units)))
    W_[0] = 1.0
    for k in sorted({key_of[u] for u in units}):
        idx = np.array([i for i, u in enumerate(units) if key_of[u] == k])
        W_[1:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=D)
    return units, W_


def summ(v):  # part (b)'s summ
    import numpy as np

    f = v[1:][np.isfinite(v[1:])]
    o = {"point": float(v[0]), "finite_draws": int(len(f)), "draws": D}
    if len(f) < 0.95 * D:
        return o | {"ci95": None, "withheld": "fewer than 95% complete draws"}
    return o | {"ci95": [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))], "degenerate": bool(np.ptp(f) == 0)}


def wmean(Wm, h):
    return (Wm * h).sum(1) / Wm.sum(1)


def units_of(keys, score, fold, canon):  # part (b)'s units_of
    import numpy as np

    keys = np.array([str(k) for k in keys])
    if not (keys.ndim == score.ndim == 1 and len(keys) == len(score)) or (fold is not None and len(fold) != len(score)):
        raise SystemExit("G2: misaligned per-row arrays")
    if not np.isfinite(score).all():
        raise SystemExit("G2: non-finite unit scores")
    world = np.array([k.split("|", 1)[0] for k in keys])
    task = np.array([k.split("|", 1)[1] for k in keys])
    if not set(world.tolist()) <= set(WORLDS) or not set(task.tolist()) <= set(canon):
        raise SystemExit("G2: unknown world or a task outside T196")
    by_run = {}
    for i, k in enumerate(keys):
        by_run.setdefault(k, []).append(i)
    return {"keys": keys, "score": np.asarray(score, np.float64), "fold": None if fold is None else np.asarray(fold, np.int64), "world": world,
            "task": task, "by_run": {k: np.array(v) for k, v in by_run.items()}}


def zscore(u, pool_runs):  # part (b)'s zscore (Append 2: per-fold unit moments, ddof 0, draw 0, file order)
    import numpy as np

    if u["fold"] is None:
        return u["score"].copy(), None
    sel = np.zeros(len(u["score"]), bool)
    for k in pool_runs:
        sel[u["by_run"][k]] = True
    mom = {}
    for f in range(5):
        v = u["score"][sel & (u["fold"] == f)]
        sd = float(np.std(v, ddof=0)) if len(v) >= 2 else 0.0
        if len(v) < 2 or not sd > 0:
            raise SystemExit(f"z: fold {f} has {len(v)} pool units, SD {sd}")
        mom[f] = {"mean": float(np.mean(v)), "sd": sd, "n": int(len(v))}
    mu, sd = np.array([mom[f]["mean"] for f in range(5)]), np.array([mom[f]["sd"] for f in range(5)])
    return (u["score"] - mu[u["fold"]]) / sd[u["fold"]], mom


def unit_points(u, zs, pool_runs, coh, Wm, uidx, grid, ui, cohort_name="all"):  # part (b)'s unit_points (ui: the universe index)
    import numpy as np

    rmax = {k: float(zs[ix].max()) for k, ix in u["by_run"].items()}
    pool = np.concatenate([zs[u["by_run"][k]] for k in pool_runs])
    d_ = pool[np.argsort(-pool, kind="mergesort")]
    n = len(d_)
    wr = {w: [k for k in u["by_run"] if k.startswith(w + "|") and k.split("|", 1)[1] in ui] for w in HW}
    wu = {w: np.concatenate([zs[u["by_run"][k]] for k in wr[w]]) if wr[w] else np.array([]) for w in HW}
    Wc = Wm[:, [uidx[t] for t in coh]]
    pts = []
    for f in grid:
        k = int(np.floor(f * n + 1e-9))
        T = float(d_[min(k, n - 1)])
        h = {w: np.array([float(rmax[f"{w}|{t}"] > T) for t in coh]) for w in A3}
        h["real_mean"] = (h["run1"] + h["run2"]) / 2
        c = {w: wmean(Wc, h[w]) for w in h}
        dif = {"sim_minus_real_mean": c["sim"] - c["real_mean"], "sim_minus_run1": c["sim"] - c["run1"], "sim_minus_run2": c["sim"] - c["run2"],
               "run2_minus_run1": c["run2"] - c["run1"]}
        pts.append({"fpr_target": f, "k": k, "n_pool_units": n, "threshold": T, "estimable": k >= 1, "pool_units_above": int((pool > T).sum()),
                    "shaded_lt10_above": int((pool > T).sum()) < 10, "cohort": cohort_name, "n_tasks": len(coh),
                    "realized_unit_fpr": {"pool": float((pool > T).mean())} | {w: float((wu[w] > T).mean()) if len(wu[w]) else None for w in HW},
                    "realized_run_false_alarm_any_unit": {"pool": float(np.mean([rmax[k_] > T for k_ in pool_runs]))}
                    | {w: float(np.mean([rmax[k_] > T for k_ in wr[w]])) if wr[w] else None for w in HW},
                    "tpr": {w: summ(v) for w, v in c.items()}, "diff": {k_: summ(v) for k_, v in dif.items()},
                    "saturated": {w: c[w][0] in (0.0, 1.0) for w in c}})
    return pts


def joint(rmax_of, pool_units, pool_unit_col, cohort_cols, Wm, grid):  # cohort_cols: the resampling column of each cohort cell
    """Panel 3 add-on B (run_panel3.py @ f5d1af8dd, `joint`), generalized only in its inputs: the threshold redrawn per draw from
    the pool units weighted by the draw's weight of each unit's resampling column. rmax_of: {world: array over cohort}."""
    import numpy as np

    if len(pool_units) == 0 or len(cohort_cols) == 0:
        return {"withheld": "an empty pool or cohort"}
    o = np.argsort(-pool_units, kind="mergesort")
    s_sorted, t_sorted = pool_units[o], pool_unit_col[o]
    n = len(s_sorted)
    Tb, kb = np.empty((len(grid), D + 1)), np.empty((len(grid), D + 1), np.int64)
    for c0 in range(0, D + 1, 500):
        cw = np.cumsum(Wm[c0:c0 + 500][:, t_sorted], 1)
        tot = cw[:, -1]
        for gi, f in enumerate(grid):
            kk = np.floor(f * tot + 1e-9)
            Tb[gi, c0:c0 + 500] = s_sorted[np.minimum((cw <= kk[:, None]).sum(1), n - 1)]
            kb[gi, c0:c0 + 500] = kk.astype(np.int64)
    cols = np.asarray(cohort_cols)
    uniq, inv = np.unique(cols, return_inverse=True)
    single = len(uniq) == len(cols) and bool((np.diff(cols) > 0).all())  # one cell per unit, columns ascending (panel 3): no averaging
    Wc = Wm[:, cols] if single else Wm[:, uniq]
    avg = (lambda h_: h_) if single else (lambda h_: np.stack([h_[:, inv == j].mean(1) for j in range(len(uniq))], 1))
    pts = []
    for gi, f in enumerate(grid):
        h = {w: avg((rmax_of[w][None, :] > Tb[gi][:, None]).astype(float)) for w in A3}
        h["real_mean"] = (h["run1"] + h["run2"]) / 2
        c = {w: wmean(Wc, h[w]) for w in h}
        dif = {"sim_minus_real_mean": c["sim"] - c["real_mean"], "sim_minus_run1": c["sim"] - c["run1"], "sim_minus_run2": c["sim"] - c["run2"],
               "run2_minus_run1": c["run2"] - c["run1"]}
        pts.append({"fpr_target": f, "threshold_draw0": float(Tb[gi, 0]), "k_draw0": int(kb[gi, 0]), "draws_k0": int((kb[gi, 1:] == 0).sum()),
                    "draws_k_lt10": int((kb[gi, 1:] < 10).sum()), "k_draws_p5_p50_p95": [float(np.percentile(kb[gi, 1:], q)) for q in (5, 50, 95)],
                    "tpr": {w: summ(v) for w, v in c.items()}, "diff": {k_: summ(v) for k_, v in dif.items()}})
    return pts


# ================================================================ the scale-up generalization (RULE.md Calibration and Error bars)
def fit_tf(score, fold, sbm, pool_idx, need_folds, any_nofold, borrow_idx=None, nofold_pool_idx=None):
    """Unit-level 09 fit_tf (21b with its Codex round 1; 21e): per-fold population moments of the pool's mapped units; a fold with no
    mapped pool unit borrows from borrow_idx (21e, empty folds only); the fold -1 group re-standardized on its own pool units of the
    five-model mean of per-fold z. Returns (tf, None) or (None, why)."""
    import numpy as np

    mu, sd, borrowed = {}, {}, []
    for f in sorted(need_folds | (set(range(5)) if any_nofold else set())):
        v = score[pool_idx][fold[pool_idx] == f]
        if not len(v) and borrow_idx is not None:
            v = score[borrow_idx][fold[borrow_idx] == f]
            borrowed.append(f)
        if len(v) < 2 or not np.isfinite(v).all() or not np.isfinite(v.std()) or not v.std() > 0:
            return None, f"fold {f}: {len(v)} pool units or SD 0"
        mu[f], sd[f] = float(v.mean()), float(v.std())
    tf = {"mu": mu, "sd": sd, "mu0": None, "sd0": None, "borrowed_folds_21e": borrowed}
    if any_nofold:
        mv, sv = np.array([mu[f] for f in range(5)]), np.array([sd[f] for f in range(5)])
        idx0 = pool_idx if nofold_pool_idx is None else nofold_pool_idx
        idx0 = idx0[fold[idx0] == -1]
        if sbm.ndim != 2 or sbm.shape[1] != 5:
            return None, "fold -1 group: score_by_model is not rows x 5"
        m = ((sbm[idx0] - mv) / sv).mean(1)
        m0, s0 = (float(m.mean()), float(m.std())) if len(m) else (float("nan"), float("nan"))
        if len(m) < 2 or not np.isfinite(m).all() or not (np.isfinite(m0) and np.isfinite(s0) and s0 > 0):
            return None, f"fold -1 group: {len(m)} pool units, or a non-finite or zero moment"
        tf["mu0"], tf["sd0"] = m0, s0
    return tf, None


def apply_tf(score, fold, sbm, tf, idx):
    import numpy as np

    z = np.full(len(idx), np.nan)
    f_ = fold[idx]
    mk = f_ >= 0
    if mk.any():
        mu, sd = np.array([tf["mu"].get(f, np.nan) for f in range(5)]), np.array([tf["sd"].get(f, np.nan) for f in range(5)])
        z[mk] = (score[idx][mk] - mu[f_[mk]]) / sd[f_[mk]]
    if (~mk).any():
        mv, sv = np.array([tf["mu"][f] for f in range(5)]), np.array([tf["sd"][f] for f in range(5)])
        z[~mk] = (((sbm[idx][~mk] - mv) / sv).mean(1) - tf["mu0"]) / tf["sd0"]
    return z


def transform(U, pools, scheme, fold_model):
    """Transformed unit scores under the selected scheme. U: arrays score, fold, sbm, stratum (per unit), run (per unit).
    pools: {stratum: unit index array of its selected honest pool}. scheme: 'union' or 'stratum'. Returns z, {group: tf}, why."""
    import numpy as np

    n = len(U["score"])
    if scheme not in ("union", "stratum"):
        raise ValueError(f"unknown scheme {scheme}")
    if not pools or any(len(pools[s]) == 0 for s in pools):
        return None, {}, "withheld: an empty threshold pool"
    if not fold_model:
        return U["score"].copy(), {"identity": True}, None
    z = np.full(n, np.nan)
    allpool = np.concatenate([pools[s] for s in sorted(pools)])
    if scheme == "union":
        groups = {"union": (np.arange(n), allpool, None)}
    else:
        groups = {}
        for s in sorted(pools):
            mapped_other = np.concatenate([pools[t][U["fold"][pools[t]] >= 0] for t in sorted(pools) if t != s] or [np.array([], int)])
            groups[s] = (np.flatnonzero(U["stratum"] == s), pools[s], mapped_other)
    tfs = {}
    for g, (idx, pidx, borrow) in groups.items():
        need = {int(f) for f in U["fold"][idx] if f >= 0}
        tf, why = fit_tf(U["score"], U["fold"], U["sbm"], pidx, need, bool((U["fold"][idx] == -1).any()), borrow)
        if tf is None:
            return None, tfs, f"{g}: {why}"
        z[idx] = apply_tf(U["score"], U["fold"], U["sbm"], tf, idx)
        tfs[g] = tf
        if not np.isfinite(z[idx]).all():
            return None, tfs, f"withheld: {g}: non-finite transformed units"
    return z, tfs, None


def sweep(U, z, pools, scheme, coh, unit_of_cell, Wm, ucol, grid, rate_runs=None):
    """Per grid point: thresholds (one union T, or one per stratum pool, part (b)'s arithmetic), run-max catches, main-level weighted
    TPRs (a resampling unit's catch = the mean over its cohort cells; equal-weighted units, multiplicity per draw), comparisons, and the
    realized rates. With one stratum, the union scheme and one cell per unit, the 'combined' block equals part (b)'s point dict."""
    import numpy as np

    if scheme not in ("union", "stratum"):
        raise ValueError(f"unknown scheme {scheme}")
    if not coh or not pools or any(len(pools[s]) == 0 for s in pools):
        return {"withheld": "an empty cohort or threshold pool"}
    runs = {}
    for i, r in enumerate(U["run"]):
        runs.setdefault(r, []).append(i)
    rmax = {r: float(z[ix].max()) for r, ix in runs.items()}
    strata = sorted(pools)
    tgroups = {"union": np.concatenate([pools[s] for s in strata])} if scheme == "union" else {s: pools[s] for s in strata}
    t_of_stratum = {s: ("union" if scheme == "union" else s) for s in strata}
    sorted_pool = {g: z[ix][np.argsort(-z[ix], kind="mergesort")] for g, ix in tgroups.items()}
    stratum_of_run = {r: U["stratum"][ix[0]] for r, ix in runs.items()}
    pool_runs = {g: list(dict.fromkeys(U["run"][ix].tolist())) for g, ix in tgroups.items()}
    cells = {s: [c for c in coh if U["cell_stratum"][c] == s] for s in strata}
    blocks = {s: cells[s] for s in strata} | ({"combined": list(coh)} if len(strata) > 1 else {})
    members = {}
    for bname, bc in blocks.items():
        mem = {}
        for c in bc:
            mem.setdefault(unit_of_cell[c], []).append(c)
        members[bname] = (sorted(mem, key=lambda x: ucol[x]), mem)
    pts = []
    for f in grid:
        th = {}
        for g, d_ in sorted_pool.items():
            n = len(d_)
            k = int(np.floor(f * n + 1e-9))
            T = float(d_[min(k, n - 1)])
            above = int((z[tgroups[g]] > T).sum())
            th[g] = {"k": k, "n_pool_units": n, "threshold": T, "estimable": k >= 1, "pool_units_above": above, "shaded_lt10_above": above < 10,
                     "realized_unit_fpr_pool": float((z[tgroups[g]] > T).mean()),
                     "realized_run_false_alarm_pool": float(np.mean([rmax[r] > T for r in pool_runs[g]]))}
        Tr = {r: th[t_of_stratum[stratum_of_run[r]]]["threshold"] for r in rmax}
        q = {"fpr_target": f, "thresholds": th, "blocks": {}}
        for bname, bc in blocks.items():
            us, mem = members[bname]
            Wc = Wm[:, [ucol[x] for x in us]]
            h = {}
            for w in A3:
                hit = {c: float(rmax[f"{w}|{c}"] > Tr[f"{w}|{c}"]) for c in bc}
                h[w] = np.array([np.mean([hit[c] for c in mem[x]]) for x in us])
            h["real_mean"] = (h["run1"] + h["run2"]) / 2
            c = {w: wmean(Wc, h[w]) for w in h}
            dif = {"sim_minus_real_mean": c["sim"] - c["real_mean"], "sim_minus_run1": c["sim"] - c["run1"], "sim_minus_run2": c["sim"] - c["run2"],
                   "run2_minus_run1": c["run2"] - c["run1"]}
            hon = {}
            for w in HW:
                rs = [r for r in rmax if r.split("|", 1)[0] == w and (bname == "combined" or stratum_of_run[r] == bname)
                      and (rate_runs is None or r in rate_runs)]
                if rs:
                    uu = np.concatenate([np.asarray(runs[r]) for r in rs])
                    hon[w] = {"unit_fpr": float(np.mean(z[uu] > np.array([Tr[U["run"][i]] for i in uu]))),
                              "run_false_alarm": float(np.mean([rmax[r] > Tr[r] for r in rs])), "runs": len(rs)}
            readable = all(not th[t_of_stratum[s]]["shaded_lt10_above"] for s in strata if bname in (s, "combined") and cells[s])
            q["blocks"][bname] = {"n_cells": len(bc), "n_units": len(us), "readable": readable, "honest": hon,
                                  "tpr": {w: summ(v) for w, v in c.items()}, "diff": {k_: summ(v) for k_, v in dif.items()},
                                  "saturated": {w: c[w][0] in (0.0, 1.0) for w in c}}
        pts.append(q)
    return pts


def part_b_view(p, coh):
    """The generalized point (one stratum, union) in part (b)'s dict format, for GR (ii)."""
    th = p["thresholds"]["union"]
    (b,) = p["blocks"].values()
    hon = b["honest"]
    return {"fpr_target": p["fpr_target"], "k": th["k"], "n_pool_units": th["n_pool_units"], "threshold": th["threshold"], "estimable": th["estimable"],
            "pool_units_above": th["pool_units_above"], "shaded_lt10_above": th["shaded_lt10_above"], "cohort": "all", "n_tasks": len(coh),
            "realized_unit_fpr": {"pool": th["realized_unit_fpr_pool"]} | {w: hon[w]["unit_fpr"] if w in hon else None for w in HW},
            "realized_run_false_alarm_any_unit": {"pool": th["realized_run_false_alarm_pool"]} | {w: hon[w]["run_false_alarm"] if w in hon else None for w in HW},
            "tpr": b["tpr"], "diff": b["diff"], "saturated": b["saturated"]}


def score_shift(U, z, coh, unit_of_cell, Wm, ucol, sigma):
    """Task-paired AUROC (paired win probability) and standardized shift (RULE.md), at the run mean and the run max of transformed
    units; the cell contrast averaged within each main, mains equal-weighted, draw weights = multiplicities. sigma: {aggregator:
    {stratum: SD of that stratum's selected-pool honest run scores}} (one value per stratum; equal across strata under union);
    each cell's contrast is divided by its own stratum's SD before averaging."""
    import numpy as np

    runs = {}
    for i, r in enumerate(U["run"]):
        runs.setdefault(r, []).append(i)
    agg = {"run_mean": {r: float(np.mean(z[ix])) for r, ix in runs.items()}, "run_max": {r: float(np.max(z[ix])) for r, ix in runs.items()}}
    H = lambda x: 1.0 if x > 0 else (0.5 if x == 0 else 0.0)
    us = sorted({unit_of_cell[c] for c in coh})
    Wc = Wm[:, [ucol[x] for x in us]]
    out = {}
    for a, s in agg.items():
        cell = {"A_sr": {c: 0.5 * (H(s[f"sim|{c}"] - s[f"run1|{c}"]) + H(s[f"sim|{c}"] - s[f"run2|{c}"])) for c in coh},
                "A_21": {c: H(s[f"run2|{c}"] - s[f"run1|{c}"]) for c in coh}}
        sg = {c: (sigma.get(a) or {}).get(U["cell_stratum"][c]) for c in coh}
        if all(v is not None and np.isfinite(v) and v > 0 for v in sg.values()):
            cell |= {"d_sr": {c: (s[f"sim|{c}"] - (s[f"run1|{c}"] + s[f"run2|{c}"]) / 2) / sg[c] for c in coh},
                     "d_21": {c: (s[f"run2|{c}"] - s[f"run1|{c}"]) / sg[c] for c in coh}}
        out[a] = {k: summ(wmean(Wc, np.array([np.mean([v[c] for c in coh if unit_of_cell[c] == x]) for x in us]))) for k, v in cell.items()}
        if "d_sr" not in cell:
            out[a]["standardized_shift"] = "withheld: sigma_H not finite and positive"
    return out


# ================================================================ remote modes
@app.function(image=image, cpu=(8, 8), memory=(32768, 32768), timeout=1800, retries=0)
def selftest() -> dict:
    """Synthetic stub data only (fixed seed). Writes stub per-row files in the scorers' layout, reads them back through
    read_units, and checks the generalized path against brute force and against the verbatim core."""
    import tempfile

    import numpy as np

    rng = np.random.default_rng(7)
    res, fails = {}, []

    def ok(name, cond, detail=""):
        res[name] = bool(cond)
        if not cond:
            fails.append(f"{name} {detail}")

    # --- (1) one stratum, union, five folds, one cell per main => generalized == verbatim core; one cell outside the universe
    # (honest runs only) exercises rate_runs, integer-rounded scores put ties at thresholds (strict >)
    cells = [f"t{i:03d}" for i in range(60)]
    extra = "x999"
    env_of = {c: f"e{i % 4}" for i, c in enumerate(cells)}
    fold_c = {c: i % 5 for i, c in enumerate(cells)} | {extra: 2}
    rows = []
    for c in cells + [extra]:
        for w in (WORLDS if c != extra else HW):
            for _ in range(int(rng.integers(1, 6))):
                rows.append((f"{w}|{c}", float(np.round(rng.normal(0.4 if w in A3 else 0.0, 1.0) + 0.3 * fold_c[c], 1)), fold_c[c]))
    keys = np.array([r[0] for r in rows])
    sc, fo = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    u = units_of(keys, sc, fo, cells + [extra])
    ui = {t: i for i, t in enumerate(cells)}
    W = draws_a24(cells, env_of)
    pool_runs = sorted(k for k in u["by_run"] if k.split("|", 1)[0] in ("honest_b", "honest_c") and k.split("|", 1)[1] in ui)
    zs, _ = zscore(u, pool_runs)
    ref = unit_points(u, zs, pool_runs, cells, W, ui, GRID, ui)
    U = {"score": sc, "fold": fo, "sbm": np.full((len(sc), 5), np.nan), "stratum": np.array(["old"] * len(sc)), "run": keys,
         "cell_stratum": {c: "old" for c in cells + [extra]}}
    pidx = np.flatnonzero(np.isin(keys, pool_runs))
    z2, _, why = transform(U, {"old": pidx}, "union", True)
    ok("degenerate_transform_defined", why is None, why)
    ok("degenerate_z_equal", z2 is not None and np.array_equal(z2, zs))
    rate = {k for k in u["by_run"] if k.split("|", 1)[1] in ui}
    gen = sweep(U, z2, {"old": pidx}, "union", cells, {c: c for c in cells}, W, ui, GRID, rate_runs=rate)
    _, _, kd, une = compare([part_b_view(p, cells) for p in gen], json.loads(json.dumps(ref, default=float)))
    ok("degenerate_points_bit_identical_to_core", une == 0 and not kd, f"unequal {une} keys {kd}")
    gen_all = sweep(U, z2, {"old": pidx}, "union", cells, {c: c for c in cells}, W, ui, GRID)
    _, _, _, une2 = compare([part_b_view(p, cells) for p in gen_all], json.loads(json.dumps(ref, default=float)))
    ok("rate_runs_matters", une2 > 0, "the outside-universe honest runs changed nothing: the rate_runs test is not exercised")
    zp = z2[pidx]
    tied = [q_ for q_ in gen if int((zp == q_["thresholds"]["union"]["threshold"]).sum()) >= 2 and q_["thresholds"]["union"]["k"] >= 1]
    ok("ties_present_at_threshold", bool(tied))
    q = tied[-1] if tied else gen[-1]
    T = q["thresholds"]["union"]["threshold"]
    rmx = {}
    for i, r in enumerate(keys):
        rmx[r] = max(rmx.get(r, -np.inf), z2[i])
    ok("strict_greater_catch", q["blocks"]["old"]["tpr"]["sim"]["point"] == np.mean([float(rmx[f"sim|{c}"] > T) for c in cells]))
    # --- (2) fold -1 ensemble (21b at unit level) on a brute-force check
    n2 = 400
    fo2 = np.where(np.arange(n2) < 300, np.arange(n2) % 5, -1)
    sbm = rng.normal(size=(n2, 5)) * np.array([1, 2, 3, 4, 5]) + np.array([0, 1, 2, 3, 4])
    sc2 = np.where(fo2 >= 0, sbm[np.arange(n2), np.clip(fo2, 0, 4)], np.nan)
    pool2 = np.arange(0, n2, 2)
    tf, why = fit_tf(sc2, fo2, sbm, pool2, set(range(5)), True)
    ok("ensemble_defined", tf is not None, why)
    zz = apply_tf(sc2, fo2, sbm, tf, np.arange(n2))
    mu = np.array([sc2[pool2][fo2[pool2] == f].mean() for f in range(5)])
    sd = np.array([sc2[pool2][fo2[pool2] == f].std() for f in range(5)])
    m_all = ((sbm - mu) / sd).mean(1)
    p0 = pool2[fo2[pool2] == -1]
    brute = np.where(fo2 >= 0, (sc2 - mu[np.clip(fo2, 0, 4)]) / sd[np.clip(fo2, 0, 4)], (m_all - m_all[p0].mean()) / m_all[p0].std())
    ok("ensemble_matches_brute_force", np.allclose(zz, brute, rtol=0, atol=1e-12))
    ok("ensemble_pool_restandardized", abs(zz[p0].mean()) < 1e-12 and abs(zz[p0].std() - 1) < 1e-12)
    ok("ensemble_rejects_bad_shape", fit_tf(sc2, fo2, sbm[:, :1], pool2, set(range(5)), True)[0] is None)
    # --- (3) 21e borrowing: an empty mapped fold borrows; a non-empty invalid fold does not, although donor data exist for it
    pool_s = np.array([i for i in pool2 if fo2[i] != 3])
    donors = np.array([i for i in range(n2) if fo2[i] in (3, 4)])
    tf3, why3 = fit_tf(sc2, fo2, sbm, pool_s, set(range(5)), True, borrow_idx=donors)
    d3 = sc2[donors][fo2[donors] == 3]
    ok("borrow_21e", tf3 is not None and tf3["borrowed_folds_21e"] == [3] and tf3["mu"][3] == float(d3.mean()) and tf3["sd"][3] == float(d3.std()), why3)
    one = np.array([i for i in pool2 if fo2[i] != 4] + [int(np.flatnonzero(fo2 == 4)[0])])
    tf4, why4 = fit_tf(sc2, fo2, sbm, one, set(range(5)), True, borrow_idx=donors)
    ok("invalid_nonempty_fold_not_borrowed", tf4 is None and "fold 4" in (why4 or ""))
    # --- (4) two strata through the file reader: stratum-specific thresholds, unequal cells per main, recipient-only mu0/sd0
    with tempfile.TemporaryDirectory() as td:
        strat = {f"t{i:03d}": "old" for i in range(30)} | {f"n{i:03d}": "new" for i in range(30)}
        fold4 = {c: (-1 if int(c[1:]) % 4 == 0 else int(c[1:]) % 5) for c in strat}
        rows4 = []
        for c, s in strat.items():
            for w in WORLDS if s == "old" else ("run1", "run2", "sim", "honest_a", "honest_b"):
                for _ in range(int(rng.integers(1, 5))):
                    bm = rng.normal(0.5 if w in A3 else 0.0, 1.0, 5) * (1.0 if s == "old" else 2.0) + (0.7 if s == "new" else 0.0)
                    rows4.append((f"{w}|{c}", float(bm[fold4[c]]) if fold4[c] >= 0 else float("nan"), fold4[c], bm))
        for s in ("old", "new"):
            rr = [r for r in rows4 if strat[r[0].split("|", 1)[1]] == s]
            seen, un = {}, []
            for r in rr:
                un.append(seen.get(r[0], 0))
                seen[r[0]] = un[-1] + 1
            np.savez(Path(td) / f"{s}.npz", key=np.array([r[0] for r in rr]), score=np.array([r[1] for r in rr]), unit=np.array(un, np.int64),
                     fold=np.array([r[2] for r in rr], np.int64), score_by_model=np.array([r[3] for r in rr]))
        files = [{"path": str(Path(td) / f"{s}.npz"), "sha256": sha_file(Path(td) / f"{s}.npz")} for s in ("old", "new")]
        U4 = read_units(files, strat, True, fold_of_cell=fold4)
        ok("reader_rows", len(U4["score"]) == len(rows4))
        try:
            read_units(files, strat, True, fold_of_cell=fold4 | {"t001": 3})
            ok("reader_rejects_wrong_fold_map", False)
        except SystemExit:
            ok("reader_rejects_wrong_fold_map", True)
        pools4 = {s: np.flatnonzero((U4["stratum"] == s) & np.isin([r.split("|", 1)[0] for r in U4["run"]], ["honest_b", "honest_c"])) for s in ("old", "new")}
        z4, tfs4, why4b = transform(U4, pools4, "stratum", True)
        ok("stratum_transform_defined", why4b is None, why4b)
        pn = pools4["new"]
        mv = np.array([tfs4["new"]["mu"][f] for f in range(5)])
        sv = np.array([tfs4["new"]["sd"][f] for f in range(5)])
        m_new = ((U4["sbm"][pn[U4["fold"][pn] == -1]] - mv) / sv).mean(1)
        ok("mu0_recipient_only", tfs4["new"]["mu0"] == float(m_new.mean()) and tfs4["new"]["sd0"] == float(m_new.std()))
        coh4 = sorted(strat)
        mains4 = {c: (c[0] + ("A" if int(c[1:]) < 1 else "B" if int(c[1:]) < 10 else "C")) for c in coh4}  # 1, 9 and 20 cells per main
        keyof = {m: ("e0", "old-only" if m.startswith("t") else "new-only") for m in set(mains4.values())}
        units4, W4 = draws_main(set(mains4.values()), keyof)
        col4 = {x: i for i, x in enumerate(units4)}
        sw = sweep(U4, z4, pools4, "stratum", coh4, mains4, W4, col4, GRID)
        p = sw[-1]  # f = 0.1: k >= 10 in both stub pools
        Ts = {s: p["thresholds"][s]["threshold"] for s in ("old", "new")}
        rm = {}
        for i, r in enumerate(U4["run"]):
            rm[r] = max(rm.get(r, -np.inf), z4[i])
        bf = np.mean([np.mean([float(rm[f"sim|{c}"] > Ts[strat[c]]) for c in coh4 if mains4[c] == m]) for m in sorted(set(mains4.values()))])
        cell_w = np.mean([float(rm[f"sim|{c}"] > Ts[strat[c]]) for c in coh4])
        ok("stratum_catch_brute_force_equal_mains", abs(p["blocks"]["combined"]["tpr"]["sim"]["point"] - bf) < 1e-12)
        ok("equal_mains_differs_from_cell_weighting", abs(bf - cell_w) > 1e-6, "pick a seed where they differ")
        th_bf = {}
        for s in ("old", "new"):
            d_ = np.sort(z4[pools4[s]])[::-1]
            th_bf[s] = d_[min(int(np.floor(GRID[-1] * len(d_) + 1e-9)), len(d_) - 1)]
        ok("stratum_thresholds_brute_force", all(Ts[s] == th_bf[s] for s in Ts))
        # joint bootstrap vs an explicit resampled computation (expanded pool per draw) at f = 0.1, every draw
        pu = z4[pools4["old"]]
        pc = np.array([col4[mains4[r.split("|", 1)[1]]] for r in U4["run"][pools4["old"]]])
        cohO = [c for c in coh4 if strat[c] == "old"]
        rmo = {w: np.array([rm[f"{w}|{c}"] for c in cohO]) for w in A3}
        jp = joint(rmo, pu, pc, [col4[mains4[c]] for c in cohO], W4, GRID)
        uo = sorted({mains4[c] for c in cohO}, key=lambda x: col4[x])
        catch = np.empty(D + 1)
        for b_ in range(D + 1):
            ex = np.sort(np.repeat(pu, W4[b_, pc].astype(int)))[::-1]
            Tb = ex[min(int(np.floor(GRID[-1] * len(ex) + 1e-9)), len(ex) - 1)]
            hm = np.array([np.mean([float(rm[f"sim|{c}"] > Tb) for c in cohO if mains4[c] == x]) for x in uo])
            wb = W4[b_, [col4[x] for x in uo]]
            catch[b_] = (wb * hm).sum() / wb.sum()
        ref_j = summ(catch)
        got_j = jp[-1]["tpr"]["sim"]
        ok("joint_matches_explicit_resampling", abs(got_j["point"] - ref_j["point"]) < 1e-12 and got_j["ci95"] is not None
           and max(abs(a_ - b_) for a_, b_ in zip(got_j["ci95"], ref_j["ci95"])) < 1e-12)
        ok("joint_draw0_threshold", all(jq["threshold_draw0"] == q_["thresholds"]["old"]["threshold"] for jq, q_ in zip(jp, sw)))
    # --- (5) score-level shift on hand-computed values (three single-unit cells, one main each)
    Us = {"run": np.array(["run1|c1", "run2|c1", "sim|c1", "run1|c2", "run2|c2", "sim|c2", "run1|c3", "run2|c3", "sim|c3"]),
          "cell_stratum": {"c1": "old", "c2": "old", "c3": "old"}}
    zsh = np.array([0.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0])
    _, W3 = draws_main({"c1", "c2", "c3"}, {"c1": ("e", "x"), "c2": ("e", "x"), "c3": ("e", "x")})
    sh = score_shift(Us, zsh, ["c1", "c2", "c3"], {"c1": "c1", "c2": "c2", "c3": "c3"}, W3, {"c1": 0, "c2": 1, "c3": 2},
                     {"run_mean": {"old": 2.0}, "run_max": {"old": 0.0}})
    exp = {"A_sr": (1 + 0.5 + 0.25) / 3, "A_21": (1 + 0.5 + 0) / 3, "d_sr": (0.75 + 0 - 0.5) / 3, "d_21": (0.5 + 0 - 1) / 3}
    ok("shift_hand_computed", all(abs(sh["run_mean"][k]["point"] - v) < 1e-12 for k, v in exp.items()))
    ok("shift_withheld_sigma0", sh["run_max"].get("standardized_shift", "").startswith("withheld") and "d_sr" not in sh["run_max"])
    Us2 = Us | {"cell_stratum": {"c1": "old", "c2": "new", "c3": "new"}}
    sh2 = score_shift(Us2, zsh, ["c1", "c2", "c3"], {"c1": "c1", "c2": "c2", "c3": "c3"}, W3, {"c1": 0, "c2": 1, "c3": 2},
                      {"run_mean": {"old": 2.0, "new": 4.0}, "run_max": {"old": 2.0, "new": 4.0}})
    ok("shift_per_stratum_sigma", abs(sh2["run_mean"]["d_sr"]["point"] - (1.5 / 2 + 0 - 1 / 4) / 3) < 1e-12
       and abs(sh2["run_mean"]["d_21"]["point"] - (1 / 2 + 0 - 2 / 4) / 3) < 1e-12)
    # --- (6) draws_main is 09's construction: equal to an independent 22d/22i-style generator on the same sorted units and strata
    units5 = [f"m{i:02d}" for i in range(20)]
    key5 = {x: ("e1" if i % 3 else "e0", "both") for i, x in enumerate(units5)}
    _, W5 = draws_main(units5, key5)
    rng5, W5b = np.random.default_rng(SEED), np.zeros((D + 1, 20))
    W5b[0] = 1.0
    for k in sorted(set(key5.values())):
        idx = np.array([i for i, x in enumerate(units5) if key5[x] == k], dtype=np.int64)
        W5b[1:, idx] = rng5.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=D)
    ok("draws_main_equals_22d_construction", np.array_equal(W5, W5b))
    ok("empty_pool_withheld", transform(U4, {"old": pools4["old"], "new": np.array([], int)}, "stratum", True)[2] is not None
       and "withheld" in sweep(U4, z4, {"old": np.array([], int)}, "union", coh4, mains4, W4, col4, GRID)
       and "withheld" in joint(rmo, np.array([]), np.array([], int), [col4[mains4[c]] for c in cohO], W4, GRID)
       and "withheld" in joint(rmo, pu, pc, [], W4, GRID))
    return {"checks": res, "failures": fails, "pass": not fails}


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_units(files, cell_stratum, fold_model, fold_of_cell=None):
    """The scale-up reader: per-row npz files in the scorers' schema (key '<world>|<cell>', score designated or NaN on fold -1,
    fold, score_by_model where fold -1 rows exist). sha256 checked; designated scores finite on mapped rows; five finite model
    scores on fold -1 rows (21b: never fewer than five, never imputed); no (key, unit) repeat."""
    import numpy as np

    if not files:
        raise SystemExit("G6: no unit files")
    K, S, F, B, ST, seen = [], [], [], [], [], set()
    for e in files:
        b = Path(e["path"]).read_bytes()
        if hashlib.sha256(b).hexdigest() != e["sha256"]:
            raise SystemExit(f"G1: {e['path']} sha256")
        z = np.load(io.BytesIO(b))
        if z["key"].ndim != 1:
            raise SystemExit(f"G2: {e['path']}: key is not one-dimensional")
        n = len(z["key"])
        if any(z[x].ndim != 1 or len(z[x]) != n for x in ("key", "score", "unit") + (("fold",) if fold_model else ())):
            raise SystemExit(f"G2: {e['path']}: misaligned per-row arrays")
        if fold_model and not np.issubdtype(z["fold"].dtype, np.integer):
            raise SystemExit(f"G2: {e['path']}: fold is not an integer array")
        if "score_by_model" in z.files and z["score_by_model"].shape != (n, 5):
            raise SystemExit(f"G2: {e['path']}: score_by_model is not rows x 5")
        k = z["key"].astype(str)
        pairs = set(zip(k.tolist(), z["unit"].tolist()))
        if len(pairs) != n or pairs & seen:
            raise SystemExit(f"G2: {e['path']}: a (key, unit) pair repeats (within or across files)")
        seen |= pairs
        f = z["fold"].astype(np.int64) if fold_model else np.zeros(n, np.int64)
        if fold_model and not set(f.tolist()) <= {-1, 0, 1, 2, 3, 4}:
            raise SystemExit(f"G2: {e['path']}: a fold outside -1..4")
        if fold_model and fold_of_cell is not None and any(fo != fold_of_cell.get(c.split("|", 1)[1], -1) for c, fo in zip(k.tolist(), f.tolist())):
            raise SystemExit(f"G3: {e['path']}: a row fold differs from the pinned map")
        s = z["score"].astype(np.float64)
        if fold_model and (f == -1).any() and ("score_by_model" not in z.files or z["score_by_model"].shape != (n, 5)):
            raise SystemExit(f"G2: {e['path']}: fold -1 rows without a rows x 5 score_by_model")
        bm = z["score_by_model"].astype(np.float64) if "score_by_model" in z.files else np.full((n, 5), np.nan)
        if fold_model and (not np.isfinite(s[f >= 0]).all() or not np.isfinite(bm[f == -1]).all()):
            raise SystemExit(f"G2: {e['path']}: a non-finite designated score, or a fold -1 row without five finite model scores")
        if not fold_model and not np.isfinite(s).all():
            raise SystemExit(f"G2: {e['path']}: non-finite scores")
        if {c.split("|", 1)[1] for c in k} - set(cell_stratum):
            raise SystemExit(f"G2: {e['path']}: a cell outside the membership")
        K.append(k), S.append(s), F.append(f), B.append(bm), ST.append(np.array([cell_stratum[c.split("|", 1)[1]] for c in k]))
    return {"run": np.concatenate(K), "score": np.concatenate(S), "fold": np.concatenate(F), "sbm": np.concatenate(B), "stratum": np.concatenate(ST),
            "cell_stratum": cell_stratum}


@app.function(image=image, cpu=(8, 8), memory=(32768, 32768), timeout=2400, retries=0, volumes={"/sol": sol_vol.read_only(), "/p2": p2_vol.read_only()})
def repro(cfg: dict, blobs: dict) -> dict:
    """GR (i) verbatim core, (ii) generalized path, (iii) the joint bootstrap vs panel 3 B. Exact (<= 1e-12; identical keys)."""
    import numpy as np

    labels = json.loads(blobs["labels"])["labels"]
    t196 = json.loads(blobs["t196"])
    canon = sorted(c["key"] for c in t196["cells"])
    env_of = {c["key"]: c["env"] for c in t196["cells"]}
    U_ = sorted(json.loads(blobs["v6"])["result"]["universe_and_split"]["universe"])
    ui = {t: i for i, t in enumerate(U_)}
    att = lambda w, t: labels.get(f"{w}|{t}") == "attempted"
    W = draws_a24(U_, env_of)
    recb, p3 = json.loads(blobs["rec_b"]), json.loads(blobs["rec_p3"])
    out, ok = {}, True
    for p in cfg["probes"]:
        pid = p["probe_id"]
        b = (Path("/sol" if p["rows"]["volume"] == "solavg" else "/p2") / p["rows"]["path"]).read_bytes()
        if hashlib.sha256(b).hexdigest() != p["rows"]["sha256"]:
            raise SystemExit(f"G1 {pid}")
        z = np.load(io.BytesIO(b))
        u = units_of(z["key"], z["score"].astype(np.float64), z["fold"].astype(np.int64) if p["folds"] == 5 else None, canon)
        pool_runs = sorted(k for k in u["by_run"] if k.split("|", 1)[0] in ("honest_b", "honest_c") and k.split("|", 1)[1] in ui)
        zs, mom = zscore(u, pool_runs)
        has = {w: {k.split("|", 1)[1] for k in u["by_run"] if k.startswith(w + "|")} for w in A3}
        coh = sorted(t for t in canon if t in ui and all(att(w, t) for w in A3) and all(t in has[w] for w in A3))
        ref = recb["probes"][pid]
        pts = unit_points(u, zs, pool_runs, coh, W, ui, GRID, ui)
        n1, w1, k1, u1 = compare({"points": pts, "z_moments_unit": mom, "pool_runs": pool_runs, "cohort": coh},
                             {"points": ref["points"], "z_moments_unit": ref["z_moments_unit"], "pool_runs": ref["pool_runs"], "cohort": ref["cohort"]})
        # (ii) generalized path: one stratum, union, cells as resampling units, A24's draws; units restricted as part (b) (all units)
        Ug = {"score": u["score"], "fold": u["fold"] if u["fold"] is not None else np.zeros(len(u["score"]), np.int64),
              "sbm": np.full((len(u["score"]), 5), np.nan), "stratum": np.array(["old"] * len(u["score"])), "run": u["keys"],
              "cell_stratum": {t: "old" for t in canon}}
        pidx = np.flatnonzero(np.isin(u["keys"], pool_runs))
        zg, _, why = transform(Ug, {"old": pidx}, "union", p["folds"] == 5)
        if why:
            raise SystemExit(f"GR(ii) {pid}: {why}")
        rate = {k for k in u["by_run"] if k.split("|", 1)[1] in ui}
        gen = sweep(Ug, zg, {"old": pidx}, "union", coh, {t: t for t in coh}, W, ui, GRID, rate_runs=rate)
        # part (b)'s honest rates use runs whose task is in the universe (A24 membership); the generalized block uses every run of the world
        view = [part_b_view(q, coh) for q in gen]
        wr_out = [k for k in u["by_run"] if k.split("|", 1)[0] in HW and k.split("|", 1)[1] not in ui]
        n2, w2, k2, u2 = compare(view, ref["points"])
        res = {"core": {"values": n1, "not_bit_identical": u1, "max_abs_diff": w1, "key_diff": k1, "pass": u1 == 0 and not k1},
               "generalized": {"values": n2, "not_bit_identical": u2, "max_abs_diff": w2, "key_diff": k2, "honest_runs_outside_universe": len(wr_out),
                               "pass": u2 == 0 and not k2}}
        lab = {"sol_a|avg_top2|corrected|none|tight": "Sol (a)", "fibs|reply_mean|corrected|none": "FIBS"}.get(pid)
        if lab:  # (iii) the joint bootstrap vs panel 3 B (part (b)'s pool and draws)
            rmax = {k: float(zs[ix].max()) for k, ix in u["by_run"].items()}
            pu = np.concatenate([zs[u["by_run"][k]] for k in pool_runs])
            pc = np.concatenate([np.full(len(u["by_run"][k]), ui[k.split("|", 1)[1]]) for k in pool_runs])
            jp = joint({w: np.array([rmax[f"{w}|{t}"] for t in coh]) for w in A3}, pu, pc, [ui[t] for t in coh], W, GRID)
            n3, w3, k3, u3 = compare(jp, p3["B"][lab]["points"])
            res["joint_vs_panel3_B"] = {"values": n3, "not_bit_identical": u3, "max_abs_diff": w3, "key_diff": k3, "pass": u3 == 0 and not k3}
        out[pid] = res
        ok &= all(v["pass"] for v in res.values())
    cover = {"probes": sorted(out) == sorted(recb["probes"]) and len(cfg["probes"]) == len(recb["probes"]) == 6,
             "joint": sum("joint_vs_panel3_B" in v for v in out.values()) == 2}
    return {"probes": out, "coverage": cover, "pass": bool(ok) and all(cover.values())}


@app.local_entrypoint()
def main(mode: str = "selftest", seal_lifted: str = "", config: str = ""):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout
    if mode == "scaleup":  # GS: nothing scale-up is opened before these checks
        if "@" not in seal_lifted:
            raise SystemExit("scaleup refused: --seal-lifted <committed decision record path>@<sha256> is required (RULE.md, GS)")
        path, sha = seal_lifted.rsplit("@", 1)
        b = git("show", f"HEAD:{path}")
        if hashlib.sha256(b).hexdigest() != sha:
            raise SystemExit("scaleup refused: the decision record's sha256 differs")
        dec = json.loads(b)
        cfgj = json.loads((HERE / config).read_text()) if config else {"TEMPLATE": True}
        fr = cfgj.get("final_record", {})
        if (dec.get("releases_final_507") is not True or dec.get("final_record", {}).get("sha256") != fr.get("sha256") or len(fr.get("sha256", "")) != 64
                or not dec.get("released_datasets") or "read_holds" not in dec):
            raise SystemExit("scaleup refused: the decision record must explicitly release the FINAL 507 record (its full sha256, as the config "
                             "pins it) and name the released datasets and the remaining READ_HOLDs")
        if cfgj.get("TEMPLATE"):
            raise SystemExit("scaleup refused: the config is the template")
        a9 = cfgj.get("answers_09", {})
        if a9.get("sha256") != ANSWERS_09 or hashlib.sha256(git("show", f"{a9.get('commit')}:{a9.get('path')}")).hexdigest() != ANSWERS_09:
            raise SystemExit("scaleup refused: 09's answers are not the pinned record (RULE.md Revision 1)")
        raise SystemExit("scaleup: the reader is wired to read_units/transform/sweep; the per-probe membership mapping is filled from 09's answers "
                         "(Q1-Q9) in a reviewed commit before the first run")
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    head = git("rev-parse", "HEAD").decode().strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    if mode == "selftest":
        out = selftest.remote()
    elif mode == "repro":
        cfg = json.loads((UNIT / "config" / "inputs_b.json").read_text())
        blobs = {}
        for k in ("v6", "t196", "labels"):
            spec = cfg["local"][k]
            b = git("show", spec["git"]) if "git" in spec else Path(spec["path"]).read_bytes()
            if hashlib.sha256(b).hexdigest() != spec["sha256"]:
                raise SystemExit(f"G1: {k}")
            blobs[k] = b
        for k, (rel, s) in (("rec_b", REC_B), ("rec_p3", REC_P3)):
            b = (UNIT / rel).read_bytes()
            if hashlib.sha256(b).hexdigest() != s:
                raise SystemExit(f"G1: {rel}")
            blobs[k] = b
        out = repro.remote(cfg, blobs)
    else:
        raise SystemExit("--mode selftest | repro | scaleup")
    rec = {"label": f"507 FPR sweep prep, {mode} (no scale-up input read)", "git": head, "stamp": stamp,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()} | out
    blob = json.dumps(rec, indent=1, default=float).encode()
    dest = HERE / "results" / f"{mode}_{stamp}.json"
    dest.open("xb").write(blob)
    print(json.dumps({"record": dest.name, "sha256": hashlib.sha256(blob).hexdigest(), "pass": out.get("pass")}), flush=True)
    if not out.get("pass"):
        raise SystemExit(f"{mode} failed (diagnostics in {dest.name})")
