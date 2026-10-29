"""Amendment 1 (RULE.md here; POST-HOC): a coverage-calibrated upper limit U_cal = X_hat + q and U_s = c * U, chosen over a
pattern grid at v6-matched structure, with split-half and leave-one-pattern-out checks, applied to v6's real rows.
Estimator, gate and draws are run_coverage.py's (copied below verbatim). Modal CPU; aggregation also on Modal. PRELIMINARY.

  XE_IN=<dir with xerr_v6_2026-10-14_161122.json> PYTHONPATH=src $MODAL run --detach \\
      experiments/2026-10-15_u_coverage_09/run_ucal.py --dump-sha <sha256 of /work/ucov/v6_real_dump.json>
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
IN = Path(os.environ.get("XE_IN", "/nonexistent"))
APP = "probe-shift_aswin-surya_sub-convaudit_ucal_kill-2026-11-14"
V6 = ("xerr_v6_2026-10-14_161122.json", "e789144b62f71e0fb168dc3106656c2b5f29aadafe12df0f86dd1f74441d7645")
DUMP = "/work/ucov/v6_real_dump.json"
ALPHAS = (0.01, 0.05)
DRAWS = 10_000
SEED = [20261014, 1]
ARMS = ("run1", "run2", "sim")
ROWS = ("R_reply_meanpooled|alpha=0.01", "R_reply_meanpooled|alpha=0.05", "C_anycommand_max|alpha=0.05")
ORIGINAL = (
    "a_null",
    "b_plus5",
    "b_minus5",
    "b_plus2p5",
    "b_minus2p5",
    "c_hetero_spread5",
    "c_hetero_spread2p5",
    "d_single_plus10",
    "e_sparse_single_plus1p5",
)
ADDED = ("h_plus10", "h_minus10", "i_hetero_spread10", "j_pair_pm8", "k_all_spread8")
# Codex optional items adopted before compute: a +-5 single-probe shift on EVERY eligible probe (l_*) and a mixed
# common-plus-heterogeneous shift (m_*); their names are built per row in prep()
FAMILIES = {  # for leave-one-family-out
    "null": ("a_null",),
    "common": ("b_plus5", "b_minus5", "b_plus2p5", "b_minus2p5", "h_plus10", "h_minus10"),
    "heterogeneous": ("c_hetero_spread5", "c_hetero_spread2p5", "i_hetero_spread10", "j_pair_pm8", "k_all_spread8"),
    "single": ("d_single_plus10", "e_sparse_single_plus1p5"),  # + every l_single_* pattern
    "mixed": ("m_mixed_common3_spread3",),
}


def family_of(p):
    if p.startswith("l_single_"):
        return "single"
    return next(f for f, ps in FAMILIES.items() if p in ps)


STRATA = ("pooled", "with_strategy")
N_DATASETS = 1000
CHUNK = 50

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "scipy==1.16.1")
if IN.exists():
    image = image.add_local_file(IN / V6[0], f"/root/in/{V6[0]}")
work_vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2)


class GateError(Exception):
    pass


def thresholds(honest) -> dict:  # verbatim from run_xerr_v6.py
    """alpha -> (threshold, n, k): the (k+1)-th largest, k = floor(alpha n + 1e-9)."""
    import numpy as np

    s = np.sort(np.asarray(honest, dtype=float), kind="mergesort")[::-1]
    if not len(s) or not np.isfinite(s).all():
        raise GateError("empty or non-finite honest pool")
    out = {}
    for a in ALPHAS:
        k = int(np.floor(a * len(s) + 1e-9))
        out[a] = (float(s[min(k, len(s) - 1)]), len(s), k)
    return out


def make_draws(tasks: list, stratum: dict, seed=SEED, draws=DRAWS):  # run_xerr_v6.py's, with the seed as an argument
    """Multinomial task multiplicities within strata; row 0 = all ones."""
    import numpy as np

    rng = np.random.default_rng(seed)
    W = np.zeros((draws + 1, len(tasks)))
    W[0] = 1.0
    g = np.array([stratum[t] for t in tasks])
    for s in sorted(set(g.tolist())):
        idx = np.flatnonzero(g == s)
        W[1:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=draws)
    return W


def xerr_core(D, Wt):
    """The X_err / U lines of run_xerr_v6.py's xerr(), verbatim. D: probes x tasks (fractions); Wt: draws x tasks."""
    import numpy as np

    wsum = Wt.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        est = (Wt @ D.T) / wsum[:, None]
    P = D.shape[0]
    comp = np.isfinite(est).all(1)
    if not comp[0] or int(comp[1:].sum()) < 3 or P < 2:
        return None
    e = est[comp]
    dbar, S2 = e.mean(1), e.var(1, ddof=1)
    wp = (e[1:] - dbar[1:, None]).var(0, ddof=1)
    spread2 = max(0.0, S2[0] - wp.sum() / (P - 1))
    bias2 = max(0.0, dbar[0] ** 2 - dbar[1:].var(ddof=1))
    plug = np.sqrt(dbar**2 + S2)
    return {
        "X_err": 100 * float(np.sqrt(bias2 + spread2)),
        "U": 100 * float(np.quantile(plug[1:], 0.95)),
        "bias_floored": bool(bias2 == 0.0),
        "spread_floored": bool(spread2 == 0.0),
        "complete_draws": int(comp[1:].sum()),
    }


def true_xerr(delta_pts):
    import numpy as np

    d = np.asarray(delta_pts, float)
    return float(np.sqrt(d.mean() ** 2 + d.var(ddof=1)))


def load_dump(sha):
    raw = Path(DUMP).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise GateError("dump sha differs from the pinned one")
    return json.loads(raw)


def sim_minus_mean_real(pp):
    """Per cohort task: hit(sim) - (hit(run1) + hit(run2)) / 2 at the dumped threshold (v6's sim_minus_mean_real)."""
    import numpy as np

    h = {w: (np.array(pp["arms"][w]) > pp["threshold"]).astype(float) for w in ARMS}
    return h["sim"] - (h["run1"] + h["run2"]) / 2


def gate(dump) -> dict:
    """G0: from the dump alone, the copied estimator reproduces v6's pooled and with-strategy X_err and U (<= 1e-12)."""
    import numpy as np

    raw = Path(f"/root/in/{V6[0]}").read_bytes()
    if hashlib.sha256(raw).hexdigest() != V6[1]:
        raise GateError("v6 record sha")
    v6 = json.loads(raw)["result"]["results"]
    cohort, c_est, env, hs = dump["cohort"], dump["c_est"], dump["env"], dump["has_strat"]
    cpos = {t: i for i, t in enumerate(cohort)}
    W = make_draws(dump["est_half"], env)
    pos = {t: i for i, t in enumerate(dump["est_half"])}
    out, worst = {}, 0.0
    for rk, r in dump["rows"].items():
        ro, al = rk.split("|alpha=")
        rec = v6[f"{ro}|mechanical|alpha={al}"]
        if r["eligible"] != rec["eligibility_append13_4"]["eligible"]:
            raise GateError(f"{rk}: eligible set differs from v6")
        Dm = (
            np.stack([sim_minus_mean_real(r["per_probe"][p]) for p in r["eligible"]])
            if r["eligible"]
            else np.zeros((0, len(cohort)))
        )
        for scope in ("pooled", "with_strategy"):
            tasks = [t for t in c_est if scope == "pooled" or hs[t]]
            mine = xerr_core(Dm[:, [cpos[t] for t in tasks]], W[:, [pos[t] for t in tasks]]) if len(Dm) else None
            ref = rec["primary_eligible_estimation_half"]["sim_minus_mean_real"][scope]
            if mine is None or ref.get("X_err") is None:
                if (mine is None) != (ref.get("X_err") is None):
                    raise GateError(f"{rk} {scope}: defined-ness differs from v6")
                out[f"{rk}|{scope}"] = "undefined in both"
                continue
            d = max(abs(mine["X_err"] - ref["X_err"]), abs(mine["U"] - ref["U_approx_upper_95"]))
            worst = max(worst, d)
            out[f"{rk}|{scope}"] = {
                "X_err": mine["X_err"],
                "U": mine["U"],
                "v6_X_err": ref["X_err"],
                "v6_U": ref["U_approx_upper_95"],
            }
    if worst > 1e-12:
        raise GateError(f"G0 failed: max |diff| {worst}")
    return {"pass": True, "max_abs_diff": worst, "rows": out}


def prep(dump, rk):
    """Super-population for one row (270 runs pooled; 255 with-strategy), honest runs by world, T*, delta per pattern."""
    import numpy as np

    r = dump["rows"][rk]
    el = r["eligible"]
    cohort, c_est, env, hs = dump["cohort"], dump["c_est"], dump["env"], dump["has_strat"]
    cpos = {t: i for i, t in enumerate(cohort)}
    runs = np.stack(
        [np.stack([np.array(r["per_probe"][p]["arms"][w])[[cpos[t] for t in c_est]] for p in el], -1) for w in ARMS],
        1,
    )  # tasks x 3 x P
    worlds = sorted({w for p in el for w in r["per_probe"][p]["pool_worlds"]})
    hk = {w: sorted({k for p in el for k in r["per_probe"][p]["honest"] if k.split("|", 1)[0] == w}) for w in worlds}
    H = {w: np.array([[r["per_probe"][p]["honest"].get(k, np.nan) for p in el] for k in hk[w]]) for w in worlds}
    Tstar = np.array([r["per_probe"][p]["threshold"] for p in el])
    for j, p in enumerate(el):
        pool = np.concatenate([H[w][:, j] for w in r["per_probe"][p]["pool_worlds"]])
        pool = pool[np.isfinite(pool)]
        th, n_, k_ = thresholds(pool)[float(rk.split("=")[1])]
        if (th, n_, k_) != (Tstar[j], r["per_probe"][p]["n"], r["per_probe"][p]["k"]):
            raise GateError(f"{rk} {p}: the rebuilt pool's (threshold, n, k) differ from the dump")
    pop = runs.reshape(-1, len(el))
    ws = np.array([hs[t] for t in c_est])
    cell_counts = {}
    for t in c_est:
        cell_counts[(env[t], hs[t])] = cell_counts.get((env[t], hs[t]), 0) + 1
    pop_ws = runs[ws].reshape(-1, len(el))
    M = len(pop)

    def flips(j, d, T):
        return int(((pop[:, j] + d) > T).sum()) - int((pop[:, j] > T).sum())

    def calibrate(j, tau):  # the main rule's delta procedure (integer tie-break)
        if tau == 0:
            return 0.0, 0.0
        b = np.unique(Tstar[j] - pop[:, j])
        h = max(1.0, float(b.max() - b.min()))
        cand = np.concatenate([[0.0], b, (b[1:] + b[:-1]) / 2, [b.min() - h, b.max() + h]])
        best = min((abs(100 * flips(j, c, Tstar[j]) - M * tau), abs(c), c) for c in cand.tolist())
        return float(best[2]), 100 * flips(j, best[2], Tstar[j]) / M

    P = len(el)
    vec = {2: [1, -1], 3: [1, -1, 0], 4: [1, -1, 1, -1]}[P]
    unit = 1 / np.sqrt(np.sum(np.square(vec)) / (P - 1))
    allnz = {2: [1 / np.sqrt(2), -1 / np.sqrt(2)], 3: [1 / np.sqrt(3), 1 / np.sqrt(3), -2 / np.sqrt(3)]}.get(
        P, [np.sqrt(3 / 4) * v for v in (1, -1, 1, -1)]
    )
    targets = {
        "a_null": [0.0] * P,
        "b_plus5": [5.0] * P,
        "b_minus5": [-5.0] * P,
        "b_plus2p5": [2.5] * P,
        "b_minus2p5": [-2.5] * P,
        "c_hetero_spread5": [float(5 * unit * v) for v in vec],
        "c_hetero_spread2p5": [float(2.5 * unit * v) for v in vec],
        "d_single_plus10": [10.0] + [0.0] * (P - 1),
        "e_sparse_single_plus1p5": [1.5] + [0.0] * (P - 1),
        "h_plus10": [10.0] * P,
        "h_minus10": [-10.0] * P,
        "i_hetero_spread10": [float(10 * unit * v) for v in vec],
        "j_pair_pm8": [8.0, -8.0] + [0.0] * (P - 2),
        "k_all_spread8": [float(8 * v) for v in allnz],
        "m_mixed_common3_spread3": [float(3 + 3 * unit * v) for v in vec],
    }
    for j, pr in enumerate(el):
        for sgn, nm in ((5.0, "plus5"), (-5.0, "minus5")):
            targets[f"l_single_{nm}_{pr}"] = [sgn if k == j else 0.0 for k in range(P)]
    cal = {}
    for sname, tv in targets.items():
        dl = [calibrate(j, t) for j, t in enumerate(tv)]
        ach = [a for _, a in dl]
        cal[sname] = {
            "target_pts": tv,
            "delta_score": [d for d, _ in dl],
            "achieved_pts_at_Tstar": ach,
            "theta_Tstar": true_xerr(ach),
            "theta_Tstar_with_strategy": true_xerr(
                [
                    100
                    * (int(((pop_ws[:, j] + dl[j][0]) > Tstar[j]).sum()) - int((pop_ws[:, j] > Tstar[j]).sum()))
                    / len(pop_ws)
                    for j in range(P)
                ]
            ),
            "theta_Tstar_target": true_xerr(tv),
            # 09's recheck note: flag probes whose achieved shift misses the target (catch ceiling / floor)
            "saturated_probes": [el[j] for j, (t, a) in enumerate(zip(tv, ach, strict=True)) if abs(a - t) > 0.5],
            "catch_room_pts": {
                el[j]: {
                    "up": 100 * float((pop[:, j] <= Tstar[j]).mean()),
                    "down": 100 * float((pop[:, j] > Tstar[j]).mean()),
                }
                for j in range(P)
            },
        }
    return {
        "el": el,
        "tasks": c_est,
        "ws": ws,
        "slots": dump["est_half"],
        "env": env,
        "hs": hs,
        "runs": runs,
        "H": H,
        "pool_worlds": [r["per_probe"][p]["pool_worlds"] for p in el],
        "Tstar": Tstar,
        "real_pool_n": [r["per_probe"][p]["n"] for p in el],
        "cell_counts": cell_counts,
        "cal": cal,
        "alpha": float(rk.split("=")[1]),
        "pop": pop,
        "pop_ws": pop_ws,
    }


def theta_at(popx, dlt, That):
    m = len(popx)
    return true_xerr(
        [
            100 * (int(((popx[:, j] + dlt[j]) > That[j]).sum()) - int((popx[:, j] > That[j]).sum())) / m
            for j in range(len(dlt))
        ]
    )


def one_dataset(sp, ri, pi, pname, i):
    import numpy as np

    rng = np.random.default_rng([20261015, 2, ri, pi, i])
    tasks, env, hs, slots = sp["tasks"], sp["env"], sp["hs"], sp["slots"]
    cells = {}
    for j, t in enumerate(tasks):  # Amendment 1: each cohort slot is filled within its environment x strategy cell
        cells.setdefault((env[t], hs[t]), []).append(j)
    pick = np.array([rng.choice(cells[(env[t], hs[t])]) for t in tasks])
    perm = np.array([rng.permutation(3) for _ in pick])
    R = sp["runs"][pick[:, None], perm]
    Hs = {w: h[rng.integers(0, len(h), len(h))] for w, h in sp["H"].items()}
    P = len(sp["el"])
    That = np.empty(P)
    pool_n = []
    for j in range(P):
        pool = np.concatenate([Hs[w][:, j] for w in sp["pool_worlds"][j]])
        pool = pool[np.isfinite(pool)]
        pool_n.append(len(pool))
        That[j] = thresholds(pool)[sp["alpha"]][0]
    dlt = np.array(sp["cal"][pname]["delta_score"])
    sim = R[:, 2, :] + dlt
    D = ((sim > That).astype(float) - ((R[:, 0, :] > That).astype(float) + (R[:, 1, :] > That).astype(float)) / 2).T
    W = make_draws(slots, env, seed=[20261015, 2, ri, pi, i, 1])
    spos = {t: k for k, t in enumerate(slots)}
    out = {"i": i, "pool_n": pool_n}
    for st in STRATA:
        idx = [k for k, t in enumerate(tasks) if st == "pooled" or hs[t]]
        xe = xerr_core(D[:, idx], W[:, [spos[tasks[k]] for k in idx]])
        th = theta_at(sp["pop"] if st == "pooled" else sp["pop_ws"], dlt, That)
        out[st] = {"X": None if xe is None else xe["X_err"], "U": None if xe is None else xe["U"], "theta": th}
    return out


@app.function(image=image, cpu=1, memory=2048, timeout=1800, retries=0, volumes={"/work": work_vol})
def batch(args) -> list:
    dump_sha, ri, pi, pname, idx = args
    sp = prep(load_dump(dump_sha), ROWS[ri])
    return [one_dataset(sp, ri, pi, pname, i) for i in idx]


@app.function(image=image, cpu=2, memory=4096, timeout=1800, retries=0, volumes={"/work": work_vol})
def setup(dump_sha) -> dict:
    try:
        dump = load_dump(dump_sha)
        g = gate(dump)
        cpg = max(
            abs(x - y)
            for kn in ((0, 50), (3, 50), (47, 50), (950, 1000), (500, 500), (480, 500))
            for x, y in zip(cp(*kn), cp_bisect(*kn), strict=True)
        )
        if cpg > 1e-9:
            raise GateError(f"CP gate: scipy and bisection intervals differ by {cpg}")
        g["cp_gate_max_abs_diff"] = cpg
        cal = {}
        for rk in ROWS:
            sp = prep(dump, rk)
            cal[rk] = {
                "eligible": sp["el"],
                "real_pool_n": sp["real_pool_n"],
                "patterns": sp["cal"],
                "cell_counts": {f"{e}|{h}": n for (e, h), n in sp["cell_counts"].items()},
            }
        return {"G0": g, "calibration": cal}
    except GateError as e:
        return {"fatal": "GateError", "gate": str(e)}


def cp_bisect(k, n, a=0.05):  # Clopper-Pearson by bisection (run_coverage.py); kept only to gate cp() below
    from math import comb

    if n == 0:
        return None

    def tail_ge(p):
        return sum(comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1))

    def tail_le(p):
        return sum(comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k + 1))

    def bis(f, target, inc):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if (f(mid) < target) == inc:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    return [0.0 if k == 0 else bis(tail_ge, a / 2, True), 1.0 if k == n else bis(tail_le, a / 2, False)]


def cp(k, n, a=0.05):
    """Exact Clopper-Pearson via beta quantiles (same interval as cp_bisect, fast enough for thousands of calls)."""
    from scipy.stats import beta

    if n == 0:
        return None
    return [
        0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1)),
        1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k)),
    ]


@app.function(image=image, cpu=4, memory=8192, timeout=1800, retries=0)
def aggregate(datasets: dict, g0_rows: dict, calib: dict) -> dict:
    """Amendment 1 (with Codex's required items): q and c by inverted_cdf 95th percentiles over the grid; all datasets
    in every denominator; split-half, leave-one-pattern-out and leave-one-family-out coverage with CP intervals; the
    X_hat = 0 diagnostic; realized truths, saturation and pool-size drift; the real v6 rows."""
    import numpy as np

    def q95(v):
        v = np.asarray(v, float)
        return float(np.quantile(v, 0.95, method="inverted_cdf"))

    def ratio(th, U):
        r = np.where(U > 0, th / np.where(U > 0, U, 1.0), np.where(th > 0, np.inf, 0.0))
        return r

    def fit(per, pats, mask=None):
        q = max(q95(per[p]["theta"][m] - per[p]["X"][m]) for p in pats for m in [mask(p) if mask else slice(None)])
        c = max(
            q95(ratio(per[p]["theta"][m], per[p]["U"][m])) for p in pats for m in [mask(p) if mask else slice(None)]
        )
        return max(0.0, q), c

    def cov(per, p, q, c, m=slice(None)):
        th, X, U = per[p]["theta"][m], per[p]["X"][m], per[p]["U"][m]
        n = len(th)
        k1 = int(((th - X) <= q).sum())  # the calibration's own arithmetic (Codex fix-check item 2)
        k2 = int((ratio(th, U) <= c).sum()) if np.isfinite(c) else None
        return {
            "n": n,
            "U_cal": k1 / n if n else None,
            "U_cal_cp95": cp(k1, n),
            "U_s": None if k2 is None else k2 / n,
            "U_s_cp95": None if k2 is None else cp(k2, n),
        }

    def worst(d, key):
        vals = [v[key] for v in d.values() if v.get(key) is not None]
        return min(vals) if vals else None

    def fin(c):
        return c if np.isfinite(c) else "no finite empirically calibrated multiplier"

    def frame(v):
        return "fine (<= 1)" if v <= 1 else "sad (>= 5)" if v >= 5 else "between 1 and 5"

    res = {}
    for rk in ROWS:
        pats = list(calib[rk]["patterns"])
        orig = [p for p in pats if p in ORIGINAL]
        for st in STRATA:
            per = {}
            for pn in pats:
                rows = datasets[f"{rk}|{pn}"]
                if any(r[st]["X"] is None for r in rows):
                    raise GateError(f"{rk}|{pn}|{st}: an undefined X_err (all datasets must enter the denominators)")
                per[pn] = {
                    "X": np.array([r[st]["X"] for r in rows]),
                    "U": np.array([r[st]["U"] for r in rows]),
                    "theta": np.array([r[st]["theta"] for r in rows]),
                    "even": np.array([r["i"] % 2 == 0 for r in rows]),
                }
            q_o, c_o = fit(per, orig)
            q_e, c_e = fit(per, pats)

            def validate(grid, per=per):
                """Split-half, leave-one-pattern-out and leave-one-family-out, fitting only on this grid's patterns."""
                qh, ch = fit(per, grid, mask=lambda p: per[p]["even"])
                split = {p: cov(per, p, qh, ch, ~per[p]["even"]) for p in grid}
                lopo = {}
                for j in grid:
                    qj, cj = fit(per, [p for p in grid if p != j])
                    lopo[j] = {"q_without_j": qj, "c_without_j": fin(cj)} | cov(per, j, qj, cj)
                lofo = {}
                for fam in FAMILIES:
                    held = [p for p in grid if family_of(p) == fam]
                    rest = [p for p in grid if family_of(p) != fam]
                    if held and rest:
                        qf, cf = fit(per, rest)
                        lofo[fam] = {
                            "q_without": qf,
                            "c_without": fin(cf),
                            "per_pattern": {p: cov(per, p, qf, cf) for p in held},
                        }
                wl = {k: worst(lopo, k) for k in ("U_cal", "U_s")}
                return {
                    "split_half": {
                        "q_even": qh,
                        "c_even": fin(ch),
                        "coverage_odd": split,
                        "min_odd": {k: worst(split, k) for k in ("U_cal", "U_s")},
                        "note": "validates the procedure at 500 datasets per pattern, not the full-data constants",
                    },
                    "leave_one_pattern_out": {
                        "per_pattern": lopo,
                        "worst": wl,
                        "diagnostic": "severe under-coverage (worst < 0.90)"
                        if wl["U_cal"] is not None and wl["U_cal"] < 0.90
                        else "no severe under-coverage diagnostic (worst >= 0.90; neither nominal coverage nor "
                        "extrapolation)",
                    },
                    "leave_one_family_out": lofo,
                }

            zero = {}
            for p in pats:
                m = per[p]["X"] == 0
                zero[p] = {
                    "n_X_zero": int(m.sum()),
                    "coverage_U_cal_given_X_zero": float((per[p]["theta"][m] <= q_e).mean()) if m.any() else None,
                    "theta_q95_given_X_zero": q95(per[p]["theta"][m]) if m.any() else None,
                }
            g = g0_rows.get(f"{rk}|{st}")
            real = None
            if isinstance(g, dict):
                x, u = g["v6_X_err"], g["v6_U"]
                real = {
                    "X_hat": x,
                    "U": u,
                    "U_cal_extended": x + q_e,
                    "U_cal_original": x + q_o,
                    "U_s_extended": None if not np.isfinite(c_e) else c_e * u,
                    "U_s_original": None if not np.isfinite(c_o) else c_o * u,
                    "frame_U": frame(u),
                    "frame_U_cal_extended": frame(x + q_e),
                    "frame_U_s_extended": None if not np.isfinite(c_e) else frame(c_e * u),
                    "caveat": "X_hat = 0 here, so U_cal = q: neither 95% coverage conditional on observing zero nor a "
                    "posterior statement; protects only against the listed simulated patterns (Codex item 6)"
                    if x == 0
                    else "protects only against the listed simulated patterns (Codex item 6)",
                }
            res[f"{rk}|{st}"] = {
                "patterns": pats,
                "q_original": q_o,
                "q_extended": q_e,
                "c_original": fin(c_o),
                "c_extended": fin(c_e),
                "per_pattern_q95_theta_minus_X": {p: q95(per[p]["theta"] - per[p]["X"]) for p in pats},
                "per_pattern_q95_ratio": {p: q95(ratio(per[p]["theta"], per[p]["U"])) for p in pats},
                "in_sample_coverage_extended": {p: cov(per, p, q_e, c_e) for p in pats},
                "in_sample_note": "descriptive: the constants were fitted on these same datasets, so these CP intervals "
                "lack the usual fixed-bound binomial interpretation",
                "validation_extended_grid": validate(pats),
                "validation_original_grid": validate(orig),
                "X_hat_zero_diagnostic": zero,
                "U_zero_datasets": {p: int((per[p]["U"] == 0).sum()) for p in pats},
                "realized_truth": {
                    p: {
                        "theta_Tstar": calib[rk]["patterns"][p][
                            "theta_Tstar" if st == "pooled" else "theta_Tstar_with_strategy"
                        ],
                        "theta_Tstar_target": calib[rk]["patterns"][p]["theta_Tstar_target"],
                        "theta_That_mean": float(per[p]["theta"].mean()),
                        "saturated_probes": calib[rk]["patterns"][p]["saturated_probes"],
                        "label": f"{p} (realized theta* "
                        f"{calib[rk]['patterns'][p]['theta_Tstar' if st == 'pooled' else 'theta_Tstar_with_strategy']:.2f})",
                    }
                    for p in pats
                },
                "pool_n_drift": {
                    pr: {
                        "real_n": calib[rk]["real_pool_n"][j],
                        "min": int(min(r["pool_n"][j] for p in pats for r in datasets[f"{rk}|{p}"])),
                        "max": int(max(r["pool_n"][j] for p in pats for r in datasets[f"{rk}|{p}"])),
                        "mean": float(np.mean([r["pool_n"][j] for p in pats for r in datasets[f"{rk}|{p}"]])),
                    }
                    for j, pr in enumerate(calib[rk]["eligible"])
                },
                "real_v6": real,
            }
    return res


@app.local_entrypoint()
def main(dump_sha: str = ""):
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    if len(dump_sha) != 64:
        raise SystemExit("--dump-sha <64 hex>")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    import time

    t0 = time.time()
    res = {"dump": {"path": DUMP, "sha256": dump_sha}, "setup": setup.remote(dump_sha)}
    if "fatal" not in res["setup"]:
        cal = res["setup"]["calibration"]
        jobs = [
            (dump_sha, ri, pi, pn, list(range(s, min(s + CHUNK, N_DATASETS))))
            for ri, rk in enumerate(ROWS)
            for pi, pn in enumerate(cal[rk]["patterns"])
            for s in range(0, N_DATASETS, CHUNK)
        ]
        by = {}
        for (_, ri, _, pn, _), rows in zip(jobs, batch.map(jobs), strict=True):
            by.setdefault(f"{ROWS[ri]}|{pn}", []).extend(rows)
        res["datasets"] = by
        res["calibrated"] = aggregate.remote(by, res["setup"]["G0"]["rows"], cal)
    res["wall_seconds"] = time.time() - t0
    res["label"] = (
        "POST-HOC (Amendment 1): coverage-calibrated upper limit over the stated grid; PRELIMINARY; exchangeable arms by "
        "permutation; eligibility and Sol moments not re-estimated; coverage concerns each listed empirical sampling "
        "model only (no guarantee between or beyond grid points, at every fixed threshold, or simultaneously); truth = "
        "theta(T-hat); never replaces U in a registered reading; not transferable to the 507-cell analysis"
    )
    dest = HERE / "results" / f"ucal_{stamp}.json"
    dest.open("x").write(json.dumps({"git": git("rev-parse", "HEAD"), "stamp": stamp, "result": res}, default=float))
    print(json.dumps({"wall_seconds": res["wall_seconds"], "setup_keys": list(res["setup"])}), flush=True)
    if "fatal" in res["setup"]:
        print(json.dumps(res["setup"]), flush=True)
    print("wrote", dest.name, flush=True)
