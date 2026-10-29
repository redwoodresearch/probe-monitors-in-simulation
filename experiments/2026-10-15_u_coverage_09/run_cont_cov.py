"""Amendment 2 (RULE.md here): coverage of the CONTINUOUS (score-based) mapped U against the CATCH truth, beside the binary
U on the same simulated datasets; location and non-location sim shifts. The mapped-U pieces are copied verbatim from
contcore.py @ 835acb130 (aswin/continuous-score-09); the binary estimator, gate and draws from run_ucal.py. Modal CPU;
aggregation on Modal. PRELIMINARY.

  XE_IN=<dir with xerr_v6_2026-10-14_161122.json and xerr_cont_v6_2026-10-15_002602.json> PYTHONPATH=src $MODAL run \\
      --detach experiments/2026-10-15_u_coverage_09/run_cont_cov.py --dump-sha <sha256 of /work/ucov/v6_real_dump.json>
"""

import hashlib
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
IN = Path(os.environ.get("XE_IN", "/nonexistent"))
APP = "probe-shift_aswin-surya_sub-convaudit_contcov_kill-2026-11-14"
V6 = ("xerr_v6_2026-10-14_161122.json", "e789144b62f71e0fb168dc3106656c2b5f29aadafe12df0f86dd1f74441d7645")
VC = ("xerr_cont_v6_2026-10-15_002602.json", "c7dafe02a6591a8a7d990420bc8e39712d19f6ec79f7feb7808c27544d93ea12")
DUMP = "/work/ucov/v6_real_dump.json"
ALPHAS = (0.01, 0.05)
DRAWS = 10_000
SEED = [20261014, 1]
ARMS = ("run1", "run2", "sim")
ROWS = ("R_reply_meanpooled|alpha=0.01", "R_reply_meanpooled|alpha=0.05", "C_anycommand_max|alpha=0.05")
STRATA = ("pooled", "with_strategy")
NONLOC = (
    "B1_scale_0p7",
    "B2_scale_1p4",
    "B3_tail_up_2sd",
    "B4_tail_down_to_minus5",
    "B5_mix_down_3sd",
    "B6_mix_up_3sd",
    "B7_mixed_sign_shape",
    "B8_band_up_at_Tstar",
    "B9_band_down_at_Tstar",
)
N_DATASETS = 1000
CHUNK = 50
G0C_TOL = 1e-10

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "scipy==1.16.1")
if IN.exists():
    image = image.add_local_file(IN / V6[0], f"/root/in/{V6[0]}").add_local_file(IN / VC[0], f"/root/in/{VC[0]}")
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


# ---- copied verbatim from contcore.py @ 835acb130 (aswin/continuous-score-09) ----
class CoreError(Exception):
    pass


def phi(z):
    return np.exp(-0.5 * np.asarray(z, float) ** 2) / math.sqrt(2 * math.pi)


def fam(est):
    """Append 19's family formula on a (draws+1) x P per-draw estimate matrix, in the matrix's own units (no conversion);
    the same expressions, in the same order, as run_xerr_v6.py's xerr()."""
    P = est.shape[1]
    comp = np.isfinite(est).all(1)
    n_complete = int(comp[1:].sum())
    n_draws = est.shape[0] - 1
    out = {
        "complete_draws": n_complete,
        "nan_draws": n_draws - n_complete,
        "unstable_over_1pct_nan": (n_draws - n_complete) > 0.01 * n_draws,
        "certification": "ok (>= 95% complete draws)"
        if n_complete >= 0.95 * n_draws
        else "no across-probe certification (< 95% complete draws)",
        "n_probes": P,
    }
    if not comp[0] or n_complete < 3:
        return out | {"X_err": None, "note": "undefined: empty task set or < 3 complete draws"}
    if P < 2:
        return out | {"X_err": None, "note": "too few eligible probes (< 2): spread undefined, no X_err"}
    e = est[comp]
    dbar, S2 = e.mean(1), e.var(1, ddof=1)
    wp = (e[1:] - dbar[1:, None]).var(0, ddof=1)
    spread2 = max(0.0, S2[0] - wp.sum() / (P - 1))
    bias2 = max(0.0, dbar[0] ** 2 - dbar[1:].var(ddof=1))
    plug = np.sqrt(dbar**2 + S2)
    x = float(np.sqrt(bias2 + spread2))
    return out | {
        "X_err": x,
        "X_err_wording": "point estimate 0 (floored)" if x == 0.0 else None,
        "bias_mean": float(dbar[0]),
        "bias_ci95": [float(np.quantile(dbar[1:], q)) for q in (0.025, 0.975)],
        "spread": float(np.sqrt(spread2)),
        "U_approx_upper_95": float(np.quantile(plug[1:], 0.95)),
    }


def scaled(d, k):
    """A fam() result with its value fields multiplied by k (used once, for the binary fractions -> points)."""
    o = dict(d)
    for f in ("X_err", "bias_mean", "spread", "U_approx_upper_95"):
        if o.get(f) is not None:
            o[f] = k * o[f]
    if o.get("bias_ci95"):
        o["bias_ci95"] = [k * v for v in o["bias_ci95"]]
    return o


def bandwidth(x):
    """Silverman on the 2N individual real scores: 0.9 min(SD ddof 1, IQR/1.34) n^-1/5; nan if not positive and finite."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan")
    sd = float(x.std(ddof=1))
    q75, q25 = np.quantile(x, [0.75, 0.25])
    h = 0.9 * min(sd, float(q75 - q25) / 1.34) * len(x) ** (-0.2)
    return h if (math.isfinite(h) and h > 0) else float("nan")


def pool_moments(pool):
    """Mean and SD (ddof 0) of a calibration pool; CoreError unless the SD is finite and positive (RULE.md)."""
    pool = np.asarray(pool, float)
    if not len(pool) or not np.isfinite(pool).all():
        raise CoreError("empty or non-finite calibration pool")
    mu, sd = float(pool.mean()), float(pool.std())
    if not (math.isfinite(sd) and sd > 0):
        raise CoreError("calibration pool SD not finite and positive")
    return mu, sd


def wmean(W, v):
    """Per-draw weighted mean of a per-task vector v (tasks along W's columns); zero-weight draws -> nan."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W @ v) / W.sum(1)


# ---- end of the contcore copy ----


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


def mapped_family(S1, S2, SS, Tp, W):
    """Mapped M per probe (contcore.cell()'s f and m lines, verbatim in effect) and Append 19 on the M matrix.
    S1, S2, SS: probes x tasks standardized scores; Tp: per-probe standardized thresholds; W: draws x tasks."""
    ms = []
    for j in range(S1.shape[0]):
        s1, s2, ss = S1[j], S2[j], SS[j]
        h = bandwidth(np.concatenate([s1, s2]))
        d = wmean(W, ss - (s1 + s2) / 2)
        if not (math.isfinite(h) and h > 0):
            f = np.full(W.shape[0], np.nan)
        else:
            f = wmean(W, (phi((Tp[j] - s1) / h) + phi((Tp[j] - s2) / h)) / (2 * h))
        ms.append(100 * f * d)
    out = fam(np.stack(ms, 1))
    return {"X": out.get("X_err"), "U": out.get("U_approx_upper_95")}


def prep(dump, rk):
    """Super-population (270 pooled / 255 with-strategy runs), honest runs by world, T*, and every pattern's transform."""
    r = dump["rows"][rk]
    el = r["eligible"]
    cohort, c_est, env, hs = dump["cohort"], dump["c_est"], dump["env"], dump["has_strat"]
    cpos = {t: i for i, t in enumerate(cohort)}
    runs = np.stack(
        [np.stack([np.array(r["per_probe"][p]["arms"][w])[[cpos[t] for t in c_est]] for p in el], -1) for w in ARMS],
        1,
    )
    worlds = sorted({w for p in el for w in r["per_probe"][p]["pool_worlds"]})
    hk = {w: sorted({k for p in el for k in r["per_probe"][p]["honest"] if k.split("|", 1)[0] == w}) for w in worlds}
    H = {w: np.array([[r["per_probe"][p]["honest"].get(k, np.nan) for p in el] for k in hk[w]]) for w in worlds}
    Tstar = np.array([r["per_probe"][p]["threshold"] for p in el])
    pools = []
    for j, p in enumerate(el):
        pool = np.concatenate([H[w][:, j] for w in r["per_probe"][p]["pool_worlds"]])
        pool = pool[np.isfinite(pool)]
        th, n_, k_ = thresholds(pool)[float(rk.split("=")[1])]
        if (th, n_, k_) != (Tstar[j], r["per_probe"][p]["n"], r["per_probe"][p]["k"]):
            raise GateError(f"{rk} {p}: the rebuilt pool's (threshold, n, k) differ from the dump")
        pools.append(pool)
    pop = runs.reshape(-1, len(el))
    ws = np.array([hs[t] for t in c_est])
    pop_ws = runs[ws].reshape(-1, len(el))
    real180 = runs[:, :2, :].reshape(-1, len(el))
    M = len(pop)
    P = len(el)
    m_p = pop.mean(0)
    q90 = np.array([float(np.quantile(real180[:, j], 0.90)) for j in range(P)])
    sig = np.array([pool_moments(pools[j])[1] for j in range(P)])

    def g(spec, x):  # the pattern's transform of sim scores (deterministic part)
        k = spec["kind"]
        if k == "loc":
            return x + spec["delta"]
        if k == "scale":
            return spec["m"] + spec["lam"] * (x - spec["m"])
        if k == "tail":
            return np.where(x > spec["q"], x + spec["delta"], x)
        if k == "mix":
            return x + spec["delta"]  # applied to the selected occurrences only
        if k == "band":  # threshold-local transport (Codex optional): scores in (lo, hi] move by delta
            return np.where((x > spec["lo"]) & (x <= spec["hi"]), x + spec["delta"], x)
        raise GateError(f"unknown transform {k}")

    def change(spec, popx, j, T):  # exact population catch change in points (mixture: expectation)
        x = popx[:, j]
        c = 100 * (int((g(spec, x) > T).sum()) - int((x > T).sum())) / len(x)
        return spec.get("pi", 1.0) * c

    def calibrate(j, tau, mask=None):  # main-rule delta procedure on the affected runs (integer tie-break)
        if tau == 0:
            return 0.0
        x = pop[:, j] if mask is None else pop[mask, j]
        b = np.unique(Tstar[j] - x)
        h = max(1.0, float(b.max() - b.min()))
        cand = np.concatenate([[0.0], b, (b[1:] + b[:-1]) / 2, [b.min() - h, b.max() + h]])

        def flips(d):
            return int(((x + d) > Tstar[j]).sum()) - int((x > Tstar[j]).sum())

        return float(min((abs(100 * flips(c) - M * tau), abs(c), c) for c in cand.tolist())[2])

    vec = {2: [1, -1], 3: [1, -1, 0], 4: [1, -1, 1, -1]}[P]
    unit = 1 / np.sqrt(np.sum(np.square(vec)) / (P - 1))
    allnz = {2: [1 / np.sqrt(2), -1 / np.sqrt(2)], 3: [1 / np.sqrt(3), 1 / np.sqrt(3), -2 / np.sqrt(3)]}.get(
        P, [np.sqrt(3 / 4) * v for v in (1, -1, 1, -1)]
    )
    loc_targets = {
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
            loc_targets[f"l_single_{nm}_{pr}"] = [sgn if k == j else 0.0 for k in range(P)]
    specs, targets = {}, {}
    for nm, tv in loc_targets.items():
        specs[nm] = [{"kind": "loc", "delta": calibrate(j, t)} for j, t in enumerate(tv)]
        targets[nm] = tv
    tail_mask = [pop[:, j] > q90[j] for j in range(P)]
    tail_down = [calibrate(j, -5.0, tail_mask[j]) for j in range(P)]
    specs["B1_scale_0p7"] = [{"kind": "scale", "lam": 0.7, "m": float(m_p[j])} for j in range(P)]
    specs["B2_scale_1p4"] = [{"kind": "scale", "lam": 1.4, "m": float(m_p[j])} for j in range(P)]
    specs["B3_tail_up_2sd"] = [{"kind": "tail", "q": float(q90[j]), "delta": float(2 * sig[j])} for j in range(P)]
    specs["B4_tail_down_to_minus5"] = [{"kind": "tail", "q": float(q90[j]), "delta": tail_down[j]} for j in range(P)]
    targets["B4_tail_down_to_minus5"] = [-5.0] * P
    specs["B5_mix_down_3sd"] = [{"kind": "mix", "pi": 0.2, "delta": float(-3 * sig[j])} for j in range(P)]
    specs["B6_mix_up_3sd"] = [{"kind": "mix", "pi": 0.2, "delta": float(3 * sig[j])} for j in range(P)]
    cyc = []
    for j in range(P):
        kind = j % 4
        if kind == 0:
            cyc.append({"kind": "scale", "lam": 1.4, "m": float(m_p[j])})
        elif kind == 1:
            cyc.append({"kind": "scale", "lam": 0.7, "m": float(m_p[j])})
        elif kind == 2:
            cyc.append({"kind": "tail", "q": float(q90[j]), "delta": tail_down[j]})
        else:
            cyc.append({"kind": "loc", "delta": calibrate(j, -5.0)})
    specs["B7_mixed_sign_shape"] = cyc
    w = 0.5 * sig  # frozen band width w_p = 0.5 x the real pool SD, anchored at T*
    specs["B8_band_up_at_Tstar"] = [
        {"kind": "band", "lo": float(Tstar[j] - w[j]), "hi": float(Tstar[j]), "delta": float(w[j])} for j in range(P)
    ]
    specs["B9_band_down_at_Tstar"] = [
        {"kind": "band", "lo": float(Tstar[j]), "hi": float(Tstar[j] + w[j]), "delta": float(-w[j])} for j in range(P)
    ]
    cal = {}
    for nm, sp_ in specs.items():
        ach = [change(sp_[j], pop, j, Tstar[j]) for j in range(P)]
        ach_ws = [change(sp_[j], pop_ws, j, Tstar[j]) for j in range(P)]
        tv = targets.get(nm)
        cal[nm] = {
            "family": "non-location" if nm in NONLOC else "location",
            "spec": sp_,
            "target_pts": tv,
            "achieved_pts_at_Tstar": ach,
            "theta_Tstar": true_xerr(ach),
            "theta_Tstar_with_strategy": true_xerr(ach_ws),
            "saturated_probes": [el[j] for j in range(P) if tv is not None and abs(ach[j] - tv[j]) > 0.5],
        }
    return {
        "el": el,
        "tasks": c_est,
        "slots": dump["est_half"],
        "env": env,
        "hs": hs,
        "runs": runs,
        "H": H,
        "pool_worlds": [r["per_probe"][p]["pool_worlds"] for p in el],
        "Tstar": Tstar,
        "alpha": float(rk.split("=")[1]),
        "pop": pop,
        "pop_ws": pop_ws,
        "specs": specs,
        "cal": cal,
        "catch_room_pts": {
            el[j]: {
                "up": 100 * float((pop[:, j] <= Tstar[j]).mean()),
                "down": 100 * float((pop[:, j] > Tstar[j]).mean()),
            }
            for j in range(P)
        },
        "g": g,
        "change": change,
    }


def one_dataset(sp, ri, pi, pname, i):
    rng = np.random.default_rng([20261015, 3, ri, pi, i])
    tasks, env, hs, slots = sp["tasks"], sp["env"], sp["hs"], sp["slots"]
    cells = {}
    for j, t in enumerate(tasks):
        cells.setdefault((env[t], hs[t]), []).append(j)
    pick = np.array([rng.choice(cells[(env[t], hs[t])]) for t in tasks])
    perm = np.array([rng.permutation(3) for _ in pick])
    R = sp["runs"][pick[:, None], perm]  # slots x 3 x P
    Hs = {w: h[rng.integers(0, len(h), len(h))] for w, h in sp["H"].items()}
    P = len(sp["el"])
    specs = sp["specs"][pname]
    mixmask = rng.random(len(pick)) < 0.2  # one occurrence-level draw, used only by the mixture patterns
    sim = R[:, 2, :].copy()
    for j in range(P):
        if specs[j]["kind"] == "mix":
            sim[mixmask, j] = sp["g"](specs[j], sim[mixmask, j])
        else:
            sim[:, j] = sp["g"](specs[j], sim[:, j])
    That, mu, sd = np.empty(P), np.empty(P), np.empty(P)
    for j in range(P):
        pool = np.concatenate([Hs[w][:, j] for w in sp["pool_worlds"][j]])
        pool = pool[np.isfinite(pool)]
        That[j] = thresholds(pool)[sp["alpha"]][0]
        mu[j], sd[j] = pool_moments(pool)
    hit = lambda x: (x > That).astype(float)
    Db = (hit(sim) - (hit(R[:, 0, :]) + hit(R[:, 1, :])) / 2).T
    S1 = ((R[:, 0, :] - mu) / sd).T
    S2 = ((R[:, 1, :] - mu) / sd).T
    SS = ((sim - mu) / sd).T
    Tp = (That - mu) / sd
    W = make_draws(slots, env, seed=[20261015, 3, ri, pi, i, 1])
    spos = {t: k for k, t in enumerate(slots)}
    out = {"i": i}
    for st in STRATA:
        idx = [k for k, t in enumerate(tasks) if st == "pooled" or hs[t]]
        Wt = W[:, [spos[tasks[k]] for k in idx]]
        xb = xerr_core(Db[:, idx], Wt)
        xm = mapped_family(S1[:, idx], S2[:, idx], SS[:, idx], Tp, Wt)
        popx = sp["pop"] if st == "pooled" else sp["pop_ws"]
        th = true_xerr([sp["change"](specs[j], popx, j, That[j]) for j in range(P)])
        out[st] = {
            "theta": th,
            "Xb": None if xb is None else xb["X_err"],
            "Ub": None if xb is None else xb["U"],
            "Xm": xm["X"],
            "Um": xm["U"],
        }
    return out


@app.function(image=image, cpu=1, memory=2048, timeout=1800, retries=0, volumes={"/work": work_vol})
def batch(args) -> list:
    dump_sha, ri, pi, pname, idx = args
    sp = prep(load_dump(dump_sha), ROWS[ri])
    return [one_dataset(sp, ri, pi, pname, i) for i in idx]


def gate_cont(dump) -> dict:
    """G0c: from the dump, the mapped X and U equal xerr_cont_v6's family_mapped_points (pooled, with-strategy) to 1e-10."""
    raw = Path(f"/root/in/{VC[0]}").read_bytes()
    if hashlib.sha256(raw).hexdigest() != VC[1]:
        raise GateError("continuous record sha")
    rec = json.loads(raw)["result"]["results"]
    cohort, c_est, hs, env = dump["cohort"], dump["c_est"], dump["has_strat"], dump["env"]
    cpos = {t: i for i, t in enumerate(cohort)}
    W = make_draws(dump["est_half"], env)
    pos = {t: i for i, t in enumerate(dump["est_half"])}
    out, worst = {}, 0.0
    for rk in ROWS:
        r = dump["rows"][rk]
        el = r["eligible"]
        ro, al = rk.split("|alpha=")
        blk = rec[f"{ro}|mechanical|alpha={al}"]["continuous"]
        sc = {}
        for p in el:
            pp = r["per_probe"][p]
            pool = np.array([v for k, v in pp["honest"].items() if k.split("|", 1)[0] in pp["pool_worlds"]])
            mu, sd = pool_moments(pool)
            if (
                abs(mu - blk["standardization"][p]["mu"]) > 1e-12
                or abs(sd - blk["standardization"][p]["sd_ddof0"]) > 1e-12
            ):
                raise GateError(f"G0c {rk} {p}: pool moments differ from the continuous record")
            sc[p] = ({w: (np.array(pp["arms"][w]) - mu) / sd for w in ARMS}, (pp["threshold"] - mu) / sd)
        for st in STRATA:
            tasks = [t for t in c_est if st == "pooled" or hs[t]]
            cols = [cpos[t] for t in tasks]
            Wt = W[:, [pos[t] for t in tasks]]
            xm = mapped_family(
                np.stack([sc[p][0]["run1"][cols] for p in el]),
                np.stack([sc[p][0]["run2"][cols] for p in el]),
                np.stack([sc[p][0]["sim"][cols] for p in el]),
                np.array([sc[p][1] for p in el]),
                Wt,
            )
            ref = blk[st]["family_mapped_points"]
            d = max(abs(xm["X"] - ref["X_err"]), abs(xm["U"] - ref["U_approx_upper_95"]))
            worst = max(worst, d)
            out[f"{rk}|{st}"] = {
                "X_map": xm["X"],
                "U_map": xm["U"],
                "ref_X": ref["X_err"],
                "ref_U": ref["U_approx_upper_95"],
            }
    if worst > G0C_TOL:
        raise GateError(f"G0c failed: max |diff| {worst}")
    return {"pass": True, "max_abs_diff": worst, "rows": out}


@app.function(image=image, cpu=2, memory=4096, timeout=1800, retries=0, volumes={"/work": work_vol})
def setup(dump_sha) -> dict:
    try:
        dump = load_dump(dump_sha)
        g = gate(dump)
        gc = gate_cont(dump)
        cpg = max(
            abs(x - y)
            for kn in ((0, 50), (3, 50), (47, 50), (950, 1000), (500, 500), (480, 500))
            for x, y in zip(cp(*kn), cp_bisect(*kn), strict=True)
        )
        if cpg > 1e-9:
            raise GateError(f"CP gate: {cpg}")
        cal = {}
        for rk in ROWS:
            sp = prep(dump, rk)
            cal[rk] = {"eligible": sp["el"], "patterns": sp["cal"], "catch_room_pts": sp["catch_room_pts"]}
        return {"G0": g, "G0c": gc, "cp_gate_max_abs_diff": cpg, "calibration": cal}
    except (GateError, CoreError) as e:
        return {"fatal": type(e).__name__, "gate": str(e)}


@app.function(image=image, cpu=4, memory=8192, timeout=1800, retries=0)
def aggregate(datasets: dict, calib: dict) -> dict:
    """Coverage (CP), conditional-on-X=0 coverage, worst patterns, sizes, and the pre-set screen (Codex items 3-4), for
    the mapped and the binary U."""

    def cov(th, U):
        n = len(th)
        k = int((th <= U).sum())
        c = cp(k, n)
        return {
            "n": n,
            "rate": k / n if n else None,
            "cp95": c,
            "evidence_of_undercoverage": bool(c is not None and c[1] < 0.95),
        }

    def cond(th, U):
        n0 = len(th)
        if n0 == 0:
            return {"n0": 0, "status": "unavailable (no dataset with X_hat = 0)"}
        c = cov(th, U)
        return c | {
            "n0": n0,
            "status": "assessed" if n0 >= 30 else "descriptive only (n0 < 30): conditional screen unassessed",
        }

    def screen(per, est):
        if any(v[est].get("unavailable") for v in per.values()):
            return {"verdict": "unavailable for a replacement decision (undefined estimator output)"}
        bad = [
            p
            for p, v in per.items()
            if not (v[est]["coverage"]["rate"] >= 0.95 and v[est]["coverage"]["cp95"][0] >= 0.93)
        ]
        badc = [p for p, v in per.items() if v[est]["cond_X0"]["n0"] >= 30 and v[est]["cond_X0"]["rate"] < 0.90]
        unass = [p for p, v in per.items() if 0 < v[est]["cond_X0"]["n0"] < 30]
        return {
            "verdict": "passes the preregistered simulation screen"
            if not bad and not badc
            else "does not pass the screen; binary U retained",
            "patterns_failing_coverage": bad,
            "patterns_failing_conditional": badc,
            "patterns_conditional_unassessed": unass,
            "patterns_with_evidence_of_undercoverage": [
                p for p, v in per.items() if v[est]["coverage"]["evidence_of_undercoverage"]
            ],
            "note": "screen labels establish neither nominal population coverage nor undercoverage",
        }

    res, overall = {}, {"mapped": [], "binary": []}
    for rk in ROWS:
        pats = list(calib[rk]["patterns"])
        for st in STRATA:
            per = {}
            for pn in pats:
                rows = [r[st] for r in datasets[f"{rk}|{pn}"]]
                th = np.array([r["theta"] for r in rows])
                ent = {
                    "family": calib[rk]["patterns"][pn]["family"],
                    "theta_Tstar": calib[rk]["patterns"][pn][
                        "theta_Tstar" if st == "pooled" else "theta_Tstar_with_strategy"
                    ],
                    "saturated_probes": calib[rk]["patterns"][pn]["saturated_probes"],
                    "mean_theta_That": float(th.mean()),
                    "share_theta_That_zero": float((th == 0).mean()),
                }
                for est, xk, uk in (("mapped", "Xm", "Um"), ("binary", "Xb", "Ub")):
                    bad = [
                        1
                        for r in rows
                        if r[xk] is None or r[uk] is None or not (math.isfinite(r[xk]) and math.isfinite(r[uk]))
                    ]
                    if bad:
                        ent[est] = {"unavailable": True, "n_undefined": len(bad)}
                        continue
                    X, U = np.array([r[xk] for r in rows]), np.array([r[uk] for r in rows])
                    z = X == 0
                    ent[est] = {
                        "coverage": cov(th, U),
                        "cond_X0": cond(th[z], U[z]),
                        "mean_U": float(U.mean()),
                        "mean_X": float(X.mean()),
                        "share_X0": float(z.mean()),
                    }
                per[pn] = ent
            out = {"per_pattern": per}
            for est in ("mapped", "binary"):
                out[f"screen_{est}"] = screen(per, est)
                for fam_ in ("location", "non-location"):
                    sub = [
                        (v[est]["coverage"]["rate"], p)
                        for p, v in per.items()
                        if v["family"] == fam_ and not v[est].get("unavailable")
                    ]
                    out[f"worst_{est}_{fam_}"] = min(sub) if sub else None
                overall[est].append(out[f"screen_{est}"]["verdict"] == "passes the preregistered simulation screen")
            res[f"{rk}|{st}"] = out
    res["overall"] = {
        est: "passes the preregistered simulation screen (all six row x stratum cells)"
        if all(v)
        else "does not pass the screen; binary U retained"
        for est, v in overall.items()
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
        res["coverage"] = aggregate.remote(by, cal)
    res["wall_seconds"] = time.time() - t0
    res["label"] = (
        "Amendment 2: coverage of the continuous mapped U against the CATCH truth theta(T-hat), beside the binary U on the "
        "same datasets; PRELIMINARY; Amendment 1's empirical sampling model (exchangeable arms by permutation, Sol moments "
        "not re-estimated); per-cell intervals; valid only over this grid"
    )
    dest = HERE / "results" / f"contcov_{stamp}.json"
    dest.open("x").write(json.dumps({"git": git("rev-parse", "HEAD"), "stamp": stamp, "result": res}, default=float))
    print(json.dumps({"wall_seconds": res["wall_seconds"], "setup_keys": list(res["setup"])}), flush=True)
    if "fatal" in res["setup"]:
        print(json.dumps(res["setup"]), flush=True)
    print("wrote", dest.name, flush=True)
