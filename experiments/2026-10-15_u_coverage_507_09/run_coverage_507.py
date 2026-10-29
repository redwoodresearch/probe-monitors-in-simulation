"""U coverage on the 507 design (RULE.md here; registered design @ e6ded1811, Panel 5 item 3). Modal CPU. PRELIMINARY.

  mode gate:  G0 only (the copied estimator reproduces Revision 6s's X_err / U from the dump)
  mode pilot: G0 + 10 datasets per row x setting (timing / cost)
  mode full:  G0 + 300 datasets per row x setting

  XE_REC=<dir holding xerr_scaleup_final_2026-10-15_102820.json> $MODAL run --detach \\
      experiments/2026-10-15_u_coverage_507_09/run_coverage_507.py --mode gate --dump-path <path> --dump-sha <sha256>
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
IN = Path(os.environ.get("XE_REC", "/nonexistent"))
APP = "probe-shift_aswin-surya_sub-convaudit_ucov507_kill-2026-11-14"
REC = ("xerr_scaleup_final_2026-10-15_102820.json", "d0933874561627c947f67af2cc0900c50d077ed0e6752eaf93620cd035a0890b")
ROWS = ("R|0.05", "R|0.01", "C|0.05")
ARMS = ("run1", "run2", "sim")
REG_DRAWS = 10_000
SIM_DRAWS = 2_000
DELTAS = (1, 3, 5, 8)
SETTINGS = ("null",) + tuple(f"{k}_{d}" for d in DELTAS for k in ("common_plus", "common_minus", "hetero", "single_sol"))
N = {"pilot": 10, "full": 300}
CHUNK = 100

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
if IN.exists():
    image = image.add_local_file(IN / REC[0], f"/root/in/{REC[0]}")
work_vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2)


class GateError(Exception):
    pass


def order_stat(values, a):  # verbatim from run_xerr_scaleup.py (6s)
    import numpy as np

    s = np.sort(np.asarray(values, float), kind="mergesort")[::-1]
    if not len(s) or not np.isfinite(s).all():
        raise GateError("empty or non-finite pool")
    k = int(np.floor(a * len(s) + 1e-9))
    return float(s[min(k, len(s) - 1)]), len(s), k


def make_draws(units, stratum, seed, draws):  # run_xerr_scaleup.py's, with the draw count as an argument
    import numpy as np

    rng = np.random.default_rng(seed)
    W = np.zeros((draws + 1, len(units)))
    W[0] = 1.0
    g = [stratum[u] for u in units]
    for s in sorted(set(g)):
        idx = np.array([i for i, v in enumerate(g) if v == s], dtype=np.intp)
        W[1:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=draws)
    return W


def xerr_core(est, draws):
    """run_xerr_scaleup.py xerr_from_est's X_err / U / bias lines, verbatim; est: (draws+1) x P."""
    import numpy as np

    P = est.shape[1]
    comp = np.isfinite(est).all(1)
    n_c = int(comp[1:].sum())
    if not comp[0] or n_c < 3 or P < 2:
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
        "bias_mean": 100 * float(dbar[0]),
        "complete_draws": n_c,
        "certified": bool(n_c >= 0.95 * draws),
    }


def true_theta(a_pts):  # Append 19's functional of achieved shifts (the U-coverage study's true_xerr)
    import numpy as np

    d = np.asarray(a_pts, float)
    return float(np.sqrt(d.mean() ** 2 + d.var(ddof=1)))


def load_dump(path, sha):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise GateError("dump sha differs from the pinned one")
    return json.loads(raw)


def row_arrays(dump, rk):
    """Analysed cells (primary cohort, estimation mains), their main / origin, scores (cells x 3 x P), T* by origin."""
    import numpy as np

    ctx, r = dump["context"], dump["rows"][rk]
    el = r["eligible"]
    cohort = ctx["cohort"]
    cpos = {c: i for i, c in enumerate(cohort)}
    mains = [m for m in ctx["mains_est"] if any(ctx["cells_eff_primary"][c] for c in ctx["cells_of_main"][m])]
    cells, cmain = [], []
    for m in mains:
        for c in ctx["cells_of_main"][m]:
            if ctx["cells_eff_primary"][c]:
                cells.append(c)
                cmain.append(m)
    Z = np.full((len(cells), 3, len(el)), np.nan)
    for j, p in enumerate(el):
        cz = r["probes"][p]["cohort_z"]
        for a, w in enumerate(ARMS):
            Z[:, a, j] = [np.nan if cz[w][cpos[c]] is None else cz[w][cpos[c]] for c in cells]
    if not np.isfinite(Z).all():
        raise GateError(f"{rk}: non-finite analysed cohort score (the FINAL row would have been withheld)")
    orig = np.array([ctx["origin"][c] for c in cells])
    Tstar = np.array([[r["probes"][p]["T"][o] for p in el] for o in ("old", "new")])  # 2 x P
    return {"el": el, "mains": mains, "cells": cells, "cmain": cmain, "Z": Z, "orig": orig, "Tstar": Tstar}


def hits_D(Z, orig, T):
    """Per cell and member: hit(sim) - (hit(run1) + hit(run2)) / 2 at T[origin]; T: 2 x P (row 0 old, row 1 new)."""
    import numpy as np

    Tc = np.where((orig == "old")[:, None], T[0][None, :], T[1][None, :])  # cells x P
    h = (Z > Tc[:, None, :]).astype(float)  # cells x 3 x P
    return h[:, 2, :] - (h[:, 0, :] + h[:, 1, :]) / 2


def main_gaps(D, cmain, mains):
    import numpy as np

    idx = {m: [] for m in mains}
    for i, m in enumerate(cmain):
        idx[m].append(i)
    return np.stack([D[idx[m]].mean(0) for m in mains], 1)  # P x mains


def gate(dump) -> dict:
    """G0: from the dump alone, the copied estimator reproduces Revision 6s's combined / S_old / S_new X_err and U."""
    import numpy as np

    raw = Path(f"/root/in/{REC[0]}").read_bytes()
    if hashlib.sha256(raw).hexdigest() != REC[1]:
        raise GateError("record sha")
    res = json.loads(raw)["result"]["results"]
    ctx = dump["context"]
    univ = ctx["univ"]
    strat = {m: tuple(v) for m, v in ctx["univ_stratum"].items()}
    W = make_draws(univ, strat, ctx["seed"], REG_DRAWS)
    pos = {m: i for i, m in enumerate(univ)}
    out, worst = {}, 0.0
    for rk in ROWS:
        ra, rec = row_arrays(dump, rk), res[rk]["primary"]
        if ra["el"] != rec["eligible_v6_frozen"]:
            raise GateError(f"{rk}: eligible set differs from the record")
        r = dump["rows"][rk]
        for j, p in enumerate(ra["el"]):  # T* is the registered order statistic of the dumped pool
            pr = r["probes"][p]
            a = float(rk.split("|")[1])
            if pr["scheme"] == "stratum_specific":
                for o, keys in (("old", pr["pool_keys_old"]), ("new", pr["pool_keys_new"])):
                    t, n, k = order_stat([pr["pool_z"][k_] for k_ in keys], a)
                    if (t, n, k) != (pr["T"][o], *pr["n_k"][o]):
                        raise GateError(f"{rk} {p} {o}: rebuilt (T, n, k) differ from the dump")
            elif pr["scheme"] == "union":
                t, n, k = order_stat([pr["pool_z"][k_] for k_ in pr["pool_keys_old"] + pr["pool_keys_new"]], a)
                if (t, n, k) != (pr["T"]["old"], *pr["n_k"]["union"]) or pr["T"]["old"] != pr["T"]["new"]:
                    raise GateError(f"{rk} {p}: rebuilt union (T, n, k) differ from the dump")
            else:
                raise GateError(f"{rk} {p}: unknown scheme {pr['scheme']}")
        D = hits_D(ra["Z"], ra["orig"], ra["Tstar"])
        for scope, keep in (("combined", None), ("S_old", "old"), ("S_new", "new")):
            sel = [i for i in range(len(ra["cells"])) if keep is None or ra["orig"][i] == keep]
            ms = [m for m in ra["mains"] if any(ra["cmain"][i] == m for i in sel)]
            Dm = main_gaps(D[sel], [ra["cmain"][i] for i in sel], ms)
            Wt = W[:, [pos[m] for m in ms]]
            with np.errstate(invalid="ignore", divide="ignore"):
                est = (Wt @ Dm.T) / Wt.sum(1)[:, None]
            mine = xerr_core(est, REG_DRAWS)
            ref = rec["sim_minus_mean_real"][scope]["pooled"]
            d = max(abs(mine["X_err"] - ref["X_err"]), abs(mine["U"] - ref["U_approx_upper_95"]))
            worst = max(worst, d)
            out[f"{rk}|{scope}"] = {"mains": len(ms), "max_abs_diff": d}
    if worst > 1e-12:
        raise GateError(f"G0 failed: max |diff| {worst}")
    return {"pass": True, "max_abs_diff": worst, "rows": out}


def prep(dump, rk):
    """The super-population for one row, the honest-pool groups, T*, and the calibrated score shifts per setting."""
    import numpy as np

    ra = row_arrays(dump, rk)
    ctx, r = dump["context"], dump["rows"][rk]
    el, P = ra["el"], len(ra["el"])
    a = float(rk.split("|")[1])
    # population weights: cell-within-main, then equal-main; each of a cell's 3 runs gets a third
    ncell = {m: ra["cmain"].count(m) for m in ra["mains"]}
    w = np.array([1.0 / (3 * ncell[m] * len(ra["mains"])) for m in ra["cmain"]])
    # honest-pool groups (origin, world), shared multiplicities across members
    groups = {}  # a key's origin is the pool list it sits in (pool cells need not be attack-cohort cells)
    for p in el:
        for o, lst in (("old", "pool_keys_old"), ("new", "pool_keys_new")):
            for k in r["probes"][p][lst]:
                groups.setdefault((o, k.split("|", 1)[0]), set()).add(k)
    seen = {}
    for (o, _), keys in groups.items():
        for k in keys:
            if seen.setdefault(k, o) != o:
                raise GateError(f"{rk}: pool key {k} listed under both origins")
    gkeys = {g: sorted(v) for g, v in sorted(groups.items())}
    H = {}
    for g, keys in gkeys.items():
        M = np.full((len(keys), P), np.nan)
        for j, p in enumerate(el):
            pz, inpool = r["probes"][p]["pool_z"], set(r["probes"][p]["pool_keys_old"] + r["probes"][p]["pool_keys_new"])
            for i, k in enumerate(keys):
                if k in inpool:
                    M[i, j] = pz[k]
        H[g] = M
    schemes = [r["probes"][p]["scheme"] for p in el]
    Zall = ra["Z"]  # cells x 3 x P
    orig3 = np.repeat(ra["orig"], 3)
    w3 = np.repeat(w, 3)

    def achieved(j, d, T):  # T: 2 x P
        z = Zall[:, :, j].reshape(-1)
        t = np.where(orig3 == "old", T[0][j], T[1][j])
        return 100 * float((w3 * ((z + d > t).astype(float) - (z > t).astype(float))).sum())

    Tstar = ra["Tstar"]

    def calibrate(j, tau):
        if tau == 0:
            return 0.0, 0.0
        z = Zall[:, :, j].reshape(-1)
        t = np.where(orig3 == "old", Tstar[0][j], Tstar[1][j])
        b = np.unique(t - z)
        h = max(1.0, float(b.max() - b.min()))
        cand = np.concatenate([[0.0], b, (b[1:] + b[:-1]) / 2, [b.min() - h, b.max() + h]])
        best = min((abs(achieved(j, c, Tstar) - tau), abs(c), c) for c in cand.tolist())
        return float(best[2]), achieved(j, best[2], Tstar)

    vec = {2: [1, -1], 3: [1, -1, 0], 4: [1, -1, 1, -1]}[P]
    unit = 1 / np.sqrt(np.sum(np.square(vec)) / (P - 1))
    if el[0] != "sol_a":
        raise GateError(f"{rk}: Sol (a) is not the first eligible member")
    targets = {"null": [0.0] * P}
    for dd in DELTAS:
        targets[f"common_plus_{dd}"] = [float(dd)] * P
        targets[f"common_minus_{dd}"] = [float(-dd)] * P
        targets[f"hetero_{dd}"] = [float(dd * unit * v) for v in vec]
        targets[f"single_sol_{dd}"] = [float(dd)] + [0.0] * (P - 1)
    cal = {}
    for sname in SETTINGS:
        dl = [calibrate(j, t) for j, t in enumerate(targets[sname])]
        cal[sname] = {
            "target_pts": targets[sname],
            "delta_score": [d for d, _ in dl],
            "achieved_pts_at_Tstar": [x for _, x in dl],
            "theta_Tstar": true_theta([x for _, x in dl]),
        }
    univ = ctx["univ"]
    strat = {m: tuple(v) for m, v in ctx["univ_stratum"].items()}
    analysed = set(ra["mains"])
    by_stratum = {}  # registered design: every universe main of the stratum, analysed or not
    for m in univ:
        by_stratum.setdefault(strat[m], []).append(m)
    cells_of = {m: [i for i, mm in enumerate(ra["cmain"]) if mm == m] for m in ra["mains"]}
    return {
        "el": el,
        "alpha": a,
        "schemes": schemes,
        "Z": Zall,
        "orig": ra["orig"],
        "cells_of": cells_of,
        "univ": univ,
        "strat": strat,
        "analysed": analysed,
        "by_stratum": by_stratum,
        "gkeys": gkeys,
        "H": H,
        "pool_groups": {
            p: {
                "old": sorted({("old", k.split("|", 1)[0]) for k in r["probes"][p]["pool_keys_old"]}),
                "new": sorted({("new", k.split("|", 1)[0]) for k in r["probes"][p]["pool_keys_new"]}),
            }
            for p in el
        },
        "Tstar": Tstar,
        "cal": cal,
        "achieved": achieved,
    }


def one_dataset(sp, ri, si, sname, i):
    import numpy as np

    rng = np.random.default_rng([20261015, 3, ri, si, i])
    P = len(sp["el"])
    # 1-2 (registered: e6ded1811 Panel 5 item 3): every universe slot is filled by a main drawn with replacement from
    # all registered universe mains of its environment x origin-membership stratum; a slot holding a non-analysed main
    # carries no data (as in the registered draws), so the number of analysed occurrences varies across datasets
    slots, occ = [], []
    for si_, m0 in enumerate(sp["univ"]):
        cand = sp["by_stratum"][sp["strat"][m0]]
        m = cand[int(rng.integers(0, len(cand)))]
        if m in sp["analysed"]:
            slots.append(si_)
            occ.append(m)
    if not occ:
        return {"i": i, "available": False, "X_err": None, "U": None, "bias_mean": None,
                "theta_That": None, "theta_Tstar": sp["cal"][sname]["theta_Tstar"], "n_occurrences": 0}
    # 3-4: honest pools, shared multiplicities across members; thresholds by the frozen scheme
    Hs = {g: M[rng.integers(0, len(M), len(M))] for g, M in sp["H"].items()}
    That = np.empty((2, P))
    for j, p in enumerate(sp["el"]):
        old = np.concatenate([Hs[g][:, j] for g in sp["pool_groups"][p]["old"]])
        new = np.concatenate([Hs[g][:, j] for g in sp["pool_groups"][p]["new"]])
        old, new = old[np.isfinite(old)], new[np.isfinite(new)]
        if sp["schemes"][j] == "stratum_specific":
            That[0, j] = order_stat(old, sp["alpha"])[0]
            That[1, j] = order_stat(new, sp["alpha"])[0]
        else:
            That[0, j] = That[1, j] = order_stat(np.concatenate([old, new]), sp["alpha"])[0]
    dlt = np.array(sp["cal"][sname]["delta_score"])
    # 5-6: per occurrence, permuted runs with the shift on the sim-labelled run; per-occurrence gaps
    Dm = np.empty((P, len(occ)))
    for q, m in enumerate(occ):
        ci = sp["cells_of"][m]
        R = sp["Z"][ci]  # cells x 3 x P
        perm = np.array([rng.permutation(3) for _ in ci])
        R = R[np.arange(len(ci))[:, None], perm]
        R = R.copy()
        R[:, 2, :] += dlt
        Dm[:, q] = hits_D(R, sp["orig"][ci], That).mean(0)
    W = make_draws(sp["univ"], sp["strat"], [20261015, 3, ri, si, i, 1], SIM_DRAWS)
    Wt = W[:, slots]
    with np.errstate(invalid="ignore", divide="ignore"):
        est = (Wt @ Dm.T) / Wt.sum(1)[:, None]
    xe = xerr_core(est, SIM_DRAWS)
    ok = xe is not None and xe["certified"]
    th = true_theta([sp["achieved"](j, dlt[j], That) for j in range(P)])
    return {
        "i": i,
        "available": bool(ok),
        "X_err": xe["X_err"] if ok else None,
        "U": xe["U"] if ok else None,
        "bias_mean": xe["bias_mean"] if ok else None,
        "theta_That": th,
        "theta_Tstar": sp["cal"][sname]["theta_Tstar"],
        "n_occurrences": len(occ),
    }


@app.function(image=image, cpu=1, memory=4096, timeout=1800, retries=0, volumes={"/work": work_vol})
def batch(args) -> list:
    dump_path, dump_sha, ri, si, idx = args
    dump = load_dump(dump_path, dump_sha)
    sp = prep(dump, ROWS[ri])
    return [one_dataset(sp, ri, si, SETTINGS[si], i) for i in idx]


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0, volumes={"/work": work_vol})
def setup(dump_path, dump_sha) -> dict:
    try:
        dump = load_dump(dump_path, dump_sha)
        g = gate(dump)
        cal = {}
        for rk in ROWS:
            sp = prep(dump, rk)
            cal[rk] = {
                "eligible": sp["el"],
                "schemes": sp["schemes"],
                "Tstar": sp["Tstar"].tolist(),
                "analysed_mains": len(sp["cells_of"]),
                "universe_slots": len(sp["univ"]),
                "settings": sp["cal"],
            }
        return {"G0": g, "calibration": cal}
    except GateError as e:
        return {"fatal": "GateError", "gate": str(e)}


def cp(k, n, a=0.05):  # Clopper-Pearson by bisection (no scipy); from the U-coverage study
    from math import comb

    def tail_ge(p):
        return sum(comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1))

    def tail_le(p):
        return sum(comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k + 1))

    def bis(f, target, inc=True):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if (f(mid) < target) == inc:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lo = 0.0 if k == 0 else bis(tail_ge, a / 2, inc=True)
    hi = 1.0 if k == n else bis(tail_le, a / 2, inc=False)
    return [lo, hi]


def summarize(rows, sname, target):
    import numpy as np

    ok = [r for r in rows if r["available"]]
    n = len(ok)
    base = {"setting": sname, "requested_target_pts": target, "n_datasets": len(rows), "n_unavailable": len(rows) - n}
    if not n:
        return base | {"label": "unavailable"}
    x = np.array([r["X_err"] for r in ok])
    u = np.array([r["U"] for r in ok])
    th = np.array([r["theta_That"] for r in ok])
    ts = np.array([r["theta_Tstar"] for r in ok])
    k1, k2 = int((u >= th).sum()), int((u >= ts).sum())
    theta = float(np.median(th))
    s = base | {
        "theta_Tstar": float(ts[0]),
        "theta_That": {"median": theta, "p05": float(np.quantile(th, 0.05)), "p95": float(np.quantile(th, 0.95))},
        "X_err": {
            "mean": float(x.mean()),
            "median": float(np.median(x)),
            "share_zero": float((x == 0).mean()),
            "mean_bias_vs_theta_That": float((x - th).mean()),
            "mean_bias_vs_theta_Tstar": float((x - ts).mean()),
        },
        "coverage_primary_theta_That": {"k": k1, "n": n, "rate": k1 / n, "cp95": cp(k1, n)},
        "coverage_secondary_theta_Tstar": {"k": k2, "n": n, "rate": k2 / n, "cp95": cp(k2, n)},
        "U": {"mean": float(u.mean()), "median": float(np.median(u))},
        "occurrences": {"median": float(np.median([r["n_occurrences"] for r in ok])), "min": int(min(r["n_occurrences"] for r in ok))},
        "bias_mean_median": float(np.median([r["bias_mean"] for r in ok])),
    }
    lab = {}
    if sname != "null":
        lab["U_covers"] = (
            "U covers (screening threshold 0.90; not nominal 95% coverage)"
            if k1 / n >= 0.90
            else "U does not cover (screening threshold 0.90)"
        )
        dd = int(sname.rsplit("_", 1)[1])
        if dd >= 3:
            gap = abs(float(np.median(x)) - theta)
            lab["X_err_recovers"] = "X_err recovers" if gap <= max(1.0, 0.25 * theta) else "X_err does not recover"
            lab["shortfall"] = bool(float(np.median(x)) < theta - max(1.0, 0.25 * theta))
    return s | {"labels": lab}


@app.local_entrypoint()
def main(mode: str = "gate", dump_path: str = "", dump_sha: str = ""):
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    if mode not in ("gate", "pilot", "full") or len(dump_sha) != 64 or not dump_path.startswith("/work/ucov507/"):
        raise SystemExit("--mode gate|pilot|full --dump-path /work/ucov507/<file> --dump-sha <64 hex>")
    import time

    t0 = time.time()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = {"mode": mode, "dump": {"path": dump_path, "sha256": dump_sha}, "setup": setup.remote(dump_path, dump_sha)}
    if "fatal" not in res["setup"] and mode != "gate":
        jobs = [
            (dump_path, dump_sha, ri, si, list(range(s, min(s + CHUNK, N[mode]))))
            for ri in range(len(ROWS))
            for si in range(len(SETTINGS))
            for s in range(0, N[mode], CHUNK)
        ]
        out = list(batch.map(jobs))
        by = {}
        for (_, _, ri, si, _), rows in zip(jobs, out, strict=True):
            by.setdefault((ri, si), []).extend(rows)
        cal = res["setup"]["calibration"]
        res["datasets"] = {f"{ROWS[ri]}|{SETTINGS[si]}": v for (ri, si), v in sorted(by.items())}
        summ = {
            f"{ROWS[ri]}|{SETTINGS[si]}": summarize(v, SETTINGS[si], cal[ROWS[ri]]["settings"][SETTINGS[si]]["target_pts"])
            for (ri, si), v in sorted(by.items())
        }
        under = {}  # "Under-recovery": shortfall in every one of common_plus, common_minus and hetero at a row x delta
        for rk in ROWS:
            for dd in DELTAS:
                if dd < 3:
                    continue
                cells = [summ.get(f"{rk}|{k}_{dd}") for k in ("common_plus", "common_minus", "hetero")]
                if any(c is None or "labels" not in c for c in cells):
                    under[f"{rk}|delta={dd}"] = "unavailable"
                else:
                    under[f"{rk}|delta={dd}"] = (
                        "under-recovery" if all(c["labels"]["shortfall"] for c in cells) else "no under-recovery"
                    )
        res["summary"], res["under_recovery"] = summ, under
    res["wall_seconds"] = time.time() - t0
    res["label"] = (
        "PRELIMINARY; simulated coverage under a model matched to the 507 data (RULE.md here): mains exchangeable within "
        "environment x origin-membership strata, a cell's three attack runs exchangeable apart from the imposed shift; "
        "transformations, scheme selection and eligibility frozen (not re-estimated); 2,000 inner draws"
    )
    res["monte_carlo_note"] = (
        "U is the 95th percentile of 2,000 inner bootstrap draws per dataset; the Monte Carlo standard error of a 95th-"
        "percentile rank is sqrt(0.95 x 0.05 / 2000) = 0.0049 in probability, so each dataset's U sits between about the "
        "94.0th and 96.0th percentile of its plug-in distribution (+-2 SE). The Clopper-Pearson intervals on coverage "
        "describe dataset-to-dataset uncertainty over 300 datasets, not this. 'U covers' uses a 0.90 screening threshold "
        "and does not establish nominal 95% coverage"
    )
    dest = HERE / "results" / f"ucov507_{mode}_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(json.dumps({"git": git("rev-parse", "HEAD"), "stamp": stamp, "result": res}, default=float))
    print(json.dumps({k: res[k] for k in ("mode", "wall_seconds")} | {"setup_keys": list(res["setup"])}), flush=True)
    if "fatal" in res["setup"]:  # status only on stdout; the gate message (with any discrepancy) is in the result file
        print(json.dumps({"fatal": res["setup"]["fatal"], "gate": "see the result file"}), flush=True)
    else:
        print(json.dumps({"G0": {"pass": res["setup"]["G0"]["pass"]}}), flush=True)
    print("wrote", dest.name, flush=True)
