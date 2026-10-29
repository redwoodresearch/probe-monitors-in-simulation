"""U coverage simulation (RULE.md here). Modal CPU. PRELIMINARY.

  mode gate:  G0 only (the copied estimator reproduces the v6 primary from the dump)
  mode pilot: G0 + 20 datasets per row x setting (timing / cost)
  mode full:  G0 + 500 datasets per row x setting

  XE_IN=<dir holding xerr_v6_2026-10-14_161122.json> PYTHONPATH=src $MODAL run --detach \\
      experiments/2026-10-15_u_coverage_09/run_coverage.py --mode gate --dump-sha <sha256 of /work/ucov/v6_real_dump.json>
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
APP = "probe-shift_aswin-surya_sub-convaudit_ucov_kill-2026-11-14"
V6 = ("xerr_v6_2026-10-14_161122.json", "e789144b62f71e0fb168dc3106656c2b5f29aadafe12df0f86dd1f74441d7645")
DUMP = "/work/ucov/v6_real_dump.json"
ALPHAS = (0.01, 0.05)
DRAWS = 10_000
SEED = [20261014, 1]
ARMS = ("run1", "run2", "sim")
ROWS = ("R_reply_meanpooled|alpha=0.01", "R_reply_meanpooled|alpha=0.05", "C_anycommand_max|alpha=0.05")
SETTINGS = (
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
N = {"pilot": 20, "full": 500}
CHUNK = 25

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
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
    """The super-population for one row: 90 tasks x 3 runs x P scores; honest runs by world; T*; delta calibration."""
    import numpy as np

    r = dump["rows"][rk]
    el = r["eligible"]
    cohort, c_est, env = dump["cohort"], dump["c_est"], dump["env"]
    cpos = {t: i for i, t in enumerate(cohort)}
    runs = np.stack(
        [np.stack([np.array(r["per_probe"][p]["arms"][w])[[cpos[t] for t in c_est]] for p in el], -1) for w in ARMS],
        1,
    )  # tasks x 3 x P
    worlds = sorted({w for p in el for w in r["per_probe"][p]["pool_worlds"]})
    hk = {w: sorted({k for p in el for k in r["per_probe"][p]["honest"] if k.split("|", 1)[0] == w}) for w in worlds}
    H = {w: np.array([[r["per_probe"][p]["honest"].get(k, np.nan) for p in el] for k in hk[w]]) for w in worlds}
    Tstar = np.array([r["per_probe"][p]["threshold"] for p in el])
    for j, p in enumerate(el):  # T* must be the order statistic of the dumped pool (sanity)
        pool = np.concatenate([H[w][:, j] for w in r["per_probe"][p]["pool_worlds"]])
        pool = pool[np.isfinite(pool)]  # a run without this probe's score is not in its real pool either
        if len(pool) != r["per_probe"][p]["n"]:
            raise GateError(f"{rk} {p}: dumped pool size differs from v6's n")
        th, n_, k_ = thresholds(pool)[float(rk.split("=")[1])]
        if (th, n_, k_) != (Tstar[j], r["per_probe"][p]["n"], r["per_probe"][p]["k"]):
            raise GateError(f"{rk} {p}: the rebuilt pool's (threshold, n, k) differ from the dump")
    pop = runs.reshape(-1, len(el))  # 270 runs

    M = len(pop)

    def flips(j, d, T):  # integer change in the hit count (exact, so the registered tie-break is exact)
        return int(((pop[:, j] + d) > T).sum()) - int((pop[:, j] > T).sum())

    def achieved(j, d, T):
        return 100 * flips(j, d, T) / M

    def calibrate(j, tau):
        """RULE (Codex items 5 and fix-check): candidates 0, every distinct breakpoint b = T* - s, adjacent midpoints,
        min(b) - h and max(b) + h (h = max(1, range)); minimise (|100 m - M tau|, |delta|, delta) with m the integer
        change in hits at T*."""
        if tau == 0:
            return 0.0, 0.0
        b = np.unique(Tstar[j] - pop[:, j])
        h = max(1.0, float(b.max() - b.min()))
        cand = np.concatenate([[0.0], b, (b[1:] + b[:-1]) / 2, [b.min() - h, b.max() + h]])
        best = min((abs(100 * flips(j, c, Tstar[j]) - M * tau), abs(c), c) for c in cand.tolist())
        return float(best[2]), achieved(j, best[2], Tstar[j])

    P = len(el)
    vec = {2: [1, -1], 3: [1, -1, 0], 4: [1, -1, 1, -1]}[P]
    unit = 1 / np.sqrt(np.sum(np.square(vec)) / (P - 1))
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
    }
    cal = {}
    for sname, tv in targets.items():
        dl = [calibrate(j, t) for j, t in enumerate(tv)]
        cal[sname] = {
            "target_pts": tv,
            "delta_score": [d for d, _ in dl],
            "achieved_pts_at_Tstar": [a for _, a in dl],
            "theta_Tstar": true_xerr([a for _, a in dl]),
        }
    pool_worlds = [r["per_probe"][p]["pool_worlds"] for p in el]
    return {
        "el": el,
        "tasks": c_est,
        "slots": dump["est_half"],
        "env": env,
        "runs": runs,
        "H": H,
        "pool_worlds": pool_worlds,
        "Tstar": Tstar,
        "cal": cal,
        "alpha": float(rk.split("=")[1]),
        "pop": pop,
    }


def one_dataset(sp, ri, si, sname, i):
    import numpy as np

    rng = np.random.default_rng([20261015, ri, si, i])
    tasks, env, slots = sp["tasks"], sp["env"], sp["slots"]
    # RULE (Codex item 2): v6's 98 estimation slots and 90-task cohort mask; each cohort slot is filled by a task drawn
    # within its environment; every occurrence gets its own run permutation (joint across probes)
    by_env = {}
    for j, t in enumerate(tasks):
        by_env.setdefault(env[t], []).append(j)
    pick = np.array([rng.choice(by_env[env[t]]) for t in tasks])
    perm = np.array([rng.permutation(3) for _ in pick])
    R = sp["runs"][pick[:, None], perm]  # cohort slots x 3 x P; columns are now the (run1, run2, sim) labels
    Hs = {w: h[rng.integers(0, len(h), len(h))] for w, h in sp["H"].items()}
    P = len(sp["el"])
    That = np.empty(P)
    for j in range(P):
        pool = np.concatenate([Hs[w][:, j] for w in sp["pool_worlds"][j]])
        That[j] = thresholds(pool[np.isfinite(pool)])[sp["alpha"]][0]
    dlt = np.array(sp["cal"][sname]["delta_score"])
    sim = R[:, 2, :] + dlt
    D = ((sim > That).astype(float) - ((R[:, 0, :] > That).astype(float) + (R[:, 1, :] > That).astype(float)) / 2).T
    W = make_draws(slots, env, seed=[20261015, ri, si, i, 1])  # all 98 slots; v6's cohort restriction below
    spos = {t: k for k, t in enumerate(slots)}
    xe = xerr_core(D, W[:, [spos[t] for t in tasks]])
    pop = sp["pop"]
    M = len(pop)
    cond = [
        100 * (int(((pop[:, j] + dlt[j]) > That[j]).sum()) - int((pop[:, j] > That[j]).sum())) / M for j in range(P)
    ]
    return {
        "i": i,
        "X_err": None if xe is None else xe["X_err"],
        "U": None if xe is None else xe["U"],
        "bias_floored": None if xe is None else xe["bias_floored"],
        "spread_floored": None if xe is None else xe["spread_floored"],
        "theta_That": true_xerr(cond),  # PRIMARY truth (the fixed-threshold estimand)
        "theta_Tstar": sp["cal"][sname]["theta_Tstar"],  # secondary (robustness to calibration noise)
    }


@app.function(image=image, cpu=1, memory=2048, timeout=1800, retries=0, volumes={"/work": work_vol})
def batch(args) -> list:
    dump_sha, ri, si, idx = args
    dump = load_dump(dump_sha)
    sp = prep(dump, ROWS[ri])
    return [one_dataset(sp, ri, si, SETTINGS[si], i) for i in idx]


@app.function(image=image, cpu=2, memory=4096, timeout=1800, retries=0, volumes={"/work": work_vol})
def setup(dump_sha) -> dict:
    try:
        dump = load_dump(dump_sha)
        g = gate(dump)
        cal = {}
        for rk in ROWS:
            sp = prep(dump, rk)
            cal[rk] = {"eligible": sp["el"], "Tstar": sp["Tstar"].tolist(), "settings": sp["cal"]}
        return {"G0": g, "calibration": cal}
    except GateError as e:
        return {"fatal": "GateError", "gate": str(e)}


def summarize(rows):
    from math import comb

    import numpy as np

    def cp(k, n, a=0.05):  # Clopper-Pearson by bisection on the binomial tail (no scipy)
        def tail_ge(p):
            return sum(comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1))

        def tail_le(p):
            return sum(comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k + 1))

        def bis(f, target, lo=0.0, hi=1.0, inc=True):
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

    ok = [r for r in rows if r["U"] is not None]
    x = np.array([r["X_err"] for r in ok])
    u = np.array([r["U"] for r in ok])
    th = np.array([r["theta_That"] for r in ok])  # PRIMARY truth
    ts = np.array([r["theta_Tstar"] for r in ok])  # secondary
    k1, k2, n = int((u >= th).sum()), int((u >= ts).sum()), len(ok)

    def q(v, p):
        return float(np.quantile(v, p)) if len(v) else None

    return {
        "n_datasets": len(rows),
        "n_defined": n,
        "theta_Tstar": float(ts[0]) if n else None,
        "theta_That": {"mean": float(th.mean()), "p05": q(th, 0.05), "p95": q(th, 0.95)} if n else None,
        "coverage_primary_theta_That": {
            "k": k1,
            "n": n,
            "rate": k1 / n if n else None,
            "cp95": cp(k1, n) if n else None,
        },
        "coverage_secondary_theta_Tstar": {
            "k": k2,
            "n": n,
            "rate": k2 / n if n else None,
            "cp95": cp(k2, n) if n else None,
        },
        "X_err": {
            "mean": float(x.mean()),
            "median": q(x, 0.5),
            "sd": float(x.std(ddof=1)) if n > 1 else None,
            "p05": q(x, 0.05),
            "p95": q(x, 0.95),
            "share_zero": float((x == 0).mean()),
            "mean_bias_vs_theta_That": float((x - th).mean()),
            "mean_bias_vs_theta_Tstar": float((x - ts).mean()),
        },
        "U": {"mean": float(u.mean()), "p05": q(u, 0.05), "p50": q(u, 0.5), "p95": q(u, 0.95)},
        "share_X_err_positive": float((x > 0).mean()),
    }


def reading(s, sname):
    """RULE (Codex item 4): per-cell CP intervals, not simultaneous guarantees."""
    if sname == "a_null":
        return "null: coverage trivially 1 (U >= 0 = truth); see share_X_err_positive and the U distribution"
    out = {}
    for key in ("coverage_primary_theta_That", "coverage_secondary_theta_Tstar"):
        lo, hi = s[key]["cp95"]
        out[key] = (
            "evidence of under-coverage"
            if hi < 0.95
            else "evidence of conservative coverage"
            if lo > 0.95
            else "inconclusive about departure from nominal"
        )
    return out


def with_reading(s, sname):
    return s | {"reading": reading(s, sname)}


@app.local_entrypoint()
def main(mode: str = "gate", dump_sha: str = ""):
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    if mode not in ("gate", "pilot", "full") or len(dump_sha) != 64:
        raise SystemExit("--mode gate|pilot|full --dump-sha <64 hex>")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    import time

    t0 = time.time()
    res = {"mode": mode, "dump": {"path": DUMP, "sha256": dump_sha}, "setup": setup.remote(dump_sha)}
    if "fatal" not in res["setup"] and mode != "gate":
        jobs = [
            (dump_sha, ri, si, list(range(s, min(s + CHUNK, N[mode]))))
            for ri in range(len(ROWS))
            for si in range(len(SETTINGS))
            for s in range(0, N[mode], CHUNK)
        ]
        out = list(batch.map(jobs))
        by = {}
        for (_, ri, si, _), rows in zip(jobs, out, strict=True):
            by.setdefault((ri, si), []).extend(rows)
        res["datasets"] = {f"{ROWS[ri]}|{SETTINGS[si]}": v for (ri, si), v in sorted(by.items())}
        res["summary"] = {
            f"{ROWS[ri]}|{SETTINGS[si]}": with_reading(summarize(v), SETTINGS[si]) for (ri, si), v in sorted(by.items())
        }
    res["wall_seconds"] = time.time() - t0
    res["label"] = (
        "PRELIMINARY; simulated coverage under a model matched to v6's data; assumes tasks exchangeable within "
        "environments and the three attack runs per task exchangeable apart from the imposed shift; does not cover "
        "threshold-transformation re-estimation (Sol moments) or eligibility re-selection (RULE.md here)"
    )
    dest = HERE / "results" / f"ucov_{mode}_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(json.dumps({"git": git("rev-parse", "HEAD"), "stamp": stamp, "result": res}, default=float))
    print(json.dumps({k: res[k] for k in ("mode", "wall_seconds")} | {"setup_keys": list(res["setup"])}), flush=True)
    if "fatal" in res["setup"]:
        print(json.dumps(res["setup"]), flush=True)
    for k, v in (res.get("summary") or {}).items():
        print(k, json.dumps({"cov": v["coverage_primary_theta_That"], "reading": v["reading"]}), flush=True)
    print("wrote", dest.name, flush=True)
