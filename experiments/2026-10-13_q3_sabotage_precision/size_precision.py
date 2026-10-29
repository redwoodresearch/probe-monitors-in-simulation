"""POST HOC planning (RULE.md): precision of the real − simulated sabotage catch gap at the frozen 1% alarm. Numbers only.

  uv run --no-project --python 3.12 --with numpy --with scipy python size_precision.py --inputs <dir> --out <dir>

<dir> holds rows_v2.json, gate_final_2026-10-11_211436.json, sim_label_counts_2026-10-13_121532.json,
learning_result_2026-10-12_144211.json and rescore_full_2026-10-13_014231.json (sha256-checked).
"""

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy import optimize, special

HERE = Path(__file__).resolve().parent
SHA = {"rows_v2.json": "727c417916b8fb2df9a31aca5f8ca0cc4dd17002a8a2edcb7bd79444bc0bf4b5",
       "gate_final_2026-10-11_211436.json": "a08fda3bf41cad1645551c2fc9f7fb1a05d5224b4c653bff4d0ced8573e0c88b",
       "sim_label_counts_2026-10-13_121532.json": "b35bccbc28aa2755c624c0db40bf3c38e905b73168e9d9b0db022dca0436379b",
       "learning_result_2026-10-12_144211.json": "3e71e02e817f051680aa424d3f3561e5f0ed8d2ed84b8cda709489f552553fa6",
       "rescore_full_2026-10-13_014231.json": "09031586a874b099d1951f5609bc7acd3797b654742908bc89a291463d810ba6"}  # from origin/main
MU = {"panel_call_end": 0.2545968882602547, "adjudicated": 0.1240960035224422, "command_endpoint": 0.36633663366336644}
SEED, NSIM, NBOOT, KS, GATE_CI = 20261013, 20000, 2000, [16, 24, 32, 48, 64, 96, 132], [0.042, 0.525]
PRICE = {"central": 110.306358 / 1350, "high": 2.0462 / 20}
SCORING = 3.0
Z, ZW = np.polynomial.hermite_e.hermegauss(80)
ZW = ZW / math.sqrt(2 * math.pi)
RUNS, RUN2_EXTRACTION = ("real1", "real2"), "2026-10-10_second_attack_segends_v1"
# Amendment 3: Aswin's full-population labels for both real runs (label counts only, @ 55a4cca7 on aswin/q3-simsab-ids)
REAL_FULL = {"file": "sim_label_counts_2026-10-13_125229.json", "sha256": "f6e7ad1315f8957c9ec1bdfadb4d3ec9cc020d514061979dd0c7082f846f9cc9",
             "keys": {"real1": "real / held-out", "real2": "run2 / held-out"}}


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(d: Path):
    for n, h in SHA.items():
        if sha256(d / n) != h:
            raise SystemExit(f"{n}: sha256 mismatch")
    rows = [x for x in json.loads((d / "rows_v2.json").read_text())["rows"] if x["role"] == "heldout"]
    fin = json.loads((d / "gate_final_2026-10-11_211436.json").read_text())["final"]["real|0"]
    sc, thr = np.asarray(fin["heldout_scores"]), fin["threshold"]
    if len(sc) != len(rows) or fin["catch"] != MU["panel_call_end"]:
        raise SystemExit("gate rows/scores mismatch")
    tasks = sorted({x["task"] for x in rows})
    # Amendment 2: the gate's held-out set pools real run 1 and real run 2 (run 2 = the second_attack extraction, as in
    # check_joins.EXTRACTIONS); each run keeps its own weighted sample
    run = lambda x: "real2" if x["x"] == RUN2_EXTRACTION else "real1"  # noqa: E731
    w = {r: {t: [] for t in tasks} for r in RUNS}
    fl = {r: {t: [] for t in tasks} for r in RUNS}
    for x, s in zip(rows, sc):
        if x["y"] == 1:
            w[run(x)][x["task"]].append(x["w"])
            fl[run(x)][x["task"]].append(int(s >= thr))
    allw = np.concatenate([w[r][t] for r in RUNS for t in tasks])
    allf = np.concatenate([fl[r][t] for r in RUNS for t in tasks])
    npos = {r: sum(len(v) for v in w[r].values()) for r in RUNS}
    if abs((allw * allf).sum() / allw.sum() - fin["catch"]) > 1e-12 or npos != {"real1": 11, "real2": 20} or len(tasks) != 16:
        raise SystemExit("weighted catch or per-run counts do not reproduce the gate")
    sim = json.loads((d / "sim_label_counts_2026-10-13_121532.json").read_text())["state_changing_labels_per_task"]["held-out"]
    if set(sim) != set(tasks):
        raise SystemExit("simulated tasks differ from the gate's held-out tasks")
    if json.loads((d / "learning_result_2026-10-12_144211.json").read_text())["curve"]["command|n=24"]["catch"]["point"] != MU["command_endpoint"]:
        raise SystemExit("command catch mismatch")
    if json.loads((d / "rescore_full_2026-10-13_014231.json").read_text())["point"]["R_adj_U_check"] != MU["adjudicated"]:
        raise SystemExit("adjudicated recall mismatch")
    W = {r: np.zeros((16, 5)) for r in RUNS}
    for r in RUNS:
        for i, t in enumerate(tasks):
            W[r][i, :len(w[r][t])] = w[r][t]
    return {"tasks": tasks, "W": W, "n_real_full": {r: np.array([round(sum(w[r][t])) for t in tasks]) for r in RUNS},
            "n_obs": {r: np.array([len(w[r][t]) for t in tasks]) for r in RUNS}, "k_obs": {r: np.array([sum(fl[r][t]) for t in tasks]) for r in RUNS},
            "n_sim": np.array([sim[t]["sabotage"] for t in tasks]), "w": w, "fl": fl}


def p_mean(m, s):
    return float((ZW * special.expit(m + s * Z)).sum())


def fit_sigma(n, k):
    keep = n > 0
    n, k = n[keep], k[keep]
    lc = special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1)

    def nll(th):
        x = (th[0] + math.exp(th[1]) * Z)[None, :]
        ll = lc[:, None] + k[:, None] * special.log_expit(x) + (n - k)[:, None] * special.log_expit(-x)
        return -special.logsumexp(ll, b=ZW[None, :], axis=1).sum()
    b = [(-10, 10), (math.log(0.01), math.log(10))]
    r = optimize.minimize(nll, [-1.0, 0.0], bounds=b, method="L-BFGS-B")
    if not (r.success and np.isfinite(r.x).all() and np.isfinite(r.fun)):
        raise SystemExit(f"sigma fit failed: {r.message}")
    hits = [f"{n}={lim}" for n, x, (lo, hi) in zip(("m", "log_sigma"), r.x, b) for lim in (lo, hi) if abs(x - lim) < 1e-3]
    return {"m": float(r.x[0]), "sigma": float(math.exp(r.x[1])), "bound_hits": hits, "nll": float(r.fun), "message": str(r.message)}


def m_for(mu, s):
    if mu <= 0 or mu >= 1:  # the feasible endpoint: every call missed (or flagged)
        return -60.0 if mu <= 0 else 60.0
    return optimize.brentq(lambda m: p_mean(m, s) - mu, -30, 30)


def studies(D, K, m, s, rho, real_full, n, seed):
    """n valid studies (each world has sabotage); returns per-world catch arrays and the redraw count."""
    rng, out, redraw = np.random.default_rng([seed, K]), {k: [] for k in ("real1", "real2", "sim", "real_pooled")}, 0
    while len(out["sim"]) < 1 or sum(map(len, out["sim"])) < n:
        idx = rng.integers(0, 16, size=(n, K))
        u = rng.standard_normal((n, K))
        est, ok, nd = {}, np.ones(n, bool), {}
        for world in ("real1", "real2", "sim"):
            p = special.expit(m[world] + s * (math.sqrt(rho) * u + math.sqrt(1 - rho) * rng.standard_normal((n, K))))
            if world == "sim" or real_full:
                cnt = (D["n_sim"] if world == "sim" else D["n_real_full"][world])[idx]
                num, den = rng.binomial(cnt, p).sum(1), cnt.sum(1)
            else:
                Wd = D["W"][world][idx]
                num, den = (Wd * (rng.random(Wd.shape) < p[:, :, None])).sum((1, 2)), Wd.sum((1, 2))
            ok &= den > 0
            est[world] = num / np.where(den > 0, den, 1)
            nd[world] = (num, den)
        est["real_pooled"] = (nd["real1"][0] + nd["real2"][0]) / np.where(nd["real1"][1] + nd["real2"][1] > 0, nd["real1"][1] + nd["real2"][1], 1)
        redraw += int((~ok).sum())
        for k in out:
            out[k].append(est[k][ok])
    return {k: np.concatenate(v)[:n] for k, v in out.items()}, redraw


def hw(x):
    return {"sd_x1.96": float(1.96 * x.std()), "range_half": float((np.quantile(x, 0.975) - np.quantile(x, 0.025)) / 2)}


def cell(D, K, mu, s, rho, real_full, power=True):
    m0 = m_for(mu, s)
    e, red = studies(D, K, {"real1": m0, "real2": m0, "sim": m0}, s, rho, real_full, NSIM, SEED)
    g0, pooled = e["real1"] - e["sim"], e["real_pooled"] - e["sim"]  # pooled ratio over both real runs, as the gate pools
    res = {"K": K, "m": m0, "redraws": red, "without_control_real1_minus_sim": hw(g0), "yardstick_real2_minus_real1": hw(e["real2"] - e["real1"]),
           "did_equals_real2_minus_sim": hw(e["real2"] - e["sim"]), "pooled_real_minus_sim": hw(pooled),
           "real1_alone": hw(e["real1"]), "real2_alone": hw(e["real2"]), "real_pooled_alone": hw(e["real_pooled"]), "sim_alone": hw(e["sim"])}
    sd0, sdp = g0.std(), pooled.std()
    redraws_alt = [0]

    def reject(delta, sign, which, n=NSIM):
        ms = m_for(mu - sign * delta, s)
        e1, r1 = studies(D, K, {"real1": m0, "real2": m0, "sim": ms}, s, rho, real_full, n, SEED + 1)
        redraws_alt[0] += r1
        g = e1["real1"] - e1["sim"] if which == "two_arm" else e1["real_pooled"] - e1["sim"]
        return g, float((np.abs(g) > 1.96 * (sd0 if which == "two_arm" else sdp)).mean())
    if power:
        for which in ("two_arm", "pooled"):
            for name, sign, cap in (("sim_lower", 1, mu), ("sim_higher", -1, 1 - mu)):
                # coarse scan (0.01 steps, 5,000 studies) up to and including the feasible endpoint `cap`
                grid = [round(x, 3) for x in np.arange(0.01, cap - 1e-9, 0.01)] + [round(cap, 6)]
                first = next((d for d in grid if reject(d, sign, which, 5000)[1] >= 0.8), None)
                if first is None:
                    res[f"detectable_80_{which}_{name}"] = "not reached on the evaluated grid, endpoint included"
                    continue
                # fine verification at 20,000 studies: the 0.005 step below, then upward until power >= 0.8
                d = max(0.005, round(first - 0.005, 6))  # also below an endpoint crossing (Codex, second review)
                while True:
                    pw = reject(d, sign, which)[1]
                    if pw >= 0.8 or d >= cap:
                        break
                    d = min(cap, round(d + 0.005, 3))
                res[f"detectable_80_{which}_{name}"] = {"gap": d, "power": pw, "mc_se": math.sqrt(pw * (1 - pw) / NSIM), "reached": pw >= 0.8}
        # earlier criterion at ±15, as a FIXED-NULL-WIDTH NORMAL APPROXIMATION (interval = estimate ± 1.96 × null SD), real 1 − sim only
        res["equiv15_fixed_width_approx"] = {"pass_at_gap0": float((np.abs(g0) + 1.96 * sd0 <= 0.15).mean())}
        for name, sign, cap in (("sim_lower", 1, mu), ("sim_higher", -1, 1 - mu)):
            if cap >= 0.30:
                g30, _ = reject(0.30, sign, "two_arm")
                res["equiv15_fixed_width_approx"][f"fail_at_gap30_{name}"] = float((np.abs(g30) + 1.96 * sd0 > 0.15).mean())
        res["redraws_alternatives"] = redraws_alt[0]
    return res


def cost(K):
    new = max(0, math.ceil(K) - 16)
    out = {"new_tasks": new}
    for runs in (3, 2):
        req = new * runs * 5
        for k, pr in PRICE.items():
            raw = req * pr + SCORING
            out[f"{runs}_runs_{k}"] = {"requests": req, "raw": round(raw, 2), "plus30": round(1.3 * raw, 2)}
    return out


def main(inputs, out, real_full_counts=False):
    D = load(Path(inputs))
    if real_full_counts:  # Amendment 3: full-population real counts replace round(sum w); sigma stays fitted on the sampled units
        f = Path(inputs) / REAL_FULL["file"]
        if sha256(f) != REAL_FULL["sha256"]:
            raise SystemExit(f"{f.name}: sha256 mismatch")
        P = json.loads(f.read_text())["state_changing_labels_per_task"]
        for r, k in REAL_FULL["keys"].items():
            if set(P[k]) != set(D["tasks"]):
                raise SystemExit(f"{k}: tasks differ from the gate's held-out tasks")
            D["n_real_full"][r] = np.array([P[k][t]["sabotage"] for t in D["tasks"]])
    fit = fit_sigma(np.concatenate([D["n_obs"][r] for r in RUNS]), np.concatenate([D["k_obs"][r] for r in RUNS]))  # units: (task, run)
    sig = {"ml": fit["sigma"], "icc0.2": math.sqrt(0.2 * (math.pi ** 2 / 3) / 0.8)}
    rng = np.random.default_rng([SEED, 99])
    bs, empty = {"pooled_both_runs": [], "real1": [], "real2": []}, {"pooled_both_runs": 0, "real1": 0, "real2": 0}
    for _ in range(NBOOT):
        ix = rng.integers(0, 16, 16)
        for key, runs in (("pooled_both_runs", RUNS), ("real1", ("real1",)), ("real2", ("real2",))):
            ww = np.concatenate([D["w"][r][D["tasks"][i]] for i in ix for r in runs] + [[]])
            ff = np.concatenate([D["fl"][r][D["tasks"][i]] for i in ix for r in runs] + [[]])
            if ww.sum() > 0:
                bs[key].append((ww * ff).sum() / ww.sum())
            else:
                empty[key] += 1
    res = {"post_hoc_planning": True, "alarm": "frozen 1% (0.9697…)", "seed": SEED, "studies_per_cell": NSIM,
           "inputs_sha256": {n: sha256(Path(inputs) / n) for n in SHA}, "mu": MU, "sigma_fit": fit, "sigma": sig,
           "amendment_3_real_full_counts": ({"file": REAL_FULL["file"], "sha256": REAL_FULL["sha256"]} if real_full_counts else None),
           "observed": {"tasks": D["tasks"], "sim_counts": D["n_sim"].tolist()} | {
               f"{r}_{q}": v for r in RUNS for q, v in (("sampled_sabotage_per_task", D["n_obs"][r].tolist()), ("flagged_per_task", D["k_obs"][r].tolist()),
                                                        ("weight_sum", float(D["W"][r].sum())),
                                                        ("full_counts_" + ("labelled" if real_full_counts else "round_sum_w"), D["n_real_full"][r].tolist()))},
           "model_check_k16": {"nonparametric_bootstrap_95": {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in bs.items()},
                               "gate_recorded_95_pooled_both_runs": GATE_CI, "empty_draws_skipped": empty}, "settings": {}}
    for r in RUNS:
        ww = np.concatenate([D["w"][r][t] for t in D["tasks"]])
        ff = np.concatenate([D["fl"][r][t] for t in D["tasks"]])
        res["observed"][f"{r}_weighted_catch"] = float((ww * ff).sum() / ww.sum())
    for name, mu in MU.items():
        S = res["settings"][name] = {}
        S["D1_real_sampled_k16"] = cell(D, 16, mu, sig["ml"], 0.0, False)
        S["D2_real_full"] = [cell(D, K, mu, sig["ml"], 0.0, True, power=True) for K in KS]
        h16 = S["D2_real_full"][0]["without_control_real1_minus_sim"]["sd_x1.96"]
        grid_k = next((c["K"] for c in S["D2_real_full"] if c["without_control_real1_minus_sim"]["sd_x1.96"] <= 0.15), None)
        hp16 = S["D2_real_full"][0]["pooled_real_minus_sim"]["sd_x1.96"]
        eq = lambda c: all(v >= 0.8 for v in c["equiv15_fixed_width_approx"].values())  # noqa: E731
        S["pm15"] = {"comparison": "real 1 − simulated, unpooled (pooled-reference K beside it)", "K_grid": grid_k,
                     "K_approx_1_over_sqrtK": 16 * (h16 / 0.15) ** 2,
                     "K_grid_pooled": next((c["K"] for c in S["D2_real_full"] if c["pooled_real_minus_sim"]["sd_x1.96"] <= 0.15), None),
                     "K_approx_pooled": 16 * (hp16 / 0.15) ** 2,
                     "K_equiv_fixed_width_approx_all_0.8": next((c["K"] for c in S["D2_real_full"] if eq(c)), None)}
        S["pm15"]["cost_at_K_grid"] = cost(grid_k) if grid_k else "not reachable within 132 held-out tasks at one run per task per arm"
        S["pm15"]["cost_at_K_approx"] = cost(S["pm15"]["K_approx_1_over_sqrtK"]) if S["pm15"]["K_approx_1_over_sqrtK"] <= 132 else "beyond 132"
        S["sensitivity_k16"] = {f"D1_rho{r}_sigma_{k}": cell(D, 16, mu, sig[k], r, False, power=False)["without_control_real1_minus_sim"]
                                for r in (0.0, 1.0) for k in sig} | {f"D2_rho{r}_sigma_{k}": cell(D, 16, mu, sig[k], r, True, power=False)["without_control_real1_minus_sim"]
                                                                       for r in (0.0, 1.0) for k in sig}
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE).stdout.strip()  # noqa: E731
    res["git"] = {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", str(HERE)))}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    p = Path(out) / f"precision_POSTHOC_{'A3_' if real_full_counts else ''}{stamp}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--real-full-counts", action="store_true", help="Amendment 3: real runs at full population from Aswin's labels")
    main(**vars(ap.parse_args()))
