"""Power and price for the changed-reply swap (PLAN.md, committed before this script runs), $0, numbers only.

Inputs, pinned: the k = 1 stage's per-site scores and sites on the Volume (sha256 below), step B's level-1 pack positions
(keys, tools and token positions only), the k = 1 records manifest and cost model (@ 796d784d, c097d16d), k = 1's billed
Modal cost ($12.2068 for 920 reads, SPEND @ 29312475), and H1's test pair counts and dev pilot noise (@ f2ba9ac3).

Model, an approximation of the plan's 2,000-draw main-task bootstrap (checked on k = 1's own data and allocation below):
each differing site's linearised residual e = (d - R g) / mean(g) is task effect + run effect + noise, with variance and
main-task intraclass correlation measured on k = 1's differing sites (per filter); the run share is 0 or 0.25 (k = 1 had
one site per run, so it can't be measured). Positions are walked in the plan's frozen order, each differing with k = 1's
rate, until N differ or the supply ends (then the shorter prefix is read under the same rule). R-hat = R + mean(e), SE
cluster-robust by main task, normal 95% interval.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy python \\
      experiments/2026-10-11_claimB_swap_changed/analyze_POWER_swap_changed.py
"""

import hashlib
import json
import random
import subprocess
from datetime import datetime
from math import erf, sqrt
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np

HERE = Path(__file__).parent
VOL = {"scores": ("claimB/k1_v1/scores_4be7958c7157.json", "b9e4eeac0d82ffef762fdf83d4dfbf7ff6242607b983abbfd11da03fd207a8af"),
       "sites": ("claimB/k1_v1/sites.json", "58fb78528074262e767631c47d15a7e51ce8c9c9a47b56a23bf15954096f9081"),
       "runs": ("claimB/level1/packedB_2cfd1aec393c/runs.L63.json", "b016c2af44d39e92b4344ac20d2d329cb8e388289aecede8a76d195149972e07")}
KR = "experiments/2026-10-11_claimB_k_replies/results/"
H1 = "experiments/2026-10-11_claimB_h1_text_vs_layout/results/"
GIT = {"records": ("796d784d", KR + "k1_records_2026-10-11_165639.json"), "k8": ("c097d16d", KR + "power_k8_2026-10-11_180315.json"),
       "h1cells": ("f2ba9ac3", H1 + "h1_test_swap_cells_2026-10-11_143409.json"), "h1pilot": ("f2ba9ac3", H1 + "h1_swap_pilot_2026-10-11_113444.json")}
TEST = ["clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"]
USD_READ, FIXED_READS, IN_FLIGHT = 12.2068 / 920, 120, 7  # billed $/read (start-up, scorer included); planted + gate; calls past the prefix
BASH_FAMILY = lambda tool: tool != "db_bash" and not tool.startswith("text_editor")  # noqa: E731  (filter F1)
SIZES, TRUE_R, REPS, SEED = [202, 300, 400, 600, 800, 1000, 1400, 1800], [0.2, 0.3, 0.4, 0.53, 0.6, 0.7, 0.8], 2000, 20261011
PHI = lambda x: 0.5 * (1 + erf(x / sqrt(2)))  # noqa: E731


def fit(rows):
    """R, residual variance, main-task ICC (one-way ANOVA) and cluster-robust SE of the ratio over k = 1 differing rows."""
    t = np.array([r["main_task"] for r in rows])
    d = np.array([r["63"]["sim"] - r["63"]["self"] for r in rows])
    g = np.array([r["63"]["stored_fixed_sim"] - r["63"]["stored_e1"] for r in rows])
    R = d.sum() / g.sum()
    e = (d - R * g) / g.mean()
    tk = np.unique(t, return_inverse=True)[1]
    k, n = tk.max() + 1, len(e)
    m, s = np.bincount(tk), np.bincount(tk, e)
    msb = (m * (s / m - e.mean()) ** 2).sum() / (k - 1)
    msw = ((e - (s / m)[tk]) ** 2).sum() / (n - k)
    n0 = (n - (m ** 2).sum() / n) / (k - 1)
    icc = max(0.0, (msb - msw) / (msb + (n0 - 1) * msw))
    se = sqrt(k / (k - 1) * ((s - m * e.mean()) ** 2).sum()) / n
    return {"sites": n, "main_tasks": int(k), "R1c": float(R), "var": float(e.var(ddof=1)), "icc": float(icc), "cluster_robust_se": se}, (t, d, g)


def main():
    vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
    data = {}
    for k, (p, sha) in VOL.items():
        raw = b"".join(vol.read_file(p))
        assert hashlib.sha256(raw).hexdigest() == sha, k
        data[k] = json.loads(raw)
    for k, (c, p) in GIT.items():
        data[k] = json.loads(subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True, cwd=HERE).stdout)
    sites = data["scores"]["sites"]
    keep = {"none": lambda tool: True, "F1": BASH_FAMILY}
    ch = {f: [r for r in sites if not r["sim_text_equals_self_text"] and keep[f](r["tool"])] for f in keep}
    p = {f: len(ch[f]) / sum(1 for r in sites if keep[f](r["tool"])) for f in keep}
    model = {f: fit(ch[f]) for f in keep}
    out = {"label": "changed-reply swap: power and price ($0, numbers only; PLAN.md)", "differing_rate": p,
           "model": {f: model[f][0] for f in keep}, "F1_drops_differing_k1_sites": len(ch["none"]) - len(ch["F1"])}
    # the plan's registered bootstrap on k = 1's differing sites, beside the normal interval the power model uses
    t, d, g = model["none"][1]
    rng = np.random.default_rng(SEED)
    tk = np.unique(t, return_inverse=True)[1]
    W = rng.multinomial(tk.max() + 1, np.full(tk.max() + 1, 1 / (tk.max() + 1)), size=2000)
    num, den = W @ np.bincount(tk, d), W @ np.bincount(tk, g)
    R0, se0 = model["none"][0]["R1c"], model["none"][0]["cluster_robust_se"]
    out["bootstrap_vs_normal_on_k1"] = {"bootstrap_ci95": np.percentile(num[den > 0] / den[den > 0], [2.5, 97.5]).tolist(),
                                       "draws_mean_G_le_0": int((den <= 0).sum()), "normal_ci95": [R0 - 1.96 * se0, R0 + 1.96 * se0]}
    # supply: eligible new positions (k = 1's rule minus its 400 sites); the frozen order; the history proxy for price
    rec = {m["site"]: m for m in data["records"]["manifest"]}
    offset = float(np.mean([rec[s["site"]]["k"] - s["earlier_outputs"] for s in data["sites"]["sites"]]))
    used = {(s["cell"], s["step"], s["action"]) for s in data["sites"]["sites"]}
    by_run = {(r["cell"], r["arm"]): r for r in data["runs"] if r["environment"] in TEST}
    elig = {}
    for (cell, arm), r in by_run.items():
        if arm != "real-attack-e1" or (cell, "sim-attack-fixed") not in by_run:
            continue
        sim_kept = {(q["step"], q["action"]) for q in by_run[(cell, "sim-attack-fixed")]["positions"] if q["kind"] == "output_end" and q["kept"]}
        outs = sorted((q["step"], q["action"], q["token"], q["tool"]) for q in r["positions"] if q["kind"] == "output_end" and q["kept"])
        ok = [(i, tool) for i, (s, a, tok, tool) in enumerate(outs) if i >= 7 and tok < 32768 - 6000 and (s, a) in sim_kept and (cell, s, a) not in used]
        if ok:
            elig[cell] = (r["main_task"], ok)
    rng_s = random.Random("swapchg-order")
    runs_o = [(c, rng_s.sample(elig[c][1], len(elig[c][1]))) for c in rng_s.sample(sorted(elig), len(elig))]
    full = [(c, elig[c][0], ix[j]) for j in range(max(len(ix) for _, ix in runs_o)) for c, ix in runs_o if len(ix) > j]
    orders = {f: [(c, tsk, pos[0]) for c, tsk, pos in full if keep[f](pos[1])] for f in keep}  # F1 skips within the frozen order
    pm = data["k8"]["price_model"]
    usd_call = lambda i: pm["usd_per_call_intercept"] + pm["usd_per_call_per_history_action"] * (i + offset)  # noqa: E731
    out["supply"] = {"positions": {f: len(o) for f, o in orders.items()}, "runs": len(elig), "main_tasks": len({v[0] for v in elig.values()}),
                     "round1_positions": len(runs_o), "history_offset_k_minus_index": offset, "k1_measured_usd_per_call": 220.4703 / 400}

    def simulate(order, mdl, p_ch, n, rho_run, costs):
        """Per replicate: sites read, mean e, SE, calls in the prefix, prefix cost, cost spent on non-differing calls."""
        tix = {x: i for i, x in enumerate(sorted({o[1] for o in order}))}
        rix = {x: i for i, x in enumerate(sorted({o[0] for o in order}))}
        ta, ra = np.array([tix[o[1]] for o in order]), np.array([rix[o[0]] for o in order])
        icc, var = mdl["icc"], mdl["var"]
        res = []
        for _ in range(REPS):
            diff = rng.random(len(order)) < p_ch
            sel = np.flatnonzero(diff)[:n]
            calls = int(sel[-1]) + 1 if len(sel) == n else len(order)
            ee = (rng.normal(0, sqrt(icc * var), len(tix))[ta[sel]] + rng.normal(0, sqrt(rho_run * var), len(rix))[ra[sel]]
                  + rng.normal(0, sqrt(max(0.0, 1 - icc - rho_run) * var), len(sel)))
            st, nt = np.bincount(ta[sel], ee, len(tix)), np.bincount(ta[sel], minlength=len(tix))
            T = (nt > 0).sum()
            se = sqrt(T / (T - 1) * ((st - nt * ee.mean()) ** 2).sum()) / len(sel) if T > 1 else float("inf")
            res.append((len(sel), float(ee.mean()), se, calls, float(costs[:calls].sum()), float(costs[:calls][~diff[:calls]].sum())))
        return res

    # model check on k = 1's own allocation: its 202 differing sites' main tasks, one site per run
    t0 = model["none"][1][0]
    chk = simulate([(f"run{i}", tt, 0) for i, tt in enumerate(t0)], model["none"][0], 1.0, len(t0), 0.0, np.zeros(len(t0)))
    out["model_check"] = {"mean_simulated_se_on_k1_allocation": float(np.mean([x[2] for x in chk])), "k1_cluster_robust_se": se0}
    power, price = {}, {}
    for f in keep:
        costs = np.array([usd_call(o[2]) for o in orders[f]])
        for n in SIZES:
            for rho in (0.0, 0.25):
                res = simulate(orders[f], model[f][0], p[f], n, rho, costs)
                R_base = model[f][0]["R1c"]
                cell = {"short_of_n_share": float(np.mean([x[0] < n for x in res])), "mean_sites": float(np.mean([x[0] for x in res])),
                        "mean_se": float(np.mean([x[2] for x in res])), "fit_R1c": R_base}
                for Rt in TRUE_R:
                    lo = np.array([Rt + x[1] - 1.96 * x[2] for x in res])
                    hi = np.array([Rt + x[1] + 1.96 * x[2] for x in res])
                    cell[f"R={Rt}"] = {"at_least_half": float((lo >= 0.5).mean()), "less_than_half": float((hi < 0.5).mean()),
                                       "clear": float(((lo >= 0.5) | (hi < 0.5)).mean())}
                power[f"filter={f}|n={n}|rho_run={rho}"] = cell
                if rho == 0.0:
                    calls, sim, waste = (np.array([x[k] for x in res]) for k in (3, 4, 5))
                    sim = sim + np.minimum(IN_FLIGHT, len(costs) - calls) * costs.mean()  # in-flight calls past the prefix, finished and paid
                    reads = (2 * np.array([x[0] for x in res]) + FIXED_READS) * USD_READ
                    tot = sim + reads
                    price[f"filter={f}|n={n}"] = {
                        "calls_mean": float(calls.mean()), "calls_p90": float(np.percentile(calls, 90)), "sim_usd_mean": float(sim.mean()),
                        "nondiffering_usd_mean": float(waste.mean()), "reads_usd_mean": float(reads.mean()), "raw_usd_mean": float(tot.mean()),
                        "raw_usd_p90": float(np.percentile(tot, 90)), "with_30pct_usd_mean": float(1.3 * tot.mean()),
                        "with_30pct_usd_p90": float(1.3 * np.percentile(tot, 90))}
    out["power"], out["price"] = power, price
    # reverse option (a real reply into a simulated history at identical bash calls, H1's matching): HYPOTHETICAL scenarios
    c1, pl = data["h1cells"]["cells"], data["h1pilot"]
    pairs, gg, se_g, tasks_r = (c1["all_sim_e1_pairs"][k] for k in ("pairs", "mean", "se", "main_tasks"))
    rev = {"pairs": pairs, "main_tasks": tasks_r, "G": gg, "G_se": se_g,
           "assumptions": "SD(T) = sd_u*sqrt(2) needs equal T and R variances and zero covariance (neither measured), or SD(T) = the "
                          "natural SD; G's SE treated as independent of T (they share scores); per-task sizes unknown, so the "
                          "design effect uses mean pairs per task (a lower bound on sum m^2 / sum m) and twice it (a sensitivity, not a bound)"}
    for name, sd_t in (("sd_u_x_sqrt2", pl["sd_u"] * sqrt(2)), ("sd_natural_d", pl["sd_natural_d"])):
        for icc_r, m_eff in ((0.0, 1.0), (0.2, pairs / tasks_r), (0.2, 2 * pairs / tasks_r)):
            de = 1 + icc_r * (m_eff - 1)
            for N in (0.2, 0.5, 0.8):
                se = sqrt(sd_t ** 2 * de / pairs / gg ** 2 + (N * se_g / gg) ** 2)
                rev[f"{name}|icc={icc_r}|m_eff={m_eff:.2f}|N={N}"] = {"se_N": se, "clear": PHI((N - 0.5) / se - 1.96) + PHI((0.5 - N) / se - 1.96)}
    n_reads = pairs + c1["pair_counts"]["e2|e1"] + FIXED_READS
    rev["price"] = {"reads": n_reads, "raw_usd": n_reads * USD_READ + 0.5, "with_30pct_usd": 1.3 * (n_reads * USD_READ + 0.5)}
    out["reverse_option"] = rev
    out |= {"inputs": {k: v[1] for k, v in VOL.items()} | {k: f"{c}:{p}" for k, (c, p) in GIT.items()},
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"power_swap_changed_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("differing_rate", "model", "F1_drops_differing_k1_sites", "bootstrap_vs_normal_on_k1", "supply", "model_check")}, indent=1))
    for k, v in power.items():
        print(k, {a: (round(b["clear"], 2) if isinstance(b, dict) else round(b, 3)) for a, b in v.items()})
    for k, v in price.items():
        print(k, {a: round(b, 1) for a, b in v.items()})
    print("reverse", {k: v for k, v in rev.items() if k != "assumptions"})
    print("->", dst)


if __name__ == "__main__":
    main()
