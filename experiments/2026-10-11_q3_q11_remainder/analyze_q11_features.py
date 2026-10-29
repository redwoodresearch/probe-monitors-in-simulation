"""POST HOC (PLAN_FEATURES_POSTHOC.md): how much of the bash per-output score gap (sim − real e1, the L63 probe's "after"
scores) simple structural features explain. Regression (Oaxaca / Gelbach) and coarsened exact matching, task-resampled
intervals, planted and null-plant checks, e2 − e1 control. Local, keys and numbers only.

  uv run --no-project --with numpy python experiments/2026-10-11_q3_q11_remainder/analyze_q11_features.py \\
      --records <records json> --gate <q11_rescore_gate json> --features <features json> --run <q11_features_run json>
"""

import argparse
import hashlib
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
SIM, REAL, E2 = "sim-attack-fixed", "real-attack-e1", "real-attack-e2"
B, SEED, PLANT = 2000, 20261011, 0.10
Q11_SHELL_PER_OUTPUT = 0.1888387412246702  # q11_remainder_POSTHOC_2026-10-11_023527.json, after/by_kind/shell/per_output
COVS = ["log_n_chars", "log_n_lines", "empty", "trailing_newline", "stderr_any", "log_stderr_lines", "err_marker", "blank_share",
        "log_max_line_len", "ansi", "log_ts_tokens", "log_path_tokens", "digit_share", "leading_ws", "trailing_ws_share", "json_like",
        "tab", "non_ascii", "repeated_line_share"]
FORMATTING = {"trailing_newline", "blank_share", "log_max_line_len", "ansi", "leading_ws", "trailing_ws_share", "tab", "non_ascii"}
RAW = ["n_chars", "n_lines", "empty", "trailing_newline", "stderr_lines", "err_marker", "blank_share", "max_line_len", "ansi",
       "ts_tokens", "path_tokens", "digit_share", "leading_ws", "trailing_ws_share", "json_like", "tab", "non_ascii", "repeated_line_share"]


def covariates(f):
    lg = np.log1p
    return {"log_n_chars": lg(f["n_chars"]), "log_n_lines": lg(f["n_lines"]), "empty": float(f["empty"]),
            "trailing_newline": float(f["trailing_newline"]), "stderr_any": float(f["stderr_lines"] > 0),
            "log_stderr_lines": lg(f["stderr_lines"]), "err_marker": float(f["err_marker"]), "blank_share": f["blank_share"],
            "log_max_line_len": lg(f["max_line_len"]), "ansi": float(f["ansi"]), "log_ts_tokens": lg(f["ts_tokens"]),
            "log_path_tokens": lg(f["path_tokens"]), "digit_share": f["digit_share"], "leading_ws": float(f["leading_ws"]),
            "trailing_ws_share": f["trailing_ws_share"], "json_like": float(f["json_like"]), "tab": float(f["tab"]),
            "non_ascii": float(f["non_ascii"]), "repeated_line_share": f["repeated_line_share"]}


def wmean(x, w):
    return float((w * x).sum() / w.sum())


def gelbach(y, sim, X, w):
    """WLS base (1, sim) and full (1, sim, X) models; Gelbach contributions gamma_k * delta_k sum to beta_raw - beta_adj."""
    sw = np.sqrt(w)
    one = np.ones(len(y))
    b0 = np.linalg.lstsq(np.c_[one, sim] * sw[:, None], y * sw, rcond=None)[0]
    b1 = np.linalg.lstsq(np.c_[one, sim, X] * sw[:, None], y * sw, rcond=None)[0]
    delta = np.array([wmean(X[sim == 1, k], w[sim == 1]) - wmean(X[sim == 0, k], w[sim == 0]) for k in range(X.shape[1])])
    contrib = b1[2:] * delta
    return {"beta_raw": float(b0[1]), "beta_adj": float(b1[1]), "contrib": contrib, "delta": delta, "gamma": b1[2:]}


def best_formatting(contrib, names, k=3):
    idx = [i for i, n in enumerate(names) if n in FORMATTING]
    best, best_set = -np.inf, ()
    for r in range(1, k + 1):
        for s in combinations(idx, r):
            v = contrib[list(s)].sum()
            if v > best:
                best, best_set = v, s
    return float(best), [names[i] for i in best_set]


def strata(f, q):
    nl = f["n_lines"]
    lb = 0 if f["empty"] else 1 if nl <= 1 else 2 if nl <= 5 else 3 if nl <= 20 else 4
    bs = 0 if f["blank_share"] == 0 else 1 if f["blank_share"] <= 0.2 else 2
    return (int(np.searchsorted(q, f["n_chars"], side="right")), lb, int(f["empty"]), int(f["trailing_newline"]),
            int(f["stderr_lines"] > 0), int(f["err_marker"]), bs)


def matched_gap(y, sim, s, w):
    """Coarsened exact matching: sim-weighted mean over strata holding both arms of (m_sim(s) - m_real(s)).
    s holds stratum ids 0..S-1."""
    S = int(s.max()) + 1
    a, b = sim == 1, sim == 0
    Wa, Wb = np.bincount(s[a], w[a], S), np.bincount(s[b], w[b], S)
    Ya, Yb = np.bincount(s[a], w[a] * y[a], S), np.bincount(s[b], w[b] * y[b], S)
    both = (Wa > 0) & (Wb > 0)
    cov = Wa[both].sum()
    num = (Wa[both] * (Ya[both] / Wa[both] - Yb[both] / Wb[both])).sum()
    tot = Wa.sum()
    return (float(num / cov) if cov else float("nan")), (float(cov / tot) if tot else float("nan"))


def ci(v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    gate, run = json.loads(Path(args.gate).read_text()), json.loads(Path(args.run).read_text())
    rraw, fraw = Path(args.records).read_bytes(), Path(args.features).read_bytes()
    if not gate["gate"]["passed"] or hashlib.sha256(rraw).hexdigest() != gate["records_local_copy_sha256"]:
        raise SystemExit("records: gate not passed, or not the gated file")
    if hashlib.sha256(fraw).hexdigest() != run["features_local_copy_sha256"]:
        raise SystemExit("features: not the file the run wrote")
    recs, feats = json.loads(rraw), json.loads(fraw)
    tasks = sorted({x["main_task"] for x in recs})
    rng = np.random.default_rng(SEED)
    draws = [np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks)).astype(float) for _ in range(B)]
    n_run = {}
    for x in recs:
        if x["after"] is not None:
            n_run[(x["cell"], x["arm"])] = n_run.get((x["cell"], x["arm"]), 0) + 1
    fk = {}
    for f in feats:
        k = (f["cell"], f["arm"], f["step"], f["action"])
        if k in fk:
            raise SystemExit(f"two feature rows for {k}")
        fk[k] = f
    bash = [x for x in recs if x["kind"] == "shell" and x["after"] is not None]
    missing = [x for x in bash if (x["cell"], x["arm"], x["step"], x["action"]) not in fk]
    if missing:
        raise SystemExit(f"{len(missing)} bash records have no feature row")
    tix = {t: i for i, t in enumerate(tasks)}
    rows = [(x, fk[(x["cell"], x["arm"], x["step"], x["action"])]) for x in bash]
    arm = np.array([x["arm"] for x, _ in rows])
    y = np.array([x["after"] for x, _ in rows])
    w0 = np.array([1.0 / n_run[(x["cell"], x["arm"])] for x, _ in rows])
    ti = np.array([tix[x["main_task"]] for x, _ in rows])
    C = np.array([[covariates(f)[c] for c in COVS] for _, f in rows])
    Fr = {c: np.array([float(f[c]) for _, f in rows]) for c in RAW}

    # gate: (p_sim + p_real)/2 x Delta reproduces q11's shell per_output term
    def p_bash(a):
        tot = {}
        for x in recs:
            if x["arm"] == a and x["after"] is not None:
                r = (x["cell"], x["arm"])
                tot.setdefault(r, [0, 0])
                tot[r][0] += 1
                tot[r][1] += x["kind"] == "shell"
        return float(np.mean([b / n for n, b in tot.values()]))
    m = lambda a: wmean(y[arm == a], w0[arm == a])
    delta_obs = m(SIM) - m(REAL)
    gate_val = (p_bash(SIM) + p_bash(REAL)) / 2 * delta_obs
    gate_ok = abs(gate_val - Q11_SHELL_PER_OUTPUT) <= 1e-9
    if not gate_ok:
        raise SystemExit(f"gate failed: {gate_val} vs {Q11_SHELL_PER_OUTPUT}")

    res = {"records_sha256": gate["records_local_copy_sha256"], "features_sha256": run["features_local_copy_sha256"], "seed": SEED,
           "draws": B, "n_bash": {a: int((arm == a).sum()) for a in (SIM, REAL, E2)},
           "gate": {"value": gate_val, "q11_shell_per_output": Q11_SHELL_PER_OUTPUT, "passed": gate_ok, "join_missing": 0},
           "delta_bash_per_output": delta_obs}

    def pair(a, b, Xc, yv, extra=None):
        sel = (arm == a) | (arm == b)
        s1 = (arm[sel] == a).astype(float)
        X = Xc[sel] if extra is None else np.c_[Xc[sel], extra[sel]]
        return sel, s1, X, yv[sel]

    # (a) feature distributions
    def dist(a, b):
        out = {}
        for c in RAW:
            v = Fr[c]
            def d(wt):
                wa, wb = wt * (arm == a), wt * (arm == b)
                return (wa * v).sum() / wa.sum() - (wb * v).sum() / wb.sum()
            out[c] = {"mean_" + a: wmean(v[arm == a], w0[arm == a]), "mean_" + b: wmean(v[arm == b], w0[arm == b]),
                      "difference": d(w0), "difference_ci95": ci([d(w0 * dr[ti]) for dr in draws])}
        return out
    res["features_sim_vs_real"] = dist(SIM, REAL)
    res["features_e2_vs_e1"] = dist(E2, REAL)

    # (b-i) Gelbach
    def decomp(a, b, yv=y, extra=None, names=COVS):
        sel, s1, X, yy = pair(a, b, C, yv, extra)
        obs = gelbach(yy, s1, X, w0[sel])
        boot = [gelbach(yy, s1, X, w0[sel] * dr[ti[sel]]) for dr in draws]
        out = {"beta_raw": obs["beta_raw"], "beta_raw_ci95": ci([g["beta_raw"] for g in boot]), "beta_adj": obs["beta_adj"],
               "beta_adj_ci95": ci([g["beta_adj"] for g in boot]),
               "share_explained_all": 1 - obs["beta_adj"] / obs["beta_raw"],
               "share_explained_all_ci95": ci([1 - g["beta_adj"] / g["beta_raw"] for g in boot]), "features": {}}
        for k, nme in enumerate(names):
            out["features"][nme] = {"class": "formatting" if nme in FORMATTING else ("plant" if nme == "P" else "content"),
                                    "delta": float(obs["delta"][k]), "gamma": float(obs["gamma"][k]),
                                    "contribution": float(obs["contrib"][k]), "contribution_ci95": ci([g["contrib"][k] for g in boot]),
                                    "share": float(obs["contrib"][k] / obs["beta_raw"]),
                                    "share_ci95": ci([g["contrib"][k] / g["beta_raw"] for g in boot])}
        bv, bs = best_formatting(obs["contrib"], names)
        bb = [best_formatting(g["contrib"], names)[0] / g["beta_raw"] for g in boot]
        out["best_formatting_set"] = {"features": bs, "share": bv / obs["beta_raw"], "share_ci95_reselected_descriptive": ci(bb),
                                      "note": "chosen and evaluated on the same data; descriptive only, the bar uses bar_heldout"}
        return out, obs, boot
    main_dec, obs_main, _ = decomp(SIM, REAL)
    res["gelbach_sim_vs_real"] = main_dec

    # the bar's test: the best <=3 formatting set chosen on one seeded half of the main tasks, its share evaluated on the
    # other half (independent selection and evaluation), with task resamples drawn within the evaluation half only
    perm = np.random.default_rng(SEED + 1).permutation(len(tasks))
    half_a = np.zeros(len(tasks), bool)
    half_a[perm[: len(tasks) // 2]] = True
    sel2 = (arm == SIM) | (arm == REAL)
    s2 = (arm[sel2] == SIM).astype(float)
    in_a = half_a[ti[sel2]]
    pick_obs = gelbach(y[sel2][in_a], s2[in_a], C[sel2][in_a], w0[sel2][in_a])
    _, chosen = best_formatting(pick_obs["contrib"], COVS)
    cidx = [COVS.index(c) for c in chosen]
    yb, sb, Xb, wb, tb = y[sel2][~in_a], s2[~in_a], C[sel2][~in_a], w0[sel2][~in_a], ti[sel2][~in_a]
    ev = gelbach(yb, sb, Xb, wb)
    b_tasks = np.flatnonzero(~half_a)
    erng = np.random.default_rng(SEED + 2)
    ev_boot = []
    for _ in range(B):
        d = np.zeros(len(tasks))
        d[b_tasks] = np.bincount(erng.integers(0, len(b_tasks), len(b_tasks)), minlength=len(b_tasks))
        g = gelbach(yb, sb, Xb, wb * d[tb])
        ev_boot.append(g["contrib"][cidx].sum() / g["beta_raw"])
    res["bar_heldout"] = {"selection_half_tasks": int(half_a.sum()), "evaluation_half_tasks": int((~half_a).sum()),
                          "chosen_on_selection_half": chosen, "share_on_evaluation_half": float(ev["contrib"][cidx].sum() / ev["beta_raw"]),
                          "share_on_evaluation_half_ci95": ci(ev_boot), "evaluation_half_beta_raw": ev["beta_raw"]}
    res["gelbach_e2_vs_e1"] = decomp(E2, REAL)[0]

    # (b-ii) matching
    sel = (arm == SIM) | (arm == REAL)
    q = np.quantile(Fr["n_chars"][sel], [0.2, 0.4, 0.6, 0.8])
    keys = [strata(f, q) for _, f in rows]
    sid = {k: i for i, k in enumerate(sorted(set(keys)))}  # stratum ids from sorted tuples, no hash()
    s_all = np.array([sid[k] for k in keys])
    s1 = (arm[sel] == SIM).astype(float)
    mg, cov = matched_gap(y[sel], s1, s_all[sel], w0[sel])
    mb = [matched_gap(y[sel], s1, s_all[sel], w0[sel] * dr[ti[sel]]) for dr in draws]
    raw_b = [wmean(y[sel][s1 == 1], (w0[sel] * dr[ti[sel]])[s1 == 1]) - wmean(y[sel][s1 == 0], (w0[sel] * dr[ti[sel]])[s1 == 0]) for dr in draws]
    res["matching"] = {"strata_quintile_edges_n_chars": [float(x) for x in q], "matched_gap": mg, "matched_gap_ci95": ci([x[0] for x in mb]),
                       "coverage_sim_weight": cov, "coverage_ci95": ci([x[1] for x in mb]),
                       "share_explained": 1 - mg / delta_obs, "share_explained_ci95": ci([1 - x[0] / r for x, r in zip(mb, raw_b)])}

    # planted checks
    prng = np.random.default_rng(SEED)
    pick = prng.random(len(y)) < 0.5
    P = ((arm == SIM) & pick).astype(float)
    y_p = y + PLANT * P
    sel_p = (arm == SIM) | (arm == REAL)
    added = PLANT * wmean(P[sel_p & (arm == SIM)], w0[sel_p & (arm == SIM)])
    pdec, pobs, _ = decomp(SIM, REAL, yv=y_p, extra=P, names=COVS + ["P"])
    others_moved = float(np.max(np.abs(pobs["contrib"][:-1] - obs_main["contrib"])))
    pc = pdec["features"]["P"]
    res["planted"] = {"expected_added_gap": added, "beta_raw_rise": pobs["beta_raw"] - obs_main["beta_raw"], "P_contribution": pc["contribution"],
                      "P_contribution_ci95": pc["contribution_ci95"], "max_other_contribution_move": others_moved,
                      "passed": bool(abs((pobs["beta_raw"] - obs_main["beta_raw"]) - added) <= 1e-9
                                     and pc["contribution_ci95"][0] <= added <= pc["contribution_ci95"][1] and pc["contribution_ci95"][0] > 0
                                     and others_moved < 0.005)}
    Pn = (prng.random(len(y)) < 0.5).astype(float)
    ndec, _, _ = decomp(SIM, REAL, extra=Pn, names=COVS + ["P"])
    npc = ndec["features"]["P"]
    res["null_plant"] = {"P_contribution": npc["contribution"], "P_contribution_ci95": npc["contribution_ci95"],
                         "passed": bool(abs(npc["contribution"]) <= 0.005 and npc["contribution_ci95"][0] <= 0 <= npc["contribution_ci95"][1])}

    # reading
    bf, alls, hb = main_dec["best_formatting_set"], main_dec, res["bar_heldout"]
    if not res["planted"]["passed"]:
        reading = "decomposition not trusted (synthetic plant failed)"
    elif hb["share_on_evaluation_half"] >= 0.5 and hb["share_on_evaluation_half_ci95"][0] >= 0.3:
        reading = "points to a procedure"
    elif alls["share_explained_all"] >= 0.5 and alls["share_explained_all_ci95"][0] >= 0.3:
        reading = "explained by content-like structure, not a formatting rule"
    else:
        reading = "no simple structural feature explains it"
    res["reading"] = reading
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"q11_features_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(res, indent=1))
    top = sorted(main_dec["features"].items(), key=lambda kv: -abs(kv[1]["contribution"]))[:8]
    print(json.dumps({"gate": res["gate"], "delta": delta_obs, "share_all": [main_dec["share_explained_all"], main_dec["share_explained_all_ci95"]],
                      "best_formatting_full_data": bf, "bar_heldout": hb, "matching": {k: res["matching"][k] for k in ("matched_gap", "coverage_sim_weight", "share_explained", "share_explained_ci95")},
                      "top": {k: [round(v["share"], 3), [round(z, 3) for z in v["share_ci95"]], v["class"]] for k, v in top},
                      "planted": res["planted"], "null_plant": res["null_plant"], "reading": reading}, indent=1, default=float))
    print(f"q11_features_POSTHOC_{stamp}.json")


if __name__ == "__main__":
    main()
