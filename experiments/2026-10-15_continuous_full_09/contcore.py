"""Shared continuous-score core (this folder's RULE.md): Append 19's family formula on per-draw estimate matrices, the
per-probe standardized shift D, the first-order KDE mapping M = 100 f(T') D, the nonlinear smoothed-CDF diagnostic, the
bandwidth sensitivity, the threshold location, the quantile profile, and the synthetic self-tests. numpy only."""

import math

import numpy as np

_erf = np.frompyfunc(math.erf, 1, 1)


class CoreError(Exception):
    pass


def phi(z):
    return np.exp(-0.5 * np.asarray(z, float) ** 2) / math.sqrt(2 * math.pi)


def Phi(z):
    return 0.5 * (1.0 + _erf(np.asarray(z, float) / math.sqrt(2)).astype(float))


def pct(v, q):
    f = v[np.isfinite(v)]
    return float(np.quantile(f, q)) if len(f) else None


def ci95(v):
    lo, hi = pct(v, 0.025), pct(v, 0.975)
    return [lo, hi] if lo is not None else None


def width(c):
    return None if not c else c[1] - c[0]


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


def cell(s1, s2, ss, hit1, hit2, hits, Tp, W):
    """One probe x stratum: per-task standardized scores (run1, run2, sim) and 0/1 hits over the stratum's tasks, the
    standardized threshold T', and the multiplicity matrix W (draws+1 x tasks). Returns per-draw arrays (fractions for the
    binary gap; SD units for D; points for M and N) and draw-0 descriptives."""
    real0 = np.concatenate([s1, s2])
    h = bandwidth(real0)
    out = {
        "b": wmean(W, hits - (hit1 + hit2) / 2),
        "br": wmean(W, hit2 - hit1),
        "d": wmean(W, ss - (s1 + s2) / 2),
        "dr": wmean(W, s2 - s1),
        "h": h,
    }

    def dens(hh):
        if not (math.isfinite(hh) and hh > 0):
            return np.full(W.shape[0], np.nan)
        return wmean(W, (phi((Tp - s1) / hh) + phi((Tp - s2) / hh)) / (2 * hh))

    out["f"] = dens(h)
    out["m"] = 100 * out["f"] * out["d"]
    out["mr"] = 100 * out["f"] * out["dr"]
    out["m_h_half"] = 100 * dens(h / 2) * out["d"]
    out["m_h_double"] = 100 * dens(2 * h) * out["d"]
    if math.isfinite(h) and h > 0:
        # nonlinear diagnostic: 100 [F_h(T') - F_h(T' - D)], kernel-smoothed real CDF with the draw's weights
        n_d = np.full(W.shape[0], np.nan)
        wsum = W.sum(1)
        ok = wsum > 0
        F_T = (W[ok] @ ((Phi((Tp - s1) / h) + Phi((Tp - s2) / h)) / 2)) / wsum[ok]
        dd = out["d"][ok]
        # F_h(T' - D_d) per draw: evaluate the per-task kernel CDF at T' - D_d (vectorized over draws x tasks)
        Fs = (Phi((Tp - dd[:, None] - s1[None, :]) / h) + Phi((Tp - dd[:, None] - s2[None, :]) / h)) / 2
        n_d[ok] = 100 * (F_T - (W[ok] * Fs).sum(1) / wsum[ok])
        out["n"] = n_d
    else:
        out["n"] = np.full(W.shape[0], np.nan)
    F = float((real0 <= Tp).mean()) if len(real0) else None
    prof = []
    if F is not None and len(ss):
        for dq in (-0.10, -0.05, 0.0, 0.05, 0.10):
            req = F + dq
            q = min(0.99, max(0.01, req))
            prof.append(
                {
                    "requested": req,
                    "used": q,
                    "clipped": q != req,
                    "Q_sim_minus_Q_real": float(np.quantile(ss, q) - np.quantile(real0, q)),
                }
            )
    out["desc"] = {
        "bandwidth": h if math.isfinite(h) else None,
        "bandwidth_note": None if math.isfinite(h) else "nonpositive or nonfinite: mapped values unavailable",
        "F_real_at_T": F,
        "tasks_with_real_score_within_h_of_T": (
            int(((np.abs(s1 - Tp) <= h) | (np.abs(s2 - Tp) <= h)).sum()) if math.isfinite(h) else None
        ),
        "quantile_profile": prof,
    }
    return out


def summ(v, k=1.0):
    c = ci95(v[1:])
    fin = int(np.isfinite(v[1:]).sum())
    return {
        "point": k * float(v[0]) if np.isfinite(v[0]) else None,
        "ci95": [k * c[0], k * c[1]] if c else None,
        "finite_draws": fin,
        "nan_draws": len(v) - 1 - fin,
    }


def discrepancy(m, b):
    """Mapped minus binary (points): draw 0 and the paired percentile interval over jointly finite bootstrap draws."""
    both = np.isfinite(m[1:]) & np.isfinite(b[1:])
    dci = ci95((m[1:] - 100 * b[1:])[both]) if both.any() else None
    return {
        "draw0_points": float(m[0] - 100 * b[0]) if np.isfinite(m[0]) and np.isfinite(b[0]) else None,
        "ci95_points": dci,
        "jointly_finite_draws": int(both.sum()),
        "flag": (
            "first-order KDE mapping disagrees with the binary estimate"
            if dci and (dci[0] > 0 or dci[1] < 0)
            else ("disagreement not detected" if dci else "unavailable")
        ),
    }


def wratio(m, b):
    cb, cm = summ(b, 100)["ci95"], summ(m)["ci95"]
    return width(cm) / width(cb) if cb and cm and width(cb) and width(cb) > 0 else None


def probe_summary(c):
    wr = wratio(c["m"], c["b"])
    return {
        "binary_delta_points": summ(c["b"], 100),
        "D_std_units": summ(c["d"]),
        "density_at_T_std": summ(c["f"]),
        "mapped_points": summ(c["m"]),
        "nonlinear_mapped_points": summ(c["n"]),
        "mapped_points_h_half": summ(c["m_h_half"]),
        "mapped_points_h_double": summ(c["m_h_double"]),
        "reexecution_binary_points": summ(c["br"], 100),
        "reexecution_D_std_units": summ(c["dr"]),
        "reexecution_mapped_points": summ(c["mr"]),
        "width_ratio_mapped_over_binary": wr,
        "width_ratio_note": None
        if wr is not None
        else "undefined (zero or missing binary interval width, or no mapped interval)",
        "mapping_discrepancy": discrepancy(c["m"], c["b"]),
        "reexecution_width_ratio_mapped_over_binary": wratio(c["mr"], c["br"]),
        "reexecution_mapping_discrepancy": discrepancy(c["mr"], c["br"]),
        "descriptives": c["desc"],
    }


def family_block(cells, names):
    """Family summaries over a list of probe cells (same stratum, same draws)."""
    B = np.stack([c["b"] for c in cells], 1)
    out = {
        "family_binary_points": scaled(fam(B), 100),
        "family_continuous_std_units": fam(np.stack([c["d"] for c in cells], 1)),
        "family_mapped_points": fam(np.stack([c["m"] for c in cells], 1)),
        "family_reexecution_continuous_std_units": fam(np.stack([c["dr"] for c in cells], 1)),
        "family_reexecution_mapped_points": fam(np.stack([c["mr"] for c in cells], 1)),
        "family_reexecution_binary_points": scaled(fam(np.stack([c["br"] for c in cells], 1)), 100),
    }
    per = {n: probe_summary(c) for n, c in zip(names, cells, strict=True)}
    ratios = [per[n]["width_ratio_mapped_over_binary"] for n in names]
    out["per_probe"] = per
    out["width_ratio_primary_mean_over_eligible"] = (
        float(np.mean(ratios)) if ratios and all(r is not None for r in ratios) else None
    )
    out["width_ratio_primary_note"] = (
        None
        if out["width_ratio_primary_mean_over_eligible"] is not None
        else "unavailable: a constituent ratio is undefined"
    )
    fb, fm = out["family_binary_points"], out["family_mapped_points"]
    wb, wm = width(fb.get("bias_ci95")), width(fm.get("bias_ci95"))
    out["bias_ci_width_ratio"] = wm / wb if wb and wm is not None and wb > 0 else None
    ub, um = fb.get("U_approx_upper_95"), fm.get("U_approx_upper_95")
    out["U_ratio_mapped_over_binary"] = um / ub if ub and um is not None else None
    return out


def selftest():
    """Synthetic fixtures (RULE.md gate 4), run before any study-data continuous output; raises CoreError on failure."""
    res = {}
    rng = np.random.default_rng(7)
    # 1. replication vs multiplicity weighting
    v = rng.normal(size=6)
    w = np.array([2, 0, 1, 3, 0, 1], float)
    rep = np.repeat(v, w.astype(int))
    got = wmean(np.vstack([np.ones(6), w]), v)
    res["replication_vs_weights"] = bool(abs(got[1] - rep.mean()) < 1e-12 and abs(got[0] - v.mean()) < 1e-12)
    # 2. direct Gaussian kernel vs hand value
    s1, s2, Tp, h = np.array([0.0]), np.array([1.0]), 0.5, 0.5
    hand = (math.exp(-0.5) / math.sqrt(2 * math.pi) * 2) / (2 * h)
    W1 = np.ones((1, 1))
    kern = wmean(W1, (phi((Tp - s1) / h) + phi((Tp - s2) / h)) / (2 * h))[0]
    res["kernel_hand_value"] = bool(abs(kern - hand) < 1e-12)
    # 3. transformation invariance of hits
    z = rng.normal(size=50)
    T = 0.7
    mu, sd = 0.3, 1.9
    res["hit_invariance"] = bool((((z - mu) / sd) > ((T - mu) / sd)).tolist() == (z > T).tolist())
    # 4. percentage conversion exactly once, analytic: real scores at -1 and +1 (N tasks per arm), T' = 0, sim = 0.2, so
    # D = 0.2 exactly and f(0) = phi(1/h)/h for the draw-0 Silverman h; M must equal 100 f D to 1e-12
    N = 50
    s1f, s2f, ssf = np.full(N, -1.0), np.full(N, 1.0), np.full(N, 0.2)
    hf = bandwidth(np.concatenate([s1f, s2f]))
    c = cell(s1f, s2f, ssf, np.zeros(N), np.zeros(N), np.zeros(N), 0.0, np.ones((1, N)))
    exact = 100 * (math.exp(-0.5 / hf**2) / math.sqrt(2 * math.pi)) / hf * 0.2
    res["percent_once_exact"] = bool(
        math.isfinite(hf) and abs(c["d"][0] - 0.2) < 1e-15 and abs(c["m"][0] - exact) < 1e-12
    )
    # 4b. zero-SD pool -> stop
    try:
        pool_moments(np.ones(5))
        res["zero_sd_stops"] = False
    except CoreError:
        res["zero_sd_stops"] = True
    # 5. zero-bandwidth -> unavailable (nan), no fallback
    cz = cell(np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), 0.0, np.ones((2, 5)))
    res["zero_bandwidth_unavailable"] = bool(np.isnan(cz["m"]).all() and cz["desc"]["bandwidth"] is None)
    # 6. empty draw -> nan, counted as incomplete by fam
    We = np.vstack([np.ones(3), np.zeros(3), np.ones(3), np.ones(3), np.ones(3)])
    e = np.stack([wmean(We, np.array([0.1, 0.2, 0.3])), wmean(We, np.array([0.0, 0.1, 0.5]))], 1)
    fe = fam(e)
    res["empty_draw_counted"] = bool(np.isnan(e[1]).all() and fe["complete_draws"] == 3)
    # 7. single probe -> family unavailable
    res["single_probe_unavailable"] = fam(e[:, :1])["X_err"] is None
    # 8. zero binary width -> ratio undefined
    zc = {k: np.zeros(4) for k in ("b", "br", "d", "dr", "f", "m", "mr", "m_h_half", "m_h_double", "n")}
    zc["desc"] = {}
    res["zero_width_ratio_undefined"] = probe_summary(zc)["width_ratio_mapped_over_binary"] is None
    if not all(res.values()):
        raise CoreError(f"selftest failed: {[k for k, v in res.items() if not v]}")
    return res
