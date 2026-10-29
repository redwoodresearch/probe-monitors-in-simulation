"""Paired-binary statistics for the first-cut paired table (RULE.md). Pure functions, numpy/scipy only.

Δ = p12 − p21, where x12 = pairs caught in the first arm only and x21 = in the second only, n = all pairs.
- tango_ci: the Tango (1998) score interval for a paired difference of proportions. Under Δ0 the restricted MLE of p21
  solves 2n·a² − B·a − C = 0 with B = x12(1 − Δ0) + x21(1 − 3Δ0) − 2·x11·Δ0 and C = x21·Δ0(1 − Δ0); the statistic is
  Z(Δ0) = (x12 − x21 − nΔ0) / sqrt(n(2a + Δ0 − Δ0²)), and the interval is {Δ0 : |Z(Δ0)| ≤ z}, found by bisection.
- mcnemar_exact: two-sided binomial test of x12 against x12 + x21 at 1/2.
- tost_label: "within X" if the 90% interval lies inside (−X, X); "mismatch" if not and the 95% interval excludes 0;
  else "inconclusive".
"""

import math

from scipy import stats


def _z(d0: float, x12: int, x21: int, n: int) -> float:
    x11 = n - x12 - x21
    b = x12 * (1 - d0) + x21 * (1 - 3 * d0) - 2 * x11 * d0
    c = x21 * d0 * (1 - d0)
    a = (b + math.sqrt(max(b * b + 8 * n * c, 0.0))) / (4 * n)
    var = n * (2 * a + d0 - d0 * d0)
    num = x12 - x21 - n * d0
    if var <= 0:
        return 0.0 if num == 0 else math.copysign(math.inf, num)
    return num / math.sqrt(var)


def tango_ci(x12: int, x21: int, n: int, level: float) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("no pairs")
    z = stats.norm.ppf(1 - (1 - level) / 2)
    est = (x12 - x21) / n

    def root(lo, hi, target):  # Z is decreasing in Δ0; find Δ0 in [lo, hi] with Z(Δ0) = target
        for _ in range(200):
            mid = (lo + hi) / 2
            if _z(mid, x12, x21, n) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    eps = 1e-12
    lower = -1.0 if _z(-1 + eps, x12, x21, n) <= z else root(-1 + eps, est, z)
    upper = 1.0 if _z(1 - eps, x12, x21, n) >= -z else root(est, 1 - eps, -z)
    return lower, upper


def mcnemar_exact(x12: int, x21: int) -> float:
    m = x12 + x21
    return 1.0 if m == 0 else float(min(1.0, 2 * stats.binom.cdf(min(x12, x21), m, 0.5)))


def tost_label(ci90: tuple[float, float], ci95: tuple[float, float], margin: float) -> str:
    if -margin < ci90[0] and ci90[1] < margin:
        return f"within {round(margin * 100):d}"
    if ci95[0] > 0 or ci95[1] < 0:
        return "mismatch"
    return "inconclusive"


def paired_t(diffs) -> dict:
    import numpy as np

    d = np.asarray(diffs, float)
    n = len(d)
    if n < 2 or not np.isfinite(d).all():
        raise ValueError("paired t needs at least 2 finite differences")
    m, s = float(d.mean()), float(d.std(ddof=1))
    se = s / math.sqrt(n)
    if se == 0:  # zero variance: no spread to test against
        return {"n": n, "mean": m, "ci95": [m, m], "t": 0.0 if m == 0 else math.copysign(math.inf, m), "p": 1.0 if m == 0 else 0.0}
    t = m / se
    h = stats.t.ppf(0.975, n - 1) * se
    return {"n": n, "mean": m, "ci95": [m - h, m + h], "t": t, "p": float(2 * stats.t.sf(abs(t), n - 1))}


def selftest() -> dict:
    """Properties a correct implementation has: the interval holds the estimate, swapping arms mirrors it, a large
    balanced sample is close to the Wald interval, and no discordant pairs gives an interval around 0."""
    out = {}
    lo, hi = tango_ci(30, 10, 200, 0.95)
    out["holds_estimate"] = lo < 0.1 < hi
    lo2, hi2 = tango_ci(10, 30, 200, 0.95)
    out["mirror"] = abs(lo + hi2) < 1e-6 and abs(hi + lo2) < 1e-6
    n, x12, x21 = 10000, 1500, 1000
    d = (x12 - x21) / n
    w = 1.96 * math.sqrt(((x12 + x21) / n - d * d) / n)
    lo3, hi3 = tango_ci(x12, x21, n, 0.95)
    out["near_wald"] = abs(lo3 - (d - w)) < 2e-3 and abs(hi3 - (d + w)) < 2e-3
    lo4, hi4 = tango_ci(0, 0, 100, 0.95)
    out["zero_discordant"] = lo4 < 0 < hi4
    out["mcnemar"] = abs(mcnemar_exact(0, 6) - 2 * 0.5 ** 6) < 1e-12
    out["pass"] = all(out.values())
    return out
