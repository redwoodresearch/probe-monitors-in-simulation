"""Low-rate (1%) thresholds and intervals for nested cross-fitted detectors, plus a coverage simulator.

Setting (#36, SCE, G3): a run in test fold f is scored by nested models (f, v), v != f; each model's threshold is set on
its calibration honest runs (fold v); a run's catch is the share of its valid models whose threshold it exceeds. Scores
come as S[(f, v)] = float array over all runs (NaN where the model doesn't score the run); bootstrap weights W are
[draws, runs] repeat counts (row 0 = all ones = the point estimate).

Threshold methods: "order" (#36's rule: the first descending score whose cumulative weight exceeds floor(rate x total)),
"hd" (Harrell-Davis (1 - rate) quantile), "gauss" (weighted mean + z(1 - rate) x weighted SD), "smooth" (the order rule
after adding N(0, h^2) noise to every calibration score, rows >= 1 only; h = Silverman). order_band() gives the
distribution-free binomial-rank interval for the (1 - rate) quantile. See experiments/2026-10-13_robust_1pct_POSTHOC.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import betainc
from scipy.stats import binom, norm


def silverman(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return float(0.9 * min(np.std(x), iqr / 1.34) * len(x) ** -0.2)


def threshold(s: np.ndarray, w: np.ndarray, rate: float, method: str = "order", h: float | None = None, rng=None) -> np.ndarray:
    """s: [m] calibration scores; w: [B, m] repeat counts. Returns [B] thresholds."""
    if method == "order" or method == "smooth":
        if method == "smooth":
            noise = rng.normal(0, h if h is not None else silverman(s), (w.shape[0], len(s)))
            noise[0] = 0.0
            S = s[None, :] + noise
            o = np.argsort(-S, 1, kind="mergesort")
            Ss, ws = np.take_along_axis(S, o, 1), np.take_along_axis(w, o, 1)
        else:
            o = np.argsort(-s, kind="mergesort")
            Ss, ws = np.broadcast_to(s[o], w.shape), w[:, o]
        cw = np.cumsum(ws, 1)
        return Ss[np.arange(w.shape[0]), np.argmax(cw > np.floor(rate * cw[:, -1:] + 1e-9), 1)]
    if method == "hd":
        o = np.argsort(s, kind="mergesort")
        ww = w[:, o]
        N = ww.sum(1, keepdims=True)
        c = np.cumsum(ww, 1) / N
        cdf = betainc((1 - rate) * (N + 1), rate * (N + 1), np.clip(c, 0, 1))
        return (np.diff(np.hstack([np.zeros((len(N), 1)), cdf]), axis=1) * s[o][None, :]).sum(1)
    if method == "gauss":
        mu = (w * s).sum(1) / w.sum(1)
        sd = np.sqrt((w * (s[None, :] - mu[:, None]) ** 2).sum(1) / w.sum(1))
        return mu + norm.ppf(1 - rate) * sd
    raise ValueError(method)


def order_band(s: np.ndarray, rate: float, level: float = 0.95) -> tuple[float, float]:
    """Distribution-free interval for the (1 - rate) quantile from binomial ranks; +inf when the upper rank passes n."""
    x = np.sort(s[np.isfinite(s)])
    n, q, a = len(x), 1 - rate, (1 - level) / 2
    lo, hi = int(binom.ppf(a, n, q)), int(binom.ppf(1 - a, n, q)) + 1
    return (float(x[lo - 1]) if lo >= 1 else -np.inf), (float(x[hi - 1]) if hi <= n else np.inf)


def catch(S: dict, fold: np.ndarray, cal: dict, W: np.ndarray, rate: float, method: str = "order", T: dict | None = None,
          h: dict | None = None, rng=None) -> tuple[np.ndarray, dict]:
    """[B, n] catch (share of a run's valid models above threshold) and the thresholds used. T overrides the computed
    thresholds (e.g. frozen ones, or band edges): T[(f, v)] is a scalar or a [B] array."""
    B, n = W.shape
    hits, cnt, used = np.zeros((B, n)), np.zeros(n), {}
    for (f, v), s in S.items():
        hv = cal[(f, v)]
        t = T[(f, v)] if T is not None else threshold(s[hv], W[:, hv], rate, method, None if h is None else h[(f, v)], rng)
        t = np.broadcast_to(np.asarray(t, float), (B,))
        used[(f, v)] = t
        te = (fold == f) & np.isfinite(s)
        hits[:, te] += s[te][None, :] > t[:, None]
        cnt[te] += 1
    with np.errstate(invalid="ignore"):
        return hits / cnt, used


def rate_of(c: np.ndarray, mask: np.ndarray, W: np.ndarray) -> np.ndarray:
    return (W[:, mask] * c[:, mask]).sum(1) / W[:, mask].sum(1)


def task_weights(tix: np.ndarray, n_tasks: int, draws: int, seed: int) -> np.ndarray:
    """Row 0 ones, then `draws` rows of main-task bootstrap counts (np.bincount of n draws), per run via tix."""
    rng = np.random.default_rng(seed)
    return np.vstack([np.ones(n_tasks)] + [np.bincount(rng.integers(0, n_tasks, n_tasks), minlength=n_tasks) for _ in range(draws)])[:, tix]


def smoothed_truth(S: dict, fold: np.ndarray, cal: dict, h: dict, rate: float) -> np.ndarray:
    """The smoothed world's true per-run catch: each model's true threshold is the (1 - rate) quantile of the mixture
    of N(s_i, h^2) over its calibration scores; a run's true catch is the mean over its models of Phi((s - T) / h)."""
    n = len(fold)
    tot, cnt = np.zeros(n), np.zeros(n)
    for (f, v), s in S.items():
        x, hh = s[cal[(f, v)]], h[(f, v)]
        F = lambda t: norm.cdf((t - x) / hh).mean() - (1 - rate)  # noqa: E731
        t = brentq(F, x.min() - 10 * hh, x.max() + 10 * hh)
        te = (fold == f) & np.isfinite(s)
        tot[te] += norm.cdf((s[te] - t) / hh)
        cnt[te] += 1
    return tot / cnt


def interval(x: np.ndarray, level: float = 0.95) -> list[float]:
    a = (1 - level) / 2
    return [float(np.percentile(x[1:], 100 * a)), float(np.percentile(x[1:], 100 * (1 - a)))]
