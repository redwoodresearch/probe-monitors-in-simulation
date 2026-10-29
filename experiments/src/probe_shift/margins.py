"""Margin (equivalence) checks on bootstrap draws, and Holm's step-down (experiments/2026-10-11_q2_margins_POSTHOC/PLAN).
A gap is inside margin m when its 95% interval lies strictly within (-m, +m); the matching bootstrap TOST p is
max(p_low, p_high), and Holm-adjusted p < 0.025 is the "inside after Holm" call."""
import numpy as np

LEVEL = 0.025


def equivalence_p(draws, m: float) -> float:
    """Bootstrap two one-sided tests against -m and +m: max of (1 + draws <= -m) / (B + 1) and (1 + draws >= m) / (B + 1)."""
    d = np.asarray(draws, float)
    return float(max((1 + (d <= -m).sum()) / (len(d) + 1), (1 + (d >= m).sum()) / (len(d) + 1)))


def difference_p(draws) -> float:
    """Two-sided bootstrap p for a gap of 0, capped at 1."""
    d = np.asarray(draws, float)
    return float(min(1.0, 2 * min(1 + (d <= 0).sum(), 1 + (d >= 0).sum()) / (len(d) + 1)))


def inside(ci, m: float) -> bool:
    return bool(-m < ci[0] and ci[1] < m)


def holm(ps: dict) -> dict:
    """Holm-adjusted p-values (step-down, monotone, capped at 1) for a family given as {name: p}."""
    names = sorted(ps, key=ps.get)
    out, running = {}, 0.0
    for i, n in enumerate(names):
        running = max(running, min(1.0, (len(names) - i) * ps[n]))
        out[n] = running
    return out
