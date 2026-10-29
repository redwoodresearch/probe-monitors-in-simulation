"""Aswin's rule for planted checks (2026-10-10, 10:51 PT): "a plant counts as detected when the estimate shifts by about
the planted amount and its 95% range excludes the unplanted estimate (not zero), or plant in the gap-widening
direction". Here "about the planted amount" means the observed shift has the planted sign and lies within 0.5x to 1.5x
of it. detected() implements the first clause only; widening() is reported beside the result and never counted as
detection."""

TOL = 0.5


def detected(planted_point: float, planted_ci: tuple[float, float], unplanted_point: float, planted_shift: float, tol: float = TOL) -> bool:
    """planted_shift: the intended signed change of the estimate. True when the planted estimate moved by about that
    amount and its 95% interval excludes the unplanted point estimate."""
    shift = planted_point - unplanted_point
    sign = 1 if planted_shift > 0 else -1
    about = planted_shift != 0 and (1 - tol) * abs(planted_shift) <= sign * shift <= (1 + tol) * abs(planted_shift)
    lo, hi = planted_ci
    return bool(about and (unplanted_point < lo or unplanted_point > hi))


def widening(unplanted_point: float, planted_shift: float) -> bool:
    """Whether a plant of this sign widens the existing gap (moves the estimate away from zero)."""
    return bool((unplanted_point >= 0) == (planted_shift > 0))
