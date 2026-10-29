"""One shared formatter for printed values (2e, 10-16: three slips came from rounding an already-rounded value). Round ONCE, to 2 dp, with Python's round()
on the stored full-precision float; never round a pre-rounded value, never round by hand. Tables keep the raw floats beside the printed strings."""


def p2(x, sign=True):
    """A stored float printed at 2 dp: f'{round(x, 2):+.2f}' (no '-0.00')."""
    if x is None:
        return "n/a"
    v = round(float(x), 2) + 0.0
    return f"{v:+.2f}" if sign else f"{v:.2f}"


def stat2(z, sign=True):
    """A summary dict {point, ci95} printed as 'point [lo, hi]' (or 'point (interval withheld)')."""
    if not z:
        return "n/a"
    if not z.get("ci95"):
        return f"{p2(z['point'], sign)} (interval withheld)"
    return f"{p2(z['point'], sign)} [{p2(z['ci95'][0], sign)}, {p2(z['ci95'][1], sign)}]"


def raw_and_printed(z):
    """Raw floats and the printed string, side by side, for value tables."""
    return None if not z else {"point": z.get("point"), "ci95": z.get("ci95"), "printed": stat2(z)}
