"""Panel check (a): false-zero rate of the floored X_err, counted from the committed U_cal record (no resampling).

usage: uv run --no-project --with scipy python count_false_zero.py <ucal_record.json> <ucov_full_record.json> <out.json>
"""

import hashlib
import json
import sys
from pathlib import Path

from scipy.stats import beta

UCAL_SHA = "02aa1bfb0e35bfd559569e2995272afa5df15627fbd87744ad0772153d51ce51"  # ucal_2026-10-15_004655 @ de300a073
UCOV_SHA = "98f37228ad9d5671ce8c77c9801446bc614c1abf22eda3f95c5f923e011843d1"  # ucov_full_2026-10-15_001850 @ 4980f707e
BINS = [("=0", None, None), ("(0,2]", 0, 2), ("(2,4]", 2, 4), ("(4,6]", 4, 6), ("(6,8]", 6, 8), (">8", 8, None)]


def cp(k, n):
    if n == 0:
        return None
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def in_bin(th, lo, hi):
    """Bins on the realized truth theta (percentage points): "=0" is |theta| < 1e-9; others are (lo, hi] or > lo."""
    if lo is None:
        return abs(th) < 1e-9
    if abs(th) < 1e-9:
        return False
    return th > lo if hi is None else lo < th <= hi


def main(ucal_p, ucov_p, out_p):
    raw = Path(ucal_p).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != UCAL_SHA:
        raise SystemExit(f"U_cal record sha {sha} does not match the pin {UCAL_SHA}")
    rec = json.loads(raw)
    rec = rec.get("result", rec)
    out = {"ucal_sha256": sha, "bins": {}, "by_pattern": {}, "cross_check_ucov": {}}
    for key, ds in rec["datasets"].items():
        row, pat = key.rsplit("|", 1)
        for stratum in ("pooled", "with_strategy"):
            vals = [(d[stratum]["X"], d[stratum]["theta"]) for d in ds]
            if any(x is None or t is None for x, t in vals):
                raise SystemExit(f"{key}|{stratum}: a dataset lacks X or theta")
            kz = sum(1 for x, _ in vals if x == 0.0)
            pp = out["by_pattern"].setdefault(f"{row}|{stratum}", {})
            pp[pat] = {
                "n": len(vals),
                "mean_theta": sum(t for _, t in vals) / len(vals) if vals else None,
                "k_zero": kz,
                "share_zero": kz / len(vals) if vals else None,
                "cp95": cp(kz, len(vals)),
            }
            bb = out["bins"].setdefault(f"{row}|{stratum}", {b[0]: {"n": 0, "k_zero": 0} for b in BINS})
            for x, t in vals:
                for name, lo, hi in BINS:
                    if in_bin(t, lo, hi):
                        bb[name]["n"] += 1
                        bb[name]["k_zero"] += x == 0.0
                        break
    for v in out["bins"].values():
        for b in v.values():
            b["share_zero"] = b["k_zero"] / b["n"] if b["n"] else None
            b["cp95"] = cp(b["k_zero"], b["n"])
    # descriptive: ucov_full's stored share_zero on its nine patterns (pooled stratum; env-only outer sampling)
    rawc = Path(ucov_p).read_bytes()
    if hashlib.sha256(rawc).hexdigest() != UCOV_SHA:
        raise SystemExit("ucov_full record sha does not match the pin")
    uc = json.loads(rawc)
    uc = uc.get("result", uc)
    for key, s in uc["summary"].items():
        out["cross_check_ucov"][key] = s["X_err"]["share_zero"]
    out["label"] = (
        "panel check (a): DESCRIPTIVE, post hoc; shares pool over the U_cal grid's simulated patterns and inherit "
        "its generator assumptions (exchangeable arms; eligibility and Sol moments fixed; truth = theta(T-hat))"
    )
    with Path(out_p).open("x") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", out_p)


if __name__ == "__main__":
    main(*sys.argv[1:4])
