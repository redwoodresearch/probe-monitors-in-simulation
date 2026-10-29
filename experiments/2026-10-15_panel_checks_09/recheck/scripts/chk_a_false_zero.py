"""Independent recount of panel check (a) from the committed U_cal record (de300a073)."""
import json, math, sys
from collections import defaultdict

ucal_path, fz_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(ucal_path))
ds = d["result"]["datasets"]

def logpmf(k, n, p):
    if p <= 0: return 0.0 if k == 0 else -math.inf
    if p >= 1: return 0.0 if k == n else -math.inf
    return math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)+k*math.log(p)+(n-k)*math.log1p(-p)

def cdf(k, n, p):  # P(X<=k)
    if k < 0: return 0.0
    if k >= n: return 1.0
    m = max(logpmf(i, n, p) for i in range(0, k+1))
    return math.exp(m) * sum(math.exp(logpmf(i, n, p)-m) for i in range(0, k+1))

def sf(k, n, p):  # P(X>=k)
    return 1.0 - cdf(k-1, n, p)

def bisect(f, target, lo=0.0, hi=1.0, increasing=True):
    for _ in range(80):
        mid = (lo+hi)/2
        v = f(mid)
        if (v < target) == increasing: lo = mid
        else: hi = mid
    return (lo+hi)/2

def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else bisect(lambda p: sf(k, n, p), a/2, increasing=True)
    hi = 1.0 if k == n else bisect(lambda p: cdf(k, n, p), a/2, increasing=False)
    return lo, hi

BINS = ["=0", "(0,2]", "(2,4]", "(4,6]", "(6,8]", ">8"]
def bin_of(t):
    if abs(t) < 1e-9: return "=0"
    if t < 0: return "NEG"
    for lo, hi, name in [(0,2,"(0,2]"),(2,4,"(2,4]"),(4,6,"(4,6]"),(6,8,"(6,8]")]:
        if lo < t <= hi: return name
    return ">8"

counts = defaultdict(lambda: [0, 0])          # (row, stratum, bin) -> [n, k]
patterns_in_bin = defaultdict(lambda: defaultdict(int))  # (row,stratum,bin) -> pattern -> n
neg = 0; nonfinite = 0
for key, lst in ds.items():
    row, alpha, pat = key.split("|")
    for rec in lst:
        for st in ("pooled", "with_strategy"):
            x, t = rec[st]["X"], rec[st]["theta"]
            if x is None or t is None or not (math.isfinite(x) and math.isfinite(t)):
                nonfinite += 1; continue
            b = bin_of(t)
            if b == "NEG": neg += 1
            c = counts[(f"{row}|{alpha}", st, b)]
            c[0] += 1; c[1] += int(x == 0.0)
            patterns_in_bin[(f"{row}|{alpha}", st, b)][pat] += 1

out = {"ucal_git": d["git"], "ucal_stamp": d["stamp"], "n_dataset_keys": len(ds),
       "neg_theta": neg, "nonfinite": nonfinite, "bins": {}, "patterns_in_bin": {}}
lines = []
for rowa in sorted({k[0] for k in counts}):
    for st in ("pooled", "with_strategy"):
        parts = []
        for b in BINS:
            n, k = counts.get((rowa, st, b), [0, 0])
            lo, hi = cp(k, n) if n else (float("nan"), float("nan"))
            out["bins"][f"{rowa}|{st}|{b}"] = {"n": n, "k": k, "share": k/n if n else None, "cp": [lo, hi]}
            out["patterns_in_bin"][f"{rowa}|{st}|{b}"] = dict(sorted(patterns_in_bin[(rowa, st, b)].items(), key=lambda kv: -kv[1]))
            parts.append(f"{b}: {k}/{n}={k/n:.2f} [{lo:.2f},{hi:.2f}]" if n else f"{b}: 0/0")
        lines.append(f"{rowa}|{st} | " + "; ".join(parts))

# compare with the committed false_zero record
fz = json.load(open(fz_path))
out["committed_fz_top_keys"] = list(fz.keys())
json.dump(out, open(out_path, "w"), indent=1)
print("\n".join(lines))
print("neg_theta", neg, "nonfinite", nonfinite)
