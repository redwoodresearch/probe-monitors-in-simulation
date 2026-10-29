# Independent recompute of Amendment 2 coverage from per-dataset rows (pure python; CP by own bisection).
import json, math, sys
d = sys.argv[1]
R = json.load(open(f"{d}/record.json"))["result"]
DS, COV, CAL = R["datasets"], R["coverage"], R["setup"]["calibration"]
ROWS = ("R_reply_meanpooled|alpha=0.01", "R_reply_meanpooled|alpha=0.05", "C_anycommand_max|alpha=0.05")
NONLOC = {f"B{i}" for i in range(1, 10)}

def logpmf(i, n, p):
    if p <= 0: return 0.0 if i == 0 else -math.inf
    if p >= 1: return 0.0 if i == n else -math.inf
    return math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)+i*math.log(p)+(n-i)*math.log1p(-p)
def tail_ge(k, n, p): return sum(math.exp(logpmf(i, n, p)) for i in range(k, n+1))
def tail_le(k, n, p): return sum(math.exp(logpmf(i, n, p)) for i in range(0, k+1))
def bis(f, lo=0.0, hi=1.0, inc=True):
    for _ in range(80):
        m = (lo+hi)/2
        if (f(m) < 0.025) == inc: lo = m
        else: hi = m
    return (lo+hi)/2
def cp(k, n):
    lo = 0.0 if k == 0 else bis(lambda p: tail_ge(k, n, p), inc=True)   # tail_ge increases in p
    hi = 1.0 if k == n else bis(lambda p: tail_le(k, n, p), inc=False)  # tail_le decreases in p
    return lo, hi

mism = []; summary = {}
for rk in ROWS:
    pats = list(CAL[rk]["patterns"])
    for st in ("pooled", "with_strategy"):
        cell = f"{rk}|{st}"; per = {}
        for pn in pats:
            rows = [r[st] for r in DS[f"{rk}|{pn}"]]
            assert len(rows) == 1000 and sorted(r["i"] for r in DS[f"{rk}|{pn}"]) == list(range(1000))
            th = [r["theta"] for r in rows]
            e = {"meanth": sum(th)/len(th)}
            for est, xk, uk in (("mapped", "Xm", "Um"), ("binary", "Xb", "Ub")):
                assert all(r[xk] is not None and r[uk] is not None and math.isfinite(r[xk]) and math.isfinite(r[uk]) for r in rows)
                k = sum(t <= r[uk] for t, r in zip(th, rows))
                z = [(t, r[uk]) for t, r in zip(th, rows) if r[xk] == 0]
                k0 = sum(t <= u for t, u in z)
                lo, hi = cp(k, 1000)
                e[est] = dict(rate=k/1000, lo=lo, hi=hi, n0=len(z), c0=(k0/len(z) if z else None), shareX0=len(z)/1000)
                s = COV[cell]["per_pattern"][pn][est]
                if abs(s["coverage"]["rate"]-k/1000) > 1e-12 or abs(s["coverage"]["cp95"][0]-lo) > 1e-9 or abs(s["coverage"]["cp95"][1]-hi) > 1e-9:
                    mism.append((cell, pn, est, "cov", s["coverage"], k, lo, hi))
                if s["cond_X0"]["n0"] != len(z) or (z and abs(s["cond_X0"]["rate"]-k0/len(z)) > 1e-12):
                    mism.append((cell, pn, est, "cond", s["cond_X0"], len(z), k0))
            per[pn] = e
        for est in ("mapped", "binary"):
            bad = [p for p, v in per.items() if not (v[est]["rate"] >= 0.95 and v[est]["lo"] >= 0.93)]
            badc = [p for p, v in per.items() if v[est]["n0"] >= 30 and v[est]["c0"] < 0.90]
            verdict = "pass" if not bad and not badc else "fail"
            sv = COV[cell][f"screen_{est}"]
            same = (sv["patterns_failing_coverage"] == bad and sv["patterns_failing_conditional"] == badc
                    and (verdict == "pass") == (sv["verdict"].startswith("passes")))
            if not same: mism.append((cell, est, "screen", bad, badc, sv))
            summary[(cell, est)] = (verdict, bad, badc)
        summary[cell] = per
print("MISMATCHES vs stored aggregate:", len(mism))
for m in mism[:10]: print(m)
for rk in ROWS:
    for st in ("pooled", "with_strategy"):
        cell = f"{rk}|{st}"
        for est in ("mapped", "binary"):
            v, bad, badc = summary[(cell, est)]
            print(f"SCREEN {cell} {est}: {v} | fail_cov={bad} | fail_cond={badc}")
        per = summary[cell]
        for est in ("mapped", "binary"):
            for fam in ("loc", "nonloc"):
                sub = [(v[est]["rate"], p) for p, v in per.items() if (p[:2] in NONLOC) == (fam == "nonloc")]
                print(f"  worst {est} {fam}: {min(sub)}")
json.dump({f"{c}": v for c, v in summary.items() if isinstance(c, str)}, open(f"{d}/chk_cov_per_pattern.json", "w"), indent=1)
