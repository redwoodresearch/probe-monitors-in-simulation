"""Per-world rates of the fake-environment judge's yeses, with exact (Clopper-Pearson) 95% intervals, over runs (runs
with any yes of that type) and over parsed turns. Reads the run's data/fake_env_<stamp>.jsonl. $0."""
import collections, json, sys
from scipy.stats import beta

def exact(k, n):
    return (beta.ppf(0.025, k, n - k + 1) if k else 0.0, beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)

rows = [json.loads(l) for l in open(sys.argv[1])]
worlds = sorted({r["world"] for r in rows})
print(f"{'world':14s} {'type':13s} {'runs with a yes':>16s} {'95% interval':>18s} {'turns':>12s} {'95% interval':>18s}")
for w in worlds:
    wr = [r for r in rows if r["world"] == w]
    runs = {r["run"] for r in wr}
    parsed = [r for r in wr if r["suspicion"] != "unparsed"]
    for t in ("test_or_eval", "looks_fake", "either"):
        hit = (lambda r: r["suspicion"] in ("test_or_eval", "looks_fake")) if t == "either" else (lambda r, t=t: r["suspicion"] == t)
        kr = len({r["run"] for r in parsed if hit(r)}); kt = sum(hit(r) for r in parsed)
        lo, hi = exact(kr, len(runs)); tlo, thi = exact(kt, len(parsed))
        print(f"{w:14s} {t:13s} {kr:>7d}/{len(runs):<3d} = {kr/len(runs):.3f} [{lo:.3f}, {hi:.3f}] {kt:>5d}/{len(parsed):<5d} [{tlo:.4f}, {thi:.4f}]")
