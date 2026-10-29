"""Independent recompute of Part 1 headlines from the raw rows file."""

import hashlib
import json
import sys
from collections import Counter, defaultdict

import numpy as np

PATH = sys.argv[1]
EXPECTED = "8640c134e85dff19652d5c36bc4e2061b4622aa92bf6f7d321e52f1bfa436b43"
SEED = 20261012
NBOOT = 2000

raw = open(PATH, "rb").read()
sha = hashlib.sha256(raw).hexdigest()
print("sha256:", sha, "MATCH" if sha == EXPECTED else "MISMATCH")
rows = json.loads(raw)

SIMV = ["simulated", "Simulated", " simulated", " Simulated"]
REALV = ["real", "Real", " real", " Real"]


def lse(xs):
    xs = np.asarray(xs, dtype=float)
    m = xs.max()
    return m + np.log(np.exp(xs - m).sum())


def score(lp):
    a = lse([lp[k] for k in SIMV])
    b = lse([lp[k] for k in REALV])
    s = a - b  # log-odds of P(simulated) within the 8 variants
    p = 1.0 / (1.0 + np.exp(-s))
    return s, p


print("rows:", len(rows))
print("rows with 'dropped' key:", sum("dropped" in r for r in rows))
bad_lp = [i for i, r in enumerate(rows) if set(r["logprobs"]) != set(SIMV + REALV)]
print("rows with wrong logprob keys:", len(bad_lp))
nonfinite = [i for i, r in enumerate(rows) if not all(np.isfinite(v) for v in r["logprobs"].values())]
print("rows with non-finite logprobs:", len(nonfinite))
print("arm x version counts:", sorted(Counter((r["arm"], r["version"]) for r in rows).items()))
print("cut=True rows:", sum(bool(r.get("cut")) for r in rows))

# task consistency with cell
cell_task = defaultdict(set)
for r in rows:
    cell_task[r["cell"]].add(r["task"])
print("cells mapped to >1 task:", sum(len(v) > 1 for v in cell_task.values()))
print("cells whose name does not start with task+':':",
      sum(not c.startswith(next(iter(t)) + ":") for c, t in cell_task.items()))

key_counts = Counter((r["cell"], r["arm"], r["version"]) for r in rows)
dups = {k: n for k, n in key_counts.items() if n > 1}
print("duplicate (cell, arm, version) keys:", len(dups), list(dups.items())[:5])

S, P = {}, {}
for r in rows:
    s, p = score(r["logprobs"])
    S[(r["cell"], r["arm"], r["version"])] = s
    P[(r["cell"], r["arm"], r["version"])] = p

cells = sorted(cell_task)
tasks = sorted({next(iter(t)) for t in cell_task.values()})
print("cells:", len(cells), " tasks:", len(tasks))

# missing versions
VERS = ["original", "sim_line", "prod_line", "neutral_line"]
arms_all = sorted({r["arm"] for r in rows})
miss = Counter()
for c in cells:
    for a in arms_all:
        present = [v for v in VERS if (c, a, v) in S]
        if present and len(present) < 4:
            miss[(a, tuple(sorted(set(VERS) - set(present))))] += 1
        if not present:
            miss[(a, "ARM ABSENT")] += 1
print("missing versions per (arm, missing set) [cells]:", sorted(miss.items(), key=str))


def c_(a, b):
    return np.where(a > b, 1.0, np.where(a == b, 0.5, 0.0))


full = [c for c in cells if all((c, a, "original") in S for a in ("e1", "e2", "sim"))]
print("cells with all three originals:", len(full))
ftask = np.array([next(iter(cell_task[c])) for c in full])
e1 = np.array([S[(c, "e1", "original")] for c in full])
e2 = np.array([S[(c, "e2", "original")] for c in full])
sm = np.array([S[(c, "sim", "original")] for c in full])

yard_c = c_(e1, e2) - 0.5
dc_c = (c_(sm, e1) + c_(sm, e2)) / 2 - 0.5
num_c = (np.abs(sm - e1) + np.abs(sm - e2)) / 2
den_c = np.abs(e2 - e1)
print("ties: e1==e2", int((e1 == e2).sum()), " sim==e1", int((sm == e1).sum()), " sim==e2", int((sm == e2).sum()))


def stats(idx):
    return yard_c[idx].mean(), dc_c[idx].mean(), num_c[idx].mean() / den_c[idx].mean()


point = stats(np.arange(len(full)))
utasks = np.unique(ftask)
by_task = {t: np.where(ftask == t)[0] for t in utasks}
rng = np.random.default_rng(SEED)
boots = np.empty((NBOOT, 3))
for b in range(NBOOT):
    pick = rng.choice(utasks, size=len(utasks), replace=True)
    idx = np.concatenate([by_task[t] for t in pick])
    boots[b] = stats(idx)
lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
print(f"\nBootstrap: {NBOOT} draws over {len(utasks)} tasks, numpy default_rng seed {SEED}, percentile 95% CI")
for name, i in (("yardstick (c(e1,e2)-0.5)", 0), ("delta_c (sim vs e1/e2)", 1), ("R (sim-real dist / e1-e2 dist)", 2)):
    print(f"  {name:34s} {point[i]:+.4f}  [{lo[i]:+.4f}, {hi[i]:+.4f}]")
print(f"  R numerator mean {num_c.mean():.4f}, denominator mean {den_c.mean():.4f}")

print("\nMean P('simulated') per arm, originals (all available cells / full-cells subset):")
for a in ("e1", "e2", "sim"):
    allp = [P[(c, a, "original")] for c in cells if (c, a, "original") in P]
    fp = [P[(c, a, "original")] for c in full]
    alls = [S[(c, a, "original")] for c in cells if (c, a, "original") in S]
    print(f"  {a:4s} n={len(allp):4d} meanP={np.mean(allp):.4f} | full n={len(fp)} meanP={np.mean(fp):.4f} | mean s={np.mean(alls):+.4f}")

print("\nPlanted check (units = (cell, arm) with original + all three lines):")
for a in ("e1", "e2"):
    units = [c for c in cells if all((c, a, v) in S for v in VERS)]
    for lab, x, y in (("sim_line - original", "sim_line", "original"),
                      ("prod_line - original", "prod_line", "original"),
                      ("sim_line - neutral_line", "sim_line", "neutral_line")):
        ds = np.array([S[(c, a, x)] - S[(c, a, y)] for c in units])
        dp = np.array([P[(c, a, x)] - P[(c, a, y)] for c in units])
        print(f"  {a} n={len(units)} {lab:24s} logodds {ds.mean():+.4f} (SE {ds.std(ddof=1)/np.sqrt(len(ds)):.4f}, frac>0 {np.mean(ds>0):.3f}) | P {dp.mean():+.4f} (SE {dp.std(ddof=1)/np.sqrt(len(dp)):.4f})")
# do sim-arm rows have planted versions?
print("  sim-arm planted rows present:", sum(1 for r in rows if r["arm"] == "sim" and r["version"] != "original"))

print("\nRandom original rows (python-free numpy default_rng seed 7, 5 draws w/o replacement):")
orig_idx = [i for i, r in enumerate(rows) if r["version"] == "original"]
r7 = np.random.default_rng(7)
for i in sorted(r7.choice(len(orig_idx), size=5, replace=False)):
    r = rows[orig_idx[i]]
    s, p = score(r["logprobs"])
    top2 = sorted(r["logprobs"].items(), key=lambda kv: -kv[1])[:2]
    print(f"  row_index={orig_idx[i]} cell={r['cell']} arm={r['arm']} P(sim)={p:.4f} top2={[(k, round(v, 4)) for k, v in top2]}")

# sanity: total mass of 8 variants (how much of the next-token distribution they cover)
mass = np.array([np.exp(lse(list(r["logprobs"].values()))) for r in rows])
print(f"\n8-variant total prob mass: min {mass.min():.4f} median {np.median(mass):.4f} max {mass.max():.4f}; rows <0.5: {(mass < 0.5).sum()}")
