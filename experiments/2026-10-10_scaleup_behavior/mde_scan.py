"""Minimum detectable difference for the scale-up behaviour comparison, under Aswin's planted-check rule (README
amendment, 10:51 PT): a copy is detected when the gap shifts by 0.5-1.5x the planted amount and the planted gap's 95%
interval excludes the unplanted estimate. Additive plants: a seeded half of the target arm's tasks get +-2 x size, so the
gap shifts by about `size`. 200 seeded copies per size, comparison and direction; 10,000 bootstrap draws over tasks.
Also: why the frozen x1.15 step plant is detected 196/200 into run 2 but 21/200 into sim (half-widths, margins, band).

  uv run --no-project --python 3.12 --with pandas --with numpy python mde_scan.py results/per_run_scaleup.csv
"""
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
exec(open(HERE / "analyze_behavior.py").read().split("commit = subprocess.run")[0].replace("sys.argv[1]", "CSV"),
     g := {"CSV": sys.argv[1] if len(sys.argv) > 1 else "results/per_run_scaleup.csv", "__name__": "mde",
           "__file__": str(HERE / "analyze_behavior.py")})
arms, gap, SEED = g["arms"], g["gap"], g["SEED"]
SIZES = {"n_steps": [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20], "sh_failtext": [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]}
PAIRS = [("run1", "sim", "sim"), ("run1", "run2", "run2")]


def copies(args):
    """Detection counts over 200 copies for one (measure, comparison, direction, size), plus the reasons for misses."""
    m, a, b, target, sign, size, mult = args
    base = gap(arms[a], arms[b], m, np.random.default_rng(SEED))[0]
    paired = (arms[a][m] - arms[target][m]).dropna().index
    det = band_miss = range_miss = 0
    halfw, margin = [], []
    for s in range(200):
        rng = np.random.default_rng(SEED + s)
        copy = arms[target].copy()
        copy[m] = copy[m].astype(float)
        hit = rng.choice(copy.index.to_numpy(), len(copy) // 2, replace=False)
        copy.loc[hit, m] = copy.loc[hit, m] * mult if mult else copy.loc[hit, m] + sign * 2 * size
        if m == "sh_failtext":
            copy[m] = copy[m].clip(0, 1)
        est, lo, hi, _ = gap(arms[a], copy, m, rng)
        want = -float((copy.loc[paired, m] - arms[target].loc[paired, m].astype(float)).mean())
        got = est - base
        in_band = bool(want) and 0.5 <= got / want <= 1.5
        excl = not (lo <= base <= hi)
        det += in_band and excl
        band_miss += not in_band
        range_miss += not excl
        halfw.append((hi - lo) / 2)
        margin.append(abs(est - base) - ((hi - est) if est < base else (est - lo)))  # > 0: the interval clears base
    return {"measure": m, "comparison": f"{a}-{b}", "target": target, "direction": "target up" if sign > 0 else "target down",
            "size": size if not mult else f"x{mult}", "detected_of_200": det, "band_misses": band_miss,
            "range_misses": range_miss, "mean_halfwidth": round(float(np.mean(halfw)), 4),
            "mean_margin": round(float(np.mean(margin)), 4), "unplanted": round(float(base), 4)}


if __name__ == "__main__":
    jobs = [(m, a, b, t, sg, sz, None) for m, sizes in SIZES.items() for a, b, t in PAIRS for sg in (+1, -1) for sz in sizes]
    jobs += [("n_steps", a, b, t, +1, 0, 1.15) for a, b, t in PAIRS]  # the frozen x1.15 plant, for the 196-vs-21 question
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(copies, jobs))
    commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"script commit {commit[:12]} | seed {SEED} | 200 copies per row | detected = shift in 0.5-1.5x AND interval excludes the unplanted estimate")
    mde = {}
    for r in res:
        if isinstance(r["size"], str):
            continue
        key = (r["measure"], r["comparison"], r["direction"])
        if r["detected_of_200"] >= 160 and key not in mde:
            mde[key] = r["size"]
    for key in sorted({(r["measure"], r["comparison"], r["direction"]) for r in res if not isinstance(r["size"], str)}):
        row = [r for r in res if (r["measure"], r["comparison"], r["direction"]) == key and not isinstance(r["size"], str)]
        print(f"  {key[0]:12s} {key[1]:9s} {key[2]:11s} smallest size detected in >=80%: {mde.get(key, '>max')} | "
              + " ".join(f"{r['size']}:{r['detected_of_200']}" for r in row))
    print("\nthe frozen x1.15 step plant (target up), with the reasons for misses:")
    for r in res:
        if isinstance(r["size"], str):
            print(f"  {r['comparison']:9s} into {r['target']:4s}: detected {r['detected_of_200']}/200 | band misses "
                  f"{r['band_misses']} | interval-includes-unplanted misses {r['range_misses']} | planted interval "
                  f"half-width {r['mean_halfwidth']} | mean margin by which it clears the unplanted estimate {r['mean_margin']}"
                  f" | unplanted {r['unplanted']}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"mde_scan_{stamp}.json"
    out.write_text(json.dumps({"script_commit": commit, "seed": SEED, "rows": res,
                               "mde_80pct": {"|".join(k): v for k, v in mde.items()}}, indent=1) + "\n")
    print("->", out.relative_to(HERE.parents[1]))
