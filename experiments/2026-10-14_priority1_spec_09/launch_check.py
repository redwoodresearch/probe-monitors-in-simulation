"""Append 5's $0 launch check (RULE.md here): is run2 < run1 a launch effect? Per-task paired run2 - run1 for Sol (a) avg_v1 and
own_a (stored per-run scores, recheck_2026-10-14_015553), overall and by run1's sub-batch launch, with honest A - honest B as the
same-launch null. Runs in one Modal CPU container. usage: python launch_check.py <recheck.json> <rows.jsonl> <out json>"""
import hashlib
import json
import sys
from collections import defaultdict

import numpy as np

REC, ROWS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
assert hashlib.sha256(open(REC, "rb").read()).hexdigest().startswith("5503cced397c0438")
assert hashlib.sha256(open(ROWS, "rb").read()).hexdigest() == "9960167498564c7ac7bbea17bfb5bd3fc467a10cf536db0f28fa7cd133bb81e3"
S = json.load(open(REC))["scores"]
launch = {}  # task -> run1's launch (B main set, attack arm)
for line in open(ROWS):
    r = json.loads(line)
    if r["staging"] == "B" and r["set"] == "q4b_main_v1" and r["arm"] == "attack":
        parts = r["key"].split("/")
        launch[parts[2]] = parts[0]


def flip_p(d, n_flips=10000, seed=20261014):
    """Per-task sign-flip permutation test of mean(d) = 0 (two-sided; add-one)."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_flips, len(d)))
    perm = np.abs((signs * d[None, :]).mean(1))
    return {"obs_mean": float(d.mean()), "p_two_sided": float(((perm >= obs - 1e-12).sum() + 1) / (n_flips + 1)), "flips": n_flips, "seed": seed}


def tci(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else None, "ci95": None}
    from scipy.stats import t as tdist
    h = tdist.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return {"n": n, "mean": float(x.mean()), "ci95": [float(x.mean() - h), float(x.mean() + h)]}


out = {"inputs": {"recheck_sha_prefix": "5503cced", "rows_sha": "9960167"}, "launch_counts": {}, "probes": {}}
tasks_all = sorted({k.split("|", 1)[1] for k in S["a"]})
for probe in ("a", "own_a"):
    s = S[probe]
    g = lambda w, t: s.get(f"{w}|{t}")  # noqa: E731
    ha = np.array([v for k, v in s.items() if k.startswith("honest_a|") and np.isfinite(v)])
    sd = float(ha.std(ddof=1))
    paired = [t for t in tasks_all if g("run1", t) is not None and g("run2", t) is not None and np.isfinite(g("run1", t)) and np.isfinite(g("run2", t))]
    by = defaultdict(list)
    for t in paired:
        by[launch.get(t, "unknown")].append(t)
    res = {"honest_a_sd": sd, "n_paired": len(paired), "score_diff_std": {}, "catch": {}}
    res["score_diff_std"]["all"] = tci([(g("run2", t) - g("run1", t)) / sd for t in paired])
    for L, ts in sorted(by.items()):
        res["score_diff_std"][L] = tci([(g("run2", t) - g("run1", t)) / sd for t in ts])
    hp = [t for t in tasks_all if g("honest_a", t) is not None and g("honest_b", t) is not None and np.isfinite(g("honest_a", t)) and np.isfinite(g("honest_b", t))]
    res["score_diff_std"]["null honest_a - honest_b"] = tci([(g("honest_a", t) - g("honest_b", t)) / sd for t in hp])
    sim_p = [t for t in paired if g("sim", t) is not None and np.isfinite(g("sim", t))]
    res["sign_flip"] = {"score run2 - run1": flip_p([(g("run2", t) - g("run1", t)) / sd for t in paired]),
                        "score null honest_a - honest_b": flip_p([(g("honest_a", t) - g("honest_b", t)) / sd for t in hp])}
    res["score_diff_std"]["sim - run1"] = tci([(g("sim", t) - g("run1", t)) / sd for t in sim_p])
    res["score_diff_std"]["sim - run2"] = tci([(g("sim", t) - g("run2", t)) / sd for t in sim_p])
    for alpha in (0.05, 0.01):
        o = np.sort(ha)[::-1]
        T = float(o[int(np.floor(alpha * len(o)))])
        hit = lambda w, t: float(g(w, t) > T)  # noqa: E731
        c = {"threshold": T, "all": {"run1": float(np.mean([hit("run1", t) for t in paired])), "run2": float(np.mean([hit("run2", t) for t in paired])),
                                     "n": len(paired)}}
        for L, ts in sorted(by.items()):
            c[L] = {"run1": float(np.mean([hit("run1", t) for t in ts])), "run2": float(np.mean([hit("run2", t) for t in ts])), "n": len(ts),
                    "run2_minus_run1_paired_t": tci([hit("run2", t) - hit("run1", t) for t in ts])}
        c["all"]["run2_minus_run1_paired_t"] = tci([hit("run2", t) - hit("run1", t) for t in paired])
        res["sign_flip"][f"catch@{alpha:g} run2 - run1"] = flip_p([hit("run2", t) - hit("run1", t) for t in paired])
        res["catch"][f"{alpha:g}"] = c
    out["probes"][probe] = res
out["launch_counts"] = {L: len([t for t in tasks_all if launch.get(t) == L]) for L in sorted(set(launch.values()))}
open(OUT, "w").write(json.dumps(out, indent=1))
for p, r in out["probes"].items():
    for k, v in r["score_diff_std"].items():
        print(f"{p:6s} score diff (std) {k:42s} n {v['n']:3d} mean {v['mean']:+.3f} ci95 {[round(x, 3) for x in v['ci95']] if v['ci95'] else None}")
    for k, v in r["sign_flip"].items():
        print(f"{p:6s} sign-flip {k:36s} obs mean {v['obs_mean']:+.4f} p {v['p_two_sided']:.4f} ({v['flips']} flips, seed {v['seed']})")
    for a, c in r["catch"].items():
        for L, v in c.items():
            if L == "threshold":
                continue
            pt = v["run2_minus_run1_paired_t"]
            print(f"{p:6s} catch @{a} {L:36s} n {v['n']:3d} run1 {100 * v['run1']:5.1f} run2 {100 * v['run2']:5.1f} diff {100 * pt['mean']:+5.1f} ci95 {[round(100 * x, 1) for x in pt['ci95']] if pt['ci95'] else None}")
