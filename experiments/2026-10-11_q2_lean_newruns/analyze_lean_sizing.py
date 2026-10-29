"""Sizing for the pooled lean test at +/-5 with new simulated runs plus a second real batch (PREREG.md @ e8787b92, committed
before this script). M2/M3 get k pseudo-tasks copied from the 163 original same-batch tasks (10 donor sets per k); M1 is
unchanged. Poisson task weights (seed 20261011); thresholds fixed as in the pooled lean test. P(clear) at true gaps of 0
= P(G_RS and G_RR both inside +/-5) x P(both margin-sized plants testable). Keys and scores only; $0.

  uv run --no-project --with numpy python experiments/2026-10-11_q2_lean_newruns/analyze_lean_sizing.py \\
      <per_task_catch_196.json> <recal_2026-10-10_225753.json>
"""
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from probe_shift.planted import detected  # noqa: E402

HERE = Path(__file__).parent
SHA = {"per_task_catch_196.json": "b608295b3563fc10ba8dc89a3f652b8a46b68bb917d32673267b193709d86935",
       "recal_2026-10-10_225753.json": "c9ba6dfe3bd028027e162ae386dc5c4ec5343eff5825d63c4ed94534b2edb9e7"}
SEED, DRAWS, SETS, COPIES, MC, M, PLANT, GOAL = 20261011, 10_000, 10, 100, 200_000, 0.05, 0.05, 0.80
TARGETS, GRID, MAXK = {"1%": 0.01, "5%": 0.05}, list(range(0, 271, 10)) + [274], 274
SIM_RUN, REAL_RUN, EXTRACT, DEVBOX = 7.86, 1.00, 0.011, 0.01  # b7's API per simulated run; df's whole job per real run
FIXED = {"real batch": 1.11, "simulated batch (assumed)": 1.11, "probe reproduction": 3.00, "scoring": 3.00}
ARMS = ("real_run1", "simulated", "real_run2")
REF_SHA = "pooled_lean_2026-10-11_142114.json on main @ 7be615a3"  # positives (run 1, sim, run 2), real - sim, real run 1 - run 2
REF = {"1%": {"M1 196-task command monitor": [[6945, 6913, 6840], 0.004552227171457424, 0.019793048135096636], "M2 data transfer": [[132, 140, 138], 0.11428571428571432, 0.018115942028985477], "M3 destructive": [[177, 191, 188], 0.012600940633596569, -0.026415434547421568]}, "5%": {"M1 196-task command monitor": [[6945, 6913, 6840], -0.0006592893659205235, 0.0028437724982002255], "M2 data transfer": [[132, 140, 138], 0.0800865800865801, 0.017457180500658698], "M3 destructive": [[177, 191, 188], 0.001893099062324377, 0.03266618583964409]}}


def threshold(neg, q):
    """The frozen convention (Q2.5, q3): sorted high to low, the value at index floor(q n)."""
    neg = np.sort(neg)[::-1]
    return float(neg[int(np.floor(q * len(neg)))])


def ci(b):
    return [float(v) for v in np.percentile(b, [2.5, 97.5])]


def gap(c_a, p_a, c_b, p_b, W):
    den_a, den_b = W @ p_a, W @ p_b
    if (den_a == 0).any() or (den_b == 0).any():
        raise SystemExit("a draw has no positives in an arm")
    return (W @ c_a) / den_a - (W @ c_b) / den_b


def m1_counts(path, key):
    """As the pooled lean test: M1's per-task counts for real run 1, simulated and real run 2 on tasks with all three."""
    d = json.loads(Path(path).read_text())
    tasks = sorted(set.intersection(*[{r["task"] for r in d["rows"] if r["arm"] == a} for a in ARMS]))
    get = {(r["arm"], r["task"]): r for r in d["rows"]}
    return {a: (np.array([get[(a, t)][key] for t in tasks], float), np.array([get[(a, t)]["positives"] for t in tasks], float)) for a in ARMS}


def sol_counts(full, name, q):
    """As the pooled lean test: per-task caught and positives for run 1, run 2 (both overlap800) and simulated, at fixed
    point thresholds, on the same-batch tasks."""
    rows = full["rows"]; s = np.array(full["label sets"][name]["test scores"], float); y = np.array([r[name] for r in rows]) == 1
    fold = np.array([r["fold"] for r in rows])
    thr = {int(k): threshold(np.array(v["scores"]), q) for k, v in full["label sets"][name]["calibration negatives"].items()}
    hit = s > np.array([thr[f] for f in fold])
    runs = {}
    for r in rows:
        if r["world"] == 0:
            runs.setdefault(r["task"], set()).add(r["run"])
    first2 = {t: sorted(v)[:2] for t, v in runs.items() if len(v) >= 2}
    same = sorted(t for t, (a, b) in first2.items() if "|overlap800_" in a and "|overlap800_" in b)
    which = {run: i for t in same for i, run in enumerate(first2[t])}
    ti = {t: i for i, t in enumerate(same)}
    out = {a: [np.zeros(len(same)), np.zeros(len(same))] for a in ARMS}
    for i, r in enumerate(rows):
        if not y[i] or r["task"] not in ti:
            continue
        a = ("real_run1", "real_run2")[which[r["run"]]] if r["world"] == 0 and r["run"] in which else "simulated" if r["world"] == 1 else None
        if a:
            out[a][0][ti[r["task"]]] += hit[i]; out[a][1][ti[r["task"]]] += 1
    return same, out


def plant(mons, arm, copies):
    """As the pooled lean test, at the margin: drop round(0.05 x positives) caught units from one arm of every monitor
    (M1, M2, M3; a fresh default_rng(copy) per monitor). mons: [(counts, W)]."""
    for cpy in range(copies):
        out = []
        for c, W in mons:
            g = np.random.default_rng(cpy); cc, pp = c[arm]; k = round(PLANT * pp.sum())
            units = np.repeat(np.arange(len(cc)), cc.astype(int))
            out.append(({**c, arm: (cc - np.bincount(g.choice(units, min(k, len(units)), replace=False), minlength=len(cc)), pp)}, W))
        yield out


def p_testable(pd, copies=200):
    return sum(math.comb(copies, j) * pd ** j * (1 - pd) ** (copies - j) for j in range(int(0.8 * copies), copies + 1))


def main(m1_p, recal_p):
    got = {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (m1_p, recal_p)}
    if got != SHA:
        raise SystemExit(f"inputs differ: {got}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    me = Path(__file__).resolve()
    if (subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode):
        raise SystemExit("the script differs from HEAD: commit it before running")
    full = json.loads(Path(recal_p).read_text())
    rng = np.random.default_rng(SEED)
    out = {"inputs sha256": got, "git": head, "prereg": "PREREG.md @ e8787b92", "targets": {}}
    for tname, q in TARGETS.items():
        m1 = m1_counts(m1_p, f"caught_{tname[:-1]}pct")
        T1 = len(m1["real_run1"][0])
        W1 = np.vstack([np.ones(T1), np.random.default_rng([SEED, 1]).poisson(1.0, (DRAWS, T1))])
        g1_rs, g1_rr = gap(*m1["real_run1"], *m1["simulated"], W1), gap(*m1["real_run1"], *m1["real_run2"], W1)
        sols = [sol_counts(full, n, q) for n in ("transfer", "destructive_raw")]
        T = len(sols[0][0])
        if T != 163 or sols[1][0] != sols[0][0]:
            raise SystemExit("the same-batch task sets differ from the pooled lean test's 163")
        ref = REF[tname]  # the committed pooled lean result's positives and point gaps
        mine = [m1] + [c for _, c in sols]
        for (name, (pos, rs, rr)), c in zip(ref.items(), mine):
            if ([int(c[a][1].sum()) for a in ARMS] != pos
                    or abs(c["real_run1"][0].sum() / c["real_run1"][1].sum() - c["simulated"][0].sum() / c["simulated"][1].sum() - rs) > 1e-9
                    or abs(c["real_run1"][0].sum() / c["real_run1"][1].sum() - c["real_run2"][0].sum() / c["real_run2"][1].sum() - rr) > 1e-9):
                raise SystemExit(f"{name} does not reproduce the pooled lean test at {tname}")
        W = np.vstack([np.ones(T + MAXK), np.random.default_rng([SEED, 23]).poisson(1.0, (DRAWS, T + MAXK))])
        res = {}
        for k in GRID:
            se_rs, se_rr, rhos, sets0 = [], [], [], None
            for j in range(SETS if k else 1):
                don = np.random.default_rng([SEED, 7, k, j]).integers(0, T, k)
                aug = [{a: (np.concatenate([c[a][0], c[a][0][don]]), np.concatenate([c[a][1], c[a][1][don]])) for a in ARMS} for _, c in sols]
                Wk = W[:, :T + k]
                G_rs = (g1_rs + sum(gap(*c["real_run1"], *c["simulated"], Wk) for c in aug)) / 3
                G_rr = (g1_rr + sum(gap(*c["real_run1"], *c["real_run2"], Wk) for c in aug)) / 3
                se_rs.append(G_rs[1:].var()); se_rr.append(G_rr[1:].var()); rhos.append(np.corrcoef(G_rs[1:], G_rr[1:])[0, 1])
                if j == 0:
                    sets0 = (aug, Wk, G_rs, G_rr)
            S_rs, S_rr, rho = float(np.sqrt(np.mean(se_rs))), float(np.sqrt(np.mean(se_rr))), float(np.mean(rhos))
            g = np.random.default_rng(SEED); z1, z2 = g.standard_normal(MC), g.standard_normal(MC)
            x, yv = S_rs * z1, S_rr * (rho * z1 + np.sqrt(1 - rho ** 2) * z2)
            p_both = float(((np.abs(x) < M - 1.96 * S_rs) & (np.abs(yv) < M - 1.96 * S_rr)).mean())
            aug, Wk, G_rs, G_rr = sets0
            pd = {}
            for arm, other, unpl in (("simulated", "simulated", G_rs), ("real_run2", "real_run2", G_rr)):
                hits = []
                for mons in plant([(m1, W1)] + [(c, Wk) for c in aug], arm, COPIES):
                    Gp = sum(gap(*c["real_run1"], *c[other], W_) for c, W_ in mons) / 3
                    hits.append(detected(float(Gp[0]), tuple(ci(Gp[1:])), float(unpl[0]), PLANT))
                pd[arm] = float(np.mean(hits))
            pt = p_testable(pd["simulated"]) * p_testable(pd["real_run2"])
            price_raw = k * (SIM_RUN + EXTRACT + DEVBOX + 2 * REAL_RUN) + sum(FIXED.values())
            price_shared = k * (SIM_RUN + EXTRACT + DEVBOX) + FIXED["simulated batch (assumed)"] + FIXED["probe reproduction"] + FIXED["scoring"]
            res[str(k)] = {"SE G_RS": S_rs, "SE G_RR": S_rr, "rho": rho, "P(both pass)": p_both,
                           "plant detected share (simulated, run 2)": [pd["simulated"], pd["real_run2"]], "P(plants testable)": pt,
                           "P(clear)": p_both * pt, "price raw": price_raw, "price with 30%": price_raw * 1.3,
                           "price shared with C's real runs, raw": price_shared, "with 30%": price_shared * 1.3}
            print(tname, k, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in res[str(k)].items() if kk in ("SE G_RS", "SE G_RR", "P(both pass)", "P(plants testable)", "P(clear)", "price with 30%")}, flush=True)
        first = next((int(k) for k, v in res.items() if v["P(clear)"] >= GOAL), None)
        out["targets"][tname] = {"by k": res, "smallest k with P(clear) >= 0.80": first,
                                 "outcome": (f"{first} new tasks" if first is not None else "too few even with spending (no k up to 274)")}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(dest := HERE / "results" / f"lean_sizing_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, separators=(",", ":")) + "\n")
    for t, r in out["targets"].items():
        print(t, r["outcome"])
    print(dest)


if __name__ == "__main__":
    main(*sys.argv[1:3])
