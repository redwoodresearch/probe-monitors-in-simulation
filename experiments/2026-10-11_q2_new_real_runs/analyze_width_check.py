"""Part A's "more real runs helps" check and sizing (PREREG.md @ a12f23b3, committed before this script). For the
data-transfer and destructive monitors: the per-task real - simulated SE against positive-bearing real runs per task
(caps 1, 2, 3, all, on a fixed task set; fit SE^2 = A + B m), the simulated side's leverage, the yardstick's SE with
donor pairs added, and the chance of a clear answer at +/-5, plant gates included. Poisson bootstrap (10,000 draws, default_rng(20261011)), thresholds re-estimated per draw.
Keys, labels and stored scores only; $0. A projection, not the analysis of record.

  uv run --no-project --with numpy python experiments/2026-10-11_q2_new_real_runs/analyze_width_check.py <recal_2026-10-10_225753.json>
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

HERE = Path(__file__).parent
SHA = "c9ba6dfe3bd028027e162ae386dc5c4ec5343eff5825d63c4ed94534b2edb9e7"
UNPAIRED_SHA = "b735a5e06016a59df327b1ad6b7b373209a8a1dfb891853d943cf9eb87414c3f"  # sorted 274 tasks, joined by newlines
SEED, DRAWS, CHUNK, SUBSETS, MC, M, Z_RS, Z_RR, GOAL = 20261011, 10_000, 2_000, 20, 200_000, 0.05, 2.24, 1.96, 0.80
MONITORS, TARGETS, CAPS, GRID = ("transfer", "destructive_raw"), (0.01, 0.05), (1, 2, 3, 99), list(range(0, 271, 10)) + [274]
RUN_RAW, BUF, BATCH, REPRO, REFIT = 1.00, 1.30, 1.11, 3.00, 3.00  # df's per run; once per batch; reproduction job; refit-and-score per monitor
COHORT = {"simulated tasks": 437, "matched real runs": 800, "same-batch pairs": 163, "tasks without a pair": 274}


def batch(run):
    return "overlap800" if "|overlap800_" in run else "base200" if "|base200_" in run else "other"


def thresholds(cs, w, q):
    """Per draw, the frozen convention on weighted negatives (as Q2.5 and #718): sorted high to low, index floor(q n)."""
    order = np.argsort(-cs, kind="stable"); cs, w = cs[order], w[:, order].astype(np.float64)
    c = np.cumsum(w, axis=1); idx = np.floor(q * c[:, -1])
    return cs[np.argmax(c > idx[:, None], axis=1)]


def p_testable(c, n, se, copies=200):
    """PREREG step 4's plant gate: per-copy shifts S from flipping caught positives (c of n per task) with probability
    pi = 0.05 / mean rate (or missed ones when pi > 1); detected when 0.025 <= |S| <= 0.075 and |S| > 1.96 se. Returns
    P(Binomial(200, p_d) >= 160). c, n: arrays (sets x tasks) of the measure's tasks."""
    c, n = np.atleast_2d(c), np.atleast_2d(n)
    rate = (c / n).mean(1, keepdims=True); flip_missed = rate < M
    pi = np.where(flip_missed, M / (1 - rate), M / rate)
    base = np.where(flip_missed, n - c, c).astype(int)
    g = np.random.default_rng(SEED)
    S = (g.binomial(np.repeat(base[None], copies, 0), np.repeat(pi[None], copies, 0)) / n).mean(2)  # copies x sets
    pd = float(((S >= 0.5 * M) & (S <= 1.5 * M) & (S > 1.96 * se)).mean())
    return sum(math.comb(copies, j) * pd ** j * (1 - pd) ** (copies - j) for j in range(int(0.8 * copies), copies + 1)), pd


def p_clear(se_rs, se_rr, rho, d_rs=0.0, d_rr=0.0, fails=False):
    """Monte Carlo chance of a clear verdict (PREREG step 4): at true gaps of 0, both pass; at today's points (fails=True),
    also real-sim failing with the yardstick passing, or the yardstick failing. Plus each comparison's pass probability."""
    g = np.random.default_rng(SEED); z1, z2 = g.standard_normal(MC), g.standard_normal(MC)
    x = d_rs + se_rs * z1; y = d_rr + se_rr * (rho * z1 + np.sqrt(1 - rho ** 2) * z2)
    rs_pass = np.abs(x) < M - Z_RS * se_rs; rr_pass = np.abs(y) < M - Z_RR * se_rr
    rs_fail = np.abs(x) > M + Z_RS * se_rs; rr_fail = np.abs(y) > M + Z_RR * se_rr
    clear = (rs_pass & rr_pass) | ((rs_fail & rr_pass) | rr_fail if fails else False)
    return {"P(clear)": float(clear.mean()), "P(real-sim passes)": float(rs_pass.mean()), "P(yardstick passes)": float(rr_pass.mean())}


def main(path):
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if sha != SHA:
        raise SystemExit(f"input differs from PREREG's: {sha}")
    me = Path(__file__).resolve()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if (subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode):
        raise SystemExit("the script differs from HEAD: commit it before running")
    full = json.loads(Path(path).read_text()); rows = full["rows"]
    world = np.array([r["world"] for r in rows]); fold = np.array([r["fold"] for r in rows]); task = np.array([r["task"] for r in rows])
    run = np.array([r["run"] for r in rows])
    sim_tasks = set(task[world == 1])
    runs = {}
    for r in rows:
        if r["world"] == 0:
            runs.setdefault(r["task"], set()).add(r["run"])
    first2 = {t: sorted(v)[:2] for t, v in runs.items() if len(v) >= 2}
    same = sorted(t for t, (a, b) in first2.items() if batch(a) == batch(b) == "overlap800")
    unpaired = sorted(sim_tasks - set(same))
    matched = (world == 0) & np.array([batch(x) == "overlap800" for x in run]) & np.isin(task, list(sim_tasks))
    mruns = {}
    for i in np.flatnonzero(matched):
        mruns.setdefault(task[i], set()).add(run[i])
    cohort = {"simulated tasks": len(sim_tasks), "matched real runs": sum(map(len, mruns.values())),
              "same-batch pairs": len(same), "tasks without a pair": len(unpaired)}
    if cohort != COHORT or hashlib.sha256("\n".join(unpaired).encode()).hexdigest() != UNPAIRED_SHA:
        raise SystemExit(f"cohort differs from PREREG: {cohort}")
    names = sorted(set(task)); T = len(names); ti = np.searchsorted(names, task); tix = {t: i for i, t in enumerate(names)}
    in_sim = np.isin(np.array(names), list(sim_tasks)); in_unpaired = np.isin(np.array(names), unpaired)
    rk = np.zeros(len(rows), int)
    for t in same:
        rk[(run == first2[t][0])] = 1; rk[(run == first2[t][1])] = 2
    W = np.ones((DRAWS + 1, T + len(unpaired)), np.float32)
    W[1:] = np.random.default_rng(SEED).poisson(1.0, (DRAWS, T + len(unpaired)))
    Wt = W[:, :T]
    print(f"frozen unpaired list: {len(unpaired)} tasks, sha256 {UNPAIRED_SHA}")
    out = {"input sha256": sha, "git": head, "prereg": "PREREG.md @ a12f23b3", "cohort": cohort, "unpaired sha256": UNPAIRED_SHA,
           "draws": DRAWS, "bootstrap": "Poisson(1) task weights", "monitors": {}}
    for name in MONITORS:
        y = np.array([r[name] for r in rows]) == 1; s = np.array(full["label sets"][name]["test scores"], float)
        cal = {int(k): (np.array(v["rows"]), np.array(v["scores"])) for k, v in full["label sets"][name]["calibration negatives"].items()}
        P = np.flatnonzero((matched | (world == 1)) & y)
        res = {"positives": {"matched real": int((matched & y).sum()), "simulated": int(((world == 1) & y).sum()),
                             "run 1": int(((rk == 1) & y).sum()), "run 2": int(((rk == 2) & y).sum())},
               "matched positives per run": float((matched & y).sum() / cohort["matched real runs"])}
        for q in TARGETS:
            thr = np.vstack([np.stack([thresholds(cal[k][1], W[lo:lo + CHUNK][:, ti[cal[k][0]]], q) for k in range(5)], axis=1)
                             for lo in range(0, DRAWS + 1, CHUNK)])
            hit = (s[P][None, :] > thr[:, fold[P]]).astype(np.float32)  # draws x positives
            pos_of = {i: j for j, i in enumerate(P)}

            def counts(sel):  # per-draw caught and per-task positives over rows sel (indices into rows)
                cols = np.array([pos_of[i] for i in sel], int)
                A = np.zeros((len(sel), T), np.float32); A[np.arange(len(sel)), ti[sel]] = 1
                return hit[:, cols] @ A, A.sum(0)

            def gap(real_sel, sim_sel, keep):
                cr, nr = counts(real_sel); cs, ns = counts(sim_sel)
                both = keep & (nr > 0) & (ns > 0)
                d = np.where(both, cr / np.maximum(nr, 1) - cs / np.maximum(ns, 1), 0)
                den = (Wt * both).sum(1)
                if (den == 0).any():
                    raise SystemExit("a bootstrap draw has no eligible task")
                return (Wt * d).sum(1) / den, int(both.sum()), both

            real_all = np.flatnonzero(matched & y); sim_all = np.flatnonzero((world == 1) & y)
            g_rs, T_all, both_all = gap(real_all, sim_all, in_sim)
            pruns = {}  # positive-bearing matched runs per task
            for i in real_all:
                pruns.setdefault(task[i], set()).add(run[i])
            Rp = np.array([len(pruns.get(t, ())) for t in names], float)
            share_pos = sum(map(len, pruns.values())) / cohort["matched real runs"]
            pts = []
            for c in CAPS:  # step 1: positive-bearing real runs per task cut to at most c; the task set stays fixed
                for j in range(SUBSETS if c < 99 else 1):
                    g = np.random.default_rng([SEED, c, j])
                    keep_runs = {t: set(sorted(v)) if c >= len(v) else set(g.choice(sorted(v), c, replace=False)) for t, v in sorted(pruns.items())}
                    sel = real_all[[run[i] in keep_runs[task[i]] for i in real_all]]
                    gc, Tc, both = gap(sel, sim_all, in_sim)
                    if Tc != T_all:
                        raise SystemExit(f"cap {c}: the task set changed ({Tc} vs {T_all})")
                    Rk = np.array([len(keep_runs.get(t, ())) for t in names], float)
                    pts.append({"cap": c, "subset": j, "T": Tc, "SE2": float(gc[1:].var()), "m": float((1 / Rk[both]).mean())})
            per_cap = [[float(np.mean([p[f] for p in pts if p["cap"] == c])) for f in ("m", "SE2")] for c in CAPS]
            X = np.array([[1.0, m_] for m_, _ in per_cap]); Y = np.array([v for _, v in per_cap])  # one point per cap (subset means)
            (A_, B_), *_ = np.linalg.lstsq(X, Y, rcond=None)
            identified = bool(A_ > 0 and B_ > 0)
            se_rs_now = float(g_rs[1:].std()); se_floor = float(np.sqrt(A_)) if identified else None
            cr0, nr0 = counts(sim_all)  # the real-sim plant acts on today's simulated positives, which don't change
            cs_pl, ns_pl = cr0[0][both_all], nr0[both_all]
            half = []
            for j in range(SUBSETS):  # step 2: the simulated side's leverage
                keep = np.random.default_rng([SEED, 99, j]).random(len(sim_all)) < 0.5
                half.append(float(gap(real_all, sim_all[keep], in_sim)[0][1:].var()))
            # step 3: the yardstick, with donor pairs added as new tasks
            c1, n1 = counts(np.flatnonzero((rk == 1) & y)); c2, n2 = counts(np.flatnonzero((rk == 2) & y))
            si = np.array([tix[t] for t in same])
            b = (n1[si] > 0) & (n2[si] > 0)
            d = np.where(b, c1[:, si] / np.maximum(n1[si], 1) - c2[:, si] / np.maximum(n2[si], 1), 0)
            num0, den0 = (Wt[:, si] * d).sum(1), (Wt[:, si] * b).sum(1)
            g_rr = num0 / den0
            rr_curve, rr_plant = {}, {}
            c2_0, n2_0 = c2[0][si], n2[si]
            for k in GRID:
                se2, sets_c, sets_n = [], [], []
                for j in range(SUBSETS if k else 1):
                    don = np.random.default_rng([SEED, 7, k, j]).integers(0, len(same), k)
                    Wk = W[:, T:T + k]
                    se2.append(float(((num0 + (Wk * d[:, don]).sum(1)) / (den0 + (Wk * b[don]).sum(1)))[1:].var()))
                    keep = np.concatenate([np.flatnonzero(b), don[b[don]]])
                    sets_c.append(c2_0[keep]); sets_n.append(n2_0[keep])
                rr_curve[k] = float(np.sqrt(np.mean(se2)))
                rr_plant[k] = (sets_c, sets_n)

            def gates(se_rs_, k):
                t_rs, pd_rs = p_testable(cs_pl, ns_pl, se_rs_)
                ts = [p_testable(cc, nn, rr_curve[k]) for cc, nn in zip(*rr_plant[k])]
                return t_rs, float(np.mean([x[0] for x in ts])), pd_rs, float(np.mean([x[1] for x in ts]))

            def clear_at(se_rs_, k, d_rs=0.0, d_rr=0.0, fails=False):
                v = p_clear(se_rs_, rr_curve[k], rho, d_rs, d_rr, fails); t_rs, t_rr, pd_rs, pd_rr = gates(se_rs_, k)
                return v | {"P(plant testable: real-sim, yardstick)": [t_rs, t_rr], "plant detected share (real-sim, yardstick)": [pd_rs, pd_rr],
                            "P(verdict clear)": v["P(clear)"], "P(clear)": v["P(clear)"] * t_rs * t_rr}
            rho = float(np.corrcoef(g_rs[1:], g_rr[1:])[0, 1])
            r = {"real - simulated (per task)": {"tasks": T_all, "point": float(g_rs[0]), "SE now": se_rs_now,
                                                 "curve points (cap, subset, T, SE2, m)": [[p["cap"], p["subset"], p["T"], p["SE2"], p["m"]] for p in pts],
                                                 "fit T SE2 = A + B m": {"A": float(A_), "B": float(B_), "identified": identified},
                                                 "SE with unlimited real runs": se_floor,
                                                 "SE with half the simulated positives (descriptive)": float(np.sqrt(np.mean(half)))},
                 "real - real (per task, same-batch pairs)": {"tasks": int(b.sum()), "point": float(g_rr[0]), "SE now": float(g_rr[1:].std()),
                                                              "share of pairs with positives in both runs": float(b.mean()),
                                                              "SE with k donor pairs": {str(k): v for k, v in rr_curve.items()}},
                 "correlation of the two estimates": rho, "share of matched runs with positives": share_pos,
                 "today": clear_at(se_rs_now, 0)}
            if identified:
                best = clear_at(se_floor, 274)
                r["best case (unlimited real runs; 274 pairs)"] = best
                r["best case at today's points"] = clear_at(se_floor, 274, float(g_rs[0]), float(g_rr[0]), fails=True)
                parts = {"real - simulated passes (unlimited real runs)": best["P(real-sim passes)"],
                         "yardstick passes (274 pairs)": best["P(yardstick passes)"],
                         "real - simulated plant testable": best["P(plant testable: real-sim, yardstick)"][0],
                         "yardstick plant testable": best["P(plant testable: real-sim, yardstick)"][1]}
                limits = [k_ for k_, v_ in parts.items() if v_ < GOAL] or ["the joint chance (each part alone reaches 0.80)"]
                few = T_all < 20 or int(b.sum()) < 20
                r["outcome"] = ("too few: under 20 tasks" if few else "sized below" if best["P(clear)"] >= GOAL else
                                "too few even with spending; limited by: " + "; ".join(limits))
                if best["P(clear)"] >= GOAL and not few:
                    elig_un = both_all & in_unpaired; sp = share_pos
                    after = (1 - sp) ** 2 / np.maximum(Rp, 1) + 2 * sp * (1 - sp) / (Rp + 1) + sp ** 2 / (Rp + 2)  # E[1/R+] with two new runs
                    r["curve (k: P(clear))"] = {}
                    for k in GRID:
                        f = k / len(unpaired)
                        inv = np.where(elig_un, (1 - f) / np.maximum(Rp, 1) + f * after, 1 / np.maximum(Rp, 1))[both_all]
                        pc = clear_at(float(np.sqrt(A_ + B_ * inv.mean())), k)
                        r["curve (k: P(clear))"][str(k)] = pc["P(clear)"]
                        if pc["P(clear)"] >= GOAL and "sizing" not in r:
                            r["sizing"] = {"new tasks": k, "new runs": 2 * k, **pc, "price raw": 2 * k * RUN_RAW + BATCH + REPRO + 2 * REFIT,
                                           "price with 30%": 2 * k * RUN_RAW * BUF + (BATCH + REPRO + 2 * REFIT) * BUF}
                    r.setdefault("sizing", "no grid k reaches 0.80")
            else:
                r["outcome"] = "unidentified split (A or B not positive): not sized"
            res[f"target {int(100 * q)}%"] = r
        out["monitors"][name] = res
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(dest := HERE / "results" / f"width_check_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, separators=(",", ":")) + "\n")
    for name, res in out["monitors"].items():
        print(name, res["positives"], f"{res['matched positives per run']:.3f} a run")
        for q in TARGETS:
            r = res[f"target {int(100 * q)}%"]; a = r["real - simulated (per task)"]; b_ = r["real - real (per task, same-batch pairs)"]
            fl = a["SE with unlimited real runs"]
            print(f"  {int(100 * q)}%: real-sim {100 * a['point']:+.1f} (T {a['tasks']}) SE {100 * a['SE now']:.2f} -> unlimited real "
                  f"{'n/a' if fl is None else f'{100 * fl:.2f}'} (fit {a['fit T SE2 = A + B m']}); half sim {100 * a['SE with half the simulated positives (descriptive)']:.2f}")
            print(f"       real-real {100 * b_['point']:+.1f} (T {b_['tasks']}) SE {100 * b_['SE now']:.2f} -> 274 pairs {100 * b_['SE with k donor pairs']['274']:.2f}; rho {r['correlation of the two estimates']:.2f}")
            print(f"       today {r['today']}; best {r.get('best case (unlimited real runs; 274 pairs)')}; -> {r['outcome']}" + (f"; sizing {r['sizing']}" if "sizing" in r else ""))
    print(dest)


if __name__ == "__main__":
    main(sys.argv[1])
