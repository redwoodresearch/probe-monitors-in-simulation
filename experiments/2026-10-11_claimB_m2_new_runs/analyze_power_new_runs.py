"""Power for confirming H2's catch-at-1% lead (M2 19.7% vs step A's A0 2.7%, fixed sim vs e1, test environments) on
new runs, written before any new run exists ($0, laptop; reads H2's saved test scores, numbers only).

From h2_summary_2026-10-11_132926.json's inputs: the real-label column of M2 (best found, picked in all 5 folds) and A0
on the 473 test-environment cells. Two calibration designs:
- (ii) fixed threshold: each model's threshold is the upper 1% point of a large held-out real set (emulated here by the
  473 e1 test runs); the new real runs only check the achieved false-alarm rate. Positives: n new sims drawn with
  replacement from the observed (M2 caught, A0 caught) pairs, the M2-only share thinned so the true gap is k times the
  observed one (k = 1, 0.5, 0.25, 0).
- (i) new reals only: n new sims and n new reals drawn with replacement from the observed runs; thresholds by the 1%
  table's two-way cross-fit on the new reals (analyze_tpr_at_fpr.crossfit), so threshold noise is included.
Rule read per draw (frozen in PLAN.md), on the paired catch flags of the n new sims: pass = the gap's one-sided 95%
lower bound > 0; fail = the gap's two-sided 95% upper bound < 0.0845 (half the test-environment +16.9 pts); otherwise
too few (pass is checked first; both cannot hold when the lower bound is above 0 and the upper bound below 0.0845 only
if the gap is small and precise, which the rule reads as pass). Bounds: Agresti-Min adjusted Wald for a paired
difference (0.5 added to each cell), which keeps width when discordant pairs are few.
Scenarios for (ii): the test-environment joint shares thinned to k x the observed gap (k = 1, 0.5, 0.25, 0), and the
dev-environment result (M2 23.6%, A0 22.7%, gap +0.9 pts; h2_devcheck_2026-10-11_144025.json @ 5baf7509) at two assumed
overlaps (both-caught share 0.15 or 0.20), since the dev check saved no joint flags.
(i) samples whole cells (a cell's sim and its real e1 together) and cross-fits by cell, with flags computed per run.
4,000 draws per cell (1,000 for (i)), default_rng(20261011).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with scikit-learn --with scipy \\
      --with pyyaml python experiments/2026-10-11_claimB_m2_new_runs/analyze_power_new_runs.py
"""

import hashlib
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np

HERE = Path(__file__).parent
H2 = HERE.parent / "2026-10-11_claimB_h2_probe_best_found"
sys.path.insert(0, str(H2))
import analyze_h2_search as A  # noqa: E402

SUMMARY = H2 / "results" / "h2_summary_2026-10-11_132926.json"
NS, KS, DRAWS, HALF_GAP = (55, 110, 165, 220, 330), (1.0, 0.5, 0.25, 0.0), 4000, 0.0845


def verdict(n11, n10, n01, n00):
    """Agresti-Min: add 0.5 to each cell of the paired 2x2, then Wald for p10 - p01."""
    a, b, c, d = n11 + 0.5, n10 + 0.5, n01 + 0.5, n00 + 0.5
    m = a + b + c + d
    g = (b - c) / m
    se = np.sqrt(((b + c) / m - g**2) / m)
    return "pass" if g - 1.645 * se > 0 else "fail" if g + 1.96 * se < HALF_GAP else "too few"


def threshold_flags(s_neg, s_pos):
    """Upper 1% point of the calibration reals (the 1% table's threshold), and the positives above it."""
    t = A.T.threshold(s_neg, 1)[0]
    return s_pos > t, s_neg


def main():
    summary = json.loads(SUMMARY.read_text())
    vol = modal.Volume.from_name(A.H.VOL, version=2)
    read = lambda p: b"".join(vol.read_file(p))  # noqa: E731
    folds_file = json.loads((A.CLAIMB / "splits/folds_v2.json").read_text())
    cfg = {"packs": {"level1": "39f842231a46"}, "seed": 0, "bootstrap": 1000, "tag": "h2",
           "main_task_to_fold": folds_file["task_folds"]["main_task_to_fold"], "environment_to_fold": folds_file["environment_folds"]["environment_to_fold"]}
    runs = json.loads(read(f"{A.H.PACK_A}/runs.L63.json"))
    env = np.array([cfg["environment_to_fold"][m["environment"]] for m in runs])
    folds = sorted({cfg["environment_to_fold"][e] for e in A.H.TEST_ENVS})
    comp = "fixed_vs_e1"
    sc, Y = A.labels(cfg, runs, comp)
    test = sc.runs_in & np.isin(env, folds)
    per = {}
    for name in summary["eligible"]:
        raw = read(f"{A.H.OUT}/{comp}/{name}.npz")
        if hashlib.sha256(raw).hexdigest() != summary["inputs"][f"{comp}/{name}.npz"]:
            raise SystemExit(f"{comp}/{name}.npz changed since the summary")
        z = np.load(io.BytesIO(raw))
        per[name] = (z["val_auroc"], z["test_run_score"])
    sel = lambda cfgs: A.select(np.concatenate([per[n][0] for n in cfgs]), np.concatenate([per[n][1] for n in cfgs]), env, folds)[:, 0]  # noqa: E731
    m2, a0, y, task = sel(summary["eligible"])[test], sel(["A0"])[test], Y[test, 0], sc.main[test]
    neg, pos = ~y, y
    out = {"label": "power for confirming M2's catch-at-1% lead on new runs (planning; no new data)", "observed": {}, "design_ii_fixed_threshold": {},
           "design_i_new_reals_only": {}}
    # design (ii): thresholds from the held-out real set (here the 473 e1 test runs), fixed
    t_m2, t_a0 = A.T.threshold(m2[neg], 1)[0], A.T.threshold(a0[neg], 1)[0]  # the 1% table's threshold, one fit on all 473 reals
    cm, ca = m2[pos] > t_m2, a0[pos] > t_a0
    p11, p10, p01 = float((cm & ca).mean()), float((cm & ~ca).mean()), float((~cm & ca).mean())
    out["observed"] = {"positives": int(pos.sum()), "negatives": int(neg.sum()), "m2_catch": float(cm.mean()), "a0_catch": float(ca.mean()),
                       "fpr_m2": float((m2[neg] > t_m2).mean()), "fpr_a0": float((a0[neg] > t_a0).mean()), "p11": p11, "p10": p10, "p01": p01}
    rng = np.random.default_rng(20261011)
    scen = {f"test_k{k}": (p11, p01 + k * (p10 - p01), p01) for k in KS}
    for ov in (0.15, 0.20):  # dev: M2 0.236, A0 0.227
        scen[f"dev_overlap{ov}"] = (ov, 0.236 - ov, 0.227 - ov)
    for name, (q11, q10, q01) in scen.items():
        probs = np.array([q11, q10, q01, 1 - q11 - q10 - q01])
        for n in NS:
            c = rng.multinomial(n, probs, size=DRAWS)
            v = [verdict(*c[i]) for i in range(DRAWS)]
            out["design_ii_fixed_threshold"][f"{name}_n{n}"] = {v_: v.count(v_) / DRAWS for v_ in ("pass", "fail", "too few")} | {"true_gap": q10 - q01}
    # design (i): thresholds from the new reals only, the 1% table's two-way cross-fit
    cell = np.array([runs[i]["cell"] for i in np.flatnonzero(test)])
    cells = sorted(set(cell[pos]) & set(cell[neg]))
    sim_of = {c_: i for i, c_ in enumerate(cell) if pos[i]}
    real_of = {c_: i for i, c_ in enumerate(cell) if neg[i]}
    for n in NS:
        res = {"pass": 0, "fail": 0, "too few": 0, "fpr_m2": [], "fpr_a0": []}
        for _ in range(1000):
            draw = rng.choice(len(cells), n)
            uniq, cnt = np.unique(draw, return_counts=True)  # copies of a cell stay in one half, as crossfit keeps them
            order = rng.permutation(len(uniq))
            cut = (len(uniq) + 1) // 2
            halves = tuple(np.repeat(uniq[order[sl]], cnt[order[sl]]) for sl in (slice(0, cut), slice(cut, None)))
            flags, fa = {"m2": [], "a0": []}, {"m2": 0, "a0": 0}
            for cal, ev in ((0, 1), (1, 0)):
                ci, ce = [cells[j] for j in halves[cal]], [cells[j] for j in halves[ev]]
                for nm, s_ in (("m2", m2), ("a0", a0)):
                    f_pos, _ = threshold_flags(s_[[real_of[c_] for c_ in ci]], s_[[sim_of[c_] for c_ in ce]])
                    t = A.T.threshold(s_[[real_of[c_] for c_ in ci]], 1)[0]
                    flags[nm].append(f_pos)
                    fa[nm] += int((s_[[real_of[c_] for c_ in ce]] > t).sum())
            fm, fa0 = np.concatenate(flags["m2"]), np.concatenate(flags["a0"])
            res[verdict(int((fm & fa0).sum()), int((fm & ~fa0).sum()), int((~fm & fa0).sum()), int((~fm & ~fa0).sum()))] += 1
            res["fpr_m2"].append(fa["m2"] / n)
            res["fpr_a0"].append(fa["a0"] / n)
        out["design_i_new_reals_only"][f"n{n}"] = {"pass": res["pass"] / 1000, "fail": res["fail"] / 1000, "too few": res["too few"] / 1000,
                                                   "achieved_fpr_m2_p5_p95": [float(np.percentile(res["fpr_m2"], 5)), float(np.percentile(res["fpr_m2"], 95))],
                                                   "achieved_fpr_a0_p5_p95": [float(np.percentile(res["fpr_a0"], 5)), float(np.percentile(res["fpr_a0"], 95))]}
    out |= {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "summary": SUMMARY.name}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"power_new_runs_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out["observed"], indent=1))
    for name in scen:
        print("ii", name, {n: {v: round(out["design_ii_fixed_threshold"][f"{name}_n{n}"][v], 3) for v in ("pass", "fail", "too few")} for n in NS})
    for n in NS:
        print("i n", n, out["design_i_new_reals_only"][f"n{n}"])
    print("->", dst)


if __name__ == "__main__":
    main()
