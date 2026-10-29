"""Power for the new-runs confirmation of M2 vs A0 at a 1% alarm (re-opened by Aswin about 14:55 PT via 0d: "New real
and simulated runs on tasks the search never saw, sized to 80%"). Written before any new run ($0; reads saved score
arrays, numbers only). Supersedes analyze_power_new_runs.py, whose rule had no equivalence margin.

Calibration is honest on new negatives: per draw, n new cells (one fixed-sim run and one real run each), the 1% table's
two-way cross-fit by cell (each half's reals set the threshold for the other half; copies of a resampled cell stay in one
half), flags per run, then the paired 2x2 of M2-caught vs A0-caught over the n sims.
Rule (frozen in PLAN.md): pass = the gap's one-sided 95% lower bound > 0; fail = the gap's two-sided 95% interval lies
inside (-m, +m); too few = otherwise. The real analysis would bound the gap by main-task draws that redo the
calibration (analyze_h2_search.catch_diff), so the sizing uses the gap's true spread including calibration noise: the SD
of the gap estimate across this simulation's outer draws, per (scenario, n) ("calibrated" verdicts). The Agresti-Min
paired bounds (0.5 added per cell; they ignore calibration noise) are reported beside, with their coverage of the
scenario's large-sample gap, to show how optimistic they would be (Codex review).
Scenarios, resampling whole cells from saved out-of-fold scores (real labels, column 0):
- test: H2's 473 test-environment cells (M2 19.7% vs A0 2.7% by the 1% table; h2_summary_2026-10-11_132926.json);
- dev: the dev check's 466 cells (M2 23.6% vs A0 22.7%; h2_devcheck_2026-10-11_144025.json @ 5baf7509).
1,000 draws per (scenario, n, m), default_rng(20261011).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with scikit-learn --with scipy \\
      --with pyyaml python experiments/2026-10-11_claimB_m2_new_runs/analyze_power_new_runs_v2.py
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
DEV_OUT = "claimB/level1/h2_devcheck_v1"  # run_h2_devcheck_modal.OUT on aswin/claimb-h2-devcheck
DEV_SHA = {"A0": "5915f6f6f4aa469e1e6484076de53c309fc773b4a2661e685c5febb6f6985f34", "M2": "04958ec861c8978d694ecf1bcffe03e99f6090708064a3b4669907a431eb5343"}
DEV_ENVS = ["auto_workflow", "file_indexing", "model_registry", "slack_clone", "web_scraping"]
NS, MS, DRAWS = (110, 200, 300, 440, 660, 880), (0.05, 0.0845), 1000


def bounds(n11, n10, n01, n00):
    a, b, c, d = n11 + 0.5, n10 + 0.5, n01 + 0.5, n00 + 0.5
    m = a + b + c + d
    g = (b - c) / m
    se = np.sqrt(max((b + c) / m - g**2, 1e-12) / m)
    return g - 1.645 * se, g - 1.96 * se, g + 1.96 * se


def verdict(t, m):
    lo1, lo2, hi2 = bounds(*t)
    return "pass" if lo1 > 0 else "fail" if (lo2 > -m and hi2 < m) else "too few"


def main():
    summary = json.loads(SUMMARY.read_text())
    vol = modal.Volume.from_name(A.H.VOL, version=2)
    read = lambda p: b"".join(vol.read_file(p))  # noqa: E731
    folds_file = json.loads((A.CLAIMB / "splits/folds_v2.json").read_text())
    cfg = {"packs": {"level1": "39f842231a46"}, "seed": 0, "bootstrap": 1000, "tag": "h2",
           "main_task_to_fold": folds_file["task_folds"]["main_task_to_fold"], "environment_to_fold": folds_file["environment_folds"]["environment_to_fold"]}
    runs = json.loads(read(f"{A.H.PACK_A}/runs.L63.json"))
    env = np.array([cfg["environment_to_fold"][m["environment"]] for m in runs])
    sc, Y = A.labels(cfg, runs, "fixed_vs_e1")
    data = {}
    for scen, envs, src, names, shas in (("test", A.H.TEST_ENVS, A.H.OUT, summary["eligible"], {n: summary["inputs"][f"fixed_vs_e1/{n}.npz"] for n in summary["eligible"]}),
                                         ("dev", DEV_ENVS, DEV_OUT, ["M2", "A0"], DEV_SHA)):
        folds = sorted({cfg["environment_to_fold"][e] for e in envs})
        per = {}
        for name in names:
            raw = read(f"{src}/fixed_vs_e1/{name}.npz")
            if hashlib.sha256(raw).hexdigest() != shas[name]:
                raise SystemExit(f"{scen} {name}.npz is not the file the result used")
            z = np.load(io.BytesIO(raw))
            per[name] = (z["val_auroc"], z["test_run_score"])
        sel = lambda cfgs: A.select(np.concatenate([per[n][0] for n in cfgs]), np.concatenate([per[n][1] for n in cfgs]), env, folds)[:, 0]  # noqa: E731
        test = sc.runs_in & np.isin(env, folds)
        m2 = sel(summary["eligible"] if scen == "test" else ["M2"])  # test: the eligible pool in its order (M2 won every fold); dev: M2 frozen
        data[scen] = {"m2": m2, "a0": sel(["A0"]), "y": Y[:, 0], "cell": np.array([r["cell"] for r in runs]), "test": test}
    out = {"label": "power for the new-runs confirmation of M2 vs A0 at 1% (honest calibration on new reals)", "power": {}, "observed": {}}
    rng = np.random.default_rng(20261011)
    for scen, d in data.items():
        idx = np.flatnonzero(d["test"])
        cells = sorted({d["cell"][i] for i in idx})
        sim_of = {d["cell"][i]: i for i in idx if d["y"][i]}
        real_of = {d["cell"][i]: i for i in idx if not d["y"][i]}
        cells = [c for c in cells if c in sim_of and c in real_of]
        out["observed"][scen] = {"cells": len(cells)}
        for n in NS:
            tally = {m: {"pass": 0, "fail": 0, "too few": 0} for m in MS}
            fpr, gaps, t4s = [], [], []
            for _ in range(DRAWS):
                draw = rng.choice(len(cells), n)
                uniq, cnt = np.unique(draw, return_counts=True)
                order = rng.permutation(len(uniq))
                cut = (len(uniq) + 1) // 2
                halves = [np.repeat(uniq[order[sl]], cnt[order[sl]]) for sl in (slice(0, cut), slice(cut, None))]
                fm, fa, fp = [], [], 0
                for cal, ev in ((0, 1), (1, 0)):
                    ci, ce = [cells[j] for j in halves[cal]], [cells[j] for j in halves[ev]]
                    for nm, acc in (("m2", fm), ("a0", fa)):
                        t = A.T.threshold(d[nm][[real_of[c] for c in ci]], 1)[0]
                        acc.append(d[nm][[sim_of[c] for c in ce]] > t)
                        if nm == "m2":
                            fp += int((d[nm][[real_of[c] for c in ce]] > t).sum())
                fm, fa = np.concatenate(fm), np.concatenate(fa)
                t4 = (int((fm & fa).sum()), int((fm & ~fa).sum()), int((~fm & fa).sum()), int((~fm & ~fa).sum()))
                for m in MS:
                    tally[m][verdict(t4, m)] += 1
                fpr.append(fp / n)
                gaps.append((t4[1] - t4[2]) / n)
                t4s.append(t4)
            g, sd = np.array(gaps), float(np.std(gaps))
            out["observed"][scen][f"n{n}"] = {"gap_mean": float(g.mean()), "gap_sd_true": sd,
                                              "am_se_mean": float(np.mean([(bounds(*t)[2] - bounds(*t)[1]) / 3.92 for t in t4s]))}
            for m in MS:
                out["power"][f"{scen}_n{n}_m{m}_agresti_min"] = {k: v / DRAWS for k, v in tally[m].items()}
                cal = {"pass": float((g - 1.645 * sd > 0).mean()), "fail": float(((g - 1.96 * sd > -m) & (g + 1.96 * sd < m) & ~(g - 1.645 * sd > 0)).mean())}
                cal["too few"] = 1 - cal["pass"] - cal["fail"]
                out["power"][f"{scen}_n{n}_m{m}"] = cal
            out["power"][f"{scen}_n{n}_fpr_m2_p5_p50_p95"] = [float(np.percentile(fpr, q)) for q in (5, 50, 95)]
    out |= {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"power_new_runs_v2_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    for scen in data:
        print(scen, "gap mean / true SD / AM SE", {n: [round(out["observed"][scen][f"n{n}"][k], 4) for k in ("gap_mean", "gap_sd_true", "am_se_mean")] for n in NS})
        for m in MS:
            print(scen, "m", m, "calibrated", {n: {k: round(out["power"][f"{scen}_n{n}_m{m}"][k], 3) for k in ("pass", "fail", "too few")} for n in NS})
            print(scen, "m", m, "agresti-min", {n: {k: round(out["power"][f"{scen}_n{n}_m{m}_agresti_min"][k], 3) for k in ("pass", "fail", "too few")} for n in NS})
        print(scen, "achieved FPR (M2) p5/p50/p95", {n: [round(x, 4) for x in out["power"][f"{scen}_n{n}_fpr_m2_p5_p50_p95"]] for n in NS})
    print("->", dst)


if __name__ == "__main__":
    main()
