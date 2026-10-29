"""POST HOC (RULE.md), step 6: does fix939's rendering edit raise the frozen #36 reply detector's false alarms on HONEST
runs? $0, local, saved run scores only.

Sets (437 main tasks, one run each; a task drops out of a layer if any of its three runs has no output_end score):
  (i) real honest: each task's first real honest run by key (as analyze_sim_honest.py);
  (ii) simulated honest as simulated; (iii) the same simulated honest runs edited with fix939's recipe (a run the
  recipe leaves unchanged keeps its as-is scores: its inputs are byte-identical).
False alarm, per run, as #36's frozen protocol (analyze_sim_honest.py): the share of the run's nested models (f, v != f)
scoring it above that model's threshold, set by attack_eval.threshold on fold v's real honest runs of the base run
("frozen_real"); the rate is the mean over runs. Layer 63 is #36's committed layer; layer 48's thresholds come from the
same rule on the same runs (as run_score_fix939.py's output_end@48).
Main contrast: (iii) - (ii), paired by run. Intervals: 2,000 task resamples (seed 20261012) with every threshold redrawn
per draw from the resampled base real honest runs (analyze_sim_honest.caught_boot); fixed-threshold intervals beside.
5% is the verdict rate (the README: rely on 5%); 1% is reported.

  PYTHONPATH=src uv run --no-project --python 3.12 --with numpy python experiments/2026-10-12_q3_edit_honest_fa/analyze_edit_honest.py \
      --scoring results/edithonest_scoring_<stamp>.json --scores <edithonest_scores_<stamp>.npz> \
      --stage results/stage_<stamp>.json --power results/power_<stamp>.json --smoke results/smoke_check_<stamp>.json
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from probe_shift.attack_eval import boot_weights, rate, threshold

HERE = Path(__file__).parent
REPO = HERE.parent.parent
Q3 = REPO / "experiments/2026-10-06_q3_attack_detector"
RUNS = Q3 / "results/sim_honest_scores_2026-10-07_124717/runs.json"
METRICS = Q3 / "results/metrics_2026-10-06_181548__2026-10-06_182739.json"
PUBLISHED = Q3 / "results/sim_honest_2026-10-07_124717__2026-10-07_125658.json"
B, SEED, PLANT_SEED, N_NULL, RATES, LAYERS, PRIMARY = 2000, 20261012, 20261013, 200, (0.05, 0.01), (63, 48), (63, 0.05)
HONEST, SIMH, EDIT = "real-honest", "sim-honest", "sim-honest-edited"

spec = importlib.util.spec_from_file_location("ash", Q3 / "analyze_sim_honest.py")
spec.loader.exec_module(ASH := importlib.util.module_from_spec(spec))
caught_boot, rate_draws = ASH.caught_boot, ASH.rate_draws  # the published false-alarm machinery, unchanged


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def assemble(runs, outer, nested, edited_rows):
    """Arrays over the 4,254 committed runs plus one edited copy per simulated honest run (appended in run order).
    edited_rows: {key: index into outer/nested of the edited extraction's score}; other copies take the as-is score."""
    n0 = len(runs)
    sim = [i for i, r in enumerate(runs) if r["arm"] == SIMH]
    src = [edited_rows.get(runs[i]["key"], i) for i in sim]
    o = np.concatenate([outer[:n0], outer[src]])
    ne = np.concatenate([nested[..., :n0], nested[..., src]], axis=2)
    arm = np.array([r["arm"] for r in runs] + [EDIT] * len(sim))
    meta = {k: np.array([r[k] for r in runs] + [runs[i][k] for i in sim]) for k in ("key", "main_task", "fold", "in_base")}
    changed = np.array([False] * n0 + [runs[i]["key"] in edited_rows for i in sim])
    return o, ne, arm, meta, changed


def masks(o, arm, meta):
    """(i), (ii), (iii) on the tasks where all three have a score; calibration = the base run's real honest runs."""
    has, task, key = np.isfinite(o), meta["main_task"], meta["key"]
    tasks = sorted(set(task[arm == SIMH]))
    first_real = np.zeros(len(o), bool)
    for t in tasks:
        idx = np.flatnonzero((arm == HONEST) & (task == t))
        first_real[idx[np.argmin(key[idx])]] = True
    two = set(task[first_real & has]) & set(task[(arm == SIMH) & has])  # the published sets (i), (ii)
    both = two & set(task[(arm == EDIT) & has])
    keep, keep2 = np.isin(task, sorted(both)), np.isin(task, sorted(two))
    M = {"i": first_real & keep, "ii": (arm == SIMH) & keep, "iii": (arm == EDIT) & keep}
    M["published_i"], M["published_ii"] = first_real & keep2, (arm == SIMH) & keep2
    return M, (arm == HONEST) & meta["in_base"] & has, len(both)


def contrast(ne, fold, cal, M, alpha, W):
    """Per draw (row 0 the point): rates of (i), (ii), (iii) with thresholds redrawn per draw, and the paired contrasts."""
    c = caught_boot(ne, fold, cal, alpha, W)
    r = {g: rate_draws(c, W, m) for g, m in M.items()}
    return c, r | {"iii-ii": r["iii"] - r["ii"], "ii-i": r["ii"] - r["i"], "iii-i": r["iii"] - r["i"]}


def summ(v):
    return {"value": float(v[0]), "ci95": ci(v[1:])}


def main():
    ap = argparse.ArgumentParser()
    for a in ("--scoring", "--scores", "--stage", "--power", "--smoke"):
        ap.add_argument(a, required=True)
    args = ap.parse_args()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    if dirty:
        raise SystemExit(f"refusing to run from a dirty tree:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    scoring, stage, power, smoke = (json.loads(Path(p).read_text()) for p in (args.scoring, args.stage, args.power, args.smoke))
    runs = json.loads(RUNS.read_text())
    out = {"label": ("POST HOC: defined after the edit's attack-run effects were published (fix939, swap_cue); "
                     "from raw data"), "git": head, "rule": "RULE.md",
           "inputs": {"scoring": {"file": Path(args.scoring).name, "sha256": sha(args.scoring)},
                      "scores": {"file": Path(args.scores).name, "sha256": sha(args.scores)},
                      "stage": {"file": Path(args.stage).name, "sha256": sha(args.stage)},
                      "power": {"file": Path(args.power).name, "sha256": sha(args.power)},
                      "smoke": {"file": Path(args.smoke).name, "sha256": sha(args.smoke)},
                      "runs_json_sha256": sha(RUNS), "metrics_sha256": sha(METRICS), "published_sha256": sha(PUBLISHED)},
           "gates": {}, "layers": {}}
    # gates before any number: the scorer's checks, the edited set, the scores file
    edited = scoring["edited_keys"]
    staged_changed = sorted(k for k, v in stage["per_run"].items() if v["changed_messages"])
    out["gates"] |= {"smoke_check_passed_for_this_stage": bool(smoke["pass"] and smoke["stage_record"] == Path(args.stage).name),
                     "scoring_checks": bool(scoring["gates"]["all"]), "edited_set_equals_stage": edited == staged_changed,
                     "scoring_used_this_stage": scoring["stage_record"] == Path(args.stage).name,
                     "scores_sha256_matches": sha(args.scores) == scoring["scores_sha256"], "runs_json_matches": sha(RUNS) == scoring["runs_json_sha256"]}
    # RULE.md amendment 1 (before any edited score): X from the measured-sigma scenario; the 2-sigma margin beside
    X = {float(a): float(np.ceil(power["scenarios"]["sigma"][f"{a:g}"]["reach_p80"] * 200) / 200) for a in RATES}
    X2 = {float(a): power["X"][f"{a:g}"] for a in RATES}
    out["X"], out["X_2sigma"] = X, X2
    z = np.load(args.scores)
    n0 = len(runs)
    rows = {k: n0 + j for j, k in enumerate(edited)}
    committed = json.loads(METRICS.read_text())["readouts"]["probe_output_end_mean"]["task"]["thresholds"]
    pub = json.loads(PUBLISHED.read_text())["results"]["output_end"]
    for l in LAYERS:
        o, ne, arm, meta, changed = assemble(runs, z[f"L{l}__outer"], z[f"L{l}__nested"], rows)
        fold = meta["fold"].astype(int)
        M, cal, n_tasks = masks(o, arm, meta)
        if l == 63:  # the rebuilt thresholds are #36's committed ones, and (i), (ii) reproduce the published rates
            worst = max(abs(threshold(ne[f, v][cal & (fold == v) & ~np.isnan(ne[f, v])], float(a)) - committed[a][f"{f}_{v}"]["threshold"])
                        for a in committed for f in range(5) for v in range(5) if v != f)
            # 1e-5: the committed values are rounded to 5 decimals (5e-6), and these scores come from the v4 features,
            # which reproduce the base run's within 1e-5 (dry run on the published scores: worst 5.19e-6)
            out["gates"]["L63_thresholds_equal_committed"] = bool(worst <= 1e-5)
            out["gates"]["L63_thresholds_worst_abs_diff"] = float(worst)
            ones = np.ones((1, len(o)))
            pts = {a: {g: float(rate_draws(caught_boot(ne, fold, cal, a, ones), ones, M[f"published_{g}"])[0]) for g in ("i", "ii")} for a in RATES}
            want = {a: {"i": pub["rates"][f"frozen_real@{a:g}"]["real_honest_first"]["value"], "ii": pub["rates"][f"frozen_real@{a:g}"]["sim_honest"]["value"]} for a in RATES}
            out["gates"]["L63_reproduces_published_rates"] = bool(int(M["published_ii"].sum()) == pub["n_tasks_both_honest_scored"] and
                                                                  all(abs(pts[a][g] - want[a][g]) <= 5e-5 for a in RATES for g in pts[a]))
            out["gates"]["published_vs_recomputed"] = {f"{a:g}": {g: [want[a][g], round(pts[a][g], 6)] for g in pts[a]} for a in RATES}
        if not all(v for k, v in out["gates"].items() if isinstance(v, bool)):
            break
        W = boot_weights(meta["main_task"], B, SEED)
        M = {g: M[g] for g in ("i", "ii", "iii")}
        res = {"tasks": n_tasks, "tasks_dropped_for_any_missing_score": int(len(set(meta["main_task"][arm == SIMH])) - n_tasks),
               "runs": {g: int(m.sum()) for g, m in M.items()}, "changed_runs_in_set": int((changed & M["iii"]).sum())}
        d = o[M["iii"]] - o[M["ii"]]  # paired: (ii) and (iii) masks hold the same runs in the same order
        dw = W[:, M["ii"]]
        res["mean_run_score_shift_iii_minus_ii"] = summ((dw * d).sum(1) / dw.sum(1)) | {"sd_per_run": float(d.std())}
        prng = np.random.default_rng(PLANT_SEED + l)
        nrng = np.random.default_rng(SEED + 1 + l)
        for a in RATES:
            c, r = contrast(ne, fold, cal, M, a, W)
            fixed = {g: rate(c[0][M[g]], W[:, M[g]]) for g in ("i", "ii", "iii")}
            rr = {"fa": {g: summ(r[g]) for g in ("i", "ii", "iii")},
                  "iii_minus_ii (main, paired)": summ(r["iii-ii"]), "ii_minus_i": summ(r["ii-i"]), "iii_minus_i": summ(r["iii-i"]),
                  "fixed_thresholds": {"iii_minus_ii": summ(fixed["iii"] - fixed["ii"]), "ii_minus_i": summ(fixed["ii"] - fixed["i"]),
                                       "iii_minus_i": summ(fixed["iii"] - fixed["i"])}}
            # planted check: +X false-alarm points on (iii), k frozen runs it never flags pushed over every threshold
            n_iii, idx_iii = int(M["iii"].sum()), np.flatnonzero(M["iii"])
            k = int(round(X[a] * n_iii))
            elig = idx_iii[c[0][idx_iii] == 0]
            if len(elig) < k:
                rr["plant"] = {"constructible": False, "eligible": int(len(elig)), "needed": k}
            else:
                pick = np.sort(prng.choice(elig, k, replace=False))
                nep = ne.copy()
                nep[:, :, pick] = np.where(np.isnan(nep[:, :, pick]), np.nan, 1e6)
                _, rp = contrast(nep, fold, cal, M, a, W)
                lo, hi = ci(rp["iii-ii"][1:])
                shift = float(rp["iii-ii"][0] - r["iii-ii"][0])
                rr["plant"] = {"constructible": True, "k": k, "of": n_iii, "planted_amount": k / n_iii, "shift": shift,
                               "planted_ci95": [lo, hi], "detected": bool(0.5 * X[a] <= shift <= 1.5 * X[a] and not (lo <= r["iii-ii"][0] <= hi))}
            # null: (ii) and (iii) swapped per run (one run per task per arm), 200 seeded copies; fires if its interval excludes 0
            cii, ciii = c[:, M["ii"]], c[:, M["iii"]]
            fires = 0
            for _ in range(N_NULL):
                flip = nrng.integers(0, 2, cii.shape[1]).astype(bool)
                a_, b_ = np.where(flip, ciii, cii), np.where(flip, cii, ciii)
                dn = ((b_ - a_) * dw).sum(1) / dw.sum(1)
                lo, hi = ci(dn[1:])
                fires += lo > 0 or hi < 0
            rr["null_swap"] = {"copies": N_NULL, "fired": int(fires), "rate": fires / N_NULL, "ok": fires / N_NULL <= 0.10}
            lo, hi = rr["iii_minus_ii (main, paired)"]["ci95"]
            stable = -X[a] <= lo and hi <= X[a]
            valid = rr["plant"].get("detected", False) and rr["null_swap"]["ok"]
            v = {"raises": bool(lo > 0), "lowers": bool(hi < 0), "within_X": bool(stable), "checks_valid": bool(valid),
                 "within_X_2sigma": bool(-X2[a] <= lo and hi <= X2[a])}
            if (l, a) == PRIMARY or a == 0.05:
                v["reading"] = ("not readable: the planted or null check failed" if not valid else
                                f"the edit raises false alarms (interval above 0){', by less than X' if stable else ''}" if v["raises"] else
                                f"the edit lowers false alarms (interval below 0){', by less than X' if stable else ''}" if v["lowers"] else
                                "false alarms stable: the edited - original interval lies inside +-X" if stable else "inconclusive")
            else:
                v["reading"] = "reported, no verdict (1% thresholds are extreme order statistics)"
            rr["verdict"] = v
            res[f"{a:g}"] = rr
        out["layers"][str(l)] = res
    out["gates"]["all"] = all(v for k, v in out["gates"].items() if isinstance(v, bool))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"edit_honest_fa_POSTHOC_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(path.name)
    if not out["gates"]["all"]:
        raise SystemExit("a gate FAILED: no reading")


if __name__ == "__main__":
    main()
