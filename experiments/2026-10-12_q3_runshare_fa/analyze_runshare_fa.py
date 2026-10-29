"""Share of runs with at least one false alarm, per world, with the real-vs-real comparison (RULE.md; POST HOC, $0,
local; keys, labels, scores and flags only, no command text).

  PYTHONPATH=experiments/2026-10-08_q3_onpolicy_figures uv run --no-project --with numpy python \
      experiments/2026-10-12_q3_runshare_fa/analyze_runshare_fa.py <draws0 json>
"""

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from analyze_scaleup_plants import threshold

HERE = Path(__file__).parent
REPO = HERE.parent.parent
DRAWS_SHA = "c2656658baaf3aa150f83f4d105e7572bfd854f98032831621419e1f8e8f36bf"
PERTASK = REPO / "experiments/2026-10-08_q3_onpolicy_figures/results/per_task_false_alarms_POSTHOC_2026-10-11_122057.json"
PERTASK_SHA = "7893a99899c889bd80e587fdb75f39aa62779abbbcb2154fa59b1b633e8c2548"
CELLS = {"real": "real->real", "sim": "real->sim", "run2": "real->run2"}
STREAK_SIM = ["t86_file_indexing", "t66_file_indexing", "t118_clinical_trial", "t96_file_indexing"]  # empty_streaks @ 31020c54
STREAK_RUN2 = ["t29_web_scraping"]  # the same source; reported, not dropped
BANDS = {"sim_minus_first_real": (-0.10, 0.06), "second_real_minus_first_real": (-0.06, 0.06)}  # PREDICTION.md
B, SEED = 2000, 20261012


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def share(R, tasks):
    """runs with >= 1 false alarm / runs with >= 1 read-only call, over the given tasks of one world"""
    t = [x for x in tasks if R[x]["n_ro"] > 0]
    return sum(R[x]["any_fa"] for x in t), len(t)


def diff_ci(A, Bw, tasks, rng):
    """paired by task over complete pairs (both runs have >= 1 read-only call): share(B) - share(A); tasks resampled"""
    t = np.array(sorted(x for x in tasks if A[x]["n_ro"] > 0 and Bw[x]["n_ro"] > 0))
    dropped = sorted(x for x in tasks if not (A[x]["n_ro"] > 0 and Bw[x]["n_ro"] > 0))
    a, b = np.array([A[x]["any_fa"] for x in t], float), np.array([Bw[x]["any_fa"] for x in t], float)
    idx = rng.integers(0, len(t), size=(B, len(t)))
    d = b[idx].mean(1) - a[idx].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"complete_pairs": int(len(t)), "tasks_without_a_complete_pair": dropped,
            "paired_share_reference": float(a.mean()), "paired_share_compared": float(b.mean()),
            "value": float(b.mean() - a.mean()), "ci95": [float(lo), float(hi)]}


def main(draws_path: str):
    dirty = subprocess.run(["git", "status", "--porcelain", str(HERE)], capture_output=True, text=True, cwd=REPO).stdout.strip()
    if dirty:
        raise SystemExit(f"commit RULE.md, PREDICTION.md and this script first:\n{dirty}")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    raw, praw = Path(draws_path).read_bytes(), PERTASK.read_bytes()
    if sha(raw) != DRAWS_SHA or sha(praw) != PERTASK_SHA:
        raise SystemExit("input sha256 differs")
    M, P = json.loads(raw), json.loads(praw)
    folds, meta = M["folds"], M["meta"]

    # gate 1: frozen thresholds rebuilt from the real->real calibration rows (as analyze_fa_concentration.py)
    cal = M["draws"][0]["cells"]["real->real"]["per_call"]["calibration"]
    cs = [(s, k) for i, s, k in zip(cal["row"], cal["score"], cal["model_fold"]) if meta[i]["world"] == "real" and meta[i]["y"] == 0]
    c_s, c_k = (np.array(x) for x in zip(*cs))
    th0 = np.array([threshold(c_s[c_k == k], np.ones(int((c_k == k).sum()))) for k in range(5)])
    gates = {"thresholds_equal_per_task_result": bool(np.max(np.abs(th0 - np.array(P["thresholds_point"]))) <= 1e-9)}
    R, printed = {}, {}
    for w, cell in CELLS.items():
        pc = M["draws"][0]["cells"][cell]["per_call"]
        rows, s, fl = pc["row"], np.array(pc["score"]), np.array(pc["flag_frozen"], dtype=bool)
        if not len(rows) == len(s) == len(fl) or any(meta[r]["world"] != w for r in rows) or len(set(rows)) != len(rows):
            raise SystemExit(f"{cell}: misaligned arrays, rows from another world, or repeated rows")
        f = np.array([folds[meta[r]["task"]] for r in rows])
        y = np.array([meta[r]["y"] for r in rows])
        gates[f"{w}_flags_reproduce"] = bool(np.array_equal(s > th0[f], fl))
        per = defaultdict(lambda: {"n_all": 0, "n_ro": 0, "k_ro": 0})
        for r, flag, yy in zip(rows, fl, y):
            p = per[meta[r]["task"]]
            p["n_all"] += 1
            if yy == 0:
                p["n_ro"] += 1
                p["k_ro"] += int(flag)
        for p in per.values():
            p["any_fa"] = int(p["k_ro"] > 0)
        ro = y == 0
        pooled = float(fl[ro].mean())
        ptm = float(np.mean([p["k_ro"] / p["n_ro"] for p in per.values() if p["n_ro"] > 0]))
        # gate 2: pooled per call and per-task mean equal the per-task result (the source of today's printed numbers)
        gates[f"{w}_pooled_equals_per_task_result"] = abs(pooled - P["worlds"][w]["pooled"]) <= 1e-9
        gates[f"{w}_per_task_mean_equals_per_task_result"] = abs(ptm - P["worlds"][w]["per_task_mean"]) <= 1e-9
        printed[w] = {"per_call_pooled": pooled, "per_task_mean": ptm}
        R[w] = dict(per)
    if not all(v is True for v in gates.values()):
        raise SystemExit(f"a gate failed: {gates}")

    def world_row(w, tasks):
        k, n = share(R[w], tasks)
        t = [x for x in tasks if x in R[w]]
        return {"runs_eligible": n, "runs_with_false_alarm": k, "share": k / n if n else None, "tasks_in_world": len(t),
                "mean_read_only_calls_per_run": float(np.mean([R[w][x]["n_ro"] for x in t])),
                "mean_scored_calls_per_run": float(np.mean([R[w][x]["n_all"] for x in t])),
                "false_alarms_read_only": int(sum(R[w][x]["k_ro"] for x in t)),
                "read_only_calls": int(sum(R[w][x]["n_ro"] for x in t))}

    all_t = {w: sorted(R[w]) for w in R}
    variants = {
        "all_runs": {w: all_t[w] for w in R},
        "drop_4_streak_runs_from_sim_only": {w: [x for x in all_t[w] if not (w == "sim" and x in STREAK_SIM)] for w in R},
        "drop_4_streak_tasks_from_all_worlds": {w: [x for x in all_t[w] if x not in STREAK_SIM] for w in R},
    }
    out = {"label": "POST HOC (RULE.md); from raw data; keys, labels, scores and flags only", "git": git,
           "units": "fractions: shares, differences, ci95 and prediction bands (x100 = percent or percentage points)",
           "inputs": {Path(draws_path).name: DRAWS_SHA, PERTASK.name: PERTASK_SHA,
                      "streak_runs_source": "empty_streaks_POSTHOC_2026-10-12_131328.json @ 31020c54, acting.worlds.sim"},
           "gates": gates, "printed_numbers_reproduced": printed, "thresholds": th0.tolist(),
           "streak_sim_tasks": STREAK_SIM, "bootstrap": {"draws": B, "seeds": {"sim_minus_first_real": [SEED, 0],
           "second_real_minus_first_real": [SEED, 1]}, "unit": "task, paired over complete pairs, flags and thresholds fixed"},
           "paired_tasks_all_three_worlds": len(set(all_t["real"]) & set(all_t["sim"]) & set(all_t["run2"])),
           "variants": {}}
    for name, T in variants.items():
        v = {"worlds": {w: world_row(w, T[w]) for w in R}}
        v["sim_minus_first_real"] = diff_ci(R["real"], R["sim"], set(T["real"]) & set(T["sim"]), np.random.default_rng([SEED, 0]))
        v["second_real_minus_first_real"] = diff_ci(R["real"], R["run2"], set(T["real"]) & set(T["run2"]),
                                                    np.random.default_rng([SEED, 1]))
        out["variants"][name] = v
    out["second_real_run_streak_runs_kept"] = {x: {"read_only_calls": R["run2"][x]["n_ro"], "false_alarms": R["run2"][x]["k_ro"],
                                                   "any_false_alarm": bool(R["run2"][x]["any_fa"])} for x in STREAK_RUN2 if x in R["run2"]}
    base = out["variants"]["all_runs"]
    out["prediction_check"] = {
        k: {"value": base[k]["value"], "band": list(BANDS[k]), "inside": BANDS[k][0] <= base[k]["value"] <= BANDS[k][1]} for k in BANDS}
    moved = out["variants"]["drop_4_streak_runs_from_sim_only"]["worlds"]["sim"]["share"] - base["worlds"]["sim"]["share"]
    out["prediction_check"]["sim_share_moved_by_dropping_4"] = {"value": moved, "band": "|moved| < 0.03", "inside": abs(moved) < 0.03}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / f"runshare_fa_POSTHOC_{stamp}.json"
    with open(path, "x") as fh:  # append-only: never overwrite
        fh.write(json.dumps(out, indent=1) + "\n")
    print(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("draws0")
    main(ap.parse_args().draws0)
