"""POST HOC, RULE.md amendment 3 — post hoc, internal check, not an untouched replication. The matched-false-alarm
comparison on the TASK-DISJOINT non-pilot cohort: the attack cells outside the 100 pilot cells whose main task is not a
pilot task (q2's coverage_overlap_2026-10-12_030025.json @ 11f83991: 729 cells, 447 tasks), with the pilot's main tasks
also left out of every world's honest calibration. $0, local, stored scores only; keys and numbers only.

Method as analyze_matched_fa.py (helpers imported unchanged): world-specific thresholds within the main-task folds (real
honest first run per task, simulated honest as simulated, simulated honest edited), #36's committed thresholds as the
frozen world, 5% and 1%, 2,000 stratified task draws (seed 20261015) with thresholds redrawn. Strata: tasks with both
disjoint attack cells and kept honest runs, attack-only, honest-only, and the base real honest runs' remaining tasks
(so the frozen world keeps all 990 runs and its committed thresholds at draw 0). No pass/fail gate on the estimates.
Gates on the inputs: the full-cohort run reproduces 2ebf7e98's points and intervals exactly (analyze_matched_fa_overlap's
build/evaluate); the cohort reproduces q2's counts and q2's frozen-threshold points on it exactly.

  PYTHONPATH=src uv run --no-project --python 3.12 --with numpy python experiments/2026-10-12_q3_edit_matched_fa/analyze_matched_fa_disjoint.py \
      --base-scores <scores_L63.npz> --fix-runs <fix939 per-run jsonl> --edit-scores <edithonest_scores_2026-10-12_022927.npz> \
      --q2-overlap <coverage_overlap_2026-10-12_030025.json from 11f83991>
"""

import argparse
import importlib.util
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("amfo", HERE / "analyze_matched_fa_overlap.py")
spec.loader.exec_module(O := importlib.util.module_from_spec(spec))
A = O.A
PILOT = A.REPO / "experiments/2026-10-07_q2_rendering_swap/config/swap_runs.json"
SHA = {"pilot": "92cf627dfa30fab92d80c6313d945cff55bb0192118d91a7e2795c1415c734e8",
       "q2": "a0afa64bda080320d6961c4eeb348b834309dafefa1be3b92a725559931b9a8e"}
LABEL = "post hoc, internal check, not an untouched replication"


def main():
    ap = argparse.ArgumentParser()
    for a in ("--base-scores", "--fix-runs", "--edit-scores", "--q2-overlap"):
        ap.add_argument(a, required=True)
    args = ap.parse_args()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=A.REPO).stdout.strip()
    if dirty:
        raise SystemExit(f"refusing to run from a dirty tree:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=A.REPO).stdout.strip()
    if A.sha(PILOT) != SHA["pilot"] or A.sha(args.q2_overlap) != SHA["q2"]:
        raise SystemExit("the pilot list or q2's overlap file differs from the pinned one")
    full = json.loads(O.FULL.read_text())
    fi = full["inputs"]
    if not (full["gates"]["all"] and A.sha(args.base_scores) == A.SHA["base"] and A.sha(args.fix_runs) == A.SHA["fix"]
            and A.sha(args.edit_scores) == fi["edit_scores_sha256"] and A.sha(A.BASE_RUNS) == fi["base_runs_sha256"]
            and A.sha(A.FIX939) == fi["fix939_result_sha256"] and A.sha(A.EH_RESULT) == fi["edit_honest_result_sha256"]
            and A.sha(A.AEH.RUNS) == json.loads(A.EH_SCORING.read_text())["runs_json_sha256"]):
        raise SystemExit("inputs or metadata differ from the full-cohort run, or its gates did not pass")
    out = {"label": LABEL, "git": head, "rule": "RULE.md amendment 3", "gates": {},
           "inputs": {"pilot_swap_runs_sha256": SHA["pilot"], "q2_overlap_sha256": SHA["q2"], "full_cohort_sha256": A.sha(O.FULL)}}
    # gate 1: the full cohort reproduces 2ebf7e98 exactly (as analyze_matched_fa_overlap.py did)
    S, F, atask, H, X, strata, draws, _ = O.build(args)
    if not (len(atask) == full["cells"] and [len(x) for x in strata] == [full["strata"][k] for k in ("both", "attack_only", "honest_only")]
            and X == full["X"]):
        raise SystemExit("cells, strata or X differ from the full-cohort run")
    allc = np.ones(len(atask), bool)
    ok = True
    for a in A.RATES:
        rf = O.evaluate(S, F, atask, H, X, draws, allc, a, np.random.default_rng(A.PLANT_SEED))
        keys = list(O.PAIRS) + ["frozen_shift_minus_matched_shift", "share_of_frozen_shift_surviving", "real_e2_catch_437run_minus_990run_calibration"]
        ok &= all(rf[k]["value"] == full["results"][a][k]["value"] and rf[k]["ci95"] == full["results"][a][k]["ci95"] for k in keys)
        ok &= all(rf["catch"][k]["value"] == v["value"] and rf["catch"][k]["ci95"] == v["ci95"] for k, v in full["results"][a]["catch"].items())
    out["gates"]["full_cohort_reproduced"] = bool(ok)
    # the cohort, by q2's rule (its file stores counts, not the list): non-pilot cells whose main task is not a pilot task
    runs = json.loads(A.BASE_RUNS.read_text())
    arm_idx = {a: {r["cell"]: i for i, r in enumerate(runs) if r["arm"] == a} for a in (A.SIM, A.E1, A.E2)}
    cells = sorted(c for c in set(arm_idx[A.SIM]) & set(arm_idx[A.E1]) & set(arm_idx[A.E2]))
    if len(cells) != len(atask) or [runs[arm_idx[A.SIM][c]]["main_task"] for c in cells] != atask.tolist():
        raise SystemExit("cells differ from the build")
    skey = np.array([runs[arm_idx[A.SIM][c]]["key"] for c in cells])
    pilot_keys = set(json.loads(PILOT.read_text())["runs"])
    pil = np.isin(skey, list(pilot_keys))
    ptasks = set(atask[pil])
    disj = ~pil & ~np.isin(atask, list(ptasks))
    q2 = json.loads(Path(args.q2_overlap).read_text())
    counts = {"pilot cells": int(pil.sum()), "pilot main tasks": len(ptasks), "non-pilot cells": int((~pil).sum()),
              "cells": int(disj.sum()), "tasks": int(len(set(atask[disj])))}
    want = {"pilot cells": q2["(I) pilot task overlap"]["pilot cells"], "pilot main tasks": q2["(I) pilot task overlap"]["pilot main tasks"],
            "non-pilot cells": q2["(I) pilot task overlap"]["non-pilot cells"], "cells": q2["(I) task-disjoint non-pilot cells"]["cells"],
            "tasks": q2["(I) task-disjoint non-pilot cells"]["tasks"]}
    out["cohort"] = counts
    out["gates"]["cohort_counts_equal_q2"] = counts == want
    # the honest calibration without the pilot's main tasks, every world but the frozen one (#36's committed thresholds)
    hruns = json.loads(A.AEH.RUNS.read_text())
    scoring = json.loads(A.EH_SCORING.read_text())
    ez = np.load(args.edit_scores)
    rows = {k: len(hruns) + j for j, k in enumerate(scoring["edited_keys"])}
    o, ne, harm, meta, _ = A.AEH.assemble(hruns, ez["L63__outer"], ez["L63__nested"], rows)
    hfold, htask = meta["fold"].astype(int), meta["main_task"]
    M, _, _ = A.AEH.masks(o, harm, meta)
    keep = ~np.isin(htask, list(ptasks))
    Hd = {w: A.cal_sets(ne, np.flatnonzero(M[g] & keep), hfold, htask) for w, g in (("real", "i"), ("orig", "ii"), ("edited", "iii"))}
    Hd["frozen"] = H["frozen"]
    out["honest_tasks_kept"] = int(len(set(htask[M["ii"] & keep])))
    out["honest_tasks_dropped_as_pilot"] = int(len(set(htask[M["ii"] & ~keep])))
    out["calibration_runs_per_nested_model"] = {w: {f"{f}_{v}": int(len(Hd[w][f, v][0])) for f, v in Hd[w]} for w in Hd}
    # stratified draws over the disjoint attack tasks, the kept honest tasks and the frozen calibration's other tasks
    base_hon_tasks = {t for fv, (_, ts) in Hd["frozen"].items() for t in ts}
    Ad, Hs = set(atask[disj]), set(htask[M["ii"] & keep])
    strata = [sorted(Ad & Hs), sorted(Ad - Hs), sorted(Hs - Ad), sorted(base_hon_tasks - Ad - Hs)]
    out["strata"] = {"both": len(strata[0]), "attack_only": len(strata[1]), "honest_only": len(strata[2]), "frozen_calibration_only": len(strata[3])}
    rng = np.random.default_rng(A.SEED)
    dd = [{t: 1.0 for st in strata for t in st}]
    for _ in range(A.B):
        cnt = {}
        for st in strata:
            for k in rng.integers(0, len(st), len(st)):
                cnt[st[k]] = cnt.get(st[k], 0.0) + 1.0
        dd.append(cnt)
    # gate 3: #36's committed thresholds at draw 0, and q2's frozen-threshold points on this cohort, exactly
    committed = json.loads(A.METRICS.read_text())["readouts"][f"probe_{A.RO}_mean"]["task"]["thresholds"]
    T0 = {a: A.thresholds(Hd["frozen"], float(a), dd[0]) for a in A.RATES}
    out["gates"]["frozen_thresholds_equal_committed"] = bool(max(abs(T0[a][int(k[0]), int(k[2])] - v["threshold"]) for a in A.RATES
                                                                 for k, v in committed[a].items()) <= 5e-6)
    g3 = {}
    for a in A.RATES:
        c = {n: A.catch(S[n][disj], F[n][disj], T0[a]) for n in ("sim", "fix", "e2")}
        q = q2["(I) task-disjoint non-pilot cells"][a]
        g3[a] = {"shift": [100 * (c["fix"] - c["sim"]).mean(), q["shift (fixed - as simulated)"]["points"]],
                 "fixed - e2": [100 * (c["fix"].mean() - c["e2"].mean()), q["fixed - e2"]["points"]],
                 "as simulated - e2": [100 * (c["sim"].mean() - c["e2"].mean()), q["as simulated - e2"]["points"]]}
    out["gate_values_q2_frozen_points"] = g3
    out["gates"]["q2_frozen_points_reproduced"] = all(abs(x - y) <= 1e-9 for a in A.RATES for x, y in g3[a].values())
    # every world, model and draw must keep positive calibration weight (diagnostics before any estimate)
    empty = [(w, f"{f}_{v}", b) for w in Hd for (f, v), (_, ts) in Hd[w].items() for b, d in enumerate(dd)
             if sum(d.get(t, 0.0) for t in ts) <= 0]
    out["gates"]["calibration_weight_positive_everywhere"] = not empty
    out["empty_calibration_examples"] = empty[:10]
    hrow = {g: (np.stack([ne[hfold[i], :, i] for i in np.flatnonzero(M[g] & keep)]), hfold[M[g] & keep], htask[M[g] & keep]) for g in ("i", "ii", "iii")}
    out["gates"]["kept_honest_scores_finite"] = all(np.isfinite(np.where(np.arange(5)[None, :] != Fg[:, None], Sg, 0.0)).all()
                                                    for Sg, Fg, _ in hrow.values())
    out["gates"]["all"] = all(v for v in out["gates"].values() if isinstance(v, bool))
    if not out["gates"]["all"]:
        _write(out)
        raise SystemExit(f"an input gate FAILED: {out['gates']}")
    # the estimates (no pass/fail gate): every full-cohort endpoint on the disjoint cells, plants as before
    prng = np.random.default_rng(A.PLANT_SEED + 2)  # once, 5% then 1%
    out["X"], out["draws"], out["seeds"] = X, A.B, {"draws": A.SEED, "plant": A.PLANT_SEED + 2}
    out["results"] = {}
    for a in A.RATES:
        r = O.evaluate(S, F, atask, Hd, X, dd, disj, a, prng)
        Tb = [{w: A.thresholds(Hd[w], float(a), d) for w in ("real", "orig", "edited")} for d in dd]
        hwt = {g: np.array([[d.get(t, 0.0) for t in hrow[g][2]] for d in dd]) for g in hrow}
        fa = {w: np.array([(hwt[g][b] * A.catch(hrow[g][0], hrow[g][1], Tb[b][w])).sum() / hwt[g][b].sum() for b in range(len(dd))])
              for w, g in (("real", "i"), ("orig", "ii"), ("edited", "iii"))}
        r["achieved_false_alarms_own_threshold"] = {w: {"value": float(v[0]), "ci95": A.ci(v[1:])} for w, v in fa.items()} | {
            "edited_minus_real": {"value": float(fa["edited"][0] - fa["real"][0]), "ci95": A.ci(fa["edited"][1:] - fa["real"][1:])},
            "orig_minus_real": {"value": float(fa["orig"][0] - fa["real"][0]), "ci95": A.ci(fa["orig"][1:] - fa["real"][1:])}}
        r["real_e2_catch_owncal_minus_990run_calibration"] = r.pop("real_e2_catch_437run_minus_990run_calibration")  # own calibration here: kept tasks only
        r["reading"] = "reported, no verdict (" + LABEL + ")"
        out["results"][a] = r
    _write(out)


def _write(out):
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"matched_fa_disjoint_POSTHOC_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("results", "calibration_runs_per_nested_model")}, indent=1))
    print(path.name)


if __name__ == "__main__":
    main()
