"""POST HOC, RULE.md amendment 2 (added after the full-cohort numbers were seen; the round-3 spec arrived after the run):
the matched-false-alarm check restricted to the attack cells of the tasks shared with the simulated honest cohort.
$0, local, stored scores only; keys and numbers only.

Same data, worlds, thresholds method, strata and draws as analyze_matched_fa.py (its helpers are imported unchanged; the
data assembly is copied from its main()). The thresholds of every world and draw are exactly those of the full-cohort
run: calibration sets are untouched; only the attack cells averaged over change. Gate: on all 939 cells this script
reproduces the full-cohort result file exactly (points and intervals) before anything else is reported.

  PYTHONPATH=src uv run --no-project --python 3.12 --with numpy python experiments/2026-10-12_q3_edit_matched_fa/analyze_matched_fa_overlap.py \
      --base-scores <scores_L63.npz> --fix-runs <fix939 per-run jsonl> --edit-scores <edithonest_scores_2026-10-12_022927.npz>
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
spec = importlib.util.spec_from_file_location("amf", HERE / "analyze_matched_fa.py")
spec.loader.exec_module(A := importlib.util.module_from_spec(spec))
FULL = HERE / "results/matched_fa_POSTHOC_2026-10-12_025014.json"
PAIRS = {"edited_minus_real_e2 (main)": ("fix", "edited", "e2", "real"), "orig_minus_real_e2": ("sim", "orig", "e2", "real"),
         "edited_minus_orig (paired)": ("fix", "edited", "sim", "orig"), "real_e2_minus_e1 (yardstick; e1 was in training)": ("e2", "real", "e1", "real"),
         "frozen: edited_minus_e2": ("fix", "frozen", "e2", "frozen"), "frozen: orig_minus_e2": ("sim", "frozen", "e2", "frozen"),
         "frozen: edited_minus_orig": ("fix", "frozen", "sim", "frozen"),
         "secondary, real at #36's frozen thresholds: edited_minus_real_e2": ("fix", "edited", "e2", "frozen")}
COMBOS = [("fix", "edited"), ("sim", "orig"), ("e2", "real"), ("e1", "real"), ("fix", "frozen"), ("sim", "frozen"), ("e2", "frozen")]


def build(args):
    """analyze_matched_fa.main()'s data assembly, copied: attack arrays, honest calibration sets, X, strata and draws."""
    scoring = json.loads(A.EH_SCORING.read_text())
    runs = json.loads(A.BASE_RUNS.read_text())
    arm, fold, rtask = (np.array([r[k] for r in runs]) for k in ("arm", "fold", "main_task"))
    Z = np.load(args.base_scores)
    nz, oz = Z[f"{A.RO}__mean__task_nested"], Z[f"{A.RO}__mean__task_outer"]
    valid = lambda i: np.isfinite(oz[i]) and np.isfinite(np.delete(nz[fold[i], :, i], fold[i])).all()
    arm_idx = {a: {r["cell"]: i for i, r in enumerate(runs) if r["arm"] == a} for a in (A.SIM, A.E1, A.E2)}
    cells = sorted(c for c in set(arm_idx[A.SIM]) & set(arm_idx[A.E1]) & set(arm_idx[A.E2]) if all(valid(arm_idx[a][c]) for a in arm_idx))
    ii = {n: np.array([arm_idx[a][c] for c in cells]) for n, a in (("sim", A.SIM), ("e1", A.E1), ("e2", A.E2))}
    S = {n: np.stack([nz[fold[i], :, i] for i in v]) for n, v in ii.items()}
    F = {n: fold[v] for n, v in ii.items()}
    atask = rtask[ii["sim"]]
    fix = {}
    for line in Path(args.fix_runs).read_text().splitlines():
        r = json.loads(line)
        if r["label"] == "FIX":
            fix[r["key"]] = r
    S["fix"], F["fix"] = S["sim"].copy(), F["sim"]
    for j, i in enumerate(ii["sim"]):
        r = fix.get(runs[i]["key"])
        if r is not None:
            row = np.full(5, np.nan)
            for v_, s in r["nested_scores"][A.RO].items():
                row[int(v_)] = np.nan if s is None else s
            if r["fold"] != fold[i] or not np.isfinite(np.delete(row, fold[i])).all():
                raise SystemExit("a FIX run's fold or scores do not fit")
            S["fix"][j] = row
    hruns = json.loads(A.AEH.RUNS.read_text())
    ez = np.load(args.edit_scores)
    rows = {k: len(hruns) + j for j, k in enumerate(scoring["edited_keys"])}
    o, ne, harm, meta, _ = A.AEH.assemble(hruns, ez["L63__outer"], ez["L63__nested"], rows)
    hfold, htask = meta["fold"].astype(int), meta["main_task"]
    M, _, _ = A.AEH.masks(o, harm, meta)
    H = {"real": A.cal_sets(ne, np.flatnonzero(M["i"]), hfold, htask), "orig": A.cal_sets(ne, np.flatnonzero(M["ii"]), hfold, htask),
         "edited": A.cal_sets(ne, np.flatnonzero(M["iii"]), hfold, htask)}
    hon = np.flatnonzero(arm == A.HON)
    H["frozen"] = A.cal_sets_frozen(nz, hon, fold, rtask)
    f939 = json.loads(A.FIX939.read_text())["readouts"][A.RO]
    X = {a: float(np.ceil(max(abs(c) for c in f939[a]["real_e2_minus_e1 (e1 was in training)"]["ci95"]) * 200) / 200) for a in A.RATES}
    Aset, Hs = set(atask), set(htask[M["ii"]])
    strata = [sorted(Aset & Hs), sorted(Aset - Hs), sorted(Hs - Aset)]
    rng = np.random.default_rng(A.SEED)
    draws = [{t: 1.0 for st in strata for t in st}]
    for _ in range(A.B):
        cnt = {}
        for st in strata:
            for k in rng.integers(0, len(st), len(st)):
                cnt[st[k]] = cnt.get(st[k], 0.0) + 1.0
        draws.append(cnt)
    per_model = {w: {f"{f}_{v}": int(len(H[w][f, v][0])) for f, v in H[w]} for w in H}  # calibration runs per nested model (f, v)
    if set(rtask[hon]) - Aset:
        raise SystemExit("a base real honest run's task is outside the attack tasks")
    return S, F, atask, H, X, strata, draws, per_model


def evaluate(S, F, atask, H, X, draws, sel, a, prng):
    """The full-cohort endpoints, averaged over the attack cells in sel only; thresholds as analyze_matched_fa.py."""
    al = float(a)
    Tb = [{w: A.thresholds(H[w], al, d) for w in H} for d in draws]
    aw = np.array([[d.get(t, 0.0) for t in atask[sel]] for d in draws])
    C = {(n, w): np.array([A.catch(S[n][sel], F[n][sel], Tb[b][w]) for b in range(len(draws))]) for n, w in COMBOS}
    m = {k: (aw * v).sum(1) / aw.sum(1) for k, v in C.items()}
    r = {name: {"value": float(m[(x, wx)][0] - m[(y, wy)][0]), "ci95": A.ci(m[(x, wx)][1:] - m[(y, wy)][1:])} for name, (x, wx, y, wy) in PAIRS.items()}
    r["catch"] = {f"{n}@{w}": {"value": float(m[(n, w)][0]), "ci95": A.ci(m[(n, w)][1:])} for n, w in C}
    sm, sf = m[("fix", "edited")] - m[("sim", "orig")], m[("fix", "frozen")] - m[("sim", "frozen")]
    rf = m[("e2", "real")] - m[("e2", "frozen")]
    r["real_e2_catch_437run_minus_990run_calibration"] = {"value": float(rf[0]), "ci95": A.ci(rf[1:])}
    r["frozen_shift_minus_matched_shift"] = {"value": float(sf[0] - sm[0]), "ci95": A.ci(sf[1:] - sm[1:])}
    if (np.abs(sf) < 1e-9).any():
        raise SystemExit("the frozen shift is zero in a draw: the surviving share is undefined")
    r["share_of_frozen_shift_surviving"] = {"value": float(sm[0] / sf[0]), "ci95": A.ci(sm[1:] / sf[1:])}
    n = int(sel.sum())
    k = int(round(X[a] * n))
    elig = np.flatnonzero(C[("fix", "edited")][0] == 1.0)
    main = r["edited_minus_real_e2 (main)"]["value"]
    if len(elig) < k:
        r["plant"] = {"constructible": False, "eligible": int(len(elig)), "needed": k}
    else:
        pick = np.sort(prng.choice(elig, k, replace=False))
        Sp = S["fix"][sel].copy()
        Sp[pick] = np.where(np.isnan(Sp[pick]), np.nan, -1e6)
        mp = (aw * np.array([A.catch(Sp, F["fix"][sel], Tb[b]["edited"]) for b in range(len(draws))])).sum(1) / aw.sum(1)
        dp = mp - m[("e2", "real")]
        lo, hi = A.ci(dp[1:])
        shift = float(dp[0] - main)
        r["plant"] = {"constructible": True, "k": k, "of": n, "planted_amount": -k / n, "shift": shift, "planted_ci95": [lo, hi],
                      "detected": bool(-1.5 * X[a] <= shift <= -0.5 * X[a] and not (lo <= main <= hi))}
    return r


def reading(r, a, X):
    lo, hi = r["edited_minus_real_e2 (main)"]["ci95"]
    inside = -X[a] <= lo and hi <= X[a]
    if a != "0.05":
        v = "reported, no verdict (1% thresholds are extreme order statistics)"
    elif not r["plant"].get("detected"):
        v = "not readable: the planted check was not detected"
    elif hi < 0:
        v = f"at a common nominal false-alarm target the edited simulation catches less than real e2{' (by less than X)' if inside else ''} (descriptive)"
    elif lo > 0:
        v = f"at a common nominal false-alarm target the edited simulation catches more than real e2{' (by less than X)' if inside else ''}"
    else:
        v = "the gap closes at a common nominal false-alarm target (inside +-X)" if inside else "inconclusive"
    alo, ahi = r["frozen_shift_minus_matched_shift"]["ci95"]
    att = ("removes part of the edit's frozen-threshold gain" if alo > 0 else "enlarges it" if ahi < 0 else "no detectable change")
    return {"inside_X": bool(inside), "below_0": bool(hi < 0), "above_0": bool(lo > 0), "reading": v,
            "attenuation": att if a == "0.05" else "reported, no verdict"}


def category(text):
    """the reading's category, independent of wording"""
    for key in ("not readable", "no verdict", "catches less", "catches more", "gap closes", "inconclusive", "removes part", "enlarges",
                "no detectable"):
        if key in text:
            return key
    raise ValueError(f"unknown reading: {text}")


def main():
    ap = argparse.ArgumentParser()
    for a in ("--base-scores", "--fix-runs", "--edit-scores"):
        ap.add_argument(a, required=True)
    args = ap.parse_args()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=A.REPO).stdout.strip()
    if dirty:
        raise SystemExit(f"refusing to run from a dirty tree:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=A.REPO).stdout.strip()
    full = json.loads(FULL.read_text())
    fi = full["inputs"]
    if not (full["gates"]["all"] and A.sha(args.base_scores) == A.SHA["base"] and A.sha(args.fix_runs) == A.SHA["fix"]
            and A.sha(args.edit_scores) == fi["edit_scores_sha256"] and A.sha(A.BASE_RUNS) == fi["base_runs_sha256"]
            and A.sha(A.FIX939) == fi["fix939_result_sha256"] and A.sha(A.EH_RESULT) == fi["edit_honest_result_sha256"]
            and A.sha(A.AEH.RUNS) == json.loads(A.EH_SCORING.read_text())["runs_json_sha256"]):
        raise SystemExit("inputs or metadata differ from the full-cohort run, or its gates did not pass")
    S, F, atask, H, X, strata, draws, per_fold = build(args)
    if not (len(atask) == full["cells"] and [len(x) for x in strata] == [full["strata"][k] for k in ("both", "attack_only", "honest_only")]
            and X == full["X"]):
        raise SystemExit("cells, strata or X differ from the full-cohort run")
    allc = np.ones(len(atask), bool)
    over = np.isin(atask, strata[0])  # attack cells of the tasks shared with the simulated honest cohort
    out = {"label": ("POST HOC amendment 2: added after the full-cohort numbers were seen (round-3 spec arrived after the run); "
                     "from raw data"), "git": head, "rule": "RULE.md amendment 2", "full_cohort_file": FULL.name,
           "full_cohort_sha256": A.sha(FULL), "X": X, "overlap_tasks": len(strata[0]), "overlap_cells": int(over.sum()),
           "all_cells": int(len(atask)), "calibration_runs_per_nested_model": per_fold, "gates": {}, "results": {}}
    for a in A.RATES:  # gates first, both rates: on all cells, every full-cohort point and interval exactly
        rf = evaluate(S, F, atask, H, X, draws, allc, a, np.random.default_rng(A.PLANT_SEED))
        keys = list(PAIRS) + ["frozen_shift_minus_matched_shift", "share_of_frozen_shift_surviving", "real_e2_catch_437run_minus_990run_calibration"]
        ok = all(rf[k]["value"] == full["results"][a][k]["value"] and rf[k]["ci95"] == full["results"][a][k]["ci95"] for k in keys)
        ok &= all(rf["catch"][k]["value"] == v["value"] and rf["catch"][k]["ci95"] == v["ci95"] for k, v in full["results"][a]["catch"].items())
        out["gates"][f"full_cohort_reproduced_{a}"] = bool(ok)
    if not all(out["gates"].values()):
        out["gates"]["all"] = False
        _write(out)
        raise SystemExit("the full-cohort reproduction gate FAILED: no overlap number")
    prng = np.random.default_rng(A.PLANT_SEED + 1)  # once, 5% then 1% (as the full-cohort run's PLANT_SEED)
    for a in A.RATES:
        ro = evaluate(S, F, atask, H, X, draws, over, a, prng)
        ro["achieved_false_alarms_unchanged_all_437_honest_tasks"] = full["results"][a]["achieved_false_alarms_own_threshold"]
        ro["verdict"] = reading(ro, a, X)
        full_v = full["results"][a]["verdict"]
        ro["compared_with_full_cohort"] = {
            "same_main_reading": category(ro["verdict"]["reading"]) == category(full_v["reading"]),
            "same_attenuation": category(ro["verdict"]["attenuation"]) == category(full_v["attenuation"]),
            "frozen_edited_minus_orig_overlap_minus_full": ro["frozen: edited_minus_orig"]["value"] - full["results"][a]["frozen: edited_minus_orig"]["value"],
            "matched_edited_minus_orig_overlap_minus_full": ro["edited_minus_orig (paired)"]["value"] - full["results"][a]["edited_minus_orig (paired)"]["value"],
            "main_point_overlap_minus_full": ro["edited_minus_real_e2 (main)"]["value"] - full["results"][a]["edited_minus_real_e2 (main)"]["value"],
            "attenuation_point_overlap_minus_full": ro["frozen_shift_minus_matched_shift"]["value"] - full["results"][a]["frozen_shift_minus_matched_shift"]["value"]}
        cw = ro["compared_with_full_cohort"]
        cw["category_change_source"] = (None if cw["same_main_reading"] and cw["same_attenuation"] else
                                        "point estimate (moved by more than 1 point)" if max(abs(cw["main_point_overlap_minus_full"]),
                                                                                           abs(cw["attenuation_point_overlap_minus_full"])) > 0.01
                                        else "interval width or plant only")
        out["results"][a] = ro
    out["gates"]["all"] = True
    _write(out)


def _write(out):
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"matched_fa_overlap_POSTHOC_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=1))
    print(path.name)


if __name__ == "__main__":
    main()
