"""POST-HOC first-k analysis (plan Amendment 10) of run_claim_c_firstk.py's records. Modal CPU selects on the Volume; the
statistics are analyze_claim_c.stats unchanged, run locally on the first-k scores (keys and numbers only).

Selection as claim C (validation AUROC on full-run scores, per fold and label column). Reproduction gate first: the
rerun's real-column picks equal claim C's (tag fb82f67348ac) in every fold, and its full-run real-column scores match claim
C's records within 1e-4; a failure stops before any first-k statistic. Then per readout and k (3 primary, 5 secondary):
Z = the picked candidate's first-k score, with the cohort fix on the first-k column; runs without a first-k score
(fewer than k rows) counted per arm, and per world the 196 tasks with no cached run at all (a missing run, or no
row of either readout); the plants' feasibility from cohort sizes, reported before the results.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal run \
      experiments/2026-10-08_q3_onpolicy_figures/analyze_claim_c_firstk.py --tag <tag> --labels <label files>
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
SCORE_VOL, REF_TAG = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08", "fb82f67348ac"
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
LAYERS, C_GRID, KS = [39, 51, 63], [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], (3, 5)
B, PLANT, PLANT_TOL, MAX_FAIL, GATE_TOL = 2000, 0.10, 0.01, 0.05, 1e-4

app = modal.App("probe-shift_aswin-surya_sub-q3onpol_claimcfk-an_kill-2026-11-10")
score_vol = modal.Volume.from_name(SCORE_VOL, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0, volumes={"/score": score_vol})
def select(tag: str) -> dict:
    import numpy as np

    score_vol.reload()
    out = {}
    for name, kind in (("command_end", "call_end"), ("reply_end", "output_end")):
        cands = [(l, kind, C) for C in C_GRID for l in LAYERS]  # argmax keeps the first maximum: smaller C, then lower layer
        R, Q = ({c: dict(np.load(Path("/score") / f"{pre}_{t}" / f"L{c[0]}_{c[1]}_C{c[2]:g}.npz")) for c in cands}
                for pre, t in (("claimcfk", tag), ("claimc", REF_TAG)))
        runs, r0 = R[cands[0]]["on_runs"], R[cands[0]]
        if any(not all(np.array_equal(r[k], r0[k]) for k in ("on_runs", "Y", "on_fold", "train")) for r in [*R.values(), *Q.values()]):
            raise ValueError(f"{name}: records disagree with each other or with claim C on the runs, labels, folds or training mask")

        def picks(S):
            ok_f, ok_v = (np.stack([S[c][f"grad_{n}"] <= S[c]["tol"] for c in cands]) for n in ("full", "val"))
            share = float(1 - np.concatenate([ok_f.ravel(), ok_v.ravel()]).mean())
            V = np.stack([np.where(ok_f[i] & ok_v[i] & np.isfinite(S[c]["val_auroc"]), S[c]["val_auroc"], -np.inf) for i, c in enumerate(cands)])
            return V.argmax(0), np.isfinite(V.max(0)), share

        pick, usable, share = picks(R)
        ref_pick = picks(Q)[0]
        diffs = [np.nanmax(np.abs(R[c]["on_score"][:, 0] - Q[c]["on_score"][:, 0])) for c in cands]
        same_nan = all(np.array_equal(np.isnan(R[c]["on_score"][:, 0]), np.isnan(Q[c]["on_score"][:, 0])) for c in cands)
        gate = {"real_column_picks_equal": bool(np.array_equal(pick[:, 0], ref_pick[:, 0])),
                "all_column_picks_equal_share": float((pick == ref_pick).mean()),
                "full_run_real_score_max_abs_diff": float(max(diffs)), "same_missing": bool(same_nan)}
        gate["passed"] = bool(gate["real_column_picks_equal"] and same_nan and gate["full_run_real_score_max_abs_diff"] <= GATE_TOL)
        if not gate["passed"] or share > MAX_FAIL or not usable.all():
            out[name] = {"gate": gate, "invalid": f"gate failed, or unconverged share {share:.4f} / fold-columns without a converged candidate {int((~usable).sum())}"}
            continue
        wo = np.array([u.split("|", 1)[0] for u in runs.astype(str)])
        for k in KS:
            col = f"on_score_k{k}"
            complete = np.all(np.stack([np.isfinite(R[c][col]).all(1) for c in cands]), 0)  # the cohort fix, first-k column
            short = np.all(np.stack([np.isnan(R[c][col]).all(1) for c in cands]), 0)  # no first-k score: fewer than k rows
            Z = np.full(R[cands[0]][col].shape, np.nan)
            for f in range(5):
                for j in range(pick.shape[1]):
                    r = R[cands[pick[f, j]]]
                    mine = (r["on_fold"] == f) & complete
                    Z[mine, j] = r[col][mine, j]
            out[f"{name}|k{k}"] = {"gate": gate, "unconverged_share": share, "runs": runs.astype(str).tolist(), "Z": Z, "Y": r0["Y"],
                                   "fewer_than_k_rows": {w: int(((wo == w) & short).sum()) for w in sorted(set(wo))},
                                   "dropped_by_cohort_fix": {w: int(((wo == w) & ~complete & ~short).sum()) for w in sorted(set(wo))},
                                   "choices_real_column": [[*cands[pick[f, 0]]] for f in range(5)]}
    return out


@app.local_entrypoint()
def main(tag: str, labels: str):
    import numpy as np

    from analyze_claim_c import stats
    from analyze_retrain import frozen_folds, task_weights

    lab = {}
    for f in labels.split(","):
        for r in map(json.loads, Path(f).read_text().splitlines()):
            w = {"attack": "run2" if "second_attack" in Path(f).name else "run1", "sim_attack@september": "sim"}.get(r["arm"])
            if w and r.get("label"):
                if f"{w}|{r['task']}" in lab:
                    raise ValueError(f"two labels for {w} {r['task']}")
                lab[f"{w}|{r['task']}"] = r["label"]
    f196 = frozen_folds(json.loads((REPO / T196).read_text()))
    tasks = sorted(f196)
    ti = {t: i for i, t in enumerate(tasks)}
    W = [task_weights([f196[t] for t in tasks], b) for b in range(1, B + 1)]
    res, sel = {}, select.remote(tag)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    if any("invalid" in d for d in sel.values()):  # Amendment 10: both readouts must reproduce before any first-k statistic
        path = HERE / "results" / f"claim_c_firstk_POSTHOC_{stamp}_GATE_FAILED.json"
        with path.open("x") as fh:
            fh.write(json.dumps({"tag": tag, "reference_tag": REF_TAG, "amendment": "10 (post hoc)",
                                 "gates": {k: {"gate": d["gate"], "invalid": d.get("invalid")} for k, d in sel.items()}}, indent=1, default=float))
        print(json.dumps({k: d["gate"] for k, d in sel.items()}, indent=1))
        print(path.name)
        return
    for name, d in sel.items():
        world = np.array([u.split("|", 1)[0] for u in d["runs"]])
        task = np.array([u.split("|", 1)[1] for u in d["runs"]])
        have = lambda w: {t for t in task[(world == w) & np.isfinite(d["Z"][:, 0])]}
        att = lambda w: {t for t in have(w) if lab.get(f"{w}|{t}") == "attempted"}
        n = len(att("run1") & att("run2") & att("sim"))  # analyze_claim_c.stats's cohort: one run per arm per task
        feas = {"cohort_N": n, "planted_amount": None if n == 0 else round(PLANT * n) / n}
        feas["within_tolerance"] = n > 0 and abs(feas["planted_amount"] - PLANT) <= PLANT_TOL
        # runs with no row of either readout are not in the cache: per world, the 196 tasks with no cached run at all
        absent = {w: len(set(tasks) - set(task[world == w])) for w in sorted(set(world))}
        res[name] = ({"feasibility": feas} | {k: v for k, v in d.items() if k not in ("runs", "Z", "Y")}
                     | {"tasks_of_196_with_no_cached_run": absent} | stats(d, lab, tasks, ti, W))
    path = HERE / "results" / f"claim_c_firstk_POSTHOC_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps({"tag": tag, "reference_tag": REF_TAG, "amendment": "10 (post hoc)", "results": res}, indent=1, default=float))
    for name, v in res.items():
        o, rp = v["observed"], v["report"]
        print(name, json.dumps({"feasibility": v["feasibility"], "fewer_than_k_rows": v["fewer_than_k_rows"],
                                "no_cached_run": v["tasks_of_196_with_no_cached_run"], "tpr_run1": o["tpr_run1"],
                                "tpr_sim": o["tpr_sim"], "tpr_run2": o["tpr_run2"], "gap": o["gap"], "gap_ci95": v["gap_interval"]["ci95"],
                                "control": o["control"], "control_ci95": v["control_interval"]["ci95"], "honest_b_fa": o["false_alarm_honest_b"],
                                "n_ref": rp["n_ref_honest_a"], "auroc_run1": rp["auroc_run1_vs_honest_b"], "auroc_sim": rp["auroc_sim_vs_honest_b"],
                                "plants": {p: x.get("detected") for p, x in v["plants"].items()}, "null_passes": v["null"]["passes"],
                                "verdicts": v["verdicts"]}, default=float))
    print(path.name)
