"""POST-HOC content-only analysis (plan Amendment 13) of run_claim_c_content.py's records, beside the original
command-end records (run_claim_c_firstk.py, tag e99a14545100, which reproduce claim C exactly). Modal CPU selects on the
Volume; the statistics run locally (keys and numbers only).

Per scope (full run, first 3, first 5 content-only commands): the content-only detector selected as claim C's
command-end selection; analyze_claim_c.stats unchanged (catch targeting 1%, controls, plants, null, AUROC, 5%), with
the plants' feasibility first. The drop comparison: content-only minus original catch of run 1 (and of the simulated
runs) targeting 1%, on the tasks in both cohorts, each threshold from its own honest_a scores, both recomputed on each
of 2,000 task resamples. "Drops" = the 95% interval of run 1's difference lies entirely below 0.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal run \
      experiments/2026-10-08_q3_onpolicy_figures/analyze_claim_c_content.py --tag <tag> --labels <label files>
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
SCORE_VOL, ORIG_TAG = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08", "e99a14545100"
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
LAYERS, C_GRID = [39, 51, 63], [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
SCOPES = {"full": "on_score", "k3": "on_score_k3", "k5": "on_score_k5"}
B, PLANT, PLANT_TOL, MAX_FAIL = 2000, 0.10, 0.01, 0.05

app = modal.App("probe-shift_aswin-surya_sub-q3onpol_claimcco-an_kill-2026-11-10")
score_vol = modal.Volume.from_name(SCORE_VOL, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0, volumes={"/score": score_vol})
def select(tag: str) -> dict:
    import numpy as np

    score_vol.reload()
    cands = [(l, "call_end", C) for C in C_GRID for l in LAYERS]  # argmax keeps the first maximum: smaller C, then lower layer
    out = {}
    for name, pre, t in (("content", "claimcco", tag), ("original", "claimcfk", ORIG_TAG)):
        R = {c: dict(np.load(Path("/score") / f"{pre}_{t}" / f"L{c[0]}_{c[1]}_C{c[2]:g}.npz")) for c in cands}
        r0 = R[cands[0]]
        if any(not all(np.array_equal(r[k], r0[k]) for k in ("on_runs", "Y", "on_fold", "train")) for r in R.values()):
            raise ValueError(f"{name}: records disagree on the runs, label columns, folds or training mask")
        ok_f, ok_v = (np.stack([R[c][f"grad_{n}"] <= R[c]["tol"] for c in cands]) for n in ("full", "val"))
        share = float(1 - np.concatenate([ok_f.ravel(), ok_v.ravel()]).mean())
        V = np.stack([np.where(ok_f[i] & ok_v[i] & np.isfinite(R[c]["val_auroc"]), R[c]["val_auroc"], -np.inf) for i, c in enumerate(cands)])
        pick, usable = V.argmax(0), np.isfinite(V.max(0))
        if share > MAX_FAIL or not usable.all():
            raise ValueError(f"{name}: unconverged share {share:.4f}; fold-columns without a converged candidate {int((~usable).sum())}")
        runs = r0["on_runs"].astype(str)
        wo = np.array([u.split("|", 1)[0] for u in runs])
        out[name] = {"runs": runs.tolist(), "Y": r0["Y"], "unconverged_share": share, "choices_real_column": [[*cands[pick[f, 0]]] for f in range(5)]}
        for scope, col in SCOPES.items():
            complete = np.all(np.stack([np.isfinite(R[c][col]).all(1) for c in cands]), 0)  # the cohort fix on this column
            none = np.all(np.stack([np.isnan(R[c][col]).all(1) for c in cands]), 0)  # no score at all (no or too few rows)
            Z = np.full(R[cands[0]][col].shape, np.nan)
            for f in range(5):
                for j in range(pick.shape[1]):
                    r = R[cands[pick[f, j]]]
                    mine = (r["on_fold"] == f) & complete
                    Z[mine, j] = r[col][mine, j]
            out[name][scope] = {"Z": Z, "no_score": {w: int(((wo == w) & none).sum()) for w in sorted(set(wo))},
                                "dropped_by_cohort_fix": {w: int(((wo == w) & ~complete & ~none).sum()) for w in sorted(set(wo))}}
    if out["content"]["runs"] != out["original"]["runs"]:
        raise ValueError("the content-only and original records list different runs")
    return out


@app.local_entrypoint()
def main(tag: str, labels: str):
    import numpy as np

    from analyze_claim_c import stats
    from analyze_retrain import frozen_folds, interval, task_weights, threshold

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
    one = np.ones(len(tasks))
    sel = select.remote(tag)
    runs = sel["content"]["runs"]
    world = np.array([u.split("|", 1)[0] for u in runs])
    task = np.array([u.split("|", 1)[1] for u in runs])
    res = {"choices_real_column": {k: sel[k]["choices_real_column"] for k in ("content", "original")},
           "unconverged_share": {k: sel[k]["unconverged_share"] for k in ("content", "original")},
           # runs with no row of either readout are not in the cache: per world, the 196 tasks with no cached run at all
           "tasks_of_196_with_no_cached_run": {w: len(set(tasks) - set(task[world == w])) for w in sorted(set(world))}}
    for scope in SCOPES:
        Zc, Zo = sel["content"][scope]["Z"][:, 0], sel["original"][scope]["Z"][:, 0]
        have = lambda Z, w: {t for t in task[(world == w) & np.isfinite(Z)]}
        att = lambda Z, w: {t for t in have(Z, w) if lab.get(f"{w}|{t}") == "attempted"}
        coh = lambda Z: att(Z, "run1") & att(Z, "run2") & att(Z, "sim")  # analyze_claim_c.stats's cohort
        n = len(coh(Zc))
        feas = {"cohort_N": n, "planted_amount": None if n == 0 else round(PLANT * n) / n}
        feas["within_tolerance"] = n > 0 and abs(feas["planted_amount"] - PLANT) <= PLANT_TOL
        d = {"runs": runs, "Z": sel["content"][scope]["Z"], "Y": sel["content"]["Y"]}
        common = coh(Zc) & coh(Zo)

        def pick(Z, w, ts):
            m = (world == w) & np.isin(task, sorted(ts)) & np.isfinite(Z)
            return Z[m], np.array([ti[t] for t in task[m]], dtype=int)

        def catch(Z, w, wt):
            ha, x = pick(Z, "honest_a", set(tasks)), pick(Z, w, common)
            if wt[ha[1]].sum() == 0 or wt[x[1]].sum() == 0:
                return None
            t = threshold(ha[0], wt[ha[1]])
            return float((wt[x[1]] * (x[0] > t)).sum() / wt[x[1]].sum())

        drop = {}
        for w in ("run1", "sim"):
            dfn = lambda wt, w=w: None if catch(Zc, w, wt) is None or catch(Zo, w, wt) is None else catch(Zc, w, wt) - catch(Zo, w, wt)
            drop[w] = {"content": catch(Zc, w, one), "original": catch(Zo, w, one), "difference": dfn(one), "interval": interval([dfn(wt) for wt in W])}
        iv = drop["run1"]["interval"]
        drop["common_tasks"] = len(common)
        drop["drops"] = bool(not iv["invalid"] and iv["ci95"][1] < 0)
        res[scope] = {"feasibility": feas, "no_score": sel["content"][scope]["no_score"],
                      "dropped_by_cohort_fix": sel["content"][scope]["dropped_by_cohort_fix"], "drop": drop} | stats(d, lab, tasks, ti, W)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"claim_c_content_POSTHOC_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps({"tag": tag, "original_tag": ORIG_TAG, "amendment": "13 (post hoc)", "results": res}, indent=1, default=float))
    for scope in SCOPES:
        v = res[scope]
        o = v["observed"]
        print(scope, json.dumps({"drop_run1": [v["drop"]["run1"]["content"], v["drop"]["run1"]["original"], v["drop"]["run1"]["difference"], v["drop"]["run1"]["interval"]["ci95"]],
                                 "drops": v["drop"]["drops"], "drop_sim": [v["drop"]["sim"]["difference"], v["drop"]["sim"]["interval"]["ci95"]],
                                 "tpr_run1": o["tpr_run1"], "tpr_sim": o["tpr_sim"], "gap": o["gap"], "gap_ci95": v["gap_interval"]["ci95"],
                                 "control": o["control"], "control_ci95": v["control_interval"]["ci95"], "honest_b_fa": o["false_alarm_honest_b"],
                                 "auroc_run1": v["report"]["auroc_run1_vs_honest_b"], "plants": {p: x.get("detected") for p, x in v["plants"].items()},
                                 "null_passes": v["null"]["passes"], "no_score": v["no_score"], "verdicts": v["verdicts"]}, default=float))
    print("no cached run", json.dumps(res["tasks_of_196_with_no_cached_run"]))
    print(path.name)
