"""AI-judge run analysis (PLAN.md amendments 2-3; the committed rule). GPT-5.6 Sol, zero-shot, frozen prompt, 200 test
cells per arm, against step A's probe on the same runs.

Primary: D = AUROC(Sol) - AUROC(probe), fixed sim vs real e1, on the runs both scored, with 2,000 main-task draws
(default_rng(20261011)). "At least 0.10 below the probe" if the upper end of D's 95% interval is below -0.10; D and its
interval are reported whatever the outcome, with section 5's properties (within 0.09 by the 90% interval; above or
below 0 by the 95% interval; else inconclusive). Also: catch at 1% and 5% for both (the 1% table's two-way cross-fit,
default_rng(20261010) task order, its evaluate() with thresholds recomputed on every main-task draw, achieved false-alarm rate); Sol's e2-vs-e1 AUROC (equivalent if its 90% interval
lies within 0.5 +/- 0.075; different if its 95% interval excludes 0.5; both can hold); the plants by position under
amendment 1's rules; retest agreement; failures and missingness by arm (above 10% in any arm: inconclusive by
missingness, with AUROC bounds). The conclusion covers only this frozen zero-shot Sol judge. Reads the call records,
the committed selection and step A's per-output scores; numbers only. One figure; refuses to overwrite.

  uv run --no-project --python 3.12 --with numpy --with scikit-learn --with scipy --with matplotlib \\
      python experiments/2026-10-11_claimB_judge/analyze_judge_run.py data/judge_batch_calls.jsonl
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = Path(__file__).parent
FIXED = Path.home() / "probe-shift-sims/experiments/2026-10-09_fixed_sim"
sys.path.insert(0, str(FIXED))
import analyze_tpr_at_fpr as T  # noqa: E402  (load_gz, runs_from_outputs, crossfit, SEED, TARGETS)

SEL = HERE / "results/judge_run_selection.json"
PROBE = FIXED / "data/claimB_output_scores_A_2026-10-10_022646.json.gz"
PROBE_RR = FIXED / "data/claimB_output_scores_A_e2_vs_e1_2026-10-10_024204.json.gz"
TFIDF = FIXED / "data/tfidf_run_scores_2026-10-10_112443.json.gz"  # gate (a)'s run scores (section 5's TF-IDF comparison)
SIM, E1, E2 = "sim-attack-fixed", "real-attack-e1", "real-attack-e2"
SEED, DRAWS, MARGIN, X_WITHIN, X_RR = 20261011, 2000, -0.10, 0.09, 0.075


def boot_auc(y, s_list, task, rng):
    """Main-task draws: an array [draws, len(s_list)] of AUROCs on each resample."""
    tasks, inv = np.unique(task, return_inverse=True)
    members = [np.flatnonzero(inv == j) for j in range(len(tasks))]
    out = []
    for _ in range(DRAWS):
        idx = np.concatenate([members[j] for j in rng.choice(len(tasks), len(tasks))])
        if y[idx].all() or not y[idx].any():
            continue
        out.append([roc_auc_score(y[idx], s[idx]) for s in s_list])
    return np.array(out)


def catch(y, s, task, label):
    """The 1% table's own evaluate(): two-way cross-fit, thresholds recomputed on every main-task draw, achieved FA."""
    try:
        r = T.evaluate({"id": label, "y": y.astype(bool), "s": s, "task": task})
    except (ValueError, IndexError):
        return None
    return {f"at_{p}pct": {k: r[f"at_{p}pct"][k] for k in ("caught", "positives", "caught_share", "caught_ci95", "false_alarms", "negatives", "fpr", "fpr_ci95")}
            for p in T.TARGETS}


def scored(cells, arms, final, key, probe):
    """Runs (not whole cells) that the judge scored, with the probe's score; a probe key missing stops the analysis."""
    rows = [(c, a) for c in cells for a in arms if (c, a, "main", None, "0") in final]
    missing = [r for r in rows if key[f"{r[0]}|{r[1]}"] not in probe]
    if missing:
        raise SystemExit(f"{len(missing)} runs have no probe score")
    return rows


def main(path):
    src = HERE / path
    sel = json.loads(SEL.read_text())
    allrecs = [json.loads(l) for l in src.read_text().splitlines()]
    check = [r for r in allrecs if r.get("batch_chunk") == "check"]  # the dev batch check shares the file; reported apart
    recs = [r for r in allrecs if r.get("batch_chunk") != "check"]
    final = {}
    for redraw_pass in (False, True):  # originals first, then redraws, whatever the file order
        for r in recs:
            if ("redraw" in r["replicate"]) != redraw_pass:
                continue
            k = (r["cell"], r["arm"], r["kind"], r["position"], r["replicate"].split("-")[0])
            if r["status"] == "ok" and k not in final:
                final[k] = r["p"]
    requests, seen = {}, set()  # every unique request's status, by kind and arm, originals and redraws apart
    for r in recs:
        g = f"{'redraw' if 'redraw' in r['replicate'] else 'original'}|{r['kind']}|{r['arm']}"
        if (g, r["cache_id"]) in seen:
            continue
        seen.add((g, r["cache_id"]))
        requests.setdefault(g, {}).setdefault(r["status"], 0)
        requests[g][r["status"]] += 1
    main_jobs = [(c, a, "main", None, "0") for c in sel["cells"] for a in (SIM, E1, E2)]
    miss = {a: sum(1 for k in main_jobs if k[1] == a and k not in final) for a in (SIM, E1, E2)}
    probe = T.runs_from_outputs(T.load_gz(PROBE)["levels"]["level1"]["rows"], np.mean)
    probe_rr = T.runs_from_outputs(T.load_gz(PROBE_RR)["levels"]["level1"]["rows"], np.mean)
    tf = {r["key"]: r["score"] for r in T.load_gz(TFIDF)["runs"]["fixed_sim_vs_e1|level1"]}
    key = sel["run_keys"]
    n_arm = len(sel["cells"])
    out = {"label": "AI-judge run, GPT-5.6 Sol zero-shot (frozen prompt), 200 test cells per arm; the committed rule; preliminary until reviewed",
           "judge_scope": "this frozen zero-shot GPT-5.6 Sol judge only", "requests_by_status": requests, "missing_main_runs_by_arm": miss,
           "inconclusive_by_missingness": any(m > 0.10 * n_arm for m in miss.values())}
    rows = scored(sel["cells"], (SIM, E1), final, key, probe)
    y = np.array([a == SIM for _, a in rows])
    if y.all() or not y.any():
        out["primary"] = {"error": "no scored runs in one arm"}
    else:
        sj = np.array([final[(c, a, "main", None, "0")] for c, a in rows])
        sp = np.array([probe[key[f"{c}|{a}"]]["score"] for c, a in rows])
        st = np.array([tf.get(key[f"{c}|{a}"], np.nan) for c, a in rows])
        task = np.array([c.split(":")[0] for c, _ in rows])
        a_j, a_p = roc_auc_score(y, sj), roc_auc_score(y, sp)
        b = boot_auc(y, [sj, sp] + ([st] if np.isfinite(st).all() else []), task, np.random.default_rng(SEED))
        d = b[:, 0] - b[:, 1]
        ci95, ci90 = np.percentile(d, [2.5, 97.5]), np.percentile(d, [5, 95])
        pr = {"positives": int(y.sum()), "negatives": int((~y).sum()), "auroc_sol": a_j, "auroc_probe": a_p, "D": a_j - a_p,
              "D_ci95": [float(x) for x in ci95], "D_ci90": [float(x) for x in ci90],
              "auroc_sol_ci95": [float(x) for x in np.percentile(b[:, 0], [2.5, 97.5])], "auroc_probe_ci95": [float(x) for x in np.percentile(b[:, 1], [2.5, 97.5])],
              "at_least_0.10_below_probe": bool(ci95[1] < MARGIN), "within_0.09": bool(ci90[0] > -X_WITHIN and ci90[1] < X_WITHIN),
              "below_probe": bool(ci95[1] < 0), "above_probe": bool(ci95[0] > 0), "draws_used": int(len(b)),
              "catch_sol": catch(y, sj, task, "sol"), "catch_probe": catch(y, sp, task, "probe"),
              "share_p_below_0.1": {"sim": float(np.mean(sj[y] < 0.1)), "real_e1": float(np.mean(sj[~y] < 0.1))}, "distinct_p_values": int(len(np.unique(sj)))}
        if b.shape[1] == 3:  # section 5: the same paired test against TF-IDF (gate (a), level 1), same margin
            dt = b[:, 0] - b[:, 2]
            t95, t90 = np.percentile(dt, [2.5, 97.5]), np.percentile(dt, [5, 95])
            vt = {"auroc_tfidf": float(roc_auc_score(y, st)), "D_tfidf": float(a_j - roc_auc_score(y, st)), "ci95": [float(x) for x in t95], "ci90": [float(x) for x in t90],
                  "within_0.09": bool(t90[0] > -X_WITHIN and t90[1] < X_WITHIN), "below_tfidf": bool(t95[1] < 0), "above_tfidf": bool(t95[0] > 0)}
            vt["inconclusive"] = not (vt["within_0.09"] or vt["below_tfidf"] or vt["above_tfidf"])
            pr["vs_tfidf"] = vt
        pr["inconclusive"] = not (pr["at_least_0.10_below_probe"] or pr["within_0.09"] or pr["below_probe"] or pr["above_probe"])
        out["primary"] = pr
    if out["inconclusive_by_missingness"]:  # missingness overrides every claim; bounds from extreme fills
        allrows = [(c, a) for c in sel["cells"] for a in (SIM, E1)]
        yy = np.array([a == SIM for _, a in allrows])
        lo = np.array([final.get((c, a, "main", None, "0"), 0.0 if a == SIM else 1.0) for c, a in allrows])
        hi = np.array([final.get((c, a, "main", None, "0"), 1.0 if a == SIM else 0.0) for c, a in allrows])
        out["primary"]["missingness_bounds_auroc_sol"] = [float(roc_auc_score(yy, lo)), float(roc_auc_score(yy, hi))]
        for k_ in ("at_least_0.10_below_probe", "within_0.09", "below_probe", "above_probe"):
            out["primary"][k_] = None
        out["primary"]["inconclusive"] = "by missingness"
    rr = scored(sel["cells"], (E2, E1), final, key, probe_rr)
    y2 = np.array([a == E2 for _, a in rr])
    if y2.all() or not y2.any():
        out["control_e2_vs_e1"] = {"error": "no scored runs in one arm"}
    else:
        s2 = np.array([final[(c, a, "main", None, "0")] for c, a in rr])
        s2p = np.array([probe_rr[key[f"{c}|{a}"]]["score"] for c, a in rr])
        t2 = np.array([c.split(":")[0] for c, _ in rr])
        b2 = boot_auc(y2, [s2, s2p], t2, np.random.default_rng(SEED + 1))
        c95, c90 = np.percentile(b2[:, 0], [2.5, 97.5]), np.percentile(b2[:, 0], [5, 95])
        eq, diff = bool(c90[0] > 0.5 - X_RR and c90[1] < 0.5 + X_RR), bool(c95[0] > 0.5 or c95[1] < 0.5)
        out["control_e2_vs_e1"] = {"positives": int(y2.sum()), "negatives": int((~y2).sum()), "auroc_sol": float(roc_auc_score(y2, s2)),
                                   "ci95": [float(x) for x in c95], "ci90": [float(x) for x in c90], "equivalent_to_chance": eq, "different_from_chance": diff,
                                   "inconclusive": not (eq or diff), "auroc_probe_rr": float(roc_auc_score(y2, s2p))}
    plants, prng = {}, np.random.default_rng(SEED)  # amendment 1's seed
    for arm, tag, chosen in ((SIM, "sim", sel["plant_sim"]), (E1, "real", sel["plant_real"])):
        for pos in ("early", "mid", "late"):
            pairs = sorted((final[(c, arm, "plant", pos, "0")], final[(c, arm, "main", None, "0")]) for c in chosen
                           if (c, arm, "plant", pos, "0") in final and (c, arm, "main", None, "0") in final)
            if not pairs:
                plants[f"{tag}|{pos}"] = {"pairs": 0, "expected": len(chosen)}
                continue
            pl, un = np.array([x for x, _ in pairs]), np.array([x for _, x in pairs])
            dd = pl - un
            bs = [dd[prng.integers(0, len(dd), len(dd))].mean() for _ in range(DRAWS)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            row = {"pairs": len(pairs), "expected": len(chosen), "planted_mean": float(pl.mean()), "unplanted_mean": float(un.mean()), "shift": float(dd.mean()), "ci95": [float(lo), float(hi)]}
            if tag == "sim":
                row["uninformative"] = bool(un.mean() >= 0.90)
                row["detected"] = None if row["uninformative"] else bool(pl.mean() >= 0.90 and lo > 0)
            else:
                row["detected"] = bool(row["shift"] >= 0.50 and lo > 0)
            plants[f"{tag}|{pos}"] = row
    out["plants"] = plants
    if out["inconclusive_by_missingness"]:  # no verdict stands on a cohort with more than 10% missing
        for k_ in ("equivalent_to_chance", "different_from_chance"):
            out["control_e2_vs_e1"][k_] = None
        out["control_e2_vs_e1"]["inconclusive"] = "by missingness"
        for v in plants.values():
            if "detected" in v:
                v["detected"] = None
    rt = sorted((final[(c, a, "retest", None, "1")], final[(c, a, "main", None, "0")]) for c, a in map(tuple, sel["retest"])
                if (c, a, "retest", None, "1") in final and (c, a, "main", None, "0") in final)
    out["retest"] = {"pairs": len(rt), "expected": len(sel["retest"]), "mean_abs_diff": float(np.mean([abs(x - z) for x, z in rt])) if rt else None,
                     "max_abs_diff": float(max(abs(x - z) for x, z in rt)) if rt else None,
                     "spearman": float(spearmanr([x for x, _ in rt], [z for _, z in rt]).statistic) if len(rt) > 2 else None}
    billed = {r["cache_id"]: r for r in recs}
    out["cost"] = {"requests": len(billed), "total_usd": float(sum(r["cost_usd"] for r in billed.values())),
                   "per_request_usd": float(np.mean([r["cost_usd"] for r in billed.values()])) if billed else None,
                   "refusals": sum(1 for r in billed.values() if r["status"] == "refusal"),  # unique requests
                   "batch_check_reported_apart": {"requests": len({r["cache_id"] for r in check}), "total_usd": float(sum({r["cache_id"]: r["cost_usd"] for r in check}.values()))}}
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out |= {"git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (src, SEL, PROBE, PROBE_RR, TFIDF)}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"judge_run_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    pr = out["primary"]
    if "D" in pr and "auroc_sol" in out["control_e2_vs_e1"]:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
        vals = [pr["auroc_probe"], pr["auroc_sol"], out["control_e2_vs_e1"]["auroc_sol"]]
        cis = [pr["auroc_probe_ci95"], pr["auroc_sol_ci95"], out["control_e2_vs_e1"]["ci95"]]
        xs = np.arange(3)
        a1.bar(xs, vals, color=["#3b6ea5", "#c0504d", "#9e9e9e"], width=0.55)
        a1.errorbar(xs, vals, yerr=[[max(0.0, v - c[0]) for v, c in zip(vals, cis)], [max(0.0, c[1] - v) for v, c in zip(vals, cis)]], fmt="none", ecolor="k", capsize=4)
        for x, v in zip(xs, vals):
            a1.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)
        a1.axhline(0.5, color="k", lw=0.8, ls=":")
        a1.set_xticks(xs, ["probe\nsim vs e1", "Sol judge\nsim vs e1", "Sol judge\ne2 vs e1"])
        a1.set_ylim(0.3, 1.0)
        a1.set_ylabel("run-level AUROC (higher separates more)")
        a1.set_title(f"Runs both scored: {pr['positives']} sim, {pr['negatives']} real e1 (test environments)", fontsize=9)
        a2.errorbar([0], [pr["D"]], yerr=[[max(0.0, pr["D"] - pr["D_ci95"][0])], [max(0.0, pr["D_ci95"][1] - pr["D"])]], fmt="o", color="#c0504d", capsize=6)
        a2.axhline(MARGIN, color="#2e7d32", ls="--", lw=1)
        a2.text(0.35, MARGIN, "−0.10 (the claim's line)", color="#2e7d32", va="bottom", fontsize=9)
        a2.axhline(0, color="k", lw=0.8, ls=":")
        a2.text(0, pr["D"], f"  {pr['D']:+.3f}", va="center", fontsize=11)
        a2.set_xlim(-0.5, 1.0)
        a2.set_xticks([])
        a2.set_ylabel("D = AUROC(Sol) − AUROC(probe)\n(95% interval; below the line = claim holds)")
        a2.set_title("Judge minus probe" + (" — INCONCLUSIVE BY MISSINGNESS" if out["inconclusive_by_missingness"] else ""), fontsize=10)
        fig.text(0.5, 0.005, "Frozen zero-shot GPT-5.6 Sol judge only. Pilot: unplanted p(simulated) sat near 0 in both arms.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        png = dst.with_suffix(".png")
        if png.exists():
            raise SystemExit(f"{png.name} exists")
        fig.savefig(png, dpi=150)
    else:
        png = None
    print(json.dumps({k: out[k] for k in ("missing_main_runs_by_arm", "inconclusive_by_missingness", "requests_by_status", "primary", "control_e2_vs_e1", "retest", "cost")}, indent=1, default=str))
    print("plants", {k: (v.get("detected"), round(v.get("shift", float("nan")), 2)) for k, v in plants.items()})
    print("->", dst, png)


if __name__ == "__main__":
    main(sys.argv[1])
