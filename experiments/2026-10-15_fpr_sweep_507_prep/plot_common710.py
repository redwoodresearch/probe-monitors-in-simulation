"""Headlines H1-H3, the 21j verdict figure (V) and the appendix Figure A for the ~710 common comparison, titled by script from the record
(the 455's title rules; appendix titles by median 95% half-widths over readable, non-saturated points, as the FPR fork's r6 @ 6c1d8e5b2).
Modal CPU; writes PNGs and a captions JSON next to the record.

  PYTHONPATH=src <modal 1.6.1> run experiments/2026-10-15_fpr_sweep_507_prep/plot_common710.py --record <name>@<sha> --commit <commit holding it>
"""

import hashlib
import io
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("probe-shift_aswin-surya_sub-fpr507prep_kill-2026-11-14")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "matplotlib==3.10.6")
PLAIN = {"Sol (a) R": "Sol-trained probe (a)", "FIBS R": "FIBS (public)", "harmfulness R": "harmfulness direction", "refusal R": "refusal direction",
         "high-stakes R": "high-stakes probe", "own_a R": "own-real probe (own_a)", "own_frac10_L48 R": "own-real, 10% data", "own_frac25_L48 R": "own-real, 25% data",
         "own_frac100_L48 R": "own-real, 100% data", "c_asis R": "stress test (sim-trained c_asis)"}
BLUE, ORANGE, GREY = "#3182bd", "#e6550d", "#7f7f7f"
PA = "FPR type: per action (share of honest tool outputs or calls flagged)"
PR = "FPR type: per run (share of honest runs alarmed)"
TS = "threshold source: real honest (pool as above)"
PAIRS = ["S_old|S_new", "S_old|S_newmains", "S_new|S_newmains"]
NICE = {"S_old": "T196", "S_new": "+311", "S_newmains": "new main tasks", "S_newmains_without_S4": "new main tasks without S4"}


def pending(rec, lb):
    return rec["probes"][lb]["relaunch"].startswith("relaunch pending")


def common_text(rec):
    P = rec["probes"]
    d = rec["design"]
    cov = P["Sol (a) R"]["coverage"]["cells_with_all_three"]
    cov2 = P["FIBS R"]["coverage"]["cells_with_all_three"]
    pb = lambda lb: ", ".join(f"{k} {v['runs']} runs / {v['units']:,} units" for k, v in P[lb]["pool_by_source"].items())  # noqa: E731
    inc = [lb for lb in P if not pending(rec, lb)]
    pend = [lb for lb in P if pending(rec, lb)]
    pool = (f"one real honest pool per probe: S_old B ∪ C, + the 958 relaunch runs where scored ({len(inc)} of {len(P)} probe rows; relaunch pending, †: "
            f"{', '.join(pend) if pend else 'none'}; DT not shown, relaunch pending (needs Aswin's approval)), + S_new B, + new-main-task B (reconstructed by key); "
            f"honest A held out. Pool for Sol (a): {pb('Sol (a) R')}; for FIBS: {pb('FIBS R')}")
    coh = (f"attempted tasks: T196 {cov['S_old']}, +311 {cov['S_new']}, new main tasks {cov['S_newmains']} for Sol (a) ({cov['S_newmains_without_S4']} "
           f"without S4) and {cov2['S_newmains']} for FIBS ({'incl. 4r' if P['FIBS R']['S4r_included'] else '4r not scored'}; "
           f"{cov2['S_newmains_without_S4']} without S4); {d['S4_cells']} S4 cells")
    lab = ("new main tasks: different construction: provenance, not equivalence validation; no Sol-success gate; descriptive comparisons do not "
           "establish cohort equivalence (21c); sim − run 1 is NOT launch-matched on +311 and new main tasks; PRELIMINARY")
    return pool, coh, lab


PANELS = [("pa1%", "unit", 0.01), ("1%", "run", 0.01), ("5%", "run", 0.05)]  # James 12:13: per action is the headline; per-action 1% first


def panel_point(pr, lv, al):
    """(index, point, catch block, key prefix, without-S4 point) for a panel: per-action = the unit sweep point at f = al (combined block),
    per-run = the run-level point at alpha = al (block all)."""
    if lv == "unit":
        i = next(i for i, q in enumerate(pr["unit"]) if abs(q["fpr_target"] - al) <= 1e-12 * al)
        return i, pr["unit"][i], "combined", f"probes.{pr['label']}.unit[{i}]", pr["unit_without_S4"][i]
    j = next(j for j, q in enumerate(pr["run"]["points"]) if abs(q["alpha"] - al) < 1e-12)
    return j, pr["run"]["points"][j], "all", f"probes.{pr['label']}.run.points[{j}]", pr["run"]["points"][j]


def titles_and_captions(rec):
    import numpy as np

    out = {}
    pool, coh, lab = common_text(rec)
    out["common"] = {"pool": pool, "cohort": coh, "labels": lab}
    for nm in ("common", "common_without_S4", "sol_all", "sol_all_without_S4"):
        H = rec[f"H1_{nm}"]
        h1 = [q for q in H["points"] if "difference_sim_minus_rerun" in q]
        pos = [q for q in h1 if q["difference_sim_minus_rerun"]["ci95"] and q["difference_sim_minus_rerun"]["ci95"][0] > 0]
        neg = [q for q in h1 if q["difference_sim_minus_rerun"]["ci95"] and q["difference_sim_minus_rerun"]["ci95"][1] < 0]
        s = [100 * q["mean_abs_sim_minus_real"]["point"] for q in h1]
        g = [100 * q["mean_abs_run2_minus_run1"]["point"] for q in h1]
        rng = f"(mean |gap| {min(s):.1f}–{max(s):.1f} points vs {min(g):.1f}–{max(g):.1f} for rerunning)" if h1 else ""
        inside = bool(h1) and all(q["difference_sim_minus_rerun"]["ci95"] and -0.02 <= q["difference_sim_minus_rerun"]["ci95"][0]
                                  and q["difference_sim_minus_rerun"]["ci95"][1] <= 0.02 for q in h1)
        wide = max((100 * max(abs(q["difference_sim_minus_rerun"]["ci95"][0]), abs(q["difference_sim_minus_rerun"]["ci95"][1])) for q in h1
                    if q["difference_sim_minus_rerun"]["ci95"]), default=float("nan"))
        if not pos and not neg and inside:
            t = f"Simulation adds about as much catch-rate error as rerunning the real eval at all {len(h1)} readable FPRs (difference within ±2 points) {rng}"
        elif not pos and not neg:
            t = f"No clear difference between simulation's and rerunning's catch-rate error at {len(h1)} readable FPRs (difference intervals up to ±{wide:.1f} points) {rng}"
        elif pos and not neg:
            t = f"Simulation adds more catch-rate error than rerunning the real eval at {len(pos)} of {len(h1)} readable FPRs (paired difference interval excludes 0) {rng}"
        elif neg and not pos:
            t = f"Simulation adds less catch-rate error than rerunning the real eval at {len(neg)} of {len(h1)} readable FPRs (paired difference interval excludes 0) {rng}"
        else:
            t = f"Simulation adds more error than rerunning at {len(pos)} and less at {len(neg)} of {len(h1)} readable FPRs (paired difference intervals exclude 0) {rng}"
        hi_ = [q for q in h1 if q["fpr_target"] >= 4e-3 - 1e-12]
        above = float(np.mean([100 * q["difference_sim_minus_rerun"]["point"] for q in hi_])) if hi_ else float("nan")
        mean_all = float(np.mean([100 * q["difference_sim_minus_rerun"]["point"] for q in h1])) if h1 else float("nan")
        n_above = sum(q["difference_sim_minus_rerun"]["point"] > 0 for q in h1)
        n = H["n_cells"]
        ix = [i for i, q in enumerate(H["points"]) if "difference_sim_minus_rerun" in q]
        kp = lambda leaf, ii=None: [f"H1_{nm}.points[{i}].{leaf}" for i in (ix if ii is None else ii)]  # noqa: E731
        src = {"readable FPR count": {"rule": "number of points carrying difference_sim_minus_rerun (all probes readable)", "keys": kp("difference_sim_minus_rerun")},
               "intervals excluding 0 / difference bound": {"rule": "ci95[0] > 0 or ci95[1] < 0; bound = max |endpoint| x 100, 1 dp", "keys": kp("difference_sim_minus_rerun.ci95")},
               "mean |gap| range (sim)": {"rule": "min and max of 100 x point, 1 dp", "keys": kp("mean_abs_sim_minus_real.point")},
               "mean |gap| range (rerunning)": {"rule": "min and max of 100 x point, 1 dp", "keys": kp("mean_abs_run2_minus_run1.point")},
               "above from 4e-3": {"rule": "mean of 100 x point over fpr_target >= 4e-3, 1 dp",
                                   "keys": kp("difference_sim_minus_rerun.point", [i for i in ix if H["points"][i]["fpr_target"] >= 4e-3 - 1e-12])},
               "n tasks": {"rule": "S_old + S_new + S_newmains; S4 as stored", "keys": [f"H1_{nm}.n_cells"]},
               "sim line above rerun line, all readable FPRs": {"rule": "mean of 100 x point (1 dp) and count of point > 0", "keys": kp("difference_sim_minus_rerun.point")}}
        out[f"H1_{nm}"] = {"title": t, "sources": src, "above_from_4e-3": above, "mean_above_all": mean_all, "n_points_above": n_above, "n_excluding_0": len(pos) + len(neg), "n_readable": len(h1), "n_cells": n,
                           "n_text": f"n = {n['S_old'] + n['S_new'] + n['S_newmains']} tasks (T196 {n['S_old']}, +311 {n['S_new']}, new main tasks {n['S_newmains']}; S4 {n['S4']})"}
    nP = len(rec["H1_common"]["probes"])
    dag = [lb for lb in rec["H1_common"]["probes"] if pending(rec, lb)]
    a, b, so, sw = out["H1_common"], out["H1_common_without_S4"], out["H1_sol_all"], out["H1_sol_all_without_S4"]
    P = rec["probes"]
    srf = {lb: P[lb]["source_ratio"]["range_f_ge_1e-3"] for lb in P if isinstance(P[lb].get("source_ratio"), dict) and not P[lb]["source_ratio"]["within_tolerance"]}
    srtxt = ("pool sources disagree beyond 2× (unit-level source-ratio check at f ≥ 1e-3, ratio range): "
             + ", ".join(f"{lb} {v[0]:.2f}–{v[1]:.2f}" for lb, v in srf.items())) if srf else "every probe's pool sources agree within 2×"
    out["source_ratio_fail"] = {"probes": srf, "text": srtxt, "sources": {"rule": "within_tolerance false; range as stored, 2 dp",
                                                                          "keys": [f"probes.{lb}.source_ratio.range_f_ge_1e-3" for lb in P if isinstance(P[lb].get("source_ratio"), dict)]}}
    h1f = [lb for lb in rec["H1_common"]["probes"] if lb in srf]
    same = rec["H1_common"].get("identical_to_without_S4")
    s4t = ("with/without S4 identical: no S4 cells in the intersection" if same else
           f"without S4 (Aswin's sensitivity): {b['title'][0].lower() + b['title'][1:]}")
    out["H1"] = {"title": a["title"], "caption": (
        f"{a['title']}. The count is of paired difference intervals (sim |gap| − rerun |gap|) that exclude 0; across all {a['n_readable']} readable FPRs the "
        f"sim line sits on average {a['mean_above_all']:.1f} points above the rerun line ({a['n_points_above']} of {a['n_readable']} points above). Left: mean over {nP} probes (stress test excluded; reply readout; registered comparator) of |sim − real| catch and of "
        f"|run 2 − run 1|, both on the SAME cells, restricted to the tasks where all {nP} probes are scored: {a['n_text']}; {s4t}. From about 4e-3 upward the "
        f"sim line sits on average {a['above_from_4e-3']:.1f} points above the rerun line; {a['n_excluding_0']} of {a['n_readable']} paired difference intervals "
        f"exclude 0. Right: Sol (a) alone on all its scored tasks ({so['n_text']}): {so['title'][0].lower() + so['title'][1:]}; without S4 (dashed; "
        f"{sw['n_text']}): {sw['title'][0].lower() + sw['title'][1:]}. {len(dag)} of the {nP} probes are relaunch pending (†). x: {PA}; top axis: {PR}, mean "
        f"over probes; {TS}. {pool}. 95% bands: shared main-level task bootstrap (10,000 draws), conditional on the thresholds; 'about as much' requires every "
        f"paired difference interval inside ±2 points; |gap| is biased upward by noise for both lines, which is why the grey rerun line is the comparison. "
        f"Hatched: fewer than all probes readable (< 10 honest units above the threshold). "
        + (f"‡ Among the {nP} probes, {', '.join(h1f)}: {srtxt}. " if h1f else "") + f"{lab}.")}
    labs = [lb for lb in PLAIN if lb in P and "points" in P[lb]["run"]]
    mon = [lb for lb in labs if not lb.startswith("c_asis")]
    for nm, lv, al in PANELS:
        pp = {lb: panel_point(P[lb], lv, al) for lb in labs}  # (index, point dict, block name, key prefix, without-S4 point)
        regk = {lb: P[lb]["registered"] for lb in labs}
        inside = lambda s: bool(s.get("ci95")) and -0.05 <= s["ci95"][0] and s["ci95"][1] <= 0.05  # noqa: E731
        resolved = lambda s: bool(s.get("ci95")) and (s["ci95"][0] > 0 or s["ci95"][1] < 0)  # noqa: E731
        cnt = {}
        for s4 in (False, True):
            D_ = {lb: (pp[lb][4] if s4 else pp[lb][1])["blocks"][pp[lb][2] if (lv == "unit" or not s4) else "all_without_S4"]["diff"] for lb in labs}
            cnt["without_S4" if s4 else "all"] = {
                "registered_within_5": sum(inside(D_[lb][regk[lb]]) for lb in mon), "real_mean_within_5": sum(inside(D_[lb]["sim_minus_real_mean"]) for lb in mon),
                "rerun_within_5": sum(inside(D_[lb]["run2_minus_run1"]) for lb in mon), "registered_resolved": sum(resolved(D_[lb][regk[lb]]) for lb in mon),
                "rerun_resolved": sum(resolved(D_[lb]["run2_minus_run1"]) for lb in mon), "n_probes": len(mon),
                "points": [100 * D_[lb][regk[lb]]["point"] for lb in mon], "stress": 100 * D_["c_asis R"][regk["c_asis R"]]["point"]}
        c = cnt["all"]
        kp = lambda lb, leaf: f"{pp[lb][3]}.blocks.{pp[lb][2]}.{leaf}"  # noqa: E731
        src = {"registered within ±5 / resolved": {"rule": "-0.05 <= ci95[0] and ci95[1] <= 0.05 / ci95 excludes 0; stress test excluded",
                                                   "keys": [kp(lb, f"diff.{regk[lb]}.ci95") for lb in mon]},
               "rerun within ±5": {"rule": "as above", "keys": [kp(lb, "diff.run2_minus_run1.ci95") for lb in mon]},
               "point gap range": {"rule": "min and max of 100 x point, 1 dp", "keys": [kp(lb, f"diff.{regk[lb]}.point") for lb in mon]},
               "stress test": {"rule": "100 x point, 1 dp", "keys": [kp("c_asis R", f"diff.{regk['c_asis R']}.point")]},
               "without S4 counts": {"rule": "as above on the without-S4 block (run level: all_without_S4; per action: unit_without_S4 combined)",
                                     "keys": [(f"probes.{lb}.unit_without_S4[{pp[lb][0]}].blocks.combined.diff" if lv == "unit" else kp(lb, "diff").replace(".blocks.all.", ".blocks.all_without_S4."))
                                              for lb in mon]},
               "n under labels": {"rule": "n_cells", "keys": [f"{pp[lb][3]}.blocks.{b_}.n_cells" for lb in labs for b_ in (pp[lb][2], "S_newmains")]}}
        if lv == "unit":
            sh = [100 * pp[lb][1]["thresholds"]["union"]["realized_run_false_alarm_pool"] for lb in mon]
            src["per-run share"] = {"rule": "mean and range over the 9 probes of 100 x realized_run_false_alarm_pool, 1 dp",
                                    "keys": [f"{pp[lb][3]}.thresholds.union.realized_run_false_alarm_pool" for lb in mon]}
            head = f"At 1% per-action FPR (≈ {np.mean(sh):.1f}% of honest runs alarmed, mean over the {len(mon)} probes; range {min(sh):.1f}–{max(sh):.1f}%)"
        else:
            head = f"At {nm} per-run FPR (share of honest runs alarmed)"
        out[f"H2_{nm}"] = {"sources": src, "title": (f"{head}, sim − real (registered comparator) is within ±5 points for {c['registered_within_5']} of {len(mon)} probes, "
                                                     f"vs {c['rerun_within_5']} of {len(mon)} for rerunning (run 2 − run 1); point gaps {min(c['points']):+.1f} to "
                                                     f"{max(c['points']):+.1f}; stress test {c['stress']:+.1f}"),
                           "counts": cnt,
                           "note": (f"resolved (interval excludes 0): registered {c['registered_resolved']} of {len(mon)}, rerun {c['rerun_resolved']} of {len(mon)}; "
                                    f"sim − real (mean) within ±5: {c['real_mean_within_5']} of {len(mon)}; without S4: registered within ±5 for "
                                    f"{cnt['without_S4']['registered_within_5']} of {len(mon)}, rerun {cnt['without_S4']['rerun_within_5']} of {len(mon)}")}
    nn_ = {lb: next(q for q in P[lb]["run"]["points"] if abs(q["alpha"] - 0.05) < 1e-12)["blocks"]["all"]["n_cells"] for lb in labs}
    out["H2"] = {"caption": (out["H2_pa1%"]["title"] + "; " + out["H2_pa1%"]["note"] + ". Panels: per-action 1% (left; any call or tool output above the "
                             "per-action threshold alarms the run), then per-run 1% and 5%. " + out["H2_5%"]["title"] + f". Bars: real catch (mean of run 1 and run 2) vs sim, all origins pooled; "
                             f"{len(mon)} probes (stress test excluded) plus the stress test; † relaunch pending. Each bar's n is its own probe's task count, printed under its "
                             f"label (new main tasks in parentheses){'; coverage differs by probe' if len({nn_[lb] for lb in labs}) > 1 else '; every probe has the same tasks'}. ‡ {srtxt}. {PR}; {TS}. {pool}. {coh}. Error bars: 95% "
                             f"main-level task bootstrap. {PA} for the left panel. {lab}.")}
    q = next(q for q in rec["H3_sol_by_origin_strategy"]["points"] if abs(q["alpha"] - 0.05) < 1e-12)
    cells = [k for k in q["blocks"] if "|" in k and not k.startswith("S_newmains_without_S4")]
    re_ = [100 * q["blocks"][k]["tpr"]["real_mean"]["point"] for k in cells]
    si = [100 * q["blocks"][k]["tpr"]["sim"]["point"] for k in cells]
    hw3 = max(100 * (q["blocks"][k]["tpr"][w]["ci95"][1] - q["blocks"][k]["tpr"][w]["ci95"][0]) / 2 for k in cells for w in ("real_mean", "sim") if q["blocks"][k]["tpr"][w].get("ci95"))
    d3 = abs(np.ptp(si) - np.ptp(re_))
    which = ("similar spread across task sets" if d3 <= hw3 else
             ("sim catch varies less across task sets than real catch" if np.ptp(si) < np.ptp(re_) else "sim catch varies more across task sets than real catch"))
    j3 = next(j for j, q_ in enumerate(rec["H3_sol_by_origin_strategy"]["points"]) if abs(q_["alpha"] - 0.05) < 1e-12)
    src3 = {"real catch range": {"rule": "min and max of 100 x point, 0 dp", "keys": [f"H3_sol_by_origin_strategy.points[{j3}].blocks.{k}.tpr.real_mean.point" for k in cells]},
            "sim catch range": {"rule": "min and max of 100 x point, 0 dp", "keys": [f"H3_sol_by_origin_strategy.points[{j3}].blocks.{k}.tpr.sim.point" for k in cells]},
            "spread wording": {"rule": "'similar spread' unless |range(sim) - range(real)| > the largest bar's 95% half-width (100 x (ci95[1] - ci95[0]) / 2 over these bars)",
                               "keys": [f"H3_sol_by_origin_strategy.points[{j3}].blocks.{k}.tpr.{w}.ci95" for k in cells for w in ("real_mean", "sim")],
                               "values": {"range_difference": float(d3), "largest_half_width": float(hw3)}}}
    t3 = f"Sol (a) reply at 5%: real catch ranges {min(re_):.0f}–{max(re_):.0f}% across task sets, sim {min(si):.0f}–{max(si):.0f}%; {which}"
    out["H3"] = {"title": t3, "sources": src3, "caption": t3 + (f". Task sets: T196, +311 and new main tasks (with and without S4), each split by whether the task has a strategy "
                                               f"(approach_chars > 0; q2_approach_chars for new main tasks); n per bar in the labels. 5% {PR}; {TS}. {pool}. Error bars: 95% "
                                               f"main-level task bootstrap. {lab}.")}
    # V: 21j verdicts at 5% and 1%
    tab = {}
    for lb in labs:
        for qq in P[lb]["run"]["points"]:
            for pr, v in qq.get("verdicts_21j", {}).items():
                tab.setdefault(f"{qq['alpha']}", {}).setdefault(pr, {})[lb] = v["verdict"]
        for qq in P[lb].get("run_7b", {}).get("points", []):
            tab.setdefault(f"{qq['alpha']}", {}).setdefault("7b S_old|S_new", {})[lb] = qq["verdicts_21j"]["S_old|S_new"]["verdict"]
    cnt5 = {pr: {v: sum(x == v for lb, x in d.items() if not lb.startswith("c_asis")) for v in ("equivalent", "inconclusive at this n", "materially different", "decision unavailable")}
            for pr, d in tab.get("0.05", {}).items()}
    md5 = cnt5.get("S_old|S_newmains", {}).get("materially different", 0) + cnt5.get("S_new|S_newmains", {}).get("materially different", 0)
    eq5 = cnt5.get("S_old|S_newmains", {}).get("equivalent", 0) + cnt5.get("S_new|S_newmains", {}).get("equivalent", 0)
    tV = (f"At 5% honest-run alarm, the origin gap comparisons involving new main tasks are 'materially different' in {md5} and 'equivalent' in {eq5} of "
          f"{2 * len(mon)} probe × pair tests (21j, X = 5 points); T196 vs +311: {cnt5.get('S_old|S_new', {}).get('materially different', 0)} materially different, "
          f"{cnt5.get('S_old|S_new', {}).get('equivalent', 0)} equivalent of {len(mon)}")
    srcV = {"verdict counts": {"rule": "count of verdict strings over the non-stress probes", "keys": [
        f"probes.{lb}.run.points[{next(j for j, q in enumerate(P[lb]['run']['points']) if abs(q['alpha'] - 0.05) < 1e-12)}].verdicts_21j.{pr}.verdict"
        for lb in mon for pr in ("S_old|S_new", "S_old|S_newmains", "S_new|S_newmains")]}, "old rule": {"rule": "as stored", "keys": ["old_rule"]}}
    orr = rec["old_rule"]
    olds = "; ".join(f"{k} at {int(100 * p_['alpha'])}%: gate {p_['gate']} (differs from 21j: {p_['differs_from_21j']})" for k in ("extended", "7b")
                     for p_ in orr[k]["points"] if p_["alpha"] in (0.01, 0.05))
    out["V"] = {"title": tV, "sources": srcV, "verdict_table": tab, "counts_5pct_non_stress": cnt5, "old_rule": orr, "caption": (
        f"{tV}. D = Δ_a − Δ_b, Δ = sim − mean(run 1, run 2) catch, per probe and ordered origin pair; 'equivalent' iff the 90% interval lies inside ±5; "
        f"'materially different' iff the 95% interval lies wholly beyond ±5; otherwise 'inconclusive at this n'; pointwise, no familywise adjustment; "
        f"verdicts never change rows, order or prominence. Hollow markers: new main tasks without S4 (Aswin's sensitivity). Registered rule before 21j "
        f"(pool-if-match, TOST ±5), for comparison only: {olds}. {PR}; {TS}. {pool}. {coh}. {lab}.")}
    # appendix Figure A titles: median 95% half-width over readable, non-saturated points (r6 rule)
    A = {}
    for lb in labs:
        u = P[lb].get("unit")
        if not isinstance(u, list):
            continue
        reg = P[lb]["registered"]
        hw = {"gap": [], "rerun": []}
        for qq in u:
            bl = qq["blocks"]["combined"]
            if not bl["readable"] or any(bl["saturated"].values()):
                continue
            for k_, key in (("gap", reg), ("rerun", "run2_minus_run1")):
                ci = bl["diff"][key].get("ci95")
                if ci:
                    hw[k_].append(100 * (ci[1] - ci[0]) / 2)
        n = len(hw["gap"])
        ia = [i for i, qq in enumerate(u) if qq["blocks"]["combined"]["readable"] and not any(qq["blocks"]["combined"]["saturated"].values())]
        A[lb] = {"sources": {"rule": "median of 100 x (ci95[1] - ci95[0]) / 2 over readable, non-saturated points, 1 dp",
                             "keys": [f"probes.{lb}.unit[{i}].blocks.combined.diff.{k_}.ci95" for i in ia for k_ in (reg, "run2_minus_run1")]}, "n_points": n, "median_halfwidth_gap": float(np.median(hw["gap"])) if n else None, "median_halfwidth_rerun": float(np.median(hw["rerun"])) if hw["rerun"] else None,
                 "title": (f"{PLAIN[lb]}{' †' if pending(rec, lb) else ''}: median 95% half-width {np.median(hw['gap']):.1f} (sim − real) vs "
                           f"{np.median(hw['rerun']):.1f} (run 2 − run 1) points, {n} readable non-saturated FPRs") if n and hw["rerun"] else f"{PLAIN[lb]}: no readable non-saturated FPR"}
    out["A"] = {"panels": A, "caption": (f"Appendix: per probe, combined catch (real mean vs sim) and the registered gap with origin points (T196, +311, new main tasks), "
                                         f"vs target {PA.lower()}; top axis {PR.lower()}; {TS}. {pool}. {coh}. Titles: median 95% half-width over readable, non-saturated "
                                         f"points. † relaunch pending. {lab}.")}
    return out


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    return buf.getvalue()


@app.function(image=image, cpu=(4, 4), memory=(16384, 16384), timeout=900, retries=0, scaledown_window=2)
def render(blob: bytes, ident: dict) -> dict:
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rec = json.loads(blob)
    P = rec["probes"]
    tc = titles_and_captions(rec)
    plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6"})
    tc = {"source_record": ident | {"key_paths": "dot-separated from the record's top level; [i] is a list index; probe labels are dict keys"}} | tc
    out = {"captions.json": json.dumps(tc, indent=1).encode()}
    srf = tc["source_ratio_fail"]["probes"]
    dag = lambda lb: PLAIN[lb] + (" †" if pending(rec, lb) else "") + (" ‡" if lb in srf else "")  # noqa: E731
    # ---- H1: left the nine probes on the shared cells; right Sol (a) alone on all its cells (dashed: without S4)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
    same = rec["H1_common"].get("identical_to_without_S4")
    for ax, versions, ttl in ((axes[0], [("common", "-")] + ([] if same else [("common_without_S4", "--")]), tc["H1_common"]["n_text"]),
                              (axes[1], [("sol_all", "-"), ("sol_all_without_S4", "--")], "Sol (a) alone: " + tc["H1_sol_all"]["n_text"])):
        for nm, ls in versions:
            pts = rec[f"H1_{nm}"]["points"]
            ok = [q for q in pts if "mean_abs_sim_minus_real" in q]
            x = [q["fpr_target"] for q in ok]
            for key, col, lab in (("mean_abs_run2_minus_run1", GREY, "|run 2 − run 1| (rerunning the real eval)"), ("mean_abs_sim_minus_real", ORANGE, "|sim − real| (registered comparator)")):
                if ls == "-":
                    ax.fill_between(x, [100 * q[key]["ci95"][0] for q in ok], [100 * q[key]["ci95"][1] for q in ok], color=col, alpha=0.22, lw=0)
                ax.plot(x, [100 * q[key]["point"] for q in ok], color=col, lw=2.2 if ls == "-" else 1.2, ls=ls, label=lab + ("" if ls == "-" else ", without S4"))
        for yv, ls_ in ((5, "--"), (1, ":")):
            ax.axhline(yv, color="black", lw=0.8, ls=ls_)
        pts = rec[f"H1_{versions[0][0]}"]["points"]
        bad = [q["fpr_target"] for q in pts if "mean_abs_sim_minus_real" not in q]
        if bad:
            ax.axvspan(1e-4 / 1.2, max(bad) * 1.1, facecolor="none", hatch="////", edgecolor="#bdbdbd", lw=0, label="too few probes readable")
        ax.set_xscale("log")
        ax.set_xlim(1e-4 / 1.2, 1.2e-1)
        ax.set_ylim(0, None)
        ax.set_xlabel("honest FPR, per action (share of honest tool outputs or calls flagged)")
        ax.set_title(textwrap.fill(ttl, 70), fontsize=9, loc="left")
        sec = ax.secondary_xaxis("top")
        at = [1e-4, 1e-3, 1e-2, 1e-1]
        lab_ = []
        for xx in at:
            q = min(pts, key=lambda q_: abs(np.log10(q_["fpr_target"]) - np.log10(xx)))
            lab_.append(f"{100 * q['mean_run_fa_pool']:.1f}%" if abs(np.log10(q["fpr_target"]) - np.log10(xx)) < 1e-9 else "")
        sec.set_xticks(at)
        sec.set_xticklabels(lab_, fontsize=8.5)
        sec.set_xlabel("per run: share of honest runs alarmed", fontsize=8.5)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel("|gap| (points), lower = sim closer to real")
    fig.suptitle(textwrap.fill(tc["H1"]["title"], 120), x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out["H1_gap_lines.png"] = png(fig)
    plt.close(fig)
    # ---- H2
    labs = [lb for lb in PLAIN if lb in P and "points" in P[lb]["run"]]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.4), sharey=True)
    for ax, (nm, lv, al) in zip(axes, PANELS):
        xs = np.arange(len(labs))
        pp = {lb: panel_point(P[lb], lv, al) for lb in labs}
        for off, w, col, lab in ((-0.2, "real_mean", BLUE, "real (mean of run 1, run 2)"), (0.2, "sim", ORANGE, "sim")):
            vals, los, his = [], [], []
            for lb in labs:
                s = pp[lb][1]["blocks"][pp[lb][2]]["tpr"][w]
                vals.append(100 * s["point"])
                los.append(100 * (s["point"] - s["ci95"][0]))
                his.append(100 * (s["ci95"][1] - s["point"]))
            bars = ax.bar(xs + off, vals, 0.4, color=col, label=lab, yerr=[los, his], capsize=2, error_kw={"lw": 0.8})
            for bb, v in zip(bars, vals):
                ax.text(bb.get_x() + bb.get_width() / 2, 2, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color="white", rotation=90)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{dag(lb)}\nn = {pp[lb][1]['blocks'][pp[lb][2]]['n_cells']} (new main {pp[lb][1]['blocks']['S_newmains']['n_cells']})" for lb in labs],
                           rotation=40, ha="right", fontsize=8)
        ax.set_title(textwrap.fill(tc[f"H2_{nm}"]["title"], 70), fontsize=9, loc="left")
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("catch (%) at the honest FPR in each panel title")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(0.01, 0.005, "† relaunch pending (pool without the 958 relaunch runs); ‡ pool sources disagree beyond 2× (source-ratio check)", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out["H2_catch_bars.png"] = png(fig)
    plt.close(fig)
    # ---- H3
    q = next(q for q in rec["H3_sol_by_origin_strategy"]["points"] if abs(q["alpha"] - 0.05) < 1e-12)
    cells = [f"{o}|{s}" for o in ("S_old", "S_new", "S_newmains", "S_newmains_without_S4") for s in ("strategy", "no_strategy")]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    xs = np.arange(len(cells))
    for off, w, col, lab in ((-0.2, "real_mean", BLUE, "real (mean of run 1, run 2)"), (0.2, "sim", ORANGE, "sim")):
        vals = [100 * q["blocks"][c]["tpr"][w]["point"] for c in cells]
        lo = [100 * (q["blocks"][c]["tpr"][w]["point"] - q["blocks"][c]["tpr"][w]["ci95"][0]) if q["blocks"][c]["tpr"][w].get("ci95") else 0 for c in cells]
        hi = [100 * (q["blocks"][c]["tpr"][w]["ci95"][1] - q["blocks"][c]["tpr"][w]["point"]) if q["blocks"][c]["tpr"][w].get("ci95") else 0 for c in cells]
        bars = ax.bar(xs + off, vals, 0.4, color=col, label=lab, yerr=[lo, hi], capsize=3)
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=8.5, color="white")
    ax.set_xticks(xs)
    short = {"S_old": "T196", "S_new": "+311", "S_newmains": "new main", "S_newmains_without_S4": "new main,\nno S4"}
    ax.set_xticklabels([f"{short[c.split('|')[0]]}\n{'with' if c.endswith('|strategy') else 'no'} strategy\n(n = {q['blocks'][c]['n_cells']})" for c in cells], fontsize=8.5)
    ax.set_ylabel("catch (%) at 5% per-run honest alarm")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(textwrap.fill(tc["H3"]["title"], 100), x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out["H3_sol_origin_strategy.png"] = png(fig)
    plt.close(fig)
    # ---- V: D per probe x pair at 5% and 1%
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True, sharex=True)
    mk = {"equivalent": "E", "materially different": "M", "inconclusive at this n": "I", "decision unavailable": "–"}
    for i, al in enumerate((0.05, 0.01)):
        for j, pr in enumerate(PAIRS):
            ax = axes[i, j]
            ax.axvspan(-5, 5, color="#f2f2f2", zorder=0)
            ax.axvline(0, color="black", lw=0.8)
            for r, lb in enumerate(labs):
                qq = next(q for q in P[lb]["run"]["points"] if abs(q["alpha"] - al) < 1e-12)
                v = qq["verdicts_21j"].get(pr)
                yy = len(labs) - r
                if v and v.get("ci95"):
                    ax.plot([100 * v["ci95"][0], 100 * v["ci95"][1]], [yy, yy], color=ORANGE, lw=1.0)
                    ax.plot([100 * v["ci90"][0], 100 * v["ci90"][1]], [yy, yy], color=ORANGE, lw=2.6)
                    ax.plot(100 * v["D_point"], yy, "o", color=ORANGE, ms=4)
                    ax.text(38, yy, mk[v["verdict"]], fontsize=8, ha="right", va="center")
                ns = pr.replace("S_newmains", "S_newmains_without_S4")
                v2 = qq["verdicts_21j"].get(ns) if ns != pr else None
                if v2:
                    ax.plot(100 * v2["D_point"], yy - 0.3, "o", mfc="none", color="black", ms=3.5)
            ax.set_xlim(-40, 40)
            a_, b_ = pr.split("|")
            ax.set_title(f"{NICE[a_]} vs {NICE[b_]}, {int(100 * al)}% per run", fontsize=9.5)
            if i == 1:
                ax.set_xlabel("D = Δ_a − Δ_b (points; Δ = sim − real mean)")
    axes[0, 0].set_yticks(range(len(labs), 0, -1))
    axes[0, 0].set_yticklabels([dag(lb) for lb in labs], fontsize=8)
    for ax in axes.flat:
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(textwrap.fill(tc["V"]["title"], 120), x=0.01, ha="left", fontsize=11)
    fig.text(0.01, 0.005, "thick: 90% interval; thin: 95% interval; E equivalent, I inconclusive at this n, M materially different; hollow: without S4; † relaunch pending",
             fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    out["V_verdicts.png"] = png(fig)
    plt.close(fig)
    # ---- appendix Figure A
    la = [lb for lb in labs if isinstance(P[lb].get("unit"), list)]
    ncol = 5
    nrow = 2 * int(np.ceil(len(la) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.1 * nrow), squeeze=False)
    for k, lb in enumerate(la):
        r0, c0 = 2 * (k // ncol), k % ncol
        u = [q for q in P[lb]["unit"] if q["thresholds"]["union"]["estimable"]]
        x = [q["fpr_target"] for q in u]
        reg = P[lb]["registered"]
        ax = axes[r0, c0]
        for w, col, labw in (("real_mean", BLUE, "real (mean)"), ("sim", ORANGE, "sim")):
            s = [q["blocks"]["combined"]["tpr"][w] for q in u]
            ax.fill_between(x, [100 * t["ci95"][0] if t.get("ci95") else np.nan for t in s], [100 * t["ci95"][1] if t.get("ci95") else np.nan for t in s], color=col, alpha=0.22, lw=0)
            ax.plot(x, [100 * t["point"] for t in s], color=col, lw=1.4, label=labw)
        ax.set_xscale("log")
        ax.set_ylim(0, 100)
        ax.set_title(textwrap.fill(tc["A"]["panels"][lb]["title"], 48), fontsize=7, loc="left")
        ax = axes[r0 + 1, c0]
        ax.axhspan(-5, 5, color="#f2f2f2", zorder=0)
        ax.axhline(0, color="black", lw=0.8)
        for key, col, labk, ls in (("run2_minus_run1", GREY, "run 2 − run 1", "--"), (reg, ORANGE, "registered gap", "-")):
            s = [q["blocks"]["combined"]["diff"][key] for q in u]
            ax.fill_between(x, [100 * t["ci95"][0] if t.get("ci95") else np.nan for t in s], [100 * t["ci95"][1] if t.get("ci95") else np.nan for t in s], color=col, alpha=0.2, lw=0)
            ax.plot(x, [100 * t["point"] for t in s], color=col, lw=1.2, ls=ls, label=labk)
        for o, ls, col in (("S_old", ":", "black"), ("S_new", "-.", "black"), ("S_newmains", "-", "#756bb1")):
            ax.plot(x, [100 * q["blocks"][o]["diff"][reg]["point"] for q in u], color=col, lw=0.8, ls=ls, label=NICE[o])
        ax.set_xscale("log")
        ax.set_ylim(-30, 30)
        ax.set_xlabel("per action FPR", fontsize=7.5)
    for k in range(len(la), (nrow // 2) * ncol):
        axes[2 * (k // ncol), k % ncol].axis("off")
        axes[2 * (k // ncol) + 1, k % ncol].axis("off")
    h, l_ = [], []
    for ax in axes.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l_:
                h.append(hh), l_.append(ll)
    fig.legend(h, l_, loc="lower center", ncol=7, frameon=False, fontsize=8)
    fig.suptitle("Appendix: catch and gap vs per-action honest FPR, per probe (threshold source: real honest)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out["A_appendix_unit.png"] = png(fig)
    plt.close(fig)
    return out


@app.local_entrypoint()
def main(record: str, commit: str):
    nm, sha = record.rsplit("@", 1)
    b = (HERE / "results" / nm).read_bytes()
    if hashlib.sha256(b).hexdigest() != sha:
        raise SystemExit(f"{nm}: sha256 differs")
    ident = {"file": f"experiments/2026-10-15_fpr_sweep_507_prep/results/{nm}", "sha256": sha, "commit": commit}
    if subprocess.run(["git", "cat-file", "-e", f"{commit}:{ident['file']}"], cwd=HERE).returncode != 0:
        raise SystemExit("the record is not in that commit")
    out = render.remote(b, ident)
    stem = nm.removesuffix(".json")
    for fn, blob in out.items():
        (HERE / "results" / f"{stem}_{fn}").open("xb").write(blob)
        print(f"{stem}_{fn}", hashlib.sha256(blob).hexdigest())
