"""For every number in the E1 and appendix titles and captions: the record file @ commit and the exact key paths, with the value re-read from
those paths and asserted equal to the printed number (0d's by-key recheck). Reads committed JSON only.

  python3 experiments/2026-10-15_e2e_sim_pipeline/make_captions.py <record stem> <record commit>
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results"
stem, commit = sys.argv[1], sys.argv[2]
A25_GIT = "63787479b:experiments/2026-10-14_simhonest_a25_09/results/a25_full_2026-10-14_232415.json"
rec = json.loads((R / f"{stem}.json").read_text())
a25 = json.loads(subprocess.run(["git", "show", A25_GIT], capture_output=True, check=True, cwd=HERE).stdout)["result"]
_e1, _ap = R / f"{stem}_E1_figure_numbers.json", R / f"{stem}_e2e_figure_numbers.json"
if not (_e1.exists() and _ap.exists()):  # a record rendered only as the per-action lead: E1/appendix sections are skipped
    e1 = {"title": "", "median_halfwidths_reply": [None, None], "numbers": []}
    ap = {"title": ""}
    SKIP_RUN = True
else:
    e1, ap = json.loads(_e1.read_text()), json.loads(_ap.read_text())
    SKIP_RUN = False
REC = f"experiments/2026-10-15_e2e_sim_pipeline/results/{stem}.json @ {commit}"
A25 = "experiments/2026-10-14_simhonest_a25_09/results/a25_full_2026-10-14_232415.json @ 63787479b (aswin/a25-multibatch-09)"
FAM5 = ["Sol (a)", "FIBS", "harmfulness", "refusal", "high-stakes"]
A25N = {"Sol (a)": "sol_a", "FIBS": "fibs", "harmfulness": "harmfulness", "refusal": "refusal", "high-stakes": "highstakes"}


def get(d, path):
    for k in path:
        d = d[k]
    return d


out = {"E1": [], "appendix": []}
if not SKIP_RUN:  # Codex v5b: a lead-only record (no E1/appendix renders) skips every run-level section
    mp = lambda lab, *k: ["rows", lab, "run", "run0.05", "matched", *k]  # noqa: E731
    # E1: within counts, median half-widths, n's
    for ro, txt in (("R", "reply: the gap point estimate lies inside the 95% interval of the two-real-pipeline contrast for {n} of 5 probes"), ("C", "command: {n} of 5")):
        paths, n = [], 0
        for f in FAM5:
            pe, py = mp(f"{f} {ro}", "gap", "e2e_vs_real_mean", "point"), mp(f"{f} {ro}", "gap", "yardstick_run2atC_minus_run1atB", "ci95")
            e, y = get(rec, pe), get(rec, py)
            n += y[0] <= e <= y[1]
            paths += [pe, py]
        out["E1"].append({"text": txt.format(n=n), "value": n, "source": REC, "key_paths": paths, "derivation": "count of probes with point (x100) inside ci95 (x100)"})
    OWN = [f for f in ("own_a", "own_frac10_L48", "own_frac25_L48", "own_frac100_L48") if f"{f} R" in rec["rows"]]
    for ro in ("R", "C"):
        paths, n = [], 0
        for f in OWN:
            pe, py = mp(f"{f} {ro}", "gap", "e2e_vs_run2", "point"), mp(f"{f} {ro}", "gap", "yardstick_run2atC_minus_run1atB", "ci95")
            e, y = get(rec, pe), get(rec, py)
            n += y[0] <= e <= y[1]
            paths += [pe, py]
        if OWN:
            assert e1["title"].count(f"{'for' if ro == 'R' else 'command:'} {n} of {len(OWN)}") >= 1, f"own-real {ro} count {n} not in the E1 title"
            out["E1"].append({"text": f"own-real against run 2, {'reply' if ro == 'R' else 'command'}: {n} of {len(OWN)}", "value": n, "source": REC, "key_paths": paths,
                              "derivation": "count of own-real probes with the e2e_vs_run2 point (x100) inside the yardstick ci95 (x100)"})
    for which in ("honest_only", "attack_only"):
        paths = [mp(f"{f} R", "halfwidth_points_by_resampling", "e2e_vs_real_mean", which) for f in FAM5]
        v = sorted(get(rec, p) for p in paths)[2]
        printed = e1["median_halfwidths_reply"][0 if which == "honest_only" else 1]
        assert v == printed
        out["E1"].append({"text": f"{'honest' if which == 'honest_only' else 'attack'} ±{v:.1f} (median half-width, reply, 5 probes)", "value": v, "source": REC,
                          "key_paths": paths, "derivation": "median of the five values (points)"})
    Ms = sorted({get(rec, ["rows", lab, "M"]) for lab in rec["rows"]})
    out["E1"].append({"text": f"{'/'.join(map(str, Ms))} sim-honest and real-honest runs (matched tasks)", "value": Ms, "source": REC, "key_paths": [["rows", "<every label>", "M"]],
                      "derivation": "matched task count M; each matched pool has one run per task"})
    out["E1"].append({"text": "183 attack tasks", "value": get(rec, ["rows", "Sol (a) R", "coverage", "attack_tasks"]), "source": REC,
                      "key_paths": [["rows", "<every label>", "coverage", "attack_tasks"], ["design", "cohort_tasks"]]})
    out["E1"].append({"text": "90 tasks (A25 estimation half)", "value": get(rec, ["design", "estimation_half_tasks"]), "source": REC, "key_paths": [["design", "estimation_half_tasks"]]})
    for x in e1["numbers"]:
        lab = f"{x['probe']} {x['readout']}"
        gk = x.get("comparator", "e2e_vs_real_mean")
        assert gk == ("e2e_vs_run2" if x["probe"].startswith("own_") else "e2e_vs_real_mean"), f"{lab}: wrong comparator {gk}"
        g_ = get(rec, mp(lab, "gap", gk))
        h_ = get(rec, mp(lab, "halfwidth_points_by_resampling", gk))
        assert all(abs(100 * u - v) < 1e-9 for u, v in zip(g_["ci95"], x["e2e_ci95"])), f"{lab}: interval"
        assert all(abs(h_[k] - x[f"halfwidth_{k}"]) < 1e-12 for k in ("joint", "honest_only", "attack_only")), f"{lab}: half-widths"
        out["E1"].append({"text": f"{lab}: e2e ({gk}) {x['e2e_point']:+.1f} [{x['e2e_ci95'][0]:+.1f}, {x['e2e_ci95'][1]:+.1f}], ±{x['halfwidth_honest_only']:.1f} / ±{x['halfwidth_attack_only']:.1f}",
                          "source": REC, "key_paths": [mp(lab, "gap", gk), mp(lab, "halfwidth_points_by_resampling", gk)],
                          "derivation": "point and ci95 x100; half-widths as stored (points)"})
        assert abs(100 * get(rec, mp(lab, "gap", gk, "point")) - x["e2e_point"]) < 1e-9
    # appendix title
    paths, n_reg, n_res = [], 0, 0
    for ro in ("R_reply_meanpooled", "C_anycommand_max"):
        r = a25["rows"][f"{ro}|mechanical|alpha=0.05"]
        for nm in r["primary"]["roster"]:
            n_reg += 1
            n_res += bool(r["primary"]["full_minus_mean_real"]["pooled"]["per_probe"][nm]["nonzero_95"])
            paths.append(["result", "rows", f"{ro}|mechanical|alpha=0.05", "primary", "full_minus_mean_real", "pooled", "per_probe", nm, "nonzero_95"])
    out["appendix"].append({"text": f"registered A25 at 5%: excludes 0 at {n_res} of {n_reg} eligible points", "value": [n_res, n_reg], "source": A25, "key_paths": paths,
                            "derivation": "eligible = primary.roster; count nonzero_95"})
    paths, n_ph, r_ph, n_yd, r_yd = [], 0, 0, 0, 0
    for f in FAM5:
        for ro in ("R", "C"):
            for pt in ("run0.05", "run0.1"):
                pe = ["rows", f"{f} {ro}", "run", pt, "matched", "gap", "e2e_vs_real_mean", "ci95"]
                py = ["rows", f"{f} {ro}", "run", pt, "matched", "gap", "yardstick_run2atC_minus_run1atB", "ci95"]
                e, y = get(rec, pe), get(rec, py)
                n_ph += 1
                r_ph += not (e[0] <= 0 <= e[1])
                if y is not None:
                    n_yd += 1
                    r_yd += not (y[0] <= 0 <= y[1])
                paths += [pe, py]
    out["appendix"].append({"text": f"post hoc 183, 5% and 10%: e2e excludes 0 at {r_ph} of {n_ph}; two real pipelines {r_yd} of {n_yd}", "value": [r_ph, n_ph, r_yd, n_yd],
                            "source": REC, "key_paths": paths, "derivation": "non-stress rows; matched design; interval excludes 0"})
    pth = ["rows", "Sol (a) C", "run", "run0.05", "matched", "gap", "threshold_shift", "point"]
    out["appendix"].append({"text": f"largest threshold-shift part {100 * get(rec, pth):+.1f} (Sol (a) C 5%)", "value": 100 * get(rec, pth), "source": REC, "key_paths": [pth],
                            "derivation": "max |threshold_shift.point| x100 over non-stress rows at 5% and 10%, matched"})
    pools = {}
    for ro in ("R_reply_meanpooled", "C_anycommand_max"):
        for f, nm in A25N.items():
            p_ = ["result", "rows", f"{ro}|mechanical|alpha=0.05", "per_probe", nm]
            q = get({"result": a25}, p_)
            pools[f"{f} {ro}"] = {"real_pool": q["real_pool"], "real_pool_n": q["real_pool_n"], "n_sim": q["n_sim"], "key_path": p_ + ["real_pool_n"]}
    out["appendix"].append({"text": "A25 threshold sources: sim honest 149 runs; real side v6 pools (B ∪ C 386 or B 194 runs, per probe)", "value": pools, "source": A25})
    cv = get(rec, ["rows", "Sol (a) R", "coverage"])
    out["appendix"].append({"text": f"coverage: {cv['attack_tasks']} attack tasks; sim honest {cv['simh_runs']} runs on {cv['simh_distinct_tasks']} tasks, {cv['simh_tasks_in_attack_cohort']} in the cohort, "
                            f"{cv['attack_tasks_without_simh']} attack tasks without", "value": cv, "source": REC, "key_paths": [["rows", "<every label>", "coverage"]]})
    assert e1["title"].count(f"for {out['E1'][0]['value']} of 5") == 1 and ap["title"].count(f"at {n_res} of {n_reg}") == 1 and ap["title"].count(f"{r_ph} of {n_ph}") == 1
# ---------- headline PA (per-action e2e, matched) and the transfer appendix (TB/TC), when rendered ----------
titles = {"E1": e1["title"], "appendix": ap["title"]}
pa_f = R / f"{stem}_PA_figure_numbers.json"
if pa_f.exists():
    pa = json.loads(pa_f.read_text())
    out["PA"] = []
    for key, blk in pa["PA"].items():
        titles[f"PA {key}"] = blk["title"]
        for x in blk["rows"]:
            lab = f"{x['probe']} {x['readout']}"
            base = ["rows", lab, "unit", key, "matched"]
            g_ = get(rec, base + ["gap", x["comparator"]])
            assert x["comparator"] in (("e2e_vs_run2", "e2e_vs_real_mean") if x["probe"].startswith(("own_", "DT_")) else ("e2e_vs_real_mean",))
            if x["e2e"]:
                assert all(abs(100 * u - v) < 1e-9 for u, v in zip([g_["point"], *g_["ci95"]], x["e2e"])), f"PA {key} {lab}"
            assert get(rec, base + ["readable"]) == x["readable"] and get(rec, base + ["run_share_alarmed_draw0"]) == x["runs_alarmed_draw0"]
            out["PA"].append({"text": f"PA {key} {lab}: e2e {x['e2e']}, yardstick {x['yardstick']}, readable {x['readable']}, runs alarmed {x['runs_alarmed_draw0']}", "source": REC,
                              "key_paths": [base + ["gap", x["comparator"]], base + ["gap", "yardstick_run2atC_minus_run1atB"], base + ["readable"], base + ["run_share_alarmed_draw0"]]})
        NC = rec["design"]["cohort_tasks"]
        cv = blk["coverage"]
        assert all(get(rec, ["rows", lab_, "coverage"]) == cv for lab_ in rec["rows"] if "coverage" in rec["rows"][lab_])
        assert blk["caption"].count(f"cover {cv['simh_tasks_in_attack_cohort']} of the {cv['attack_tasks']} attack tasks; {cv['attack_tasks_without_simh']} attack tasks have no") == 1
        out["PA"].append({"text": f"PA {key} caption: sim-honest runs cover {cv['simh_tasks_in_attack_cohort']} of {cv['attack_tasks']} attack tasks; {cv['attack_tasks_without_simh']} without",
                          "value": cv, "source": REC, "key_paths": [["rows", "<every label>", "coverage"]], "derivation": "identical on every row (asserted)"})
        DT4 = [f for f in ("DT_destructive_primary", "DT_destructive_secondary", "DT_transfer_primary", "DT_transfer_secondary") if f"{f} C" in rec["rows"]]
        if DT4:
            cp = lambda f, w: ["rows", f"{f} C", "unit", key, "matched", "catch", w, "point"]  # noqa: E731
            hs = [round(get(rec, cp(f, "full")) * NC) for f in DT4]
            hr = [round(get(rec, cp(f, w)) * NC) for f in DT4 for w in ("run1", "run2")]
            assert hs == blk["dt_caption"]["sim_hits"] and hr == blk["dt_caption"]["real_hits_run1_run2"], f"PA {key} DT caption"
            txt = f"sim catch {'/'.join(map(str, hs))} of {NC} attack tasks"
            txt2 = f"real catch (run 1 and run 2) {min(hr)}–{max(hr)} of {NC} ({100 * min(hr) / NC:.0f}–{100 * max(hr) / NC:.0f}%)"
            assert blk["caption"].count(txt) == 1 and blk["caption"].count(txt2) == 1, f"PA {key} DT caption text"
            out["PA"].append({"text": f"PA {key} caption, DT: {txt}; {txt2}", "value": {"sim_hits": hs, "real_hits_run1_run2": hr, "NC": NC}, "source": REC,
                              "key_paths": [cp(f, w) for f in DT4 for w in ("full", "run1", "run2")], "derivation": "catch point x cohort_tasks, rounded; real range over run 1 and run 2 of the four DT rows"})
        DTI = [f"{f}_imp" for f in ("DT_destructive_primary", "DT_destructive_secondary", "DT_transfer_primary", "DT_transfer_secondary") if f"{f}_imp C" in rec["rows"]]
        if DTI:  # v5b
            cp = lambda f, w: ["rows", f"{f} C", "unit", key, "matched", "catch", w, "point"]  # noqa: E731
            hs = [round(get(rec, cp(f, "full")) * NC) for f in DTI]
            hr = [round(get(rec, cp(f, w)) * NC) for f in DTI for w in ("run1", "run2")]
            nu = [get(rec, ["rows", f"{f} C", "unit", key, "matched", "n_units"]) for f in DTI]
            assert hs == blk["dti_caption"]["sim_hits"] and hr == blk["dti_caption"]["real_hits_run1_run2"] and nu == blk["dti_caption"]["n_units"], f"PA {key} DTI caption"
            txt = (f"sim catch {'/'.join(map(str, hs))} of {NC}; real catch (run 1 and run 2) {min(hr)}–{max(hr)} of {NC}; honest units "
                   "(real honest B / sim-honest, per design) "
                   f"{', '.join(str(x_['real']) + '/' + str(x_['sim']) for x_ in nu)}")
            assert blk["caption"].count(txt) == 1, f"PA {key} DTI caption text"
            out["PA"].append({"text": f"PA {key} caption, DT impactful-only: {txt}", "value": blk["dti_caption"], "source": REC,
                              "key_paths": [cp(f, w) for f in DTI for w in ("full", "run1", "run2")] + [["rows", f"{f} C", "unit", key, "matched", "n_units"] for f in DTI],
                              "derivation": "catch point x cohort_tasks, rounded; real range over run 1 and run 2 of the four rows; n_units as stored"})
        n_in = {ro: sum(1 for x in blk["rows"] if x["readout"] == ro and not x["probe"].startswith(("own_", "DT_")) and x["probe"] != "c_asis" and x["e2e"] and x["yardstick"]
                        and x["readable"]["sim"] and x["readable"]["real"] and not x.get("saturated") and x["yardstick"][1] <= x["e2e"][0] <= x["yardstick"][2]) for ro in ("R", "C")}
        assert blk["title"].count(f"contrast for {n_in['R']} of ") == 1 and blk["title"].count(f"and {n_in['C']} of ") >= 1, f"PA {key} title counts"
        out["PA"].append({"text": f"PA {key} title: inside counts reply {n_in['R']}, command {n_in['C']}", "value": n_in, "source": REC,
                          "derivation": "non-stress, non-own probes, readable on both sides, e2e point inside the yardstick ci95"})
        five = [x["runs_alarmed_draw0"] for x in blk["rows"] if not x["probe"].startswith(("own_", "DT_")) and x["probe"] != "c_asis" and "unit" in rec["rows"][f"{x['probe']} {x['readout']}"]]
        if five:
            ms = sorted(v["sim"] for v in five)
            mr = sorted(v["real"] for v in five)
            import statistics
            assert blk["title"].count(f"sim {100 * statistics.median(ms):.0f}% / real {100 * statistics.median(mr):.0f}%") == 1, f"PA {key} run-share medians"
            out["PA"].append({"text": f"PA {key} title: five-probe runs-alarmed medians sim {100 * statistics.median(ms):.0f}% / real {100 * statistics.median(mr):.0f}%",
                              "source": REC, "key_paths": "rows[<five headline probes> R and C].unit[key].matched.run_share_alarmed_draw0", "derivation": "median over both readouts' rows"})
tf = R / f"{stem}_transfer_figure_numbers.json"
if tf.exists():
    tr = json.loads(tf.read_text())
    out["transfer"] = [{"text": tr["TB"][k], "source": REC, "key_paths": "rows[*].unit[pt].as_is or rows[*].run[pt].as_is: catch.{full,real_mean_at_sim,run1_at_sim,run2_at_sim}, "
                        "gap.{transfer_real_mean_minus_sim,yardstick_transfer_run2_minus_run1_at_B}, cross_unit_fpr / cross_false_alarm"} for k in tr["TB"] if k.startswith("title_")]
    out["transfer"] += [{"text": v["title"], "source": REC, "key_paths": [["curve", ro, "points"]]} for ro, v in tr["TC"].items()]
dest = R / f"{stem}_captions.json"
dest.write_text(json.dumps({"label": "every title and caption number with its record @ commit and key paths; values re-read and asserted", "titles": titles,
                            "E1_title": e1["title"], "appendix_title": ap["title"], "numbers": out}, indent=1) + "\n")
print(dest, sum(len(v) for v in out.values()), "entries")
