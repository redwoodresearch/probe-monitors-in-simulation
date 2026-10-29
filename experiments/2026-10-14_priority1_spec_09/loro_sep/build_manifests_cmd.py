"""Build the +311 (S_new) and pooled-455 separate-threshold LORO manifest entries for the COMMAND readout from METADATA ONLY:
row keys, readouts, coverage counts, fold -1 drops, moments sources, calibration-main deviations, the npz pointer and the producer
code's key construction. Never reads a gap, catch, threshold or FPR value.
Usage: python3 -I build_manifests_cmd.py <pooled455_C.json> <pooled455_R.json> <hotdeck_C.json> <hotdeck_R.json> <hotdeck_O.json> <hotdeck_X.json> <out_311.json> <out_455.json>"""

import hashlib
import json
import sys

pooledC_p, pooledR_p, hdC_p, hdR_p, hdO_p, hdX_p, out311, out455 = sys.argv[1:9]


def load(p):
    b = open(p, "rb").read()
    return json.loads(b), hashlib.sha256(b).hexdigest()


PP = {"C": load(pooledC_p), "R": load(pooledR_p)}
HD = {g: load(p) for g, p in (("C", hdC_p), ("R", hdR_p), ("O", hdO_p), ("X", hdX_p))}
assert PP["C"][1] == "587739d0b7d9a7fc7e10a292927fc06dc5d460fe91e606b3ae306c94c508b3c4" and PP["R"][1] == "c794e175a324b17e2eac45cac06824463acc0eff655def7d07efc6d371214bbf"
NPZ = {ro: {"volume": PP[ro][0]["npz"]["volume"], "path": PP[ro][0]["npz"]["path"], "sha256": PP[ro][0]["npz"]["sha256"], "n_arrays": PP[ro][0]["npz"]["n_arrays"],
            "key_format": PP[ro][0]["npz"]["key_format"]} for ro in ("C", "R")}
assert NPZ["C"]["sha256"] == "84112f5061455f9b1970ac90ff963e31d38f8c75b26b3cf28510e2c06476dbc8" and NPZ["R"]["sha256"] == "bd5875fbd09fc9177783f7bd84f6b862dae1d2eb8818f2d25b1461b1bd165a11"
assert PP["C"][0]["meta"]["script_sha256"] == PP["R"][0]["meta"]["script_sha256"]
P = {"rows": {**PP["C"][0]["rows"], **PP["R"][0]["rows"]}, "pooled": {**PP["C"][0]["pooled"], **PP["R"][0]["pooled"]}}
PROD = {"script": "experiments/2026-10-15_fpr_sweep_507_prep/run_peraction311.py", "git_C": PP["C"][0]["meta"]["git"], "git_R": PP["R"][0]["meta"]["git"], "script_sha256": PP["C"][0]["meta"]["script_sha256"],
        "key_construction": "pool_reduce(): arrs[f'{label}|{mode}|{f}|{gap}'] (pooled) and arrs[f'{label}|{mode}|{f}|{origin}|{catch}'] (per origin), saved with '|'->'__', ' '->'_', '(' and ')' removed, '@'->'_at_'; f in ('0.01', '0.001', '0.05') (F = (0.01, 0.001, 0.05), f'{f:g}')"}
REC_POOLED = {"C": {"path": "experiments/2026-10-15_fpr_sweep_507_prep/results/pooled455_C_2026-10-15_141921.json", "commit": "6d2a2fbc5", "branch": "origin/aswin/fpr-sweep-507-prep", "sha256": PP["C"][1]},
              "R": {"path": "experiments/2026-10-15_fpr_sweep_507_prep/results/pooled455_R_2026-10-15_142548.json", "commit": "45db6b238", "branch": "origin/aswin/fpr-sweep-507-prep", "sha256": PP["R"][1]}}
HD_NAMES = {"C": "peraction311_all_hotdeck_C_2026-10-15_140206.json", "R": "peraction311_all_hotdeck_R_2026-10-15_140206.json", "O": "peraction311_all_hotdeck_O_2026-10-15_140209.json", "X": "peraction311_all_hotdeck_X_2026-10-15_140208.json"}
REC_HD = {g: {"path": f"experiments/2026-10-15_fpr_sweep_507_prep/results/{HD_NAMES[g]}", "commit": "c838127d3", "sha256": HD[g][1]} for g in HD}


def san(s):
    return s.replace("|", "__").replace(" ", "_").replace("(", "").replace(")", "").replace("@", "_at_")


FAMS = ["Sol (a)", "FIBS", "harmfulness", "refusal", "high-stakes", "c_asis", "own_frac10_L48", "own_frac25_L48", "own_frac100_L48"]
LABS = [f"{f} {ro}" for ro in ("C", "R") for f in FAMS]
assert sorted(LABS) == sorted(P["rows"]) == sorted(P["pooled"])
PUBLIC = {"FIBS", "harmfulness", "refusal", "high-stakes"}
PTS = [("0.01", "LEAD"), ("0.001", "beside"), ("0.05", "beside (exact 5e-2)")]
MODES = ("joint", "fixed")


def group_of(fam):
    if fam == "Sol (a)":
        return "other-model-real", "Sol's real runs (other model; no Qwen3.5 simulated data)", "fitted member"
    if fam in PUBLIC:
        return f"public:{fam}", f"public direction ({fam}); not trained on the study's runs", "fitted member"
    if fam == "c_asis":
        return "STRESS", "uses simulated training or adaptation data (STRESS per LORO RULE @ 8d59e2989 and 25h item 8)", "STRESS (never in any fit; descriptive position)"
    return "own-real", "Qwen3.5-27B's own real runs (run 1; LORO RULE @ 8d59e2989); own_frac = own-real-trained variants of the own_a recipe", "fitted member"


def hd_source(lab):
    fam, ro = lab[:-2], lab[-1]
    return "X" if fam == "c_asis" else ("O" if fam.startswith("own_") else ro)


def rows_for(kind):
    out = []
    for i, lab in enumerate(LABS):
        fam = lab[:-2]
        g, world, role = group_of(fam)
        k = san(lab)
        cov_new, cov_old = P["rows"][lab]["S_new"]["coverage"], P["rows"][lab]["S_old"]["coverage"]
        r = {"row_index": i, "inventory_key": lab, "npz_label": k, "family": fam, "readout": lab[-1], "npz": f"input_provenance.npz.{lab[-1]}", "loro_group": g, "training_world": world, "role": role,
             "S_new": {"attack_cohort_cells": f"{cov_new['cohort_cells']} of {cov_new['of']}", "pools_runs": {"H_B": cov_new["H_B_runs"], "H_A": cov_new["H_A_runs"], "H_sim": cov_new["H_sim_runs"]},
                       "H_sim_by_batch": cov_new["H_sim_runs_by_batch"], "fold_minus1_rule": P["rows"][lab]["S_new"]["fold_minus1_rule"],
                       "fold_minus1_dropped_counts": {kk: v["n"] for kk, v in P["rows"][lab]["S_new"]["fold_minus1_dropped"].items()}}}
        if kind == "311":
            r["npz_keys"] = {m: {f: {"Δ_full": [f"{k}__{m}__{f}__S_new__sim_at_S__primary", "minus", f"{k}__{m}__{f}__S_new__real_mean_at_B"],
                                     "Δ_rr": [f"{k}__{m}__{f}__S_new__run2_at_A", "minus", f"{k}__{m}__{f}__S_new__run1_at_B"]} for f, _ in PTS} for m in MODES}
            r["draw0_gates"] = {"hotdeck_record": f"{hd_source(lab)}: rows['{lab}'].points.{{joint,fixed}}[f].gap.{{delta_full,delta_rr}}.point",
                                "pooled_record_per_set": f"rows['{lab}'].S_new.points.{{joint,fixed}}[f].gap.{{delta_full,delta_rr}}.point"}
            r["readability_reported_not_used_for_membership"] = f"pooled record rows['{lab}'].S_new.readable[f].{{S,B,A}} and .pools_at_draw0[f].{{S,B,A}}.{{n_units,units_above}}"
        else:
            r["S_old"] = {"attack_cohort_cells": f"{cov_old['cohort_cells']} of {cov_old['of']}", "pools_runs": {"H_B (B ∪ C)": cov_old["H_B_runs"], "H_A (honest C)": cov_old["H_A_runs"], "H_sim": cov_old["H_sim_runs"]},
                          "fold_minus1_rule": P["rows"][lab]["S_old"]["fold_minus1_rule"]}
            r["npz_keys"] = {m: {f: {"Δ_full": f"{k}__{m}__{f}__delta_full", "Δ_rr": f"{k}__{m}__{f}__delta_rr",
                                     "per_origin_consistency": {o: [f"{k}__{m}__{f}__{o}__{c}" for c in (["sim_at_S__primary", "real_mean_at_B", "run2_at_A", "run1_at_B", "D"] + (["run1_at_B2"] if o == "S_old" else []))]
                                                                for o in ("S_old", "S_new")}} for f, _ in PTS} for m in MODES}
            r["draw0_gates"] = {"pooled_record": f"pooled['{lab}'].pooled.{{joint,fixed}}[f].gap.{{delta_full,delta_rr}}.point",
                                "per_draw_consistency": "every draw: pooled Δ_full == (sim_old·D_old + sim_new·D_new)/(D_old + D_new) − (real_mean_old·D_old + real_mean_new·D_new)/(D_old + D_new), and pooled Δ_rr likewise with run2@A and (S_old: run1@B2; S_new: run1@B) (≤ 1e-12, NaN-matched)"}
            r["S_old_per_set_npz_keys"] = {m: {f: {"Δ_full": [f"{k}__{m}__{f}__S_old__sim_at_S__primary", "minus", f"{k}__{m}__{f}__S_old__real_mean_at_B"],
                                                   "Δ_rr": [f"{k}__{m}__{f}__S_old__run2_at_A", "minus", f"{k}__{m}__{f}__S_old__run1_at_B2"]} for f, _ in PTS} for m in MODES}
            r["draw0_gates"]["S_old_per_set"] = f"pooled record rows['{lab}'].S_old.points.{{joint,fixed}}[f].gap.{{delta_full,delta_rr}}.point"
            r["readability_reported_not_used_for_membership"] = f"pooled record rows['{lab}'].{{S_old,S_new}}.readable[f].{{S,B,A}} with .pools_at_draw0[f].{{S,B,A}}.{{n_units,units_above}}, per origin"
        out.append(r)
    return out


def summary(rows):
    gs = {}
    for r in rows:
        gs.setdefault(r["loro_group"], []).append(r["inventory_key"])
    return {"per_readout": {g: {"n_members_per_readout": len(v) // 2, "members": v} for g, v in gs.items()}, "fitted_groups": sorted(g for g in gs if g != "STRESS"),
            "k_max_per_cell": len([g for g in gs if g != "STRESS"]), "stress_rows": gs["STRESS"], "unclassified_rows": []}


COMMON = {
    "governing": {
        "loro_rule": {"path": "experiments/2026-10-15_loro_09/RULE.md", "commit": "8d59e2989cc35f33b9db11ea38f9e4676b0f9217",
                      "used": "groups by training world; equal-weight members; finite-draw member rule (draw 0 finite and >= 9,500 of 10,000 finite); |D| >= 9,500; k >= 3; Σ, (a), τ̂², (b) ranges, leave-one-out, stress positions; every caveat. UNCHANGED (cell() as in run_loro_sep.py @ 6b4cd3a3e)."},
        "amendment": {"path": "experiments/2026-10-14_priority1_spec_09/RULE.md", "item_8_commit": "0f9927f3e", "fix_D_commit": "8e985cde5"},
        "first_entry": "experiments/2026-10-14_priority1_spec_09/loro_sep/loro_sep_manifest.json @ 9b8692ac1 (183, S_old)",
        "input_definition": {"path": "experiments/2026-10-15_fpr_sweep_507_prep/RULE.md", "sections": ["## +311 (S_new) per-action end-to-end gap, separate thresholds", "## Pooled 455 per-action e2e"],
                             "commit": "e4e2504287e3bcf3b161e9f4a6caeabf96a6951f (the producer commit)", "sha256": "71ca98b44c0b83972f100fd28ccc1f6ce6e900208d919b5b9dec39249b6f52cc",
                             "same_content_at": ["6d2a2fbc5", "b75a631f53932cc4ad644af93c70a84aae91c87e (Codex-reviewed)"], "branch_locator": "origin/aswin/fpr-sweep-507-prep"},
    },
    "readout_scope": "COMMAND (C) from the pooled-455 command npz and REPLY (R) from the pooled-455 reply npz; command first, then reply (2e's +311 convention)",
    "points": {"registered": [{"label": f, "role": role} for f, role in PTS], "per_action": True},
    "units": "the producer stores catches as fractions (TPR = weighted share of cells caught); all gaps × 100 to percentage points. Gate: the FINITE entries of every catch array's draw 0 lie in [0, 1], else stop; non-finite entries follow the LORO finite-draw eligibility",
    "sign": "Δ = sim (or later real run) − comparator: positive means the simulated pipeline overstates catch",
    "eligibility": {"membership": "LORO RULE finite-draw eligibility only; readability reported per member, never used for membership", "k": "k >= 3 present groups, else unavailable", "filter": "unfiltered"},
    "own_a": "WITHHELD on S_new (no mapped units on +311; 25h §1) and absent from this npz; its all-units sensitivity is UNAVAILABLE from this input",
    "cells": {"grid": "readout {C, R} × point {0.01 (LEAD), 0.001, 0.05} × input {primary, fixed_threshold, yardstick, yardstick_fixed_threshold}", "n_cells": 24, "pooled_entry_extra": "the pooled entry also registers the same 24 cells for S_old per-set (48 in all)"},
    "gates_common": ["each readout npz sha256 == input_provenance.npz.<readout>.sha256; every pinned record's bytes sha256 == pinned; else stop",
                     "every key in rows[*].npz_keys present with shape (10001,); else stop",
                     "draw 0 of every derived or stored gap equals the pinned record keys (≤ 1e-9; NaN matched), fixed draw 0 == joint draw 0 (≤ 1e-12); else stop",
                     "finite catch entries at draw 0 in [0, 1] (fractions); else stop"],
    "absent_from_input": {"would_be_STRESS": ["b_asis", "step-3 sim-trained variants", "own_sim", "ad_k*"], "other": ["own_a (withheld)", "own_frac L16/L32", "DT", "Q2 roots"]},
    "exposure_disclosure": [
        "The drafting agent read, while looking for metadata, the string rows['FIBS C'].points.joint['0.01'].exclusion_label of the +311 C hot-deck record (whether Δ_full's joint 95% interval excludes 0 under each exclusion scenario; no number)",
        "While scanning git logs for 2e's export, it read the subject line of commit fc27bce68, which states the range of the hot-deck scenario's move in Δ_full at 1e-2 on five +311 headline probes (a sensitivity move, not a level). Hot-deck is a named, unfitted sensitivity here",
        "It had computed and read the 183 (S_old) separate-threshold LORO values (@ b5b3796e7) before drafting these entries. No other +311 or pooled gap, catch, threshold or FPR value was read",
    ],
}

m311 = {
    "label": "Separate-threshold LORO manifest, +311 (S_new) entry (09; 25h amendment item 8 and fix D). Built by script from METADATA only; no +311 gap, catch, threshold or FPR value read.",
    "status": "DRAFT until committed; once committed it freezes mapping, cohort, draws and input provenance BEFORE any +311 gap is inspected by this LORO",
    **COMMON,
    "input_provenance": {"npz": NPZ, "arrays_used": "the S_new per-origin catch arrays of the pooled-455 npz (sim@S|primary, real_mean@B, run2@A, run1@B) per row × mode × f; this IS the +311 per-set computation (same code, roots and pools as the hot-deck records), exported by 2e's pooled job",
                         "producer": PROD, "records": {"pooled_455": REC_POOLED, "hotdeck": REC_HD}},
    "cohort": {"origin": "S_new (+311) only. Labels: 'Q4 FINAL; Q1, Q3 interim (routed roots so far)'; Q2 roots not included; POST HOC (+311 values were read by 2e); PRELIMINARY",
               "attack_cohort": "the FINAL record's S_new cells: 272, equal-main weighting. Sol (a) and c_asis rows use 268 (4 fold −1 cells dropped: t271_slack_clone, t291_tiktok, t388_media_processing, t505_tiktok; 2e's stated deviation); public and own_frac rows use 272",
               "pools": "H_B = 09's FINAL pool_new (S_new honest B): 306 runs (300 for Sol (a) / c_asis). H_A = S_new honest A on exactly H_sim's cells: 194 (189). H_sim = simh311 Q1 and Q3 (s1–s3, interim) and Q4 (s1–s3, FINAL) after the primary exclusions: 195 runs (190): Q1 50, Q3 63 (61), Q4 82 (79). Pool sizes are not matched (n units printed per pool)",
               "fold_minus1": "Sol (a) / c_asis: fold −1 runs dropped from every pool and fold −1 cells from the cohort (stated deviation). own_frac: weighted 21b on each pool's own fold −1 units",
               "sensitivities_named_not_fitted": ["exclusion LOW", "exclusion HIGH", "hot-deck (donor stream [20261015, 7])", "RL rulings (t298 (i)/(ii))", "own_a all-units (unavailable from this input)"]},
    "draws": {"which": "run455's main-level draws over the 506-main sameness universe (T196 196 mains ∪ 310 S_new mains): strata (environment, class), fresh default_rng([20261014, 1]), multinomial per stratum, 10,000 draws; draw 0 all ones; every calibration and attack unit carries its main's multiplicity; joint = pool moments and thresholds re-estimated per draw, fixed = draw 0's",
              "matches_25f_7": "NO. 25f §7 resamples mains within environment over E (the +311 estimation universe) with seed [20261014, 1], shares multiplicities across A, B and sim-honest, and gives calibration mains outside E the seed-31 draw (seed-32 sensitivity beside). Here S_new mains are one class, so the within-stratum law is within-environment main resampling, but the universe, ordering and generator differ, so these are not 25f §7's registered draws; the one calibration-only main outside the 506 universe (s3_http_debugging) keeps multiplicity 1 (seed-31 stream not implemented); no seed-32 sensitivity",
              "multiplicities_field": "per-draw cohort main weight D (<label>__<mode>__<f>__S_new__D) is in the npz; the main-level multiplicity matrix itself is not"},
    "comparators": {
        "primary": {"name": "Δ_full (joint)", "definition": "sim@S|primary − real_mean@B = TPR_sim(τ_sim) − ½[TPR_run1 + TPR_run2](τ_B); mean of the real runs for every group (fix D)", "label": "thresholds re-estimated per draw on each pool"},
        "fixed_threshold": {"name": "Δ_full (fixed threshold)", "label": "FIXED-THRESHOLD INPUT: pool moments and thresholds at draw 0; attack tasks resampled"},
        "yardstick": {"name": "Δ_rr (joint), beside", "definition": "run2@A − run1@B = TPR_run2(τ_A) − TPR_run1(τ_B): H_A as 25h §3 names", "label": "thresholds re-estimated per draw on each pool"},
        "yardstick_fixed_threshold": {"name": "Δ_rr (fixed threshold), beside", "label": "FIXED-THRESHOLD INPUT"}},
    "target": "a new recipe's MEAN Δ_full (per action, separate thresholds) on S_new, per readout; Δ_rr analysed the same way and printed beside",
    "rows": rows_for("311"),
    "caveats": ["own-real has 3 members (own_a withheld); own-real trained on run 1, so the real-run mean includes that launch",
                "c_asis is not a positive control under separate thresholds (25h item 12)",
                "interim: Q1 and Q3 routed roots so far, Q2 absent; not the registered +311 row",
                "Sol (a) and c_asis on 268 cells, others on 272", "draws are not 25f §7's registered joint draws",
                "one origin, one reader (Qwen3.5-27B), one simulator; recipes are not a random sample; k ≤ 6, τ̂² very imprecise",
                "not like-for-like with the 183 LORO (other origin, cells, pools, own-real membership)"],
}
m311["group_summary"] = summary(m311["rows"])

m455 = {
    "label": "Separate-threshold LORO manifest, POOLED 455 entry (09; 25h amendment item 8 and fix D). Built by script from METADATA only; no pooled or +311 gap, catch, threshold or FPR value read.",
    "status": "DRAFT until committed; once committed it freezes mapping, cohort, draws and input provenance BEFORE any pooled gap is inspected by this LORO",
    **COMMON,
    "input_provenance": {"npz": NPZ, "arrays_used": "the pooled gap arrays delta_full and delta_rr per row × mode × f (pooled cells); the S_old per-origin catch arrays sim@S|primary, real_mean@B, run2@A and run1@B2 for the registered S_old per-set cells; and all per-origin catch arrays with D for the per-draw pooled-consistency gate",
                         "producer": PROD, "records": {"pooled_455": REC_POOLED}},
    "cohort": {"origin": "POOLED S_old ∪ S_new. Label: 'POST HOC; not an authorized registered row (25h §5 authorizes only S_new ∪ S_newmains); S_old's sim-honest is A25 T196, read earlier; S_new interim (Q4 FINAL; Q1, Q3 interim); origin strata not shown equivalent (21j: inconclusive at this n)'; per-set rows beside (S_old: LORO @ b5b3796e7 is a different S_old design, see below; S_new: the +311 entry)",
               "attack_cohort": "S_old 183 T196 cells + S_new 272 cells (Sol (a) and c_asis: 268) = 455 (451), equal-main weighting over both origins' attack-estimation mains: X_b = (X_old,b D_old,b + X_new,b D_new,b)/(D_old,b + D_new,b)",
               "S_old_part": "τ_sim on all 149 A25 sim-honest runs; τ_real on B ∪ C (09's FINAL pool_old keys, 386 runs); yardstick run2 at τ_C (honest C, 192) − run1 at τ_B (honest B alone). This differs from the 183 LORO's matched design (pools restricted to M)",
               "S_new_part": "as in the +311 entry",
               "sensitivities_named_not_fitted": ["exclusion LOW / HIGH / hot-deck (S_new)", "RL rulings (S_new, t298 (i)/(ii))", "own_a all-units (unavailable: own_a absent)", "single-pooled-line sensitivity (2e, later record)"],
               "per_origin_S_old_cells": "REGISTERED beside the pooled cells, from the same npzs: S_old Δ_full = sim@S|primary − real_mean@B (τ_real on B ∪ C) and S_old Δ_rr = run2@A − run1@B2 (honest C at τ_C minus honest-B-alone at τ_B), same groups, points, modes, readouts, estimators and gates (draw 0 vs rows[label].S_old.points). Labelled 'S_old per-set, pooled job's design'; the 183 LORO @ b5b3796e7 (matched on M) stays a separate comparison. The S_new per-set cells are the +311 entry"},
    "draws": {"which": "per origin, separate constructions combined draw by draw: S_old = A24's task draws (195-task v6 universe, environment strata, seed [20261014, 1]); S_new = run455's main draws (506-main universe, (environment, class) strata, seed [20261014, 1]); the producer treats them as independent",
              "matches_25f_7": "NO (S_new part: see the +311 entry; S_old part: A24's draws, not 25f §7's)",
              "multiplicities_field": "per-draw cohort main weights D per origin (<label>__<mode>__<f>__<origin>__D)"},
    "comparators": {
        "primary": {"name": "pooled Δ_full (joint)", "definition": "pooled sim − pooled real_mean (each origin at its own thresholds); npz key <label>__joint__<f>__delta_full", "label": "thresholds re-estimated per draw on each pool"},
        "fixed_threshold": {"name": "pooled Δ_full (fixed threshold)", "label": "FIXED-THRESHOLD INPUT"},
        "yardstick": {"name": "pooled Δ_rr (joint), beside", "definition": "pooled run2@A − pooled run1@B: S_new run2 at τ_A − run1 at τ_B; S_old run2 at τ_C − run1 at τ_B (honest B alone)", "label": "thresholds re-estimated per draw on each pool; labelled 'Δ_rr: S_new with H_A, S_old with honest C'"},
        "yardstick_fixed_threshold": {"name": "pooled Δ_rr (fixed threshold), beside", "label": "FIXED-THRESHOLD INPUT"}},
    "target": "a new recipe's MEAN pooled Δ_full (per action, separate thresholds), per readout; pooled Δ_rr beside",
    "rows": rows_for("455"),
    "caveats": ["POST HOC; not an authorized registered row; origin strata not shown equivalent (21j inconclusive at this n)",
                "own-real has 3 members (own_a withheld on S_new; pooled own_a 'partial', not computed)",
                "S_old part's pools differ from the 183 LORO's matched design", "c_asis is not a positive control under separate thresholds",
                "per-origin draws are separate constructions from the same seed, treated as independent", "k ≤ 6; τ̂² very imprecise; recipes not a random sample"],
}
m455["group_summary"] = summary(m455["rows"])
for m, p in ((m311, out311), (m455, out455)):
    json.dump(m, open(p, "x"), indent=1, ensure_ascii=False)
print("311 rows", len(m311["rows"]), m311["group_summary"]["k_max_per_cell"], "| 455 rows", len(m455["rows"]))
