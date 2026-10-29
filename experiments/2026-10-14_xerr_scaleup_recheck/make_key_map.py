"""Write config/full_override.json's key_map for 09's final record (key NAMES and decision strings only; no estimate,
interval or catch value is read or printed).

  python3 make_key_map.py <target record json> <my full record json> > config/full_override.json
"""

import json
import sys

FAM = ["complete_draws", "unstable_over_1pct_nan", "X_err", "X_err_wording", "bias_mean", "bias_ci95", "bias2_corrected_floored",
       "spread", "spread_floored", "spread_plugin_ci95_inflated_by_noise", "X_err_plugin_no_deconvolution", "U_approx_upper_95",
       "fixed_family_rms", "certification", "U_label", "note", "strata_label"]
PP = ["delta", "ci95", "U_abs_approx", "discordant_tasks"]
COMPS = ["sim_minus_mean_real", "sim_minus_run1", "sim_minus_run2", "reexecution_run2_minus_run1"]
SUBS = ["combined", "S_old", "S_old|with_strategy", "S_old|without_strategy", "S_new", "S_new|with_strategy", "S_new|without_strategy"]
CELLS = ["R|0.01", "R|0.05", "C|0.01", "C|0.05"]
FIVE = ["sol_a", "fibs", "harmfulness", "refusal", "highstakes"]
# Decision-free family labels mapped from my phrasing to the spec's documented vocabulary (RULE.md, "Label value maps"):
# the scale-up RULE (09, @ 3ce6ea55e) says Append 19's estimator is used "with every v6 label" (line 63) and documents U
# (line 88); v6's record (0ed287c95, xerr_v6_2026-10-14_161122.json) carries the U label with " (Append 19)" and the
# fewer-than-two note as below. No decision-bearing label is mapped.
U_MINE = ("approximate bootstrap upper limit, conditional on fixed thresholds, frozen eligible sets and transformations; "
          "coverage not guaranteed")
# 09's ENUMERATED vocabulary for the decision labels (LABEL_VOCAB.md @ 83ee8d924, from run_xerr_scaleup.py @ 11be43945;
# vocabulary only, no record). The comparator composes the label MY category implies (compose_label) from these.
LABEL_VOCAB = {"_source": "experiments/2026-10-14_xerr_scaleup_09/LABEL_VOCAB.md @ 83ee8d924",
               "underpowered_suffix": "; underpowered (n < 30 mains)",
               "pooled": "pooled (TOST passed)",
               "differ": "strata differ (equivalence not established); stratum rows lead",
               "inconclusive": "inconclusive",
               "beyond_prefix": "difference beyond 5 established for ",
               "combined_pooled": "pooled (old/new TOST passed)"}
# Append 30 readings: my phrasing -> 09's enumerated strings (LABEL_VOCAB.md "Append 30 reading"), per cell kind.
A30_PRIMARY_FINAL = {"label withheld (< 95% jointly finite draws)":
                     "label withheld (< 95% jointly finite draws or an undefined point)",
                     "replicated": "replicated (direction replicated; magnitude not established)",
                     "not replicated: a gap as large as half the prediction is excluded":
                     "not replicated: a gap as large as half the prediction is excluded (negative direction not established; "
                     "C <= -2.5 points excluded)",
                     "inconclusive at this n": "inconclusive at this n"}
A30_OTHER = {"descriptive (not the primary cell, or not final)":
             "descriptive (the confirmatory labels apply to the final reply 5% S_new cell only)"}
FIELD_VALUE_MAP = {"U_label": {U_MINE: U_MINE + " (Append 19)"},
                   "note": {"too few members (< 2): spread undefined, no X_err":
                            "too few eligible probes (< 2): spread undefined, no X_err"}}


def ex(mine, why, checks=()):
    return {"mine": mine, "excuse": why, "checks": list(checks)}


def absent(path):
    return {"side": "target", "path": path, "absent": True}


def main(target_path, mine_path):
    tfull = json.load(open(target_path))
    t, tmode = tfull["result"], (tfull.get("spec") or {}).get("mode")
    m = json.load(open(mine_path))["result"]["mine"]
    km = []
    fam = {"fields": FAM, "per_probe_fields": PP, "per_probe_rename": {"discordant_tasks": "discordant_mains"},
           "field_value_map": FIELD_VALUE_MAP}
    for c in CELLS:
        R = f"result/results/{c}/primary"
        mf = m["rows"][c].get("families") or {}
        for cn in COMPS:
            for s in SUBS:
                fc = ({"strata_label": {"kind": "strata_combined_primary", "field": "strata_label",
                                        "test": f"rows/{c}/pooling_tests/old_new (S_new minus S_old)"}}
                      if s == "combined" else  # my per-stratum blocks emit no strata_label (absent allowed)
                      {"strata_label": {"kind": "tost_pooled_label", "field": "strata_label", "absent_ok": True,
                                        "test": f"rows/{c}/pooling_tests/strategy within {s}"}}
                      if s in ("S_old", "S_new") else {})
                km.append({"mine": f"rows/{c}/families/{cn}/{s}", "target": f"{R}/{cn}/{s}/pooled", **fam,
                           **({"field_compose": fc} if fc else {})})
                km.append({"mine": f"rows/{c}/families/{cn}/{s}/n_cells", "target": f"{R}/{cn}/{s}/cells"})
                km.append({"mine": f"rows/{c}/families/{cn}/{s}/n_mains", "target": f"{R}/{cn}/{s}/mains"})
                for extra in ("stratum_label", "label"):
                    if extra in (mf.get(cn, {}).get(s) or {}):
                        km.append(ex(f"rows/{c}/families/{cn}/{s}/{extra}", "label text; the target encodes the <30 rule as "
                                     "under_30_mains and n_mains is compared"))
            for s in ("combined|with_strategy", "combined|without_strategy"):
                km.append(ex(f"rows/{c}/families/{cn}/{s}", "not registered for the scale-up (strategy split is within each "
                             "origin stratum); computed here only", [absent(f"{R}/{cn}/{s}")]))
            if cn != "reexecution_run2_minus_run1":
                km.append(ex(f"rows/{c}/families/{cn}/S_new|same_launch_pairs", "same-launch subset registered only for run2 - run1"))
            if "diagnostic_alt_stratum_order" in mf.get(cn, {}):
                km.append(ex(f"rows/{c}/families/{cn}/diagnostic_alt_stratum_order", "my diagnostic (RULE ambiguity 3)"))
        sl = "rows/{}/families/reexecution_run2_minus_run1/S_new|same_launch_pairs".format(c)
        km.append({"mine": sl, "target": f"{R}/S_new_same_launch_run2_minus_run1/pooled", **fam})
        km.append({"mine": f"{sl}/n_cells", "target": f"{R}/S_new_same_launch_run2_minus_run1/cells"})
        km.append({"mine": f"{sl}/n_mains", "target": f"{R}/S_new_same_launch_run2_minus_run1/mains"})
        km.append({"mine": f"rows/{c}/eligible", "target": f"{R}/eligible_v6_frozen"})
        km.append({"mine": f"rows/{c}/status", "target": f"result/required_rows/primary|{c}", "value_map": {"complete": "ok"}})
        km.append({"mine": f"rows/{c}/headline/label", "target": f"result/results/{c}/headline_21i/decision"})
        for k in m["rows"][c].get("headline", {}):
            if k != "label":
                km.append(ex(f"rows/{c}/headline/{k}", "selected-row vocabulary differs; the 21i decision is compared"))
        # cell identity per row: the target lists the S_new cells each row used
        km.append({"mine": f"rows/{c}/cells_used/S_new/sha256", "target_list": f"{R}/cells_used/cells"})
        km.append({"mine": f"rows/{c}/cells_used/S_new/n_cells", "target": f"{R}/cells_used/by_stratum/S_new/cells"})
        km.append({"mine": f"rows/{c}/cells_used/S_new/n_mains", "target": f"{R}/cells_used/by_stratum/S_new/mains"})
        km.append({"mine": f"rows/{c}/cells_used/S_new|same_launch_pairs/n_cells",
                   "target": f"{R}/S_new_same_launch_run2_minus_run1/cells_same_launch"})
        for sub in m["rows"][c].get("cells_used", {}):
            if sub == "S_new":
                continue
            if sub == "S_new|same_launch_pairs":
                km.append(ex(f"rows/{c}/cells_used/{sub}/sha256", "LIMIT: the target records no list for this subset (a count, "
                             "compared); the subset is the run-table rule applied to an S_new list proven identical"))
                km.append(ex(f"rows/{c}/cells_used/{sub}/n_mains", "compared in the family block (same-launch n_mains)"))
            else:
                km.append(ex(f"rows/{c}/cells_used/{sub}", "LIMIT: the target lists only each row's S_new cells; S_old cells are "
                             "v6's (selftest-gated); strategy subsets are my rule (RULE ambiguity 2) applied to an S_new list "
                             "proven identical, and their counts and statistics are compared in the family blocks"))
        # pooling tests
        pt = m["rows"][c].get("pooling_tests") or {}
        for mk, tk in (("old_new (S_new minus S_old)", "pooling_tests_tost5/S_new-S_old"),
                       ("strategy within S_old", "strategy_pooling_tests_tost5/S_old"),
                       ("strategy within S_new", "strategy_pooling_tests_tost5/S_new")):
            P, T = f"rows/{c}/pooling_tests/{mk}", f"{R}/{tk}"  # always mapped: a block missing here shows as unmapped
            km += [{"mine": f"{P}/per_member", "target": f"{T}/per_probe"}, {"mine": f"{P}/complete_draws", "target": f"{T}/complete_draws"},
                   {"mine": f"{P}/defined", "target": f"{T}/defined"}, {"mine": f"{P}/n_mains_with_without", "target": f"{T}/mains"},
                   {"mine": f"{P}/pooled_label", "target": f"{T}/pooled_label", "compose": {"kind": "tost_pooled_label", "test": P}},
                   {"mine": f"{P}/difference_reading", "target": f"{T}/difference_reading",
                    "compose": {"kind": "tost_difference_reading", "test": P}},
                   ex(f"{P}/n_cells_with_without", "the target stores main counts only (compared)", [absent(f"{T}/cells")]),
                   ex(f"{P}/complete_draws_warning", "no such field in the target", [absent(f"{T}/complete_draws_warning")])]
        if "strategy combined" in pt:
            km.append(ex(f"rows/{c}/pooling_tests/strategy combined", "not registered for the scale-up",
                         [absent(f"{R}/strategy_pooling_tests_tost5/combined")]))
        # calibration
        rd, a = c.split("|")
        calrec = t["calibration_records"]["primary"]
        for p in FIVE:
            keys = [k for k in calrec if k.startswith(f"{p}|{rd}|{a}|primary|")]
            mc = (m["rows"][c].get("calibration") or {}).get(p)
            if len(keys) != 1 or mc is None:
                km.append(ex(f"rows/{c}/calibration/{p}", f"no single primary calibration record ({len(keys)})"))
                continue
            T, P = f"result/calibration_records/primary/{keys[0]}", f"rows/{c}/calibration/{p}"
            km += [{"mine": f"{P}/sameness_new_vs_old/auroc", "target": f"{T}/sameness_new_vs_old/auroc"},
                   {"mine": f"{P}/sameness_new_vs_old/ci95", "target": f"{T}/sameness_new_vs_old/ci95"},
                   {"mine": f"{P}/sameness_new_vs_old/pass", "target": f"{T}/sameness_new_vs_old/pass"},
                   {"mine": f"{P}/sameness_new_vs_old/nan_draws", "target": f"{T}/sameness_new_vs_old/finite_draws",
                    "transform": "ndraw_minus"},
                   {"mine": f"{P}/scheme", "target": f"{T}/primary",
                    "value_map": {"stratum-specific thresholds": "stratum_specific", "union": "union"}},
                   {"mine": f"{P}/S_old/threshold", "target": f"{T}/schemes/stratum_specific/T/old"},
                   {"mine": f"{P}/S_old", "target": f"{T}/schemes/stratum_specific/n_k/old", "as_list": ["n", "k"]},
                   ex(f"{P}/S_old/transformation", "transformation representation differs; thresholds and catches are compared"),
                   ex(f"{P}/union_transformation", "transformation representation differs; thresholds and catches are compared")]
            if mc.get("scheme") == "union":
                km += [{"mine": f"{P}/threshold", "target": f"{T}/schemes/union/T/old"},
                       {"mine": P, "target": f"{T}/schemes/union/n_k/union", "as_list": ["n", "k"]}]
            else:
                km += [{"mine": f"{P}/S_new/threshold", "target": f"{T}/schemes/stratum_specific/T/new"},
                       {"mine": f"{P}/S_new", "target": f"{T}/schemes/stratum_specific/n_k/new", "as_list": ["n", "k"]},
                       ex(f"{P}/S_new/transformation", "transformation representation differs; thresholds and catches are compared")]
        km.append(ex(f"extended_rows/{c}", "the target's primary spec has no S_newmains folder; the extended analysis is "
                     "withheld in both", [{"side": "mine", "path": f"extended_rows/{c}/status", "startswith": "extended analysis unavailable"},
                                          {"side": "target", "path": f"result/results/{c}/extended/note", "startswith": "INCOMPLETE"}]))
        # Append 30
        A = f"result/expansion/{c}/append30_training_source_contrast"
        mine30 = m["append30"].get(c) or {}
        for k in mine30:
            if k not in ("primary", "rerun_reference"):
                km.append(ex(f"append30/{c}/{k}", "note text"))
        for mk, tk in (("primary", "delta"), ("rerun_reference", "run2_minus_run1")):
            blk = mine30.get(mk) or {}
            if "C" not in blk:
                km.append(ex(f"append30/{c}/{mk}", f"not computed here ({mine30.get('status', 'no block')})"))
                continue
            P = f"append30/{c}/{mk}"
            for f in ("C", "G_real_mean", "G_public_mean", "C_b"):
                if f in blk:
                    km.append({"mine": f"{P}/{f}", "target": f"{A}/{tk}/{f}/point"})
            km += [{"mine": f"{P}/C_ci95", "target": f"{A}/{tk}/C/ci95"},
                   {"mine": f"{P}/jointly_finite_draws", "target": f"{A}/{tk}/C/finite_draws"},
                   {"mine": f"{P}/G_real", "target": f"{A}/G_real"}, {"mine": f"{P}/G_public", "target": f"{A}/G_public"},
                   ex(f"{P}/n_cells", "the target stores no cell count here", [absent(f"{A}/{tk}/n_cells")])]
            for n in blk.get("per_member_delta", {}):
                tgt_n = f"{A}/{tk}/b_asis/point" if n == "b_asis" else f"{A}/{tk}/per_member/{n}/point"
                km.append({"mine": f"{P}/per_member_delta/{n}", "target": tgt_n})
            if mk == "primary":
                vm = (dict(A30_PRIMARY_FINAL) if tmode == "final" else
                      {"descriptive (not the primary cell, or not final)":
                       f"no confirmatory label outside final mode ({tmode}); estimates descriptive"}) if c == "R|0.05" else A30_OTHER
                km.append({"mine": f"{P}/reading", "target": f"{A}/reading", "value_map": vm})
            else:
                km.append(ex(f"{P}/reading", "the target has one reading (on the sim contrast)"))
    km.append({"mine": "cohort/S_new/cohort_cells_sha256", "target_list": "result/results/R|0.05/primary/cells_used/cells"})
    km.append(ex("cohort/S_newmains/cohort_cells_sha256", "no S_newmains cohort in the target",
                 [{"side": "target", "path": "result/cohort/S_newmains_attempted_all_three", "equals": 0}]))
    out = {"_note": ("key map for 09's final record (identical from @ 6d6732c2e and @ 3ce6ea55e), written by make_key_map.py "
                     "from key NAMES of the target and of my record; label value maps are decision-free and cite the spec "
                     "(RULE.md, 'Label value maps'); no estimate value read"),
           "labels": "labels", "run_tables": "run_table", "label_vocab": LABEL_VOCAB, "key_map": km}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
