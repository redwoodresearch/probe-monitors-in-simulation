"""APPEND-ONLY CORRECTION NOTE (ownall-loo, 10-16): the "descriptive cut-32768 refit equals the registered own_all_rs_L48 weights EXACTLY (max diff 0.0)"
line I relayed at ~03:17 was my summary print's rounding (round(x, 4) of values ~1e-6..1e-5), not the fit record. This NOTE reads both committed records
and states, by script: the fit record's 10 descriptive diffs at full precision; that the diagnosis job's recomputed (B) reproduces fold 0's record values;
(A) the gate's 0.0; (E) the shuffled-order dW; (F) the row equality (bitwise as a multiset, not in order). Git only."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _b(rel):
    b = (REPO / rel).read_bytes()
    if subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{rel}"], capture_output=True).stdout != b:
        raise SystemExit(f"{rel}: not committed as read")
    return b


def main(fit_rel, diag_rel, out_rel):
    fb, db = _b(fit_rel), _b(diag_rel)
    desc = json.loads(fb)["fit"]["descriptive_cut32768"]
    gate = json.loads(fb)["fit"]["gate"]["refit_match"]
    dg = json.loads(db)
    per = {k: v["weights_max_abs_diff_vs_registered"] for k, v in desc.items()}
    rec = {}
    for rd, r in dg["results"].items():
        f_ = r["fits"]
        b_rec = per[f"own_all_rs_L48|{rd}|f0"]
        b_new = max(f_["B_descriptive_uncut_below_32768"]["max_abs_diff_vs_registered_file"].values())
        rec[rd] = {"fit_record_descriptive_f0": b_rec, "diagnosis_B_recomputed_f0": b_new, "B_reproduces_fit_record": b_new == b_rec,
                   "A_gate_diff": max(f_["A_gate_registered_rows"]["max_abs_diff_vs_registered_file"].values()),
                   "E_shuffled_order_max_abs_dW_vs_A": f_["E_shuffled_order"]["max_abs_dW_vs_A"],
                   "E_shuffled_order_max_abs_diff_vs_file_all_params": max(f_["E_shuffled_order"]["max_abs_diff_vs_registered_file"].values()),
                   "F_rows_bitwise_equal_as_multiset": r["F_rows_in_fit_order"]["bitwise_equal_as_multiset"], "F_rows_bitwise_equal_in_fit_order": r["F_rows_in_fit_order"]["bitwise_equal_in_sequence"]}
    note = {"label": "CORRECTION (APPEND-ONLY): own_all_rs_uncut's descriptive cut-32768 refit is NOT bitwise equal to the registered own_all_rs_L48 weights; DESCRIPTIVE, never a gate",
            "what_was_wrong": "my 03:17 relay said max diff 0.0 for all 10 fold models; it came from a print that rounded the diffs to 4 decimals. The fit record itself always "
                              "held the full values below. Nothing downstream used the line; 4f held it from citation at 03:2x.",
            "fit_record": {"path": fit_rel, "sha256": hashlib.sha256(fb).hexdigest()}, "diagnosis_record": {"path": diag_rel, "sha256": hashlib.sha256(db).hexdigest()},
            "descriptive_diffs_full_precision": per, "descriptive_max": max(per.values()), "descriptive_min": min(per.values()),
            "gate_refit_match_max_diff": max(v["weights_max_abs_diff"] for v in gate.values()), "fold0": rec,
            "explanation": "the uncut rows below token 32768 equal own_all's registered rows bitwise as a multiset (F), in another order; the certified float64 LBFGS fit is "
                           "order-sensitive at the 1e-6..1e-5 level (E: the registered rows shuffled); so the descriptive differences are row-order noise of the same size. "
                           "The gate's 0.0 is real (same rows, same order).",
            "codex_r22_note": "the diagnosis record's fitter_order_invariant_bitwise checks W only; it reads false (W differs), so the order-sensitivity conclusion stands; "
                              "an invariance claim would have needed all four parameters"}
    (REPO / out_rel).open("x").write(json.dumps(note, indent=1))
    print(json.dumps({"descriptive_min": note["descriptive_min"], "descriptive_max": note["descriptive_max"], "fold0": rec}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
