"""Provenance AMENDMENT 1 to the +311 and pooled-455 separate-threshold LORO manifests (@ cd1afb33e): the corrected pooled-455 inputs
(Sol (a) / c_asis under 25h §1, @ 2892fa260) supersede the fold -1-dropped ones. Groups, comparators, points, estimators unchanged.
Metadata only. Usage: python3 -I build_amend1.py <pooled_C_new.json> <pooled_R_new.json> <man311.json> <man455.json> <out311> <out455>"""

import hashlib
import json
import sys

pc, pr, m311p, m455p, o311, o455 = sys.argv[1:7]


def load(p):
    b = open(p, "rb").read()
    return json.loads(b), hashlib.sha256(b).hexdigest()


NEW = {"C": load(pc), "R": load(pr)}
M = {"311": load(m311p), "455": load(m455p)}
assert NEW["C"][1] == "edfef938d3520cb736ae7bed970e82b6e1371dfbf37f5621afdf2156fa317146" and NEW["R"][1] == "54ffd0778514bfd0c0b7ef0d266889a6f951dcfd4297c6dcc384baf7eed1313b"
assert M["311"][1] == "2dd2d5397b8ccb9f070e4301a41112740259560f9387692c1f6335ad7c1cfa73" and M["455"][1] == "f83c8880298ffe1a28fe8a761e9c024b0709e17cb7b9ba734e30a36cfffa0688"
RD = "experiments/2026-10-15_fpr_sweep_507_prep/results"
NAMES = {"C": "pooled455_C_2026-10-15_151852.json", "R": "pooled455_R_2026-10-15_151846.json"}
npz = {ro: {"volume": NEW[ro][0]["npz"]["volume"], "path": NEW[ro][0]["npz"]["path"], "sha256": NEW[ro][0]["npz"]["sha256"], "n_arrays": NEW[ro][0]["npz"]["n_arrays"],
            "key_format": NEW[ro][0]["npz"]["key_format"]} for ro in NEW}
assert npz["C"]["sha256"].startswith("219681d83278c349") and npz["R"]["sha256"].startswith("6470f012fda86ff9")
assert all(npz[ro]["n_arrays"] == 1026 for ro in npz)
recs = {ro: {"path": f"{RD}/{NAMES[ro]}", "commit": "2892fa260", "branch": "origin/aswin/fpr-sweep-507-prep", "sha256": NEW[ro][1]} for ro in NEW}
prod = {"script": "experiments/2026-10-15_fpr_sweep_507_prep/run_peraction311.py", "git": NEW["C"][0]["meta"]["git"], "script_sha256": NEW["C"][0]["meta"]["script_sha256"],
        "key_construction": "unchanged from e4e250428: pool_reduce() keys and the per-origin draws_out catch keys are identical (checked by diff); the only additions are readability fields"}
assert NEW["C"][0]["meta"]["script_sha256"] == NEW["R"][0]["meta"]["script_sha256"]
cov = {}
for ro in NEW:
    for lab in (f"Sol (a) {ro}", f"c_asis {ro}"):
        r = NEW[ro][0]["rows"][lab]["S_new"]
        c = r["coverage"]
        cov[lab] = {"S_new_cohort_cells": f"{c['cohort_cells']} of {c['of']}", "H_B_runs": c["H_B_runs"], "H_A_runs": c["H_A_runs"], "H_sim_runs": c["H_sim_runs"],
                    "fold_minus1_rule": r["fold_minus1_rule"], "fold_minus1_dropped": {k: v["n"] for k, v in r["fold_minus1_dropped"].items()}}
        assert c["cohort_cells"] == 272 and (c["H_B_runs"], c["H_A_runs"], c["H_sim_runs"]) == (306, 194, 195)

for nm, out in (("311", o311), ("455", o455)):
    a = {
        "label": f"AMENDMENT 1 (provenance) to the {'+311 (S_new)' if nm == '311' else 'pooled 455'} separate-threshold LORO manifest; committed BEFORE the rerun on the corrected inputs. Metadata only.",
        "amends": {"path": f"experiments/2026-10-14_priority1_spec_09/loro_sep/loro_sep_manifest_{nm}.json", "commit": "cd1afb33e", "sha256": M[nm][1]},
        "supersedes_inputs": M[nm][0]["input_provenance"]["npz"],
        "input_provenance": {"npz": npz, "records": {"pooled_455": recs}, "producer": prod,
                             "hotdeck_records": "kept for draw-0 gates on every row EXCEPT Sol (a) and c_asis (their hot-deck rows are fold −1-dropped); those two are gated against the corrected pooled records' per-set S_new rows"},
        "what_changed": {
            "Sol (a) and c_asis, S_new": "now under 25h §1 (weighted 21b on each pool's own fold −1 units): 272 cells and H_B / H_A / H_sim 306 / 194 / 195 runs, replacing the fold −1-dropped 268 cells and 300 / 189 / 190",
            "coverage_by_row": cov,
            "own family": "bootstrap draws reclassified under 09's clarification @ 7310476ae (fold −1 moments needed only when a fold −1 unit is present in the draw): 2e reports finite draws 9,938 → 9,960; draw-0 values unchanged",
            "public probes and S_old": "unchanged (2e's gate: public rows equal by metadata; Sol / c_asis S_old unchanged)"},
        "unchanged": "rows, inventory keys, npz key names, groups (own-real 3; other-model-real; 4 public; STRESS c_asis; own_a WITHHELD), comparators, points (1e-2 lead, 1e-3, 5e-2), estimators (LORO RULE @ 8d59e2989), eligibility (finite-draw only), units, labels"
                     + ("; the S_old per-set cells beside the pooled cells" if nm == "455" else ""),
        "added_gates": ["npz sha256 == input_provenance.npz.<readout>.sha256 (full shas); record sha256 == pinned",
                        "draw 0 of every row other than Sol (a) and c_asis equals the hot-deck records (+311) or the previous pooled record keys (≤ 1e-9)",
                        "every group other than other-model-real: draw-0 group values equal the previous LORO record's group_gaps_draw0 (≤ 1e-9), cell by cell"
                        + ("; for the S_old per-set cells, ALL groups, since S_old is unchanged" if nm == "455" else ""),
                        "Sol (a) and c_asis: draw 0 equals the corrected pooled records' per-set S_new rows (≤ 1e-9)"],
        "supersedes_records": "the fold −1-dropped results (loro_sep311 / loro_sep455 _2026-10-15_145541 @ 2351051cc, launch_check311 @ dc34efd81, sameattack311 @ 729952a13) stay, with their NOTE sidecars @ 3302b84df; the reruns are labelled as superseding them",
        "labels": "POST HOC (+311 values were read by 2e; the fold −1-dropped LORO values were read); PRELIMINARY; '+311: Q4 FINAL; Q1, Q3 interim'" + ("; pooled: not an authorized registered row; origin strata not shown equivalent" if nm == "455" else ""),
    }
    json.dump(a, open(out, "x"), indent=1, ensure_ascii=False)
print("ok")
