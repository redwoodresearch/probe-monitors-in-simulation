"""Append-only NOTE sidecars (0d's recheck via 09): the +311 S_new rows of Sol (a) and c_asis come from fold -1-dropped inputs.
Reads only record METADATA (fold_minus1_dropped keys, coverage counts) from the pooled records, plus each target record's sha256.
Writes results/<record stem>__NOTE_fold_minus1_2026-10-15.json with mode "x" (never overwrites). Run from this folder's repo root:
  python3 -I experiments/2026-10-15_loro_sep_09/make_fm1_notes.py"""

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
POOLED = {"C": ("6d2a2fbc5", "experiments/2026-10-15_fpr_sweep_507_prep/results/pooled455_C_2026-10-15_141921.json", "587739d0b7d9a7fc7e10a292927fc06dc5d460fe91e606b3ae306c94c508b3c4"),
          "R": ("45db6b238", "experiments/2026-10-15_fpr_sweep_507_prep/results/pooled455_R_2026-10-15_142548.json", "c794e175a324b17e2eac45cac06824463acc0eff655def7d07efc6d371214bbf")}
NPZ = {"C": ("xfer/fpr507prep/pooled455_2026-10-15_141921_perdraw.npz", "84112f5061455f9b1970ac90ff963e31d38f8c75b26b3cf28510e2c06476dbc8"),
       "R": ("xfer/fpr507prep/pooled455_2026-10-15_142548_perdraw.npz", "bd5875fbd09fc9177783f7bd84f6b862dae1d2eb8818f2d25b1461b1bd165a11")}
TARGETS = {  # record file in results/: (commit it was committed at, what it is)
    "loro_sep311_2026-10-15_145541.json": ("2351051cc", "separate-threshold LORO, +311 (S_new)"),
    "loro_sep455_2026-10-15_145541.json": ("2351051cc", "separate-threshold LORO, pooled 455 (S_new part affected; S_old per-set cells unaffected)"),
    "launch_check311_2026-10-15_150414.json": ("dc34efd81", "POST HOC launch check on the +311 reply offset"),
    "sameattack311_2026-10-15_150800.json": ("729952a13", "POST HOC same-attack add-on"),
}
AFFECTED = ["Sol (a) C", "Sol (a) R", "c_asis C", "c_asis R"]


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout


dropped, coverage = {}, {}
for ro, (c_, p_, s_) in POOLED.items():
    b = git("show", f"{c_}:{p_}")
    if hashlib.sha256(b).hexdigest() != s_:
        raise SystemExit(f"pooled {ro} sha256 differs")
    rows = json.loads(b)["rows"]
    for lab in AFFECTED:
        if lab[-1] != ro:
            continue
        new, old = rows[lab]["S_new"], rows[lab]["S_old"]
        dropped[lab] = {"S_new": {k: {"n": v["n"], "keys": v["keys"]} for k, v in new["fold_minus1_dropped"].items()},
                        "S_old": {k: v["n"] for k, v in old["fold_minus1_dropped"].items()}}
        cv = new["coverage"]
        coverage[lab] = {"S_new_cohort_cells": f"{cv['cohort_cells']} of {cv['of']}", "H_B_runs": cv["H_B_runs"], "H_A_runs": cv["H_A_runs"], "H_sim_runs": cv["H_sim_runs"],
                         "fold_minus1_rule": new["fold_minus1_rule"]}
for name, (commit, what) in TARGETS.items():
    b = git("show", f"{commit}:experiments/2026-10-15_loro_sep_09/results/{name}")
    note = {
        "note": "NOTE (append-only, 2026-10-15, after 0d's recheck via 09): the S_new rows of Sol (a) and c_asis in this record come from the fold −1-dropped pooled npzs. "
                "Fold −1 cells were dropped from the cohort (268 of 272 cells), and fold −1 runs were dropped from H_B, H_A and H_sim (300 / 189 / 190 runs instead of 306 / 194 / 195). "
                "This is a deviation from 25h §1: its 21b fold −1 rule is not implemented for these rows. A rerun is pending once per-model fold −1 scores exist.",
        "provisional": "The other-model-real group (Sol (a)) is PROVISIONAL until the corrected inputs land, and so is every 6-group mean and every LORO quantity that includes it "
                       "(m, τ², Σ, the (b) and true-gap ranges, leave-one-out). The c_asis stress positions and contrasts are provisional too. The other groups' per-group values do not depend on these inputs.",
        "record": {"path": f"experiments/2026-10-15_loro_sep_09/results/{name}", "commit": commit, "sha256": hashlib.sha256(b).hexdigest(), "what": what},
        "inputs": {"npz": {ro: {"path": p_, "sha256": s_} for ro, (p_, s_) in NPZ.items()},
                   "pooled_records": {ro: {"commit": c_, "path": p_, "sha256": s_} for ro, (c_, p_, s_) in POOLED.items()}},
        "affected_rows": AFFECTED, "dropped": dropped, "coverage_after_drop": coverage,
        "S_old": "no fold −1 drops on S_old (counts 0), so the pooled record's S_old per-set cells are unaffected" if "455" in name else "not part of this record",
    }
    dest = HERE / "results" / f"{name.removesuffix('.json')}__NOTE_fold_minus1_2026-10-15.json"
    with dest.open("x") as fh:
        fh.write(json.dumps(note, indent=1, ensure_ascii=False))
    print(dest.name, hashlib.sha256(dest.read_bytes()).hexdigest())
