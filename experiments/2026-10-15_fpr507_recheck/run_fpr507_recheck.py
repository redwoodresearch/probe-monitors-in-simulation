"""Independent recheck of 09's 507 registered unit FPR sweep. PRELIMINARY until reviewed.

Target: experiments/2026-10-15_fpr_sweep_507_prep/results/registered507_2026-10-15_103452.json @ 404b6d241
(aswin/fpr-sweep-507-run-09; app ap-m8Mjh9NRnrEOWsDch6ealH).

Spec implemented: that folder's RULE.md with Revision 1 (answers_09 @ 4c61a67ad), the registered inputs and the GS
decision @ 56ffba5ea, and 09's keys-only pool export @ 886533127 (addendum 1). 09's runner and sweep core
(run_registered507.py, run_fprsweep_507.py) were NOT read for the computation. Only I/O formats were read: the unit-file
wiring in run_common455.py (load_probe, arms_from_spec: which file, which npz field) and the G2 reference fields in
run_registered507.py (g2_reference).

Modal CPU only: one container for the global checks, then one container per registered row (16), retries 0, Volumes
mounted read-only. The console and the committed JSON carry verdicts, counts and key paths only. Recomputed values exist
only in container memory and are never printed or written.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with "numpy==2.3.3" \
      modal run --detach experiments/2026-10-15_fpr507_recheck/run_fpr507_recheck.py
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-fpr507recheck_xcheck_kill-2026-11-14"
BRANCH = "aswin/xerr-scaleup-recheck"
# Volume names as the configs give them (inputs_b.json "volumes"; run_common455.py VOLS for the arm folders)
VOLS = {
    "/sol": "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13",
    "/p2": "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13",
    "/conv": "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14",
}
VROOT = {"solavg": "/sol", "p2": "/p2", "conv": "/conv"}

# Tolerances, stated before any run: counts, ints, booleans and checked strings exact; floats from deterministic
# arithmetic (moments, thresholds, rates, percentiles, standardized shifts) absolute 1e-12.
TOL_FLOAT = 1e-12

R507 = "experiments/2026-10-15_fpr_sweep_507_prep"
GIT = {  # name -> (commit, path, full sha256)
    "target": (
        "404b6d241",
        f"{R507}/results/registered507_2026-10-15_103452.json",
        "6e58128d65717dc71d3c75a39afadced2831c2f5a4a3de0a5c6d871977c92d4c",
    ),
    "inputs_registered": (
        "56ffba5ea",
        f"{R507}/config/inputs_507_registered_09.json",
        "19a709ce2b23766423850845f2d075b0553fc01e5f7284695d1cd4d196b19c64",
    ),
    "inputs_as_run": (
        "46a50873e",
        f"{R507}/config/inputs_507_registered_09.json",
        "c7c3c24697b8df310281a3a345990708a2f64c24751f5f751d680c825387e78a",
    ),
    "seal_decision": (
        "56ffba5ea",
        f"{R507}/config/seal_decision_507_09.json",
        "1679f0487c078316e065156c279b9e56d1bba60a016e69c407c3b21ee953ecd0",
    ),
    "answers_09": (
        "4c61a67ad",
        "experiments/2026-10-14_priority1_spec_09/answers/fpr507_answers_09_2026-10-15.md",
        "6a0ae013a7e778c8c840fa315999c920a7a2bf94d21961da6f6770cc3c40b2ee",
    ),
    "pool_export": (
        "886533127",
        "experiments/2026-10-14_priority1_spec_09/answers/fpr507_pool_export_09_2026-10-15.json",
        "e57e934d3a8a828a8f15505c9629a295bd79267d0f1dfad5ec0c3bd437d5e3e6",
    ),
    "final": (
        "3ce6ea55e",
        "experiments/2026-10-14_xerr_scaleup_09/results/xerr_scaleup_final_2026-10-15_013255.json",
        "7dceb8bb25752aa3619cfde64add702fd4f7d13a1309a56dbc8880a6b27af9f5",
    ),
    "final_spec": (
        "3ce6ea55e",
        "experiments/2026-10-14_xerr_scaleup_09/specs/final_spec_3_launch.json",
        "6bd2ab192337e1023a5454c17217eb2b44f0f7c94165cd8671639443d5f90a5a",
    ),
    "release": (
        "c901ff2e3",
        "experiments/2026-10-14_xerr_scaleup_09/results/RELEASE_507.md",
        "95ffc7d1f440a75909efc2f057e16aa3be9ac2f75635db5dfc28a83cdbdf2a51",
    ),
    "t196": (
        "debef0af2",
        "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json",
        "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb",
    ),
    "plus311": (
        "0caadbfb5",
        "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json",
        "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b",
    ),
    "common455": (
        "78fe04ba5",
        f"{R507}/config/common455.json",
        "37a02bf889bf5395d54a6e003d1559863d68271a035d217b2b214b93f649d467",
    ),
    "owna_pilot": (
        "404b6d241",
        f"{R507}/results/owna507_2026-10-15_031024.json",
        "fa0c87be5d065842f6d4be58027ef8e5884b5ed5d53bc1a88ddf30d69366fe1a",
    ),
    "owna_rest": (
        "404b6d241",
        f"{R507}/results/owna507_2026-10-15_031320.json",
        "f5856826cd22c7f56ae4a4314429778ede68004313d3e56f7abcade6c0ba4a04",
    ),
    "owna_record": (
        "cc53afc3a",
        "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_scores_2026-10-14_164934.json",
        "c002d5f822e49d7877f4e8b79a80ca9dd2e2f16c664b921d3ad12cb3da4c49ac",
    ),
    "inputs_b": (
        "f5d1af8dd",
        "experiments/2026-10-14_fpr_sweep_unit/config/inputs_b.json",
        "75fa4dc2f891aaea294f34e4b78d641300ec99258395b09326824b6f13962bf0",
    ),
    "fibs_record": (
        "d74ccaba0",
        "experiments/2026-10-14_p2_fibs_pair/results/fibs_scores_2026-10-14_124948.json",
        "366d594cd719d52e9bb11d34bf4d3491b6cbd82d64f61efa73f476773cb49b1a",
    ),
    "rh_record": (
        "a8d786425",
        "experiments/2026-10-14_p2_refusal_harm/results/scores_2026-10-14_145748.json",
        "d0b7f028f8ac3795dd946a261f384f474748a296772fb1f2f2f22cecaac6c961",
    ),
    "hs_record": (
        "3d04b7936",
        "experiments/2026-10-14_p2_highstakes/results/scores_2026-10-14_133519.json",
        "2d251dee71a4a0366bd9549fcc3726511fe576aafce0bb6a913353d7fc06f38a",
    ),
    "asis_record": (
        "74992a373",
        "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/asis_scores_2026-10-14_142228.json",
        "59c051a4ff9cc40811745a119251224f031cdaf479b8d3fb909c1059ab94da31",
    ),
    # the target's own code at its as-run commit (pinned so the target's meta shas are checked, never read for the computation)
    "target_script": (
        "46a50873e",
        f"{R507}/run_registered507.py",
        "cd76ea40e701a2ef9cdf06c7e86209485940f2b8d456db050bc37678f0bc3983",
    ),
    "target_loader": (
        "46a50873e",
        f"{R507}/run_common455.py",
        "8d6c27c073acb2f8d0a2fdac7a081fe83d2c5ef3de03cde63ee32a88113f1ecd",
    ),
    "target_core": (
        "46a50873e",
        f"{R507}/run_fprsweep_507.py",
        "28dcf5414861f42356616d96192553194d7b6d2a60bd0c78f958dd7aaa384ba1",
    ),
    "gr_record": (
        "404b6d241",
        f"{R507}/results/repro_2026-10-15_034516.json",
        "a29dbafa5440a0a72d4f0ff7b7e4aab64943dcc7867c831ebbeeee38b143dc01",
    ),
    "famexp_index": (
        "84df4d069",
        "experiments/2026-10-14_p2_family_expansion/results/own_family_unit_files_2026-10-15_000120.json",
        "4a8156107399630f791d65802044d35f21be76645205a026e661c46b1cc8084e",
    ),
}
# r36 (the Sol / own_a_adapted fold map) is pinned by sha only in 09's spec; this is 4f's harness copy, sha-checked
LOCAL = {
    "r36": (
        Path(
            "/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/"
            "823cc0a0-23fc-47f8-b84f-0087b7bffcbd/scratchpad/p2/xr_in/runs36.json"
        ),
        "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96",
    )
}

WORLDS = ("run1", "run2", "sim", "honest_a", "honest_b", "honest_c")
ATTACK = ("run1", "run2", "sim")
NOT_LM = "NOT launch-matched (sim ran in a separate follow-up launch)"
# RULE Comparisons: "Combined rows mix S_old's rerun-plus-launch reference with the new strata's within-launch reference,
# and say so"; answers_09 Q5: combined rows are descriptive. Compared by meaning (required substrings), not wording.
COMBINED_LABEL = "descriptive; mixes S_old's rerun-plus-launch reference with S_new's within-launch reference"
SEMANTIC = {"stratum_labels.combined": ["descriptive", "rerun-plus-launch", "within-launch"]}
STRATUM_LABELS = {
    "S_old": {"sim_minus_run1": "same launch", "run2_minus_run1": "rerun + launch"},
    "S_new": {"sim_minus_run1": NOT_LM, "run2_minus_run1": "within-launch rerun only"},
}
# row key -> (common455 probe label, S_new arm file, G2 reference json (None: famexp), routing, evidence key)
ROWS = {
    "sol_a|R": (
        "Sol (a) R",
        "sol/a_tight__avg_top2.npz",
        "sol/a_tight__avg_top2.json",
        "r36",
        "sol",
    ),
    "fibs|R": (
        "FIBS R",
        "p2/fibs__reply_mean.npz",
        "p2/fibs__run_means.json",
        "single",
        None,
    ),
    "harmfulness|R": (
        "harmfulness R",
        "p2/harmfulness__reply_mean.npz",
        "p2/harmfulness__run_means.json",
        "single",
        None,
    ),
    "refusal|R": (
        "refusal R",
        "p2/refusal__reply_mean.npz",
        "p2/refusal__run_means.json",
        "single",
        None,
    ),
    "highstakes|R": (
        "high-stakes R",
        "p2/highstakes__reply_mean.npz",
        "p2/highstakes__run_means.json",
        "single",
        None,
    ),
    "sol_a|C": (
        "Sol (a) C",
        "sol/a_tight__cmd_A.npz",
        "sol/a_tight__avg_top2.json",
        "r36",
        "sol",
    ),
    "fibs|C": (
        "FIBS C",
        "p2/fibs__command.npz",
        "p2/fibs__run_means.json",
        "single",
        None,
    ),
    "harmfulness|C": (
        "harmfulness C",
        "p2/harmfulness__command.npz",
        "p2/harmfulness__run_means.json",
        "single",
        None,
    ),
    "refusal|C": (
        "refusal C",
        "p2/refusal__command.npz",
        "p2/refusal__run_means.json",
        "single",
        None,
    ),
    "highstakes|C": (
        "high-stakes C",
        "p2/highstakes__command.npz",
        "p2/highstakes__run_means.json",
        "single",
        None,
    ),
    "exp:own_frac100_L48|R": (
        "own_frac100_L48 R",
        "famexp/own_frac100_L48__R.npz",
        None,
        "r36",
        None,
    ),
    "exp:own_frac100_L48|C": (
        "own_frac100_L48 C",
        "famexp/own_frac100_L48__C.npz",
        None,
        "r36",
        None,
    ),
    "c_asis|R": (
        "c_asis R",
        "sol/c_asis_tight__avg_top2.npz",
        "sol/c_asis_tight__avg_top2.json",
        "r36",
        "c_asis",
    ),
    "c_asis|C": (
        "c_asis C",
        "sol/c_asis_tight__cmd_A.npz",
        "sol/c_asis_tight__avg_top2.json",
        "r36",
        "c_asis",
    ),
    "own_a|R": (
        "own_a R",
        "own_a/own_a_tight__avg_top2.npz",
        "own_a/own_a_tight__avg_top2.json",
        "own_a",
        "own_a",
    ),
    "own_a|C": (
        "own_a C",
        "own_a/own_a_tight__cmd_A.npz",
        "own_a/own_a_tight__avg_top2.json",
        "own_a",
        "own_a",
    ),
}

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
VOLMAP = {k: modal.Volume.from_name(v, version=2).read_only() for k, v in VOLS.items()}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _read(path: str, sha: str) -> bytes:
    b = Path(path).read_bytes()
    if _sha(b) != sha:
        raise SystemExit(f"G1: sha256 differs for {path}")
    return b


# ------------------------------------------------------------------ container: pins, membership, universe, design, arm files
def _all_true(d) -> bool:
    """True when every boolean leaf of a nested dict is True (non-boolean leaves are ignored)."""
    if isinstance(d, dict):
        return all(_all_true(v) for v in d.values())
    return d is not False


@app.function(image=image, cpu=2.0, memory=8192, timeout=900, retries=0, volumes=VOLMAP)
def global_job(g: dict) -> dict:
    import numpy as np

    J = g["J"]
    reg, asrun, dec, exp, c455, ib = (
        J["inputs_registered"],
        J["inputs_as_run"],
        J["seal_decision"],
        J["pool_export"],
        J["common455"],
        J["inputs_b"],
    )
    fin = J["final"]
    checks, counts = {}, {}
    # ---- GS and the registered-input pins
    checks["seal_decision_releases_final"] = (
        dec.get("releases_final_507") is True and dec["final_record"]["sha256"] == GIT["final"][2]
    )
    checks["registered_inputs_pin_answers_09"] = reg["answers_09"]["sha256"] == GIT["answers_09"][2]
    checks["registered_inputs_pin_final"] = reg["final_record"]["sha256"] == GIT["final"][2]
    checks["final_cell_list_digest_equals_pin"] = fin["cohort"]["cell_list_sha256"] == reg["final_record"][
        "cohort_cell_list_sha256"
    ] and fin["cohort"]["cell_list_sha256"].startswith("8352805f2d9ec226")
    checks["release_names_final_record"] = g["release_names_final"]
    checks["pool_export_source_is_final"] = exp["source"]["sha256"] == GIT["final"][2]
    checks["final_spec_arms_equal_final_record_spec_arms"] = J["final_spec"]["arms"] == fin["spec_arms"]
    strip = lambda d: {
        k: ([{kk: vv for kk, vv in r.items() if kk not in ("unit_files", "status")} for r in v] if k == "rows" else v)
        for k, v in d.items()
        if k not in ("INCOMPLETE", "pool_export")
    }
    checks["as_run_inputs_equal_registered_except_unit_files_status_and_export_pin"] = strip(reg) == strip(asrun)
    checks["as_run_pool_export_pin_is_09s_export"] = (
        asrun.get("pool_export", {}).get("sha256") == GIT["pool_export"][2]
        and asrun.get("pool_export", {}).get("commit") == GIT["pool_export"][0]
        and sorted(asrun.get("pool_export", {}).get("rows", [])) == sorted(exp["rows"])
    )
    checks["as_run_status_changes_only_on_the_five_export_rows"] = all(
        (r1.get("status") == r2.get("status")) or (r1["K"] is None) for r1, r2 in zip(reg["rows"], asrun["rows"])
    )
    checks["as_run_unit_files_name_common455_at_78fe04ba5"] = all(
        "common455:" in r["unit_files"] and "78fe04ba5" in r["unit_files"] for r in asrun["rows"]
    )
    tm = g["target_meta"]
    checks["target_meta_inputs_pins"] = {
        "pool_export": tm["inputs"]["pool_export"] == GIT["pool_export"][2],
        "final": tm["inputs"]["final"] == GIT["final"][2],
        "cells311": tm["inputs"]["cells311"] == GIT["plus311"][2],
        "t196": tm["inputs"]["t196"] == GIT["t196"][2],
        "r36": tm["inputs"]["r36"] == LOCAL["r36"][1],
    }
    checks["target_meta_seal_decision_pin"] = (
        tm["seal_decision"] == f"{R507}/config/seal_decision_507_09.json@{GIT['seal_decision'][2]}"
    )
    checks["target_meta_git_is_46a50873e"] = tm["git"].startswith("46a50873e")
    checks["target_meta_code_shas"] = {
        "script": tm["script_sha256"] == GIT["target_script"][2],
        "loader": tm["loader_sha256"] == GIT["target_loader"][2],
        "core": tm["core_sha256"] == GIT["target_core"][2],
        "config_is_the_as_run_inputs": tm["config_sha256"]
        == hashlib.sha256(json.dumps(asrun, sort_keys=True).encode()).hexdigest(),
    }
    checks["target_envelope"] = {
        "root_keys": sorted(g["target_root_keys"]) == ["design", "meta", "pass", "rows"],
        "meta_keys": sorted(tm)
        == sorted(
            [
                "git",
                "stamp",
                "script_sha256",
                "core_sha256",
                "loader_sha256",
                "seal_decision",
                "config_sha256",
                "gr_record",
                "inputs",
                "label",
            ]
        ),
        "meta_inputs_keys": sorted(tm["inputs"]) == ["cells311", "final", "pool_export", "r36", "t196"],
        "meta_label_and_stamp_are_strings": isinstance(tm.get("label"), str) and isinstance(tm.get("stamp"), str),
        "design_keys": sorted(g["target_design"]) == ["by_origin", "cohort", "draws", "grid", "universe_mains"],
    }
    gr = J["gr_record"]
    checks["target_GR_record_pinned_and_passing"] = (
        tm["gr_record"] == f"results/repro_2026-10-15_034516.json@{GIT['gr_record'][2]}"
        and gr.get("pass") is True
        and gr.get("script_sha256") == tm["core_sha256"]
    )
    # ---- S_old cross-pins: the wiring config's S_old files against the scorer records and part (b) / (c) inputs
    byl = {p["label"]: p for p in c455["probes"]}
    ibp = {p["probe_id"].split("|")[0]: p["rows"]["sha256"] for p in ib["probes"]}
    asis = J["asis_files"]
    xp = {
        "Sol (a) R": byl["Sol (a) R"]["old"]["sha256"] == ibp["sol_a"],
        "Sol (a) C": byl["Sol (a) C"]["old"]["sha256"] == ib["g4_command_file"]["sha256"],
        "c_asis R": byl["c_asis R"]["old"]["sha256"] == ibp["c_asis"] == asis["c_asis_tight__avg_top2"]["sha256"],
        "c_asis C": byl["c_asis C"]["old"]["sha256"] == asis["c_asis_tight__cmd_A"]["sha256"],
        "FIBS R": byl["FIBS R"]["old"]["sha256"] == ibp["fibs"] == J["fibs_rows"]["reply_mean"]["sha256"],
        "FIBS C": byl["FIBS C"]["old"]["sha256"] == J["fibs_rows"]["command"]["sha256"],
    }
    for lab, pr in (
        ("harmfulness", "harmfulness"),
        ("refusal", "refusal"),
        ("high-stakes", "highstakes"),
    ):
        rows_ = J["p2_rows"][pr]
        xp[f"{lab} R"] = byl[f"{lab} R"]["old"]["sha256"] == ibp[pr] == rows_["reply_mean"]["sha256"]
        xp[f"{lab} C"] = byl[f"{lab} C"]["old"]["sha256"] == rows_["command"]["sha256"]
    for ro in ("R", "C"):
        o = byl[f"own_frac100_L48 {ro}"]["old"]
        xp[f"own_frac100_L48 {ro}"] = all(
            o[x]["sha256"] == ib["own_frac100_L48"]["files"][ro][x]["sha256"]
            and o[x]["sha256"] in g["famexp_index_text"]
            for x in ("open", "sim")
        )
    xp["own_a record pinned by part (b) and the wiring config"] = (
        ib["own_a"]["record"]["sha256"] == GIT["owna_record"][2] == c455["own_a_record"]["sha256"]
    )
    xp["own_a layer choice equals part (b)"] = ib["own_a"]["layer_choice"] == c455["own_a_layer_choice"]
    checks["S_old_cross_pins"] = xp
    # ---- S_new arm files: manifests pinned by FINAL's spec (Sol, p2), spec.expansion.arms (famexp), the own_a records
    owna_sha = {x["arm"]: x["manifest_sha256"] for rec in J["owna_records"] for x in rec["arms"]}
    arms, n_man = {}, 0
    for arm, e in J["final_spec"]["arms"].items():
        files = {}
        for kind, ent in (
            ("sol", e["sol_manifest"]),
            ("p2", e["p2_manifest"]),
            ("famexp", J["final_spec"]["expansion"]["arms"][arm]["manifest"]),
            (
                "own_a",
                {
                    "path": f"/work/scaleup/{arm}/own_a/manifest_owna.json",
                    "sha256": owna_sha[arm],
                },
            ),
        ):
            rel = ent["path"].replace("/work/", "", 1)
            m = json.loads(_read(f"/conv/{rel}", ent["sha256"]))
            n_man += 1
            d = rel.rsplit("/", 1)[0]
            for fn, s in m["files"].items():
                base = fn.rsplit("/", 1)[-1]
                if fn.startswith("/") and fn.replace("/work/", "", 1).rsplit("/", 1)[0] != d:
                    raise SystemExit(f"{arm} {kind}: manifest file outside its folder")
                files[f"{kind}/{base}"] = [f"{d}/{base}", s]
        arms[arm] = files
    counts["manifests_sha_verified"] = n_man
    counts["arms"] = len(arms)
    # ---- cells, universe (sameness universe: mains of all T196 cells and of every registered S_new cell) and cohort
    cme = {c["key"]: [c["main"], c["env"]] for c in J["t196_cells"]}
    cme |= {c["key"]: [c["main"], c["env"]] for c in J["plus311_cells"]}
    t196 = {c["key"] for c in J["t196_cells"]}
    reg_new = sorted(fin["registered_new_cells"])
    old_m = {cme[c][0] for c in t196}
    new_m = {cme[c][0] for c in reg_new}
    env_sets = {}
    for c in list(t196) + reg_new:
        env_sets.setdefault(cme[c][0], set()).add(cme[c][1])
    checks["universe_env_unique_per_main"] = all(len(v) == 1 for v in env_sets.values())
    uni = sorted(old_m | new_m)
    uenv = {m: next(iter(env_sets[m])) for m in uni}
    counts |= {
        "universe_mains": len(uni),
        "universe_old_only": len(old_m - new_m),
        "universe_new_only": len(new_m - old_m),
        "universe_both": len(old_m & new_m),
        "registered_new_cells": len(reg_new),
    }
    checks["universe_equals_final_env_of_main_keys"] = set(uni) == set(fin["env_of_main"])
    checks["universe_env_equals_final"] = all(uenv[m] == fin["env_of_main"].get(m) for m in uni)
    checks["universe_origin_equals_final"] = all(
        set(fin["origin_of_main"].get(m, [])) == ({"old"} if m in old_m else set()) | ({"new"} if m in new_m else set())
        for m in uni
    )
    coh = fin["cohort"]["cohort_cells"]
    counts |= {
        "cohort_cells": len(coh),
        "cohort_S_old": sum(c in t196 for c in coh),
        "cohort_S_new": sum(c not in t196 for c in coh),
    }
    checks["cohort_unique"] = len(set(coh)) == len(coh)
    checks["cohort_new_cells_registered"] = all(c in t196 or c in set(reg_new) for c in coh)
    checks["cohort_main_of_equals_cell_lists"] = all(fin["main_of"][c] == cme[c][0] for c in coh)
    checks["cohort_completeness_lists_empty"] = all(
        not v for x in fin["cohort"]["incomplete_by_scope_readout"].values() for v in x.values()
    ) and all(not v for x in fin["cohort"]["honest_b_pool_missing_by_scope_readout"].values() for v in x.values())
    ms = {c for x in fin["missing_scores"].values() for c in x}
    checks["missing_score_cells_outside_cohort"] = not (ms & set(coh))
    checks["non_finite_transformed_runs_empty"] = not fin["non_finite_transformed_runs"]
    checks["approach_excluded_cells_empty"] = not fin["approach_excluded_cells"]
    checks["omitted_registered_cells_outside_cohort"] = not (set(fin["omitted_registered_cells"]) & set(coh))
    excl = set(fin["staging_applied"]) | set(fin["setup_failures"]) | set(fin["excluded_before_counting"])
    counts["cohort_attack_keys_also_in_exclusion_records_info"] = sum(f"{w}|{c}" in excl for c in coh for w in ATTACK)
    # launch labels on the analysed S_new keys (FINAL's run table, d9's launch IDs); S_old's come from v6, not rechecked here
    rt = fin["run_table_rows"]
    cnew = [c for c in coh if c not in t196]
    counts["S_new_cohort_cells_with_run_table_row"] = sum(c in rt for c in cnew)
    counts["S_new_cohort_sim_launch_differs_from_run1"] = sum(
        c in rt and rt[c]["sim_launch"] != rt[c]["run1_launch"] for c in cnew
    )
    counts["S_new_cohort_run1_run2_same_launch"] = sum(
        c in rt and rt[c]["run1_launch"] == rt[c]["run2_launch"] for c in cnew
    )
    counts["S_new_cohort_attack_pair_launch_same"] = sum(
        c in rt and rt[c]["attack_pair_launch"] == "same" for c in cnew
    )
    checks["S_new_label_not_launch_matched_holds_on_every_cohort_cell"] = counts[
        "S_new_cohort_sim_launch_differs_from_run1"
    ] == len(cnew)
    checks["S_new_label_within_launch_holds_on_every_cohort_cell"] = counts[
        "S_new_cohort_run1_run2_same_launch"
    ] == len(cnew)
    W = draws(uni, uenv, old_m, new_m)
    checks["draw0_all_ones"] = bool((W[0] == 1).all())
    checks["each_draw_resamples_every_stratum_to_its_size"] = bool((W.sum(1) == len(uni)).all())
    # ---- design (target) against mine
    grid = sorted([10 ** (-4 + 0.125 * i) for i in range(25)] + [0.05])
    mydesign = {
        "cohort": len(coh),
        "by_origin": {"S_old": counts["cohort_S_old"], "S_new": counts["cohort_S_new"]},
        "universe_mains": len(uni),
        "grid": grid,
        "draws": W.shape[0] - 1,
    }
    design_cmp = compare_tree(g["target_design"], mydesign, set(), {})
    # ---- info: the two S_old Sol (a) files that exist per readout (22g perrow vs asis rows): identical by row?
    pairs = {}
    for name, (fa, fb) in g["sol_old_pairs"].items():
        za = np.load(io.BytesIO(_read(f"/sol/{fa[0]}", fa[1])), allow_pickle=False)
        zb = np.load(io.BytesIO(_read(f"/sol/{fb[0]}", fb[1])), allow_pickle=False)
        out = {"fields_a": sorted(za.files), "fields_b": sorted(zb.files)}
        ia, ib_ = _row_ids(za), _row_ids(zb)
        out["same_row_ids"] = sorted(ia) == sorted(ib_)
        if out["same_row_ids"] and "score" in za.files and "score" in zb.files:
            sa = dict(zip(ia, za["score"].astype(np.float64).tolist()))
            sb = dict(zip(ib_, zb["score"].astype(np.float64).tolist()))
            out["scores_equal_within_1e-12"] = all(abs(sa[k] - sb[k]) <= 1e-12 for k in sa)
        pairs[name] = out
    # ---- rows: K records (cross-alpha), or 09's export for the five rows without one; scheme; S_old/S_new files
    cr = fin["calibration_records"]
    cfg_rows = {f"{r['probe']}|{r['readout']}": r for r in reg["rows"]}
    asrun_rows = {f"{r['probe']}|{r['readout']}": r for r in asrun["rows"]}
    membership, spec_rows = {}, {}
    for rk, (label, new, g2j, routing, evid) in ROWS.items():
        r = cfg_rows[rk]
        if label not in asrun_rows[rk]["unit_files"] or byl[label]["new"] != new:
            raise SystemExit(f"{rk}: unit-file wiring differs")
        mem = {}
        if r["K"]:
            recs = [cr[k] for k in r["K"]]
            sch = [x["primary"] for x in recs]
            lists = [
                (
                    sorted(x["schemes"][x["primary"]]["pool_keys_old"]),
                    sorted(x["schemes"][x["primary"]]["pool_keys_new"]),
                )
                for x in recs
            ]
            mem["cross_alpha_same_pool_config"] = len({(x["pool_old"], x["pool_new"]) for x in recs}) == 1
            mem["cross_alpha_same_scheme"] = len(set(sch)) == 1
            mem["cross_alpha_identical_pool_keys"] = lists[0] == lists[1]
            mem["pool_config_strings_info"] = [recs[0]["pool_old"], recs[0]["pool_new"]]
            scheme, (pold, pnew), pexp = sch[0], lists[0], None
        else:
            pexp = f"{rk}|BuC|B"
            e = exp["rows"][pexp]
            pold, pnew = sorted(e["pool_keys_old"]), sorted(e["pool_keys_new"])
            mem["export_counts_match"] = e["n_old"] == len(pold) and e["n_new"] == len(pnew)
            ref = cr["sol_a|R|0.01|primary|BuC|B"]["schemes"]["stratum_specific"]
            mem["export_equals_final_sol_a_R_BuC_B_stratum_lists"] = (pold, pnew) == (
                sorted(ref["pool_keys_old"]),
                sorted(ref["pool_keys_new"]),
            )
            scheme = (
                "stratum_specific"  # answers_09 Q8: explicit, stratum-specific primary with the union as sensitivity
            )
        mem["scheme_equals_registered_config"] = (scheme == "stratum_specific") == (r["scheme"] == "stratum_specific")
        mem["union_sensitivity_flag_consistent"] = r["union_sensitivity"] == (scheme == "stratum_specific")
        mem["n_pool_old"], mem["n_pool_new"] = len(pold), len(pnew)
        mem["pool_old_by_world"] = {
            w: sum(k.startswith(w + "|") for k in pold) for w in ("honest_a", "honest_b", "honest_c")
        }
        mem["pool_new_by_world"] = {
            w: sum(k.startswith(w + "|") for k in pnew) for w in ("honest_a", "honest_b", "honest_c")
        }
        mem["pool_has_no_honest_a"] = (
            mem["pool_old_by_world"]["honest_a"] == 0 and mem["pool_new_by_world"]["honest_a"] == 0
        )
        mem["pool_old_cells_in_T196"] = all(k.split("|", 1)[1] in t196 for k in pold)
        mem["pool_new_cells_registered_S_new"] = all(k.split("|", 1)[1] in set(reg_new) for k in pnew)
        mem["no_duplicate_pool_keys"] = len(set(pold)) == len(pold) and len(set(pnew)) == len(pnew)
        mem["pool_keys_also_in_exclusion_records_info"] = len((set(pold) | set(pnew)) & excl)
        membership[rk] = mem
        o = dict(byl[label]["old"])
        if o["kind"] == "own_a_R":
            o["files"] = J["own_saved"]["tight"]["avg"]
        elif o["kind"] == "own_a_C":
            e = J["own_saved"]["cmd"]["cmdA"]["rows|own_a_tight"]
            o |= {"path": e["path"], "sha256": e["sha256"]}
        oref = {"name": J["old_ref_rows"][rk]} if rk in J["old_ref_rows"] else None
        spec_rows[rk] = {
            # Sol (a)'s S_old run score is its own npz mean (part (b) G2); the asis record is a cross-check only
            "old_ref_is_gate": bool(oref) and not rk.startswith("sol_a|"),
            "old_ref_name": oref["name"]
            if oref
            else "none pinned (famexp packs: no separate run-score file is pinned)",
            "shared_rows": (
                {"old": byl["FIBS R"]["old"], "new": byl["FIBS R"]["new"]} if rk in ("sol_a|R", "c_asis|R") else None
            ),
            "label": label,
            "probe": r["probe"],
            "readout": r["readout"],
            "units": r["units"],
            "routing_text": r["routing"],
            "scheme_source": r["scheme_source"],
            "caveats": reg["caveats"],
            "K": r["K"],
            "scheme": scheme,
            "union_sensitivity": r["union_sensitivity"],
            "old": o,
            "new": new,
            "g2_json": g2j,
            "routing": routing,
            "evidence": evid,
            "own_a_layer_choice": c455["own_a_layer_choice"],
            "pool_old": pold,
            "pool_new": pnew,
            "row_label": r.get("label"),
            "arm_labels": r.get("labels"),
            "pool_export_row": pexp,
        }
    r36 = {}
    for row in J["r36"]:
        if r36.setdefault(row["main_task"], int(row["fold"])) != int(row["fold"]):
            raise SystemExit("r36: a main with two folds")
    counts["r36_mains"] = len(r36)
    base = {
        "arms": arms,
        "arm_order": list(J["final_spec"]["arms"]),
        "t196_cells": sorted(t196),
        "registered_new_cells": reg_new,
        "cell_main_env": cme,
        "cohort": coh,
        "r36_main_fold": r36,
        "routing_evidence": fin["routing_evidence"],
        "sol_training_manifests": fin["sol_training_manifests"],
        "exclusion_keys": sorted(excl),
        "universe": uni,
        "universe_env": uenv,
        "universe_old": sorted(old_m),
        "universe_new": sorted(new_m),
        "grid": grid,
    }
    st = _selftest()
    rep = {
        "selftest": st,
        "checks": checks,
        "counts": counts,
        "membership": membership,
        "design_comparison": design_cmp,
        "sol_old_file_pairs": pairs,
        "ok": {
            "selftest": _all_true(st),
            "checks": _all_true(checks),
            "membership": all(
                _all_true({k: v for k, v in m.items() if isinstance(v, bool)}) for m in membership.values()
            ),
            "design": design_cmp["n_mismatch"] == 0
            and not design_cmp["not_produced"]
            and not design_cmp["structure_mismatch"]
            and not design_cmp["only_mine"],
        },
    }
    return {"report": rep, "base": base, "spec_rows": spec_rows}


def _row_ids(z):
    k = z["key"].astype(str)
    for f in ("unit", "call_index"):
        if f in z.files:
            return [f"{a}#{b}" for a, b in zip(k, z[f].tolist())]
    return [f"{a}#{i}" for i, a in enumerate(k)]


def weighted_thresholds(zs_desc, w, grid):
    """A24's weighted arithmetic (09 spec section 3): scores in stable descending order with weights w; k = floor(f * sum(w)
    + 1e-9); T = the first score whose cumulative weight exceeds k. A zero total weight leaves T undefined (NaN)."""
    import numpy as np

    c = np.cumsum(w)
    tot = c[-1] if len(c) else 0
    k = np.floor(np.asarray(grid) * tot + 1e-9).astype(np.int64)
    t = np.full(len(k), np.nan)
    if tot > 0:
        j = np.searchsorted(c, k, side="right")
        ok = j < len(zs_desc)
        t[ok] = zs_desc[j[ok]]
    return t, k


def _selftest() -> dict:
    """Synthetic negative tests of the comparator, the weighted threshold and the draws (no project data)."""
    import numpy as np

    out = {}
    t = {"a": 1, "b": [1.0, 2.0], "c": "x", "d": None, "e": True}
    out["identical_matches"] = compare_tree(t, dict(t), {"c"}, {})["n_mismatch"] == 0
    out["int_off_by_one_flagged"] = compare_tree({"a": 10**17}, {"a": 10**17 + 1}, set(), {})["n_mismatch"] == 1
    out["float_beyond_tol_flagged"] = compare_tree({"a": 0.5}, {"a": 0.5 + 1e-9}, set(), {})["n_mismatch"] == 1
    out["float_within_tol_matches"] = compare_tree({"a": 0.5}, {"a": 0.5 + 1e-14}, set(), {})["n_mismatch"] == 0
    out["nan_pair_flagged"] = compare_tree({"a": float("nan")}, {"a": float("nan")}, set(), {})["n_mismatch"] == 1
    out["list_length_flagged"] = bool(compare_tree({"b": [1, 2]}, {"b": [1]}, set(), {})["structure_mismatch"])
    out["missing_key_flagged"] = bool(compare_tree({"a": 1, "z": 2}, {"a": 1}, set(), {})["not_produced"])
    out["extra_key_flagged"] = bool(compare_tree({"a": 1}, {"a": 1, "z": 2}, set(), {})["only_mine"])
    out["checked_string_flagged"] = compare_tree({"c": "x"}, {"c": "y"}, {"c"}, {})["n_mismatch"] == 1
    out["string_vs_null_flagged"] = compare_tree({"c": "x"}, {"c": None}, set(), {})["n_mismatch"] == 1
    out["bool_vs_int_flagged"] = compare_tree({"e": True}, {"e": 1}, set(), {})["n_mismatch"] == 1
    rng = np.random.default_rng(7)
    z = np.sort(rng.normal(size=500))[::-1]
    grid = sorted([10 ** (-4 + 0.125 * i) for i in range(25)] + [0.05])
    tw, kw = weighted_thresholds(z, np.ones(500, np.int64), grid)
    ku = [math.floor(f * 500 + 1e-9) for f in grid]
    out["weighted_equals_unweighted_at_unit_weights"] = bool(
        all(tw[i] == z[min(ku[i], 499)] for i in range(len(grid))) and list(kw) == ku
    )
    w = np.ones(500, np.int64)
    w[:3] = 0  # the three top scores carry no weight: k = 0 must land on the fourth
    out["zero_weight_units_skipped"] = bool(weighted_thresholds(z, w, [1e-4])[0][0] == z[3])
    out["zero_total_weight_undefined"] = bool(np.isnan(weighted_thresholds(z, np.zeros(500, np.int64), [0.01])[0][0]))
    u = [f"m{i:03d}" for i in range(20)]
    env = {m: ("e1" if i < 12 else "e2") for i, m in enumerate(u)}
    W1, W2 = (
        draws(u, env, set(u[:15]), set(u[10:]), 50),
        draws(u, env, set(u[:15]), set(u[10:]), 50),
    )
    out["draws_deterministic"] = bool((W1 == W2).all())
    old_s, new_s = set(u[:15]), set(u[10:])
    cls = {m: "both" if (m in old_s and m in new_s) else ("old-only" if m in old_s else "new-only") for m in u}
    expected = {
        ("e1", "old-only"): 10,
        ("e1", "both"): 2,
        ("e2", "both"): 3,
        ("e2", "new-only"): 5,
    }
    out["draws_stratum_sizes_kept"] = bool(
        all(
            (W1[1:, [i for i, m in enumerate(u) if (env[m], cls[m]) == key]].sum(1) == n).all()
            and len([m for m in u if (env[m], cls[m]) == key]) == n
            for key, n in expected.items()
        )
    )
    return out


def draws(uni, env, old_m, new_m, n_draws=10000):
    """RULE draws (Q4; 09's draw clarification items 2-3): units = sorted main IDs; stratum key (environment, class) with
    class in {old-only, new-only, both}; one fresh default_rng([20261014, 1]) stream over strata in lexicographic order;
    multinomial(n_s, uniform, size=n_draws) per stratum gives draws 1..n_draws; draw 0 all ones."""
    import numpy as np

    cls = {m: "both" if (m in old_m and m in new_m) else ("old-only" if m in old_m else "new-only") for m in uni}
    strata = {}
    for i, m in enumerate(uni):
        strata.setdefault((env[m], cls[m]), []).append(i)
    W = np.ones((n_draws + 1, len(uni)), dtype=np.int64)
    rng = np.random.default_rng([20261014, 1])
    for key in sorted(strata):
        idx = np.array(strata[key], dtype=np.int64)
        n = len(idx)
        W[1:, idx] = rng.multinomial(n, np.full(n, 1.0 / n), size=n_draws)
    return W


# ------------------------------------------------------------------ container: one registered row
@app.function(image=image, cpu=4.0, memory=16384, timeout=1800, retries=0, volumes=VOLMAP)
def row_job(a: dict) -> dict:
    import numpy as np

    rk, spec = a["row_key"], a["spec"]
    gates, info = {}, {}
    alt = {}  # exact key path -> {variant: value}; consulted only to name an interpretation that matches a mismatch
    folds5 = spec["routing"] != "single"

    # ---------------------------------------------------------------- units
    cols = {
        "run": [],
        "stratum": [],
        "s": [],
        "fold": [],
        "sbm": [],
        "uid": [],
        "arm": [],
        "file_fold": [],
    }
    uid_fields, nfiles = set(), 0

    def add(z, stratum, arm, force_m1=False, s_override=None):
        nonlocal nfiles
        nfiles += 1
        k = z["key"].astype(str)
        n = len(k)
        fold = z["fold"].astype(np.int64) if (folds5 and "fold" in z.files) else np.zeros(n, np.int64)
        if folds5 and "fold" not in z.files:
            raise SystemExit(f"{rk}: fold column absent")
        sbm = z["score_by_model"].astype(np.float64) if "score_by_model" in z.files else np.full((n, 5), np.nan)
        if sbm.shape != (n, 5):
            raise SystemExit(f"{rk}: score_by_model is not rows x 5")
        if s_override is not None:
            s = s_override
        elif "score" in z.files:
            s = z["score"].astype(np.float64)
        else:  # famexp scale-up files: the designated model's column; fold -1 rows use the ensemble
            s = np.where(fold >= 0, sbm[np.arange(n), np.clip(fold, 0, 4)], np.nan)
        uf = next((f for f in ("unit", "call_index") if f in z.files), None)
        uid_fields.add(uf or "row_order")
        uid = z[uf].astype(str) if uf else np.array([str(i) for i in range(n)])
        # every column of one file must be one row per key (no silent misalignment across files)
        if k.ndim != 1 or s.shape != (n,) or fold.shape != (n,) or uid.shape != (n,):
            raise SystemExit(f"{rk}: per-row columns misaligned in {arm}")
        ff = fold.copy()
        if force_m1:
            fold = np.full(n, -1, np.int64)
        for name, val in (
            ("run", k),
            ("stratum", np.full(n, stratum)),
            ("s", s),
            ("fold", fold),
            ("sbm", sbm),
            ("uid", uid),
            ("arm", np.full(n, arm)),
            ("file_fold", ff),
        ):
            cols[name].append(val)

    def npz(vol, path, sha):
        return np.load(
            io.BytesIO(_read(f"{VROOT[vol]}/{path.lstrip('/')}", sha)),
            allow_pickle=False,
        )

    old = spec["old"]
    if old["kind"] == "npz":
        add(npz(old["volume"], old["path"], old["sha256"]), "S_old", "S_old_file")
    elif old["kind"] == "fam":
        for x in ("open", "sim"):
            add(npz("p2", old[x]["path"], old[x]["sha256"]), "S_old", f"S_old_{x}")
    elif (
        old["kind"] == "own_a_R"
    ):  # per fold: the mean of the fold's two chosen layers (aligned files of fold f's runs)
        for f, (la, lb) in enumerate(spec["own_a_layer_choice"]):
            za = npz(
                "solavg",
                old["files"][f"{la}|{f}"]["rows"],
                old["files"][f"{la}|{f}"]["rows_sha256"],
            )
            zb = npz(
                "solavg",
                old["files"][f"{lb}|{f}"]["rows"],
                old["files"][f"{lb}|{f}"]["rows_sha256"],
            )
            aligned = all(x in za.files and x in zb.files and np.array_equal(za[x], zb[x]) for x in ("key", "fold"))
            aligned &= all(
                np.array_equal(za[x], zb[x]) for x in ("unit", "call_index") if x in za.files and x in zb.files
            )
            n_ = len(za["key"])
            aligned &= za["key"].ndim == 1 and za["score"].shape == (n_,) and zb["score"].shape == (n_,)
            if not aligned or set(za["fold"].tolist()) != {f}:
                raise SystemExit(f"{rk}: own_a layer files of fold {f} misaligned")
            add(
                za,
                "S_old",
                f"S_old_fold{f}",
                s_override=(za["score"].astype(np.float64) + zb["score"].astype(np.float64)) / 2,
            )
    elif old["kind"] == "own_a_C":
        add(npz("solavg", old["path"], old["sha256"]), "S_old", "S_old_file")
    else:
        raise SystemExit(f"{rk}: unknown S_old kind")
    for arm in a["arm_order"]:
        path, sha = a["arms"][arm][spec["new"]]
        add(npz("conv", path, sha), "S_new", arm, force_m1=(spec["routing"] == "own_a"))
    run = np.concatenate(cols["run"])
    stratum = np.concatenate(cols["stratum"])
    s = np.concatenate(cols["s"])
    fold = np.concatenate(cols["fold"])
    sbm = np.concatenate(cols["sbm"])
    uid = np.concatenate(cols["uid"])
    armv = np.concatenate(cols["arm"])
    file_fold = np.concatenate(cols["file_fold"])
    n_u = len(run)
    info["files_read"] = nfiles
    info["uid_fields"] = sorted(uid_fields)
    world = np.array([k.split("|", 1)[0] for k in run])
    cell = np.array([k.split("|", 1)[1] if "|" in k else "" for k in run])
    t196 = set(a["t196_cells"])
    p311 = set(a["cell_main_env"]) - t196
    gates["worlds_known"] = bool(np.isin(world, WORLDS).all())
    gates["S_old_file_cells_in_T196"] = all(c in t196 for c in set(cell[stratum == "S_old"].tolist()))
    gates["S_new_file_cells_in_plus311"] = all(c in p311 for c in set(cell[stratum == "S_new"].tolist()))
    pairs = np.char.add(np.char.add(run.astype(str), "#"), uid.astype(str))
    gates["no_repeated_key_unit_pair"] = len(np.unique(pairs)) == n_u
    gates["no_run_key_in_two_files"] = _runs_in_one_source(run, armv)
    main_of = {c: me[0] for c, me in a["cell_main_env"].items()}

    # ---------------------------------------------------------------- folds (G3) and finiteness
    if folds5:
        r36 = a["r36_main_fold"]
        exp_fold = np.array([r36.get(main_of.get(c, ""), -1) for c in cell], dtype=np.int64)
        newm = stratum == "S_new"
        if spec["routing"] == "own_a":  # Q9: every S_new own_a row is fold -1; the file must say so
            g3 = bool((file_fold[newm] == -1).all())
        else:
            g3 = bool((fold[newm] == exp_fold[newm]).all())
        m1 = np.flatnonzero(newm & (fold == -1))
        ev = {}
        if spec["evidence"] in (
            "sol",
            "c_asis",
        ):  # FINAL's routing evidence, bound to the arm's five training manifests
            rev, tms = a["routing_evidence"], a["sol_training_manifests"]
            ok = True
            for i in m1:
                e = rev.get(str(armv[i]), {}).get(main_of.get(cell[i], ""), {})
                e = e.get("c_asis_tight", {}) if spec["evidence"] == "c_asis" else e
                ok &= e.get("excluded_by_all_five") is True
                if spec["evidence"] == "sol":
                    ok &= sorted(e.get("training_manifest_sha256") or []) == sorted(
                        tms.get(str(armv[i])) or ["missing"]
                    )
            ev["final_routing_evidence"] = bool(ok)
        if spec["evidence"] in (
            "sol",
            "c_asis",
            "own_a",
        ):  # direct: the main is in none of the five models' training mains
            tag = {
                "sol": "sol/training_mains__a_tight",
                "c_asis": "sol/training_mains__c_asis_tight",
                "own_a": "own_a/training_mains__own_a_tight",
            }[spec["evidence"]]
            r36_mains = set(r36)
            ok, nonvacuous = True, True
            for arm in a["arm_order"]:
                names = set()
                for f in range(5):
                    path, sha = a["arms"][arm][f"{tag}__f{f}.json"]
                    names |= _strings(json.loads(_read(f"/conv/{path}", sha)))
                # the files may name mains, cells or run keys: map cells and run keys to their mains
                names |= {main_of[x] for x in names if x in main_of}
                names |= {main_of[x.split("|", 1)[1]] for x in names if "|" in x and x.split("|", 1)[1] in main_of}
                nonvacuous &= bool(names & r36_mains)
                sel = m1[armv[m1] == arm]
                ok &= not any(main_of.get(cell[i], "") in names for i in sel)
            ev["training_mains_exclude_fold_m1_mains"] = bool(ok)
            ev["training_mains_format_names_mains"] = bool(nonvacuous)
        info["G3_evidence"] = ev
        info["G3_fold_m1_rows_S_new"] = len(m1)
        g3_old = bool((fold[~newm] == exp_fold[~newm]).all())  # S_old: r36 (= own_a's T196 training-fold map; part (c))
        gates["G3_S_old_fold_map"] = g3_old
        gates["G3_S_new"] = g3 and all(ev.values())
        alt["G3_S_new_fold_map"] = {"fold_only": g3}
    req_finite = np.where(fold >= 0, np.isfinite(s), np.isfinite(sbm).all(axis=1)) if folds5 else np.isfinite(s)
    non_finite_rows = int((~req_finite).sum())
    alt["non_finite_rows"] = {
        "S_new_only": int((~req_finite[stratum == "S_new"]).sum()),
        "raw_designated_incl_fold_m1_placeholders": int((~np.isfinite(s)).sum()),
    }
    gates["all_required_scores_finite"] = non_finite_rows == 0

    # ---------------------------------------------------------------- runs
    order = np.argsort(run, kind="stable")
    rs = run[order]
    starts = np.r_[0, np.flatnonzero(rs[1:] != rs[:-1]) + 1]
    ends = np.r_[starts[1:], n_u]
    units_of = {str(rs[b]): order[b:e] for b, e in zip(starts, ends)}

    # ---------------------------------------------------------------- membership
    coh = a["cohort"]
    coh_old = [c for c in coh if c in t196]
    coh_new = [c for c in coh if c not in t196]
    cells_of = {"S_old": coh_old, "S_new": coh_new}
    attack_runs = [f"{w}|{c}" for c in coh for w in ATTACK]
    cohort_missing = sorted(r for r in attack_runs if r not in units_of)
    pool = {"S_old": a["pool_old"], "S_new": a["pool_new"]}
    pool_missing = {st: sorted(k for k in pool[st] if k not in units_of) for st in pool}
    wrong = sorted(
        [k for k in pool["S_old"] if k.split("|", 1)[1] not in t196]
        + [k for k in pool["S_new"] if k.split("|", 1)[1] not in p311]
    )
    excl = set(a["exclusion_keys"])
    hon_a_all = {
        st: sorted(k for k in units_of if k.startswith("honest_a|") and (k.split("|", 1)[1] in t196) == (st == "S_old"))
        for st in ("S_old", "S_new")
    }
    coh_set = set(a["cohort"])
    # READING (identified from runs 1-4, adopted in run 5): honest A = the scored honest A runs of COHORT cells, per
    # stratum. The pre-stated reading (every scored honest A run of the stratum) is kept as the variant
    # "prestated_all_scored".
    hon_a = {st: [k for k in hon_a_all[st] if k.split("|", 1)[1] in coh_set] for st in hon_a_all}
    hon_a_x = {st: [k for k in hon_a_all[st] if k not in excl] for st in hon_a_all}
    reg_cells = set(a["registered_new_cells"]) | t196
    pool_cells = {st: {k.split("|", 1)[1] for k in a[f"pool_{'old' if st == 'S_old' else 'new'}"]} for st in hon_a}
    hon_a_var = {
        "prestated_all_scored": hon_a_all,
        "minus_final_exclusions": hon_a_x,
        "cohort_cells_only": {st: [k for k in hon_a_all[st] if k.split("|", 1)[1] in coh_set] for st in hon_a},
        "cohort_cells_minus_exclusions": {
            st: [k for k in hon_a_x[st] if k.split("|", 1)[1] in coh_set] for st in hon_a
        },
        "registered_cells_only": {st: [k for k in hon_a_all[st] if k.split("|", 1)[1] in reg_cells] for st in hon_a},
        "pool_cells_only": {st: [k for k in hon_a_all[st] if k.split("|", 1)[1] in pool_cells[st]] for st in hon_a},
    }
    coverage = {
        "cohort_runs_missing": cohort_missing,
        "n_cohort_runs_missing": len(cohort_missing),
        "pool_runs_missing": pool_missing,
        "n_pool_runs_missing": {st: len(v) for st, v in pool_missing.items()},
        "pool_keys": {st: len(v) for st, v in pool.items()},
        "pool_keys_wrong_stratum": wrong,
        "honest_a_runs": len(hon_a["S_old"]) + len(hon_a["S_new"]),
    }
    alt["coverage.honest_a_runs"] = {n: len(v["S_old"]) + len(v["S_new"]) for n, v in hon_a_var.items()}
    gates["G6_cohort_runs_have_units"] = not cohort_missing
    gates["G6_pool_runs_have_units"] = not any(pool_missing.values())
    gates["pool_keys_right_stratum"] = not wrong
    gates["pool_disjoint_from_cohort_and_honest_a"] = all(
        k.split("|", 1)[0] in ("honest_b", "honest_c") for st in pool for k in pool[st]
    )
    pool_src = {st: sorted({k.split("|", 1)[0] for k in pool[st]}) for st in pool}

    def idx_of(keys):
        parts = [units_of[k] for k in keys if k in units_of]
        return np.concatenate(parts) if parts else np.zeros(0, np.int64)

    pool_idx = {st: idx_of(pool[st]) for st in pool}
    # calibration demand (which routing groups need moments, withholding, finiteness): every scored honest A run, as in
    # runs 1-4, so the adopted honest A reading cannot change a transform or a withholding decision
    active_runs = (
        set(attack_runs) | set(pool["S_old"]) | set(pool["S_new"]) | set(hon_a_all["S_old"]) | set(hon_a_all["S_new"])
    )
    # the analysed runs under the adopted reading (G2's scope): cohort attack runs, the pool, cohort-cell honest A
    analysed_runs = (
        set(attack_runs) | set(pool["S_old"]) | set(pool["S_new"]) | set(hon_a["S_old"]) | set(hon_a["S_new"])
    )
    act = np.zeros(n_u, bool)
    for k in active_runs:
        if k in units_of:
            act[units_of[k]] = True
    withhold = []
    if cohort_missing:
        withhold.append("cohort runs without units")
    if any(pool_missing.values()):
        withhold.append("pool runs without units (the pool is never shrunk)")
    if not req_finite[act].all():
        withhold.append("non-finite required scores on the calibration-demand units")
    alt["non_finite_rows"]["active_units_only"] = int((~req_finite[act]).sum())

    # ---------------------------------------------------------------- transformation (RULE Calibration 3; Q2; 21e)
    def fit(pidx, sidx, donor_idx):
        """Moments from pool units pidx for the units sidx of one stratum (or the union). Returns (tf, z over sidx) or
        (None, reason)."""
        if not folds5:
            return {"identity": True}, s[sidx].copy()
        used = set(fold[sidx][fold[sidx] >= 0].tolist())
        need_m1 = bool((fold[sidx] == -1).any())
        need = set(range(5)) if need_m1 else used
        mu, sd, borrowed = {}, {}, []
        for f in range(5):
            pf = pidx[fold[pidx] == f]
            if len(pf) == 0 and f in need:
                pf = donor_idx[fold[donor_idx] == f]
                borrowed.append(f)
                if len(pf) == 0:
                    return None, f"fold {f}: empty in pool and donor"
            if len(pf) == 0:
                mu[str(f)], sd[str(f)] = None, None
                continue
            v = s[pf]
            m_, d_ = float(np.mean(v)), float(np.std(v))
            if f in need and (len(pf) < 2 or not math.isfinite(m_) or not math.isfinite(d_) or d_ <= 0):
                return None, f"fold {f}: invalid moments"
            mu[str(f)], sd[str(f)] = m_, d_
        z = np.full(len(sidx), np.nan)
        fs = fold[sidx]
        for f in used:
            msk = fs == f
            z[msk] = (s[sidx][msk] - mu[str(f)]) / sd[str(f)]
        mu0 = sd0 = None
        if need_m1:
            p0 = pidx[fold[pidx] == -1]
            if len(p0) < 2:
                return None, "fold -1: fewer than two pool units"
            mus = np.array([mu[str(f)] for f in range(5)])
            sds = np.array([sd[str(f)] for f in range(5)])
            m_pool = np.mean((sbm[p0] - mus) / sds, axis=1)
            mu0, sd0 = float(np.mean(m_pool)), float(np.std(m_pool))
            if not math.isfinite(mu0) or not math.isfinite(sd0) or sd0 <= 0:
                return None, "fold -1: invalid moments"
            msk = fs == -1
            m_u = np.mean((sbm[sidx][msk] - mus) / sds, axis=1)
            z[msk] = (m_u - mu0) / sd0
        return {
            "mu": mu,
            "sd": sd,
            "mu0": mu0,
            "sd0": sd0,
            "borrowed_folds_21e": borrowed,
        }, z

    sidx = {st: np.flatnonzero((stratum == st) & act) for st in ("S_old", "S_new")}
    donor = np.concatenate([pool_idx["S_old"], pool_idx["S_new"]])

    def transform(scheme):
        zt = np.full(n_u, np.nan)
        if scheme == "union":
            allidx = np.flatnonzero(act)
            tf, z = fit(donor, allidx, donor)
            if tf is None:
                return None, None, z
            zt[allidx] = z
            if not np.isfinite(zt[act]).all():
                return None, None, "non-finite transformed score"
            return ({"identity": True} if not folds5 else {"union": tf}), zt, None
        tfs = {}
        for st in ("S_new", "S_old"):
            tf, z = fit(pool_idx[st], sidx[st], donor)
            if tf is None:
                return None, None, z
            zt[sidx[st]] = z
            tfs[st] = tf
        if not np.isfinite(zt[act]).all():
            return None, None, "non-finite transformed score"
        return ({"identity": True} if not folds5 else tfs), zt, None

    # ---------------------------------------------------------------- sweep machinery
    W = draws(a["universe"], a["universe_env"], set(a["universe_old"]), set(a["universe_new"]))
    mpos = {m: i for i, m in enumerate(a["universe"])}
    grid = a["grid"]
    nd = W.shape[0] - 1

    def summ(v):
        d = v[1:]
        fin = np.isfinite(d)
        nf = int(fin.sum())
        ci = [float(x) for x in np.percentile(d[fin], [2.5, 97.5])] if nf >= 0.95 * nd else None
        return {
            "point": float(v[0]) if math.isfinite(v[0]) else None,
            "finite_draws": nf,
            "draws": nd,
            "ci95": ci,
            "degenerate": bool(nf > 0 and np.ptp(d[fin]) == 0),
        }

    def block_mains(cells):
        """-> (M, Wm): M averages a per-cell array over each main's cells (a main's value is the mean over its cohort
        cells); Wm holds the draws' multiplicities of those mains."""
        mains = sorted({main_of[c] for c in cells})
        mi = {m: j for j, m in enumerate(mains)}
        ci = np.array([mi[main_of[c]] for c in cells], dtype=np.int64)
        order = np.argsort(ci, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(ci[order])) + 1]
        cnt = np.diff(np.r_[starts, len(ci)]).astype(np.float64)

        def M(h):  # h: (..., n_cells) -> (..., n_mains)
            return np.add.reduceat(h[..., order], starts, axis=-1) / cnt

        Wm = W[:, [mpos[m] for m in mains]].astype(np.float64)
        return M, Wm

    def wmean(Wm, A):  # A: (n_mains,) or (draws+1, n_mains)
        num = Wm @ A if A.ndim == 1 else (Wm * A).sum(1)
        den = Wm.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    def run_red(zt, keys, how):
        f = np.max if how == "max" else np.mean
        return np.array([f(zt[units_of[k]]) for k in keys])

    def sweep(zt, scheme):
        pools = (
            {"union": np.concatenate([pool_idx["S_old"], pool_idx["S_new"]])} if scheme == "union" else dict(pool_idx)
        )
        pool_runs = {"union": pool["S_old"] + pool["S_new"]} if scheme == "union" else dict(pool)
        tpool = (lambda st: "union") if scheme == "union" else (lambda st: st)
        sorted_pool = {p: np.sort(zt[ix], kind="stable")[::-1] for p, ix in pools.items()}
        pool_runmax = {p: run_red(zt, [k for k in pool_runs[p] if k in units_of], "max") for p in pools}
        att_max = {st: {w: run_red(zt, [f"{w}|{c}" for c in cells_of[st]], "max") for w in ATTACK} for st in cells_of}
        blk = {}
        for b, sts in (
            ("S_old", ["S_old"]),
            ("S_new", ["S_new"]),
            ("combined", ["S_old", "S_new"]),
        ):
            cells = [c for st in sts for c in cells_of[st]]
            M, Wm = block_mains(cells)
            blk[b] = (sts, cells, M, Wm)
        # honest sources per stratum: the pool's sources (by membership) and honest A (held out; cohort-cell runs, adopted reading)
        hon = {}
        for st in ("S_old", "S_new"):
            hon[st] = {"honest_a": hon_a[st]}
            for src in pool_src[st]:
                hon[st][src] = [k for k in pool[st] if k.startswith(src + "|")]
        pts, Tsel, cache_nv = [], [], {}
        for i, f in enumerate(grid):
            th, T = {}, {}
            for p, srt in sorted_pool.items():
                n = len(srt)
                k = math.floor(f * n + 1e-9)
                t = float(srt[min(k, n - 1)])
                above = int((zt[pools[p]] > t).sum())
                th[p] = {
                    "k": k,
                    "n_pool_units": n,
                    "threshold": t,
                    "estimable": k >= 1,
                    "pool_units_above": above,
                    "shaded_lt10_above": above < 10,
                    "realized_unit_fpr_pool": above / n,
                    "realized_run_false_alarm_pool": float((pool_runmax[p] > t).mean()),
                }
                T[p] = t
            Tsel.append(T)
            blocks = {}
            for b, (sts, cells, M, Wm) in blk.items():
                hits = {
                    w: np.concatenate([(att_max[st][w] > T[tpool(st)]).astype(np.float64) for st in sts])
                    for w in ATTACK
                }
                hits["real_mean"] = (hits["run1"] + hits["run2"]) / 2
                pm = {w: M(hits[w]) for w in hits}
                v = {w: wmean(Wm, pm[w]) for w in pm}
                tpr = {w: summ(v[w]) for w in ("run1", "run2", "sim", "real_mean")}
                diff = {
                    "sim_minus_real_mean": summ(wmean(Wm, pm["sim"] - pm["real_mean"])),
                    "sim_minus_run1": summ(wmean(Wm, pm["sim"] - pm["run1"])),
                    "sim_minus_run2": summ(wmean(Wm, pm["sim"] - pm["run2"])),
                    "run2_minus_run1": summ(wmean(Wm, pm["run2"] - pm["run1"])),
                }
                honest = {}
                srcs = sorted({x for st in sts for x in hon[st]})
                for src in srcs:
                    nu = ab = nr = ra = 0
                    for st in sts:
                        ks = hon[st].get(src, [])
                        t = T[tpool(st)]
                        for kk in ks:
                            zz = zt[units_of[kk]]
                            nu += len(zz)
                            ab += int((zz > t).sum())
                            nr += 1
                            ra += int(zz.max() > t)
                    honest[src] = {
                        "unit_fpr": ab / nu if nu else None,
                        "run_false_alarm": ra / nr if nr else None,
                        "runs": nr,
                    }
                    if src == "honest_a":
                        base = f"{scheme}#points[{i}].blocks.{b}.honest.honest_a"
                        for fld in ("runs", "unit_fpr", "run_false_alarm"):
                            alt.setdefault(f"{base}.{fld}", {})
                        for vname, vset in hon_a_var.items():
                            nu2 = ab2 = nr2 = ra2 = 0
                            for st in sts:
                                t = T[tpool(st)]
                                for kk in vset[st]:
                                    zz = zt[units_of[kk]]
                                    nu2 += len(zz)
                                    ab2 += int((zz > t).sum())
                                    nr2 += 1
                                    ra2 += int(zz.max() > t)
                            alt[f"{base}.runs"][vname] = nr2
                            alt[f"{base}.unit_fpr"][vname] = ab2 / nu2 if nu2 else None
                            alt[f"{base}.run_false_alarm"][vname] = ra2 / nr2 if nr2 else None
                read = all(not th[tpool(st)]["shaded_lt10_above"] for st in sts)
                n_att_units = sum(len(units_of[f"{w}|{c}"]) for c in cells for w in ATTACK)
                n_pool_units = sum(len(pools[tpool(st)]) for st in sts) if scheme != "union" else len(pools["union"])
                n_hon_units = sum(len(units_of[kk]) for st in sts for src in hon[st] for kk in hon[st][src])
                blocks[b] = {
                    "n_cells": len(cells),
                    # READING (identified, adopted in run 5): the block's draw units = its cohort mains
                    "n_units": len({main_of[c] for c in cells}),
                    "readable": read,
                    "honest": honest,
                    "tpr": tpr,
                    "diff": diff,
                    "saturated": {w: tpr[w]["point"] in (0.0, 1.0) for w in tpr},
                }
                cellset = set(cells)
                n_cell_units = int(sum(1 for c_ in cell.tolist() if c_ in cellset)) if i == 0 else None
                nv = {
                    "prestated_attack_units": n_att_units,
                    "pool_units": n_pool_units,
                    "attack_plus_pool_plus_honest_a_units": n_att_units + n_hon_units,
                    "attack_plus_pool_units": n_att_units + n_pool_units,
                    "sim_units": sum(len(units_of[f"sim|{c}"]) for c in cells),
                    "real_units": sum(len(units_of[f"{w}|{c}"]) for c in cells for w in ("run1", "run2")),
                    "all_units_in_stratum_files": int(sum((stratum == st).sum() for st in sts)),
                }
                for vname, vset in hon_a_var.items():
                    nv[f"attack_plus_pool_plus_honest_a_{vname}"] = (
                        n_att_units + n_pool_units + sum(len(units_of[kk]) for st in sts for kk in vset[st])
                    )
                nkey = f"{scheme}#points[{i}].blocks.{b}.n_units"
                if i == 0:
                    nv["all_units_of_block_cells_any_world"] = n_cell_units
                    cache_nv[b] = nv["all_units_of_block_cells_any_world"]
                else:
                    nv["all_units_of_block_cells_any_world"] = cache_nv[b]
                alt[nkey] = nv
            pts.append({"fpr_target": f, "thresholds": th, "blocks": blocks})
        return pts, Tsel, pools, att_max, blk

    def joint(zt, scheme, pools, att_max, blk, Tsel):
        """Panel 3 B: each draw re-estimates T_b with A24's weighted arithmetic over the pool units, a unit weighted by
        W[b, main of its run's cell]; draws 1..N give the intervals; draw 0 must equal the fixed path (GB1)."""
        out, gb1 = {}, {}
        tpool = (lambda st: "union") if scheme == "union" else (lambda st: st)
        cache = {}
        for p, ix in pools.items():
            o = np.argsort(-zt[ix], kind="mergesort")
            zs = zt[ix][o]
            mains_u = np.array([mpos[main_of[run[j].split("|", 1)[1]]] for j in ix[o]])
            Tb = np.full((nd + 1, len(grid)), np.nan)
            Kb = np.zeros((nd + 1, len(grid)), np.int64)
            gr = np.array(grid)
            for b in range(nd + 1):
                Tb[b], Kb[b] = weighted_thresholds(zs, W[b, mains_u], gr)
            cache[p] = (Tb, Kb)
        entries = (
            {"S_old": ["S_old"], "S_new": ["S_new"]}
            if scheme != "union"
            else {
                "S_old": ["S_old"],
                "S_new": ["S_new"],
                "combined": ["S_old", "S_new"],
            }
        )
        for b, sts in entries.items():
            _, _cells, M, Wm = blk[b]
            p = tpool(sts[0])
            Tb, Kb = cache[p]
            lst, ok0 = [], True
            for i, f in enumerate(grid):
                tcol = Tb[:, i]
                hits = {}
                for w in ATTACK:
                    am = np.concatenate([att_max[st][w] for st in sts])
                    h = (am[None, :] > tcol[:, None]).astype(np.float64)
                    h[~np.isfinite(tcol)] = np.nan
                    hits[w] = h
                hits["real_mean"] = (hits["run1"] + hits["run2"]) / 2
                pm = {w: M(hits[w]) for w in hits}
                v = {w: wmean(Wm, pm[w]) for w in pm}
                tpr = {w: summ(v[w]) for w in ("run1", "run2", "sim", "real_mean")}
                diff = {
                    "sim_minus_real_mean": summ(wmean(Wm, pm["sim"] - pm["real_mean"])),
                    "sim_minus_run1": summ(wmean(Wm, pm["sim"] - pm["run1"])),
                    "sim_minus_run2": summ(wmean(Wm, pm["sim"] - pm["run2"])),
                    "run2_minus_run1": summ(wmean(Wm, pm["run2"] - pm["run1"])),
                }
                kd = Kb[1:, i]
                kall = Kb[:, i]
                jb_path = f"{scheme}#joint.{b}[{i}]"
                alt[f"{jb_path}.draws_k0"] = {"draws_0_to_N": int((kall == 0).sum())}
                alt[f"{jb_path}.draws_k_lt10"] = {"draws_0_to_N": int((kall < 10).sum())}
                alt[f"{jb_path}.k_draws_p5_p50_p95"] = {
                    "draws_0_to_N": [float(x) for x in np.percentile(kall, [5, 50, 95])]
                }
                lst.append(
                    {
                        "fpr_target": f,
                        "threshold_draw0": float(tcol[0]),
                        "k_draw0": int(Kb[0, i]),
                        "draws_k0": int((kd == 0).sum()),
                        "draws_k_lt10": int((kd < 10).sum()),
                        "k_draws_p5_p50_p95": [float(x) for x in np.percentile(kd, [5, 50, 95])],
                        "tpr": tpr,
                        "diff": diff,
                    }
                )
                ok0 &= tcol[0] == Tsel[i][p]
                for st in sts:
                    am = att_max[st]
                    ok0 &= all(bool(((am[w] > tcol[0]) == (am[w] > Tsel[i][p])).all()) for w in ATTACK)
            out[b] = lst
            gb1[b] = bool(ok0)
        if scheme != "union":
            out["combined"] = {"withheld": "stratum-specific thresholds: no single joint pool for the combined block"}
        return out, gb1

    def score_shift(zt, scheme):
        """RULE 'Score-level shift': run mean and run max of transformed units; paired H contrasts and standardized
        shifts per cell, averaged within main, mains equal-weighted with the shared draws."""
        pools_ = {"union": pool["S_old"] + pool["S_new"]} if scheme == "union" else dict(pool)
        sig, sig_alt, res = {}, {}, {}
        for agg in ("run_mean", "run_max"):
            how = "mean" if agg == "run_mean" else "max"
            if scheme == "union":
                su = float(np.std(run_red(zt, [k for k in pools_["union"] if k in units_of], how)))
                sig[agg] = {"S_old": su, "S_new": su}
                sig_alt[agg] = {
                    st: float(np.std(run_red(zt, [k for k in pool[st] if k in units_of], how)))
                    for st in ("S_old", "S_new")
                }
            else:
                sig[agg] = {
                    st: float(np.std(run_red(zt, [k for k in pools_[st] if k in units_of], how)))
                    for st in ("S_old", "S_new")
                }
        for b, sts in (
            ("S_old", ["S_old"]),
            ("S_new", ["S_new"]),
            ("combined", ["S_old", "S_new"]),
        ):
            cells = [c for st in sts for c in cells_of[st]]
            M, Wm = block_mains(cells)
            res[b] = {}
            for agg in ("run_mean", "run_max"):
                how = "mean" if agg == "run_mean" else "max"
                sc = {
                    w: np.concatenate([run_red(zt, [f"{w}|{c}" for c in cells_of[st]], how) for st in sts])
                    for w in ATTACK
                }
                sigv = np.concatenate([np.full(len(cells_of[st]), sig[agg][st]) for st in sts])
                H = lambda x: np.where(x > 0, 1.0, np.where(x == 0, 0.5, 0.0))
                with np.errstate(invalid="ignore", divide="ignore"):
                    cc = {
                        "A_sr": 0.5 * (H(sc["sim"] - sc["run1"]) + H(sc["sim"] - sc["run2"])),
                        "A_21": H(sc["run2"] - sc["run1"]),
                        "d_sr": (sc["sim"] - (sc["run1"] + sc["run2"]) / 2) / sigv,
                        "d_21": (sc["run2"] - sc["run1"]) / sigv,
                    }
                res[b][agg] = {}
                for st_name, v in cc.items():
                    if st_name.startswith("d_") and not (np.isfinite(sigv).all() and (sigv > 0).all()):
                        res[b][agg][st_name] = None
                        continue
                    res[b][agg][st_name] = summ(wmean(Wm, M(v)))
                if scheme == "union":
                    sigv2 = np.concatenate([np.full(len(cells_of[st]), sig_alt[agg][st]) for st in sts])
                    for st_name, v in (
                        ("d_sr", (sc["sim"] - (sc["run1"] + sc["run2"]) / 2) / sigv2),
                        ("d_21", (sc["run2"] - sc["run1"]) / sigv2),
                    ):
                        sm = summ(wmean(Wm, M(v)))
                        for kk in ("point", "ci95"):
                            alt[f"score_shift.{b}.{agg}.{st_name}.{kk}"] = {"sigma_H_per_stratum_pool": sm[kk]}
        if scheme == "union":
            for agg, per in sig_alt.items():
                for st, val in per.items():
                    alt[f"sigma_H.{agg}.{st}"] = {"per_stratum_pool": val}
        return sig, res

    # ---------------------------------------------------------------- G2: S_new run-level references in the arm folders
    def g2():
        if spec["g2_json"] is None:
            return {"reference": "NA: no raw run reference in the arm folder (famexp): G2 not applicable, stated"}
        if spec["g2_json"].startswith("p2/"):
            fields = ("reply_mean", None) if spec["readout"] == "R" else ("command", None)
        else:
            fields = ("run_scores", "by_model") if spec["readout"] == "R" else ("cmd_A", "cmd_A_by_model")
        ref_d, ref_m = {}, {}
        for arm in a["arm_order"]:
            path, sha = a["arms"][arm][spec["g2_json"]]
            j = json.loads(_read(f"/conv/{path}", sha))
            ref_d.update(j[fields[0]])
            if fields[1]:
                ref_m.update(j.get(fields[1]) or {})
        newruns = sorted({str(k) for k in run[stratum == "S_new"]})

        def check(keys):
            checked, without, mx = 0, [], 0.0
            for k in keys:
                u = units_of[k]
                if folds5 and (fold[u] == -1).all():
                    if k not in ref_m:
                        without.append(k)
                        continue
                    d = np.abs(sbm[u].mean(axis=0) - np.array(ref_m[k], dtype=np.float64))
                elif k in ref_d and ref_d[k] is not None:
                    d = np.array([abs(float(np.mean(s[u])) - float(ref_d[k]))])
                else:
                    without.append(k)
                    continue
                checked += 1
                mx = max(mx, float(np.max(d))) if np.isfinite(d).all() else float("inf")
            return checked, without, mx

        # READING (identified, adopted in run 5): G2 checks the ANALYSED S_new runs (cohort attack runs, the pool, cohort-
        # cell honest A); the pre-stated reading (every S_new run in the files) is kept as a variant
        checked_all, without, mx_all = check(newruns)  # all-runs pass, for the variants below
        checked, without_an, mx = check([k for k in newruns if k in analysed_runs])
        newm = stratum == "S_new"
        fin = bool(req_finite[newm].all())
        alt["G2.all_finite"] = {"all_rows": bool(req_finite.all())}
        newset, without_set, attack_set = set(newruns), set(without), set(attack_runs)
        chk = [k for k in newruns if k not in without_set]
        cohort_pool = (set(attack_runs) | set(pool["S_new"])) & newset
        alt["G2.max_abs_diff"] = {"prestated_all_S_new_runs": mx_all}
        alt["G2.n_runs_without_reference"] = {"prestated_all_S_new_runs": len(without)}
        alt["G2.runs_checked"] = {
            "prestated_all_S_new_runs": checked_all,
            "reference_keys": len(set(ref_d) | set(ref_m)),
            "active_S_new_runs_with_reference": sum(k in active_runs for k in chk),
            "cohort_attack_and_pool_S_new_runs_with_reference": sum(k in cohort_pool for k in chk),
            "cohort_attack_S_new_runs_with_reference": sum(k in attack_set for k in chk),
            "values_compared": sum(5 if (folds5 and (fold[units_of[k]] == -1).all()) else 1 for k in chk),
            "reference_keys_present_in_rows": len((set(ref_d) | set(ref_m)) & newset),
        }
        return {
            "reference": f"{spec['g2_json']} in {len(a['arm_order'])} arm folders",
            "runs_checked": checked,
            "runs_without_reference": without_an,
            "n_runs_without_reference": len(without_an),
            "max_abs_diff": mx,
            "all_finite": fin,
            "pass_1e-9": bool(mx <= 1e-9 and not without_an and fin),
        }

    # ---------------------------------------------------------------- my further G2 checks (RULE G2; not in the target's G2 fields)
    def g2_old():
        """S_old: each run's mean over its rows (R and C) against the pinned scorer record's run scores, on the same key set
        after dropping reference entries that are null or non-finite (runs without rows). famexp: none pinned."""
        ref = a.get("old_ref")
        out = {"reference": spec.get("old_ref_name")}
        oldruns = sorted({str(k) for k in run[stratum == "S_old"]})
        if ref is None:
            out["status"] = "no pinned S_old run reference"
            return out

        def finite(v):
            return v is not None and math.isfinite(float(v))

        only_rows = [k for k in oldruns if k not in ref or not finite(ref[k])]
        only_ref = [k for k in ref if finite(ref[k]) and (k not in units_of or stratum[units_of[k][0]] != "S_old")]
        out["n_reference_entries_null_or_nonfinite"] = sum(not finite(v) for v in ref.values())
        mx = 0.0
        for k in oldruns:
            if k in ref and finite(ref[k]):
                mx = max(mx, abs(float(np.mean(s[units_of[k]])) - float(ref[k])))
        out |= {
            "runs_checked": len(oldruns) - len(only_rows),
            "n_runs_only_in_rows": len(only_rows),
            "n_runs_only_in_reference": len(only_ref),
            "pass_1e-9": bool(mx <= 1e-9 and not only_rows and not only_ref),
        }
        return out

    def shared_rows():
        """Sol and c_asis reply rows against p2's reply rows (FIBS reply_mean files), run by run, where both score a run."""
        if spec.get("shared_rows") is None:
            return None
        sr = spec["shared_rows"]
        cnt = {}
        zo = npz(sr["old"]["volume"], sr["old"]["path"], sr["old"]["sha256"])
        keys = [zo["key"].astype(str)]
        for arm in a["arm_order"]:
            path, sha = a["arms"][arm][sr["new"]]
            keys.append(npz("conv", path, sha)["key"].astype(str))
        kk, nn = np.unique(np.concatenate(keys), return_counts=True)
        cnt = dict(zip(kk.tolist(), nn.tolist()))
        shared = [k for k in units_of if k in cnt]
        bad = [k for k in shared if len(units_of[k]) != cnt[k]]
        return {
            "runs_shared": len(shared),
            "runs_with_different_row_counts": len(bad),
            "pass": not bad,
        }

    # ---------------------------------------------------------------- composition search for two descriptive counts
    def subset_search(components, target_value, cap=8):
        """Names of component subsets whose counts sum to the target's count (names only; no count is returned)."""
        import itertools

        if not isinstance(target_value, int) or isinstance(target_value, bool):
            return {"searched": False}
        names = sorted(components)
        hits = []
        for r_ in range(1, len(names) + 1):
            for comb in itertools.combinations(names, r_):
                if sum(components[x] for x in comb) == target_value:
                    hits.append(list(comb))
        return {
            "searched": True,
            "n_components": len(names),
            "n_matching_subsets": len(hits),
            "matching_subsets": sorted(hits, key=len)[:cap],
        }

    def categories(sts):
        """Unit and run counts per category for the strata sts: attack worlds in / out of the cohort, honest worlds in /
        out of the pool or the cohort cells."""
        cohs = {c for st in sts for c in cells_of[st]}
        pk = {k for st in sts for k in pool[st]}
        cat_u, cat_r = {}, {}
        for k, u in units_of.items():
            st_ = stratum[u[0]]
            if st_ not in sts:
                continue
            w, c_ = k.split("|", 1)
            if w in ATTACK:
                name = f"{w}_{'cohort' if c_ in cohs else 'noncohort'}"
            elif w == "honest_a":
                name = f"honest_a_{'cohortcell' if c_ in cohs else 'noncohortcell'}"
            else:
                name = f"{w}_{'pool' if k in pk else 'nonpool'}"
            cat_u[name] = cat_u.get(name, 0) + len(u)
            cat_r[name] = cat_r.get(name, 0) + 1
        return cat_u, cat_r

    diag = {}
    tp = a["target_row"].get("primary", {}).get("points") if isinstance(a["target_row"].get("primary"), dict) else None
    if tp:
        uold, unew = set(a["universe_old"]), set(a["universe_new"])
        for b, sts in (("S_old", ["S_old"]), ("S_new", ["S_new"]), ("combined", ["S_old", "S_new"])):
            cu, cr_ = categories(sts)
            tv = tp[0].get("blocks", {}).get(b, {}).get("n_units")
            diag[f"n_units_{b}"] = subset_search(cu, tv)
            diag[f"n_units_{b}_as_run_counts"] = subset_search(cr_, tv)
            cells_b = [c for st in sts for c in cells_of[st]]
            umains = (uold if "S_old" in sts else set()) | (unew if "S_new" in sts else set())
            diag[f"n_units_{b}_named"] = {
                "equals_cohort_mains_in_block": tv == len({main_of[c] for c in cells_b}),
                "equals_universe_mains_of_block_strata": tv == len(umains),
                "equals_attack_runs_in_block": tv == 3 * len(cells_b),
                "equals_cohort_cells_in_block": tv == len(cells_b),
            }
    _, cr_new = categories(["S_new"])
    tg = a["target_row"].get("G2", {})
    diag["G2_runs_checked_S_new_runs"] = subset_search(cr_new, tg.get("runs_checked") if isinstance(tg, dict) else None)
    info["composition_search"] = diag

    gold = g2_old()
    info["G2_S_old_reference"] = gold
    if "pass_1e-9" in gold and spec.get("old_ref_is_gate"):
        gates["G2_S_old_reference"] = gold["pass_1e-9"]
    shr = shared_rows()
    if shr is not None:
        info["G2_sol_p2_shared_row_counts"] = shr
        gates["G2_sol_p2_shared_row_counts"] = shr["pass"]

    # ---------------------------------------------------------------- assemble the row
    mine = {
        "label": spec["label"],
        "probe": spec["probe"],
        "readout": spec["readout"],
        "units": spec["units"],
        "routing": spec["routing_text"],
        "scheme_source": spec["scheme_source"],
        "stratum_labels": {**STRATUM_LABELS, "combined": COMBINED_LABEL},
        "caveats": spec["caveats"],
        **({"K": spec["K"]} if spec["K"] is not None else {}),
        "scheme": "stratum" if spec["scheme"] == "stratum_specific" else "union",
        "coverage": coverage,
        "non_finite_rows": non_finite_rows,
    }
    if spec.get("row_label") is not None:
        mine["row_label"] = spec["row_label"]
    if spec.get("arm_labels") is not None:
        mine["arm_labels"] = spec["arm_labels"]
    if spec.get("pool_export_row") is not None:
        mine["pool_export_row"] = spec["pool_export_row"]
    if folds5:
        mine["G3_S_new_fold_map"] = gates["G3_S_new"]
    mine["G2"] = g2()
    tf, zt, why = (None, None, "; ".join(withhold)) if withhold else transform(spec["scheme"])
    status = "computed"
    if tf is None:
        status = f"withheld: {why}"
    else:
        pts, Tsel, pools_used, att_max, blk = sweep(zt, spec["scheme"])
        mine["primary"] = {"transform": tf, "points": pts}
        if spec["union_sensitivity"]:
            tfu, ztu, whyu = transform("union")
            if tfu is not None:
                ptsu, *_ = sweep(ztu, "union")
                mine["union_sensitivity"] = {"transform": tfu, "points": ptsu}
            else:
                mine["union_sensitivity"] = {"withheld": whyu}
        jb, gb1 = joint(zt, spec["scheme"], pools_used, att_max, blk, Tsel)
        mine["joint_threshold_bootstrap"] = jb
        mine["GB1_joint_draw0_equals_fixed"] = gb1
        sig, ss = score_shift(zt, spec["scheme"])
        mine["sigma_H"] = sig
        mine["score_shift"] = ss
        gates["GB1"] = all(gb1.values())
    g2ok = mine["G2"].get("pass_1e-9", mine["G2"]["reference"].startswith("NA:"))
    gates["G2_S_new_reference"] = bool(g2ok)
    mine["pass"] = bool(status == "computed" and g2ok and all(gates.values()))
    mine["status"] = status
    # draw-key lookups for alt variants are scheme-prefixed; translate to the target's key paths
    alt2 = {}
    for k, v in alt.items():
        if k.startswith(spec["scheme"] + "#joint."):
            alt2["joint_threshold_bootstrap." + k.split("#joint.", 1)[1]] = v
        elif k.startswith(spec["scheme"] + "#"):
            alt2["primary." + k.split("#", 1)[1]] = v
        elif k.startswith("union#") and spec["scheme"] != "union":
            alt2["union_sensitivity." + k.split("#", 1)[1]] = v
        elif "#" not in k:
            alt2[k] = v
    cmp = compare_tree(a["target_row"], mine, a["checked_strings"], alt2)
    findings = []
    tg2 = a["target_row"].get("G2", {}).get("reference") if isinstance(a["target_row"].get("G2"), dict) else None
    want = spec["g2_json"] if spec["g2_json"] else "NA:"
    info["target_G2_reference_names_the_expected_reference"] = isinstance(tg2, str) and want in tg2
    if spec["K"] is None and "K" not in a["target_row"]:
        findings.append("target omits K on this row (null in the registered config); compared as absent on both sides")
    jtb = a["target_row"].get("joint_threshold_bootstrap")
    jt = jtb.get("combined") if isinstance(jtb, dict) else None
    if isinstance(jt, dict) and "withheld" in jt:
        findings.append(
            "target withholds joint_threshold_bootstrap.combined (stratum-specific thresholds); the RULE asks for the joint "
            "intervals beside the fixed ones at every point and does not say to withhold the combined block"
        )
    if folds5:
        findings.append(
            "RULE Calibration 5 asks realized rates also by routing group (mapped vs fold -1); the target has no such field"
        )
    return {
        "row": rk,
        "gates": gates,
        "info": info,
        "pool_sources": pool_src,
        "comparison": cmp,
        "my_pass": mine["pass"],
        "my_status": status,
        "target_findings": findings,
    }


def _strings(x) -> set:
    """Every string inside a JSON value (keys and values)."""
    if isinstance(x, str):
        return {x}
    if isinstance(x, dict):
        return set().union(*(({k} | _strings(v)) for k, v in x.items())) if x else set()
    if isinstance(x, list):
        return set().union(*(_strings(v) for v in x)) if x else set()
    return set()


def _runs_in_one_source(run, armv):
    seen = {}
    for k, a in zip(run.tolist(), armv.tolist()):
        if seen.setdefault(k, a) != a:
            return False
    return True


# ------------------------------------------------------------------ comparison (runs in containers; no value leaves it)
def compare_tree(t, m, checked_strings, alt):
    """Walk the target tree; compare each leaf with mine at the same key path. Output: counts, key paths, and for
    mismatching numbers a coarse |diff| bucket label. Strings are compared only where the registered config or RULE gives
    the text (checked_strings, by generic path); other strings are counted as descriptive."""
    import re

    checked_strings = set(checked_strings)
    out = {
        "numeric_compared": 0,
        "numeric_match": 0,
        "bool_compared": 0,
        "bool_match": 0,
        "none_compared": 0,
        "none_match": 0,
        "str_checked": 0,
        "str_checked_match": 0,
        "str_semantic_checked": 0,
        "str_semantic_match": 0,
        "str_descriptive_type_checked": 0,
        "not_produced": [],
        "only_mine": [],
        "structure_mismatch": [],
        "mismatch": [],
    }

    def gen(p):
        return re.sub(r"\[\d+\]", "[]", p)

    def num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def same(tv, mv):
        if isinstance(tv, bool) or isinstance(mv, bool):
            return isinstance(tv, bool) and isinstance(mv, bool) and tv == mv
        if tv is None or mv is None:
            return tv is None and mv is None
        if isinstance(tv, int):
            return isinstance(mv, int) and not isinstance(mv, bool) and mv == tv
        if num(tv) and num(mv):  # NaN never matches (the target writes withheld values as null)
            return math.isfinite(tv) and math.isfinite(mv) and abs(float(tv) - float(mv)) <= TOL_FLOAT
        return False

    def variant_of(tv, p):
        cands = dict(alt.get(p, {}))
        mm = re.match(r"^(.*)\[(\d+)\]$", p)
        if mm and mm.group(1) in alt:  # list-valued variant (an interval): compare the element
            for name, val in alt[mm.group(1)].items():
                if isinstance(val, list) and int(mm.group(2)) < len(val):
                    cands[name] = val[int(mm.group(2))]
        names = sorted(n for n, val in cands.items() if not isinstance(val, (list, dict)) and same(tv, val))
        return "+".join(names) if names else None

    def leaf(tv, mv, p):
        if isinstance(tv, str):
            if gen(p) in SEMANTIC:
                kind = "str_semantic"
                out["str_semantic_checked"] += 1
                if isinstance(mv, str) and all(x in tv for x in SEMANTIC[gen(p)]):
                    out["str_semantic_match"] += 1
                    return
            elif gen(p) not in checked_strings:
                kind = "str_type"
                out["str_descriptive_type_checked"] += 1
                if isinstance(mv, str):
                    return
            else:
                kind = "str_checked"
                out["str_checked"] += 1
                if tv == mv:
                    out["str_checked_match"] += 1
                    return
        else:
            kind = "bool" if isinstance(tv, bool) else ("none" if tv is None else ("numeric" if num(tv) else "other"))
            out[f"{kind}_compared"] = out.get(f"{kind}_compared", 0) + 1
            if same(tv, mv):
                out[f"{kind}_match"] = out.get(f"{kind}_match", 0) + 1
                return
        bucket = None
        if num(tv) and num(mv):
            dd = abs(float(tv) - float(mv))
            bucket = "le1e-9" if dd <= 1e-9 else "le1e-6" if dd <= 1e-6 else "le1e-3" if dd <= 1e-3 else "gt1e-3"
        elif tv is None or mv is None:
            bucket = "one_side_null"
        out["mismatch"].append(
            {
                "key_path": p,
                "kind": kind,
                "diff_bucket": bucket,
                "matched_by_variant": variant_of(tv, p),
            }
        )

    def walk(tv, mv, p):
        if isinstance(tv, dict):
            if not isinstance(mv, dict):
                out["structure_mismatch"].append(p)
                return
            for k in tv:
                q = f"{p}.{k}" if p else k
                if k not in mv:
                    out["not_produced"].append(q)
                else:
                    walk(tv[k], mv[k], q)
            out["only_mine"].extend(f"{p}.{k}" if p else k for k in mv if k not in tv)
        elif isinstance(tv, list):
            if mv is None:
                out["mismatch"].append(
                    {
                        "key_path": p,
                        "kind": "list_vs_null",
                        "diff_bucket": "one_side_null",
                        "matched_by_variant": None,
                    }
                )
                return
            if not isinstance(mv, list) or len(mv) != len(tv):
                out["structure_mismatch"].append(p)
                return
            for i, (x, y) in enumerate(zip(tv, mv)):
                walk(x, y, f"{p}[{i}]")
        else:
            if isinstance(mv, (dict, list)):
                out["structure_mismatch"].append(p)
                return
            leaf(tv, mv, p)

    walk(t, m, "")
    by = {}
    for x in out["mismatch"]:
        by.setdefault(gen(x["key_path"]), []).append(x)
    out["mismatch_by_generic_path"] = {
        gp: {
            "count": len(v),
            "examples": [x["key_path"] for x in v[:5]],
            "diff_buckets": sorted({str(x["diff_bucket"]) for x in v}),
            "matched_by_variant": sorted({str(x["matched_by_variant"]) for x in v}),
        }
        for gp, v in by.items()
    }
    out["n_mismatch"] = len(out["mismatch"])
    out["mismatch"] = out["mismatch"][:300]
    return out


# ------------------------------------------------------------------ local entrypoint: pinned git blobs only, then Modal
def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, check=True, cwd=HERE).stdout


@app.local_entrypoint()
def main():
    if _git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("refused: uncommitted tracked changes")
    head = _git("rev-parse", "HEAD").decode().strip()
    _git("fetch", "-q", "origin", BRANCH)
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", head, f"origin/{BRANCH}"],
            check=False,
            cwd=HERE,
        ).returncode
        != 0
    ):
        raise SystemExit("refused: HEAD is not on origin; push first")
    blobs, pins = {}, {}
    for name, (c, p, sha) in GIT.items():
        b = _git("show", f"{c}:{p}")
        pins[name] = {"commit": c, "path": p, "sha256": sha, "match": _sha(b) == sha}
        blobs[name] = b
    for name, (p, sha) in LOCAL.items():
        b = p.read_bytes()
        pins[name] = {"path": str(p), "sha256": sha, "match": _sha(b) == sha}
        blobs[name] = b
    if not all(x["match"] for x in pins.values()):
        raise SystemExit(
            "G1: a pinned input's sha256 differs: " + ", ".join(k for k, x in pins.items() if not x["match"])
        )
    L = {
        k: json.loads(v)
        for k, v in blobs.items()
        if k not in ("answers_09", "release", "target_script", "target_loader", "target_core")
    }
    tgt, fr = L["target"], L["final"]["result"]
    # pass only the keys the spec names (membership, pools, maps); FINAL's outcome fields are not passed
    cr = fr["calibration_records"]["primary"]
    ks = {k for r in L["inputs_registered"]["rows"] if r["K"] for k in r["K"]} | {"sol_a|R|0.01|primary|BuC|B"}
    fin = {
        "cohort": fr["cohort"],
        "env_of_main": fr["maps"]["env_of_main"],
        "origin_of_main": fr["maps"]["origin_of_main"],
        "main_of": fr["maps"]["main_of"],
        "registered_new_cells": sorted(fr["run_table_summary"]["rows"]),
        "routing_evidence": fr["routing_evidence"],
        "sol_training_manifests": {
            arm: sorted(v["sol_training_manifests"].values()) for arm, v in fr["scale_up_inputs"].items()
        },
        "run_table_rows": {
            c: {
                k: r[k]
                for k in (
                    "run1_launch",
                    "run2_launch",
                    "sim_launch",
                    "attack_pair_launch",
                )
            }
            for c, r in fr["run_table_summary"]["rows"].items()
        },
        "missing_scores": fr["missing_scores"],
        "non_finite_transformed_runs": fr["non_finite_transformed_runs"],
        "approach_excluded_cells": fr["approach_excluded_cells"],
        "omitted_registered_cells": sorted(fr["omitted_registered_cells"]),
        "staging_applied": sorted(fr["staging_exclusions"]["applied"]),
        "setup_failures": sorted(fr["staging_exclusions"]["setup_failures"]),
        "excluded_before_counting": sorted(fr["staging_exclusions"]["excluded_before_counting"]),
        "spec_arms": L["final"]["spec"]["arms"],
        "calibration_records": {
            k: {
                "pool_old": cr[k]["pool_old"],
                "pool_new": cr[k]["pool_new"],
                "primary": cr[k]["primary"],
                "schemes": {
                    s: {
                        "pool_keys_old": v["pool_keys_old"],
                        "pool_keys_new": v["pool_keys_new"],
                    }
                    for s, v in cr[k]["schemes"].items()
                },
            }
            for k in sorted(ks)
        },
    }
    asis = L["asis_record"]["rows_file"]["files"]
    c455 = L["common455"]
    # S_old run-level references (RULE G2), by key, from the pinned scorer records
    ars, oscores = L["asis_record"]["run_scores"], L["owna_record"]["scores"]
    rh, hs, fb = (
        L["rh_record"]["probes"],
        L["hs_record"]["probes"]["highstakes"],
        L["fibs_record"],
    )
    old_refs = {
        "sol_a|R": {
            "name": "asis_scores run_scores.avg_top2.a_tight (asis rows file; see sol_old_file_pairs)",
            "ref": ars["avg_top2"]["a_tight"],
        },
        "sol_a|C": {
            "name": "asis_scores run_scores.cmd_A.a_tight (asis rows file; see sol_old_file_pairs)",
            "ref": ars["cmd_A"]["a_tight"],
        },
        "c_asis|R": {
            "name": "asis_scores run_scores.avg_top2.c_asis_tight",
            "ref": ars["avg_top2"]["c_asis_tight"],
        },
        "c_asis|C": {
            "name": "asis_scores run_scores.cmd_A.c_asis_tight",
            "ref": ars["cmd_A"]["c_asis_tight"],
        },
        "fibs|R": {
            "name": "fibs_scores run_scores.reply_mean",
            "ref": fb["run_scores"]["reply_mean"],
        },
        "fibs|C": {
            "name": "fibs_scores run_scores.command",
            "ref": fb["run_scores"]["command"],
        },
        "highstakes|R": {
            "name": "highstakes run_scores.reply_mean",
            "ref": hs["run_scores"]["reply_mean"],
        },
        "highstakes|C": {
            "name": "highstakes run_scores.command",
            "ref": hs["run_scores"]["command"],
        },
        "own_a|R": {
            "name": "correctedb_scores scores.own_a_tight_substituted.avg_top2",
            "ref": oscores["own_a_tight_substituted"]["avg_top2"],
        },
        "own_a|C": {
            "name": "correctedb_scores scores.cmd.A_content_only_call_end.own_a_tight",
            "ref": oscores["cmd"]["A_content_only_call_end"]["own_a_tight"],
        },
    }
    for pr in ("harmfulness", "refusal"):
        old_refs[f"{pr}|R"] = {
            "name": f"{pr} run_scores.reply_mean",
            "ref": rh[pr]["run_scores"]["reply_mean"],
        }
        old_refs[f"{pr}|C"] = {
            "name": f"{pr} run_scores.command",
            "ref": rh[pr]["run_scores"]["command"],
        }
    byl = {p["label"]: p for p in c455["probes"]}
    J = {
        "inputs_registered": L["inputs_registered"],
        "inputs_as_run": L["inputs_as_run"],
        "seal_decision": L["seal_decision"],
        "pool_export": L["pool_export"],
        "common455": c455,
        "inputs_b": L["inputs_b"],
        "final": fin,
        "final_spec": L["final_spec"],
        "asis_files": asis,
        "fibs_rows": L["fibs_record"]["rows"],
        "p2_rows": {
            "harmfulness": L["rh_record"]["probes"]["harmfulness"]["rows"],
            "refusal": L["rh_record"]["probes"]["refusal"]["rows"],
            "highstakes": L["hs_record"]["probes"]["highstakes"]["rows"],
        },
        "own_saved": {
            "tight": {"avg": L["owna_record"]["saved"]["tight"]["avg"]},
            "cmd": {"cmdA": {"rows|own_a_tight": L["owna_record"]["saved"]["cmd"]["cmdA"]["rows|own_a_tight"]}},
        },
        "owna_records": [
            {"arms": L["owna_pilot"]["arms"]},
            {"arms": L["owna_rest"]["arms"]},
        ],
        "t196_cells": L["t196"]["cells"],
        "plus311_cells": L["plus311"]["cells"],
        "r36": L["r36"],
        "gr_record": {k: L["gr_record"].get(k) for k in ("pass", "script_sha256")},
        "old_ref_rows": {k: v["name"] for k, v in old_refs.items()},
    }
    g = {
        "J": J,
        "release_names_final": GIT["final"][1].encode() in blobs["release"],
        "famexp_index_text": blobs["famexp_index"].decode(),
        "target_meta": tgt["meta"],
        "target_root_keys": sorted(tgt),
        "target_design": tgt["design"],
        "sol_old_pairs": {
            "R": [
                [byl["Sol (a) R"]["old"]["path"], byl["Sol (a) R"]["old"]["sha256"]],
                [
                    asis["a_tight__avg_top2"]["path"],
                    asis["a_tight__avg_top2"]["sha256"],
                ],
            ],
            "C": [
                [byl["Sol (a) C"]["old"]["path"], byl["Sol (a) C"]["old"]["sha256"]],
                [asis["a_tight__cmd_A"]["path"], asis["a_tight__cmd_A"]["sha256"]],
            ],
        },
    }
    gl = global_job.remote(g)
    checked = sorted(
        {
            "probe",
            "readout",
            "units",
            "routing",
            "scheme_source",
            "scheme",
            "status",
            "K[]",
            "row_label",
            "pool_export_row",
            "arm_labels.S_old",
            "arm_labels.S_new",
            "stratum_labels.S_old.sim_minus_run1",
            "stratum_labels.S_old.run2_minus_run1",
            "stratum_labels.S_new.sim_minus_run1",
            "stratum_labels.S_new.run2_minus_run1",
            "caveats.every_output[]",
            "caveats.every_command_row",
            "caveats.every_S_new_row",
            "caveats.next_to_X_err",
            "caveats.combined_rows",
        }
    )
    if set(tgt["rows"]) != set(ROWS):
        raise SystemExit("target rows differ from the 16 registered rows")
    jobs = [
        gl["base"]
        | {
            "row_key": rk,
            "spec": gl["spec_rows"][rk],
            "pool_old": gl["spec_rows"][rk]["pool_old"],
            "pool_new": gl["spec_rows"][rk]["pool_new"],
            "old_ref": old_refs[rk]["ref"] if rk in old_refs else None,
            "target_row": tgt["rows"][rk],
            "checked_strings": checked,
        }
        for rk in ROWS
    ]
    res = list(row_job.map(jobs, order_outputs=True))
    tot = dict.fromkeys(
        (
            "numeric_compared",
            "numeric_match",
            "bool_compared",
            "bool_match",
            "none_compared",
            "none_match",
            "str_checked",
            "str_checked_match",
            "str_semantic_checked",
            "str_semantic_match",
            "str_descriptive_type_checked",
            "n_mismatch",
        ),
        0,
    )
    tot |= {"not_produced": 0, "only_mine": 0, "structure_mismatch": 0}
    rows_out = {}
    for r in res:
        c = r["comparison"]
        for k in tot:
            tot[k] += len(c[k]) if isinstance(c.get(k), list) else c.get(k, 0)
        ok = c["n_mismatch"] == 0 and not c["structure_mismatch"] and not c["not_produced"] and not c["only_mine"]
        rows_out[r["row"]] = {
            "verdict": "MATCH" if ok else "NOT MATCH",
            "target_pass": tgt["rows"][r["row"]].get("pass"),
            "my_pass": r["my_pass"],
            "my_status": r["my_status"],
            "gates": r["gates"],
            "info": r["info"],
            "pool_sources": r["pool_sources"],
            "target_findings": r["target_findings"],
            "comparison": c,
        }
    grep = gl["report"]
    top_pass_ok = tgt.get("pass") is all(v["my_pass"] for v in rows_out.values())
    validation = {
        "target_top_level_pass": tgt.get("pass"),
        "target_rows_pass": sum(v["target_pass"] is True for v in rows_out.values()),
        "my_rows_pass": sum(v["my_pass"] is True for v in rows_out.values()),
        "my_failed_gates_by_row": {k: sorted(g for g, ok in v["gates"].items() if not ok) for k, v in rows_out.items()},
    }
    rows_ok = len(rows_out) == 16 and all(v["verdict"] == "MATCH" for v in rows_out.values())
    verdict = "MATCH" if (rows_ok and all(grep["ok"].values()) and top_pass_ok) else "NOT MATCH"
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {
        "label": "PRELIMINARY: independent recheck of 09's 507 registered unit FPR sweep; verdicts, counts and key paths only",
        "meta": {
            "git": head,
            "stamp_PT": stamp,
            "script_sha256": _sha(Path(__file__).read_bytes()),
            "app": APP,
            "target": {
                "commit": GIT["target"][0],
                "path": GIT["target"][1],
                "sha256": GIT["target"][2],
            },
            "tolerance": {
                "floats_abs": TOL_FLOAT,
                "ints_bools_counts": "exact",
                "strings": "exact where the registered config or RULE gives the text; other strings counted as descriptive",
            },
        },
        "verdict": verdict,
        "verdict_scope": "agreement of every target field by key with the recomputation, plus input pins, membership and design; "
        "gate status is reported separately under validation",
        "top_level_pass_matches": top_pass_ok,
        "validation": validation,
        "totals": tot
        | {
            "rows": len(rows_out),
            "rows_match": sum(v["verdict"] == "MATCH" for v in rows_out.values()),
        },
        "input_pins": {
            "git_and_local": pins,
            "checks": grep["checks"],
            "volume_counts": grep["counts"],
            "sol_old_file_pairs": grep["sol_old_file_pairs"],
        },
        "pool_membership": grep["membership"],
        "global_ok": grep["ok"],
        "design_comparison": grep["design_comparison"],
        "rows": rows_out,
    }
    blob = json.dumps(out, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)).encode()
    dest = HERE / "results" / f"recheck_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    with dest.open("xb") as fh:
        fh.write(blob)
    print(
        json.dumps(
            {
                "record": dest.name,
                "sha256": _sha(blob),
                "verdict": verdict,
                "rows_match": out["totals"]["rows_match"],
                "n_mismatch": tot["n_mismatch"],
                "global_ok": grep["ok"],
                "top_level_pass_matches": top_pass_ok,
            }
        ),
        flush=True,
    )
