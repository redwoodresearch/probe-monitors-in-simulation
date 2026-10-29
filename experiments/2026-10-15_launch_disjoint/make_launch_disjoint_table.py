"""Launch-disjoint ELIGIBILITY FILE v5 (the deciding version) for the on-policy probes.

Rule: 09's RULE @ 1ffffca72 (last section), amendment 1 @ 24fd07866, clarifications @ 47565ecc2 / bf2ab0ade /
4e0c17de7, and 09's v3 rulings relayed by 2e:
- A cell is eligible for probe p iff its run1 AND run2 are both launch-disjoint from p's training launches.
  run3 is never substituted.
- A set's row is withheld if ANY contributing H_A or tau_real-pool member (H_B; B u C on S_old) shares a training
  launch, or has unknown provenance (fail closed). Pools are whole: S4 is scoped out of the attack cohort only.
- A set counts only if it has structurally available scoring coverage frozen in this version. own_frac on S_newmains:
  "S_newmains withheld: extension coverage of real arms not pinned in the frozen build".
- Available iff >= 30 counted eligible mains; global choice own_all_L48 -> own_frac100_L48 -> none; no fallback.
  own_frac's choice is provisional until famexp's by-key read of spec 5e2409a7 confirms its training launches.

A launch id is a run's launch directory (`launch_id` in the upload manifests; first path component of a key). Cells and
mains come from each manifest file's `task` field and the committed cell lists, never from path parsing.

Reads ONLY committed fit / inputs records, upload manifests, pool-membership and cohort key lists, and cell lists, from
git by sha:path. No score value is read. Writes results/launch_disjoint_eligibility_v5_<stamp PT>.json (create-only).

Usage (repo root, script committed first):  python3 -I experiments/2026-10-15_launch_disjoint/make_launch_disjoint_table.py
"""

import collections
import datetime
import hashlib
import json
import re
import subprocess
import zoneinfo
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAM = "experiments/2026-10-14_p2_family_expansion/results"
CHAIN_SHA, CHAIN_CFG = "0abe8f1af", "experiments/2026-10-14_q3_overnight_chain/config"
RULE = ("24fd07866", "experiments/2026-10-14_priority1_spec_09/RULE.md")
STAGE_A_PIN = ("214f74de4", "experiments/2026-10-15_fpr_sweep_507_prep/run_common455.py")
SRC = {  # name: (commit, path, pinned sha256 or None)
    "own_all_inputs": ("924ccd43b", f"{FAM}/own_all_inputs_2026-10-15.json", "26f9d8d1fe08dd2ef430a1bef8af50f336da47234fe364e1a94879bcb54491d1"),
    "own_all_fit": ("66800141f", f"{FAM}/own_all_fit_2026-10-15_173309.json", None),
    "own_frac_fit": ("a7bf423e5", f"{FAM}/fit_2026-10-14_175636.json", None),
    "own_frac_heldout": ("39ec0db8f", f"{FAM}/heldout_overlap_own_frac_2026-10-15.json", None),
    "own_frac_pins": ("924ccd43b", f"{FAM}/simh_ownfam_2026-10-15_114221.json", None),
    "nm360_routing": ("924ccd43b", f"{FAM}/nm360_routing_2026-10-14_233508.json", None),
    "inputs_final": ("32abe4120", "experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FINAL_2026-10-15_203213.json", None),
    "final_record": ("3ce6ea55e", "experiments/2026-10-14_xerr_scaleup_09/results/xerr_scaleup_final_2026-10-15_013255.json", "7dceb8bb25752aa3619cfde64add702fd4f7d13a1309a56dbc8880a6b27af9f5"),
    "membership710": ("32abe4120", "experiments/2026-10-15_fpr_sweep_507_prep/config/membership710.json", "6516d15fab77d7f344497e72db10b60c5f709b35c3a1859a2c9e016d2fc6e4ad"),
    "nm360_ext_list": ("ec858af33", "experiments/2026-10-15_e2e_sim_pipeline/results/check_alltasks_nm_inputs_2026-10-15_163558.json", None),
    "t196_upload": (CHAIN_SHA, f"{CHAIN_CFG}/manifest_q4b_eval_upload_v1.json", None),
    "batch_c_upload": (CHAIN_SHA, f"{CHAIN_CFG}/manifest_q4f_c_eval_upload_v1.json", None),
    "t196_cells": ("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb"),
    "plus311_cells": ("0caadbfb5", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json", "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b"),
    "nm360_cells": ("ef171c813", "experiments/2026-10-14_q3_new_mains_mode/config/new_mains_cells_v1.json", "98bfb97e26e7ec580b08643a3a38a5e66a0f6b63333f4013aa57559383402e8e"),
}
SET_MANIFESTS = {"S_new": r"^manifest_su311_q[1-4]r?_(main|run2|sim)(_s[0-9]+)?\.json$", "S_newmains": r"^manifest_nm360_s[0-9a-z]+_(main|run2|sim)(_s[0-9]+)?\.json$"}
NM_SCOPE = {"S1", "S2", "S3", "S5"}
SUPERSEDES = ("supersedes v1 @ 9268c66b4 and v2 @ 2c58702d0; reasons: v2 added run2 / H_B per-key lists from committed records "
              "(v1's own_frac 0 was fail-closed on missing inputs, not a determination); v3 applies Codex round-2 fixes (whole "
              "pools, registered H_A coverage, verbatim readouts, sim arms, independent scored column) and 09's rulings "
              "(S_newmains withheld for own_frac; choice by availability, provisional). v4 supersedes v3 @ 0186e44af: metadata-only: Q2 sim rerun "
              "provenance added (manifest_su311_q2r_sim.json). v5 supersedes v4 @ 4efea7d83: 09's corrected criterion (A), applied "
              "uniformly on v4's counting basis (launch-disjoint scored mains, cohort or not); the Q2 sim-provenance fix is kept. "
              "No version contains or used any score.")
CRITERION = ("eligible = all touched real arms and pools launch-disjoint, and the scored main in no training fold of the scoring "
             "models; fold −1 scoring allowed")
PROXY_ERROR = ("v3/v4 counted a main as scored only if it had a Sol #36 (r36) fold. That proxy was wrong: own_frac scores a main "
               "without an r36 fold as fold −1 (five-model mean, 25h §1 extension), and Stage A does so on S_new (pooled455v2_C "
               "@ 106262746: fold_minus1_rule 'weighted 21b on each pool's own fold -1 units', fold_minus1_dropped {}). What "
               "matters is that the scored main is in no training fold of the models that score it; every +311 main satisfies "
               "this (held-out record @ 39ec0db8f: plus311_overlap 0 for all 30 fits).")
STAGE_A_REC = ("106262746", "experiments/2026-10-15_fpr_sweep_507_prep/results/pooled455v2_C_2026-10-15_221159.json")
FOLD_M1_CELLS = ["t271_slack_clone", "t291_tiktok", "t388_media_processing", "t505_tiktok"]
DECIDING = "THIS IS THE DECIDING VERSION. No score (restricted or otherwise) was read before it, by this generator or for it."
NM_WITHHELD = "S_newmains withheld: extension coverage of real arms not pinned in the frozen build"
OWN_FRAC_LABEL = "provisional: training launches inferred by refit match"
RULE_NEEDLES = ["Points: per-action 1e-2 leads; 1e-3 and 5e-2 beside it; both readouts (reply, command)",
                "**Δ_full = TPR_sim(τ_sim) − TPR_real(τ_real)**", "τ_sim = T on H_sim's units with g_sim; τ_real = T on H_B's units with g_B",
                "**Δ_rr = TPR_run2(τ_A) − TPR_run1(τ_B)**", "1e-2 leading, draws, ≥ 9,500 jointly complete draws, the 32k cut",
                "registered validity test (moments, ties, readability, < 9,500 jointly complete draws)"]


def git_bytes(sha, path):
    return subprocess.run(["git", "show", f"{sha}:{path}"], check=True, capture_output=True).stdout


def load(name):
    sha, path, want = SRC[name]
    b = git_bytes(sha, path)
    h = hashlib.sha256(b).hexdigest()
    if want and h != want:
        raise SystemExit(f"{name}: sha256 {h} != pinned {want}")
    return json.loads(b), {"commit": sha, "path": path, "sha256": h}


def done(f):
    return f.get("state") == "done" and str(f.get("policy_called")) == "True"


def main():
    src, d = {}, {}
    for n in SRC:
        d[n], src[n] = load(n)
    ins, fit = d["own_all_inputs"], d["own_all_fit"]
    if fit["inputs"]["sha256"] != src["own_all_inputs"]["sha256"] or not (fit["fit"]["gate"]["refit_match_pass"] and fit["fit"]["gate"]["all_certified"]):
        raise SystemExit("own_all fit record does not name these inputs, or its gate did not pass")
    stage_a = d["inputs_final"]["pools"]["stage_a"]
    if any(stage_a[s]["H_B"]["sha256"] != src["final_record"]["sha256"] for s in ("S_old", "S_new")) or stage_a["S_newmains"]["H_B"]["sha256"] != src["membership710"]["sha256"]:
        raise SystemExit("INPUTS_FINAL pool refs != pinned membership records")
    if "own_frac100_L48" not in d["inputs_final"]["nm360_fold_extension"]["gates"]:
        raise SystemExit("INPUTS_FINAL nm360_fold_extension has no own_frac100 gate")

    cell_main, set_cells = {}, {}
    for n, s in (("t196_cells", "S_old"), ("plus311_cells", "S_new"), ("nm360_cells", "S_newmains")):
        cells = d[n]["cells"]
        if len({c["main"] for c in cells}) != len(cells):
            raise SystemExit(f"{n}: not one cell per main")
        cell_main.update({c["key"]: c["main"] for c in cells})
        set_cells[s] = [c["key"] for c in cells]

    # ---- training launches and fold basis (deployed fit inputs, by key)
    train = collections.defaultdict(set)
    for e in ins["training_epochs"]:
        train[e["pool"]].add(e["key"].split("/", 1)[0])
    fold_basis = {m: v[1] for m, v in ins["folds"]["by_main"].items()}
    universe = {p: set() for p in ("T196", "S_new", "S_newmains")}
    tr_cells = collections.defaultdict(lambda: collections.defaultdict(set))
    for e in ins["training_epochs"]:
        tr_cells[e["pool"]][e["world"]].add(e["cell"])
    for p in universe:
        universe[p] = {cell_main[c] for c in tr_cells[p]["run1"] & tr_cells[p]["honest_a"]}
    probes = {
        "own_all_L48": {"training_launch_ids": sorted(set().union(*train.values())), "training_worlds": ["run1 (attack, y=1)", "honest_a (y=0)"],
                        "basis": "by key: own_all_inputs training_epochs (the inputs named by sha256 in the deployed fit record)",
                        "fit_record": src["own_all_fit"] | {"fit_record_sha256": fit["fit"]["fit_record_sha256"]},
                        "checkpoints": [{k: f[k] for k in ("variant", "readout", "fold", "weights_sha256", "preproc_sha256", "training_manifest_sha256")}
                                        for f in fit["fit"]["fits"] if f["variant"] == "own_all_L48"],
                        "in_fold": {"violations": len(ins["gate"]["in_fold_violations"]), "gate_pass": ins["gate"]["pass"], "in_job_stop": fit["fit"]["gate"]["in_job_stop"]}},
        "own_frac100_L48": {"training_launch_ids": sorted(train["T196"]), "training_worlds": ["run1 (attack, y=1)", "honest_a (y=0)"],
                            "basis": "inferred: T196 run1 + honest_a rows of spec 5e2409a7 (the refit-match control @ 66800141f reproduced its weights exactly)",
                            "label": OWN_FRAC_LABEL,
                            "fit_record": src["own_frac_fit"] | {"fits": d["own_frac_fit"]["result"]["fits"], "fit_manifest_sha256": d["own_frac_pins"]["pins"]["pass1"]["fit"],
                                                                 "fit_manifest_source": src["own_frac_pins"] | {"field": "pins.pass1.fit"}},
                            "checkpoints": "MISSING in git: per-fold weight sha256s are on the Volume only; the row is provisional until famexp's spec read",
                            "in_fold": {"plus311_in_fold_violations": sum(len(v.get("plus311_in_fold_violations", [])) for v in d["own_frac_heldout"]["fits"].values()),
                                        "nm360_overlap": sum(len(v.get("nm360_overlap", [])) for v in d["own_frac_heldout"]["fits"].values()),
                                        "fits_checked": len(d["own_frac_heldout"]["fits"]), "source": src["own_frac_heldout"]}},
    }

    # ---- per-run provenance: (set, world) -> runs {key, launch, cell, main, source}
    runs = collections.defaultdict(list)
    for w, entries in d["t196_upload"]["worlds"].items():
        if w in ("run1", "run2", "honest_a", "honest_b", "sim"):
            for e in entries:
                runs[("S_old", w)].append({"key": e["path"], "launch": e["path"].split("/", 1)[0], "cell": e["task"], "source": "t196_upload.worlds"})
    for f in d["batch_c_upload"]["files"]:
        runs[("S_old", "honest_c")].append({"key": f["path"], "launch": f["launch"], "cell": f["task"], "source": "batch_c_upload.files"})
    chain_files = subprocess.run(["git", "ls-tree", "--name-only", f"{CHAIN_SHA}:{CHAIN_CFG}"], check=True, capture_output=True, text=True).stdout.split()
    used_manifests, sha_index = {}, {}
    for s, pat in SET_MANIFESTS.items():
        seen = {}
        for fn in sorted(x for x in chain_files if re.match(pat, x)):
            b = git_bytes(CHAIN_SHA, f"{CHAIN_CFG}/{fn}")
            used_manifests[fn] = hashlib.sha256(b).hexdigest()
            for f in json.loads(b)["files"]:
                sha_index[f["sha256"]] = f["path"]
                if f["launch_id"] != f["path"].split("/", 1)[0]:
                    raise SystemExit(f"{fn}: launch_id disagrees with path for {f['path']}")
                meta = (f["task"], f["world"], f["launch_id"], f["sha256"], f.get("state"), str(f.get("policy_called")))
                if f["path"] in seen:
                    if seen[f["path"]] != meta:
                        raise SystemExit(f"{fn}: duplicate path with different metadata: {f['path']}")
                    continue
                seen[f["path"]] = meta
                if done(f):
                    runs[(s, f["world"])].append({"key": f["path"], "launch": f["launch_id"], "cell": f["task"], "source": fn})
    for e in ins["training_epochs"]:  # the 25f SELECTED honest_a attempts (references/ copies), by key from the deployed inputs
        if str(e.get("ha25f_selected")) == "True":
            runs[(e["pool"], "honest_a")].append({"key": e["key"], "launch": e["key"].split("/", 1)[0], "cell": e["cell"], "source": "own_all_inputs training_epochs (ha25f_selected)"})
    for (s, w), rs in runs.items():
        for r in rs:
            if r["cell"] not in cell_main:
                raise SystemExit(f"{s} {w}: task {r['cell']} not in the committed cell lists")
            r["main"] = cell_main[r["cell"]]

    batch = d["membership710"]["batch_of_cell"]

    def in_attack_scope(s, cell):  # S4 leaves the ATTACK cohort only (INPUTS_FINAL stage_a rule); pools stay whole
        return s != "S_newmains" or batch.get(cell) in NM_SCOPE

    # ---- registered calibration membership (world|cell keys), whole pools
    xr = d["final_record"]["result"]["calibration_records"]["primary"]["sol_a|R|0.01|primary|BuC|B"]
    xs = xr["schemes"][xr["primary"]]
    m710 = d["membership710"]
    # an H_sim cell needs its own honest_a reference, so registered H_A lies within the cells that have any run in the committed launches
    launched_new = sorted({r["cell"] for (s_, w_), rs in runs.items() if s_ == "S_new" for r in rs})
    never_launched_new = sorted(set(set_cells["S_new"]) - set(launched_new))
    excluded_new = {k.split("|", 1)[1] for k in stage_a["S_new"]["excluded_cells"]}
    pools = {
        "S_old": {"tau_real (B u C)": list(xs["pool_keys_old"]), "H_A (honest_c runs of H_B)": [k for k in xs["pool_keys_old"] if k.startswith("honest_c|")]},
        "S_new": {"tau_real (H_B)": list(xs["pool_keys_new"]),
                  "H_A (honest_a on every launched S_new cell: a superset of the registered H_sim cells)": [f"honest_a|{c}" for c in launched_new]},
        "S_newmains": {"tau_real (H_B)": m710["pool_honest_b"] + m710["pool_honest_b_4r_sol_only"],
                       "H_A (held_out_honest_a)": m710["held_out_honest_a"] + m710["held_out_honest_a_4r_sol_only"]},
    }

    def pool_check(s, members, tids, is_ha):
        idx = collections.defaultdict(list)
        for w in ("honest_a", "honest_b", "honest_c"):
            for r in runs.get((s, w), []):
                idx[(w, r["cell"])].append(r)
        unknown, bad, ids = [], [], set()
        for m in members:
            w, cell = m.split("|", 1)
            rs = idx.get((w, cell), [])
            if not rs:
                unknown.append(m)
            for r in rs:
                ids.add(r["launch"])
                if r["launch"] in tids:
                    bad.append(r["key"])
        exempt = {x.split("_")[0] for x in excluded_new} if (s == "S_new" and is_ha) else set()  # registered H_A drops H_sim-excluded cells
        unknown_short = sorted(u for u in unknown if u.split("|", 1)[1].split("_")[0] not in exempt)
        return {"members": len(members), "unknown": sorted(unknown), "unknown_not_in_registered_exclusions": unknown_short,
                "sharing_training_launch": len(bad), "launch_ids": sorted(ids), "pass": bool(members) and not unknown_short and not bad}

    # cohort source = Stage A's pinned FINAL blob (run_peraction311.py: coh455 = fin["cohort"]["cohort_cells"]); assert the pin first
    pin_src = git_bytes(STAGE_A_PIN[0], STAGE_A_PIN[1]).decode()
    pin = re.search(r'FINAL = \("([0-9a-f]+)", "([^"]+)",\s*"([0-9a-f]{64})"\)', pin_src)
    if not pin or (pin.group(1), pin.group(2), pin.group(3)) != (SRC["final_record"][0], SRC["final_record"][1], src["final_record"]["sha256"]):
        raise SystemExit("STOP: cohort source sha256 differs from Stage A's pin")
    cohort_source = src["final_record"] | {"field": "result.cohort.cohort_cells", "stage_a_pin": f"{STAGE_A_PIN[0]}:{STAGE_A_PIN[1]} FINAL", "pin_equal": True,
                                           "reader": "run_peraction311.py @ 214f74de4: coh455 = sorted(fin['cohort']['cohort_cells']); S_old = T196 cells, S_new = the rest"}
    cohort = d["final_record"]["result"]["cohort"]["cohort_cells"]
    coh = {"S_old": sorted(c for c in cohort if c in set(set_cells["S_old"])), "S_new": sorted(c for c in cohort if c in set(set_cells["S_new"])),
           "S_newmains": sorted(m710["cohort_without_S4"])}
    if (len(coh["S_old"]), len(coh["S_new"])) != (183, 272) or len(coh["S_old"]) + len(coh["S_new"]) != len(cohort):
        raise SystemExit("pooled455 cohort is not 183 + 272")

    # ---- check 1 (Stage A's fold -1 columns -> cells, by key) and check 2 (out of training, by key)
    srec_b = git_bytes(*STAGE_A_REC)
    srec = json.loads(srec_b)
    if not srec["meta"]["git"].startswith(STAGE_A_PIN[0]):
        raise SystemExit("Stage A record not from the pinned code")
    t196_main_of = {c["key"]: c["main"] for c in d["t196_cells"]["cells"]}
    reg_new = sorted(set(d["final_record"]["result"]["design"]["primary_universe_mains"]) - set(t196_main_of.values()))
    units = sorted(set(t196_main_of.values()) | set(reg_new))  # draws_main sorts its units (run_fprsweep_507.py:103 @ 214f74de4)
    p311_cell_of = {c["main"]: c["key"] for c in d["plus311_cells"]["cells"]}
    plus311_mains = set(p311_cell_of)
    srow = srec["rows"]["own_frac100_L48 C"]["S_new"]
    cols = srow["fold_minus1_draws"]["attack_fold_minus1_cols"]
    mapped = {k: sorted(p311_cell_of.get(units[i], "NOT +311") for i in v) for k, v in cols.items()}
    if any(v != FOLD_M1_CELLS for v in mapped.values()) or srow["fold_minus1_dropped"] != {}:
        raise SystemExit("STOP: check 1 failed")
    held = d["own_frac_heldout"]
    if held["counts"]["plus311_mains"] != 311 or not all(v["plus311_overlap"] == 0 and v["plus311_in_fold_violations"] == [] for v in held["fits"].values()):
        raise SystemExit("STOP: check 2 failed (held-out record)")
    own_frac_train_mains = {t196_main_of[e["cell"]] for e in ins["training_epochs"] if e["pool"] == "T196"}
    fm1_mains = {c: d["plus311_cells"]["cells"][[x["key"] for x in d["plus311_cells"]["cells"]].index(c)]["main"] for c in FOLD_M1_CELLS}
    if any(m in set(t196_main_of.values()) or m in own_frac_train_mains for m in fm1_mains.values()):
        raise SystemExit("STOP: check 2 failed (direct)")
    checks = {
        "check_1_fold_minus1_columns": {"record": {"commit": STAGE_A_REC[0], "path": STAGE_A_REC[1], "sha256": hashlib.sha256(srec_b).hexdigest(), "meta_git": srec["meta"]["git"]},
                                        "field": 'rows["own_frac100_L48 C"]["S_new"]["fold_minus1_draws"]["attack_fold_minus1_cols"]', "columns": cols,
                                        "mapping": "column = index into sorted(T196 mains | FINAL design.primary_universe_mains), 506 units", "cells_by_pool": mapped,
                                        "fold_minus1_dropped": srow["fold_minus1_dropped"], "fold_minus1_rule": srow["fold_minus1_rule"], "pass": True},
        "check_2_out_of_training": {"heldout_record": src["own_frac_heldout"], "fits": len(held["fits"]), "plus311_overlap_all_zero": True, "plus311_in_fold_violations_all_empty": True,
                                    "universe": "all 311 +311 mains", "the_4_mains": fm1_mains,
                                    "direct": "none of the 4 mains is a T196 main or in own_frac's T196 training rows (own_all_inputs T196 part)",
                                    "run_launches": {c: sorted({(w, r["launch"]) for w in ("run1", "run2") for r in runs[("S_new", w)] if r["cell"] == c}) for c in FOLD_M1_CELLS},
                                    "pass": True},
    }

    # ---- eligibility
    sets_out, avail, prov_out = {}, {}, {}
    for probe, p in probes.items():
        tids = set(p["training_launch_ids"])
        total, per_set = 0, {}
        for s in ("S_old", "S_new", "S_newmains"):
            pool = {"S_old": "T196", "S_new": "S_new", "S_newmains": "S_newmains"}[s]
            by_cell = {w: collections.defaultdict(list) for w in ("run1", "run2", "sim")}
            for w in by_cell:
                for r in runs.get((s, w), []):
                    if in_attack_scope(s, r["cell"]):
                        by_cell[w][r["cell"]].append(r)
            both = sorted(set(by_cell["run1"]) & set(by_cell["run2"]))
            elig = sorted(c for c in both if all(r["launch"] not in tids for w in ("run1", "run2") for r in by_cell[w][c]))
            # scored (structural coverage), independent of launch eligibility
            run1_mains = {cell_main[c] for c in by_cell["run1"]}
            if probe == "own_frac100_L48":
                scored_mains = {m for m in run1_mains if fold_basis.get(m) == "r36"}
                if s == "S_newmains":
                    scored_note = f"{NM_WITHHELD} (r36-mapped mains: {len(scored_mains)}; the fold -1 extension's frozen list @ ec858af33 holds H_sim files only)"
                    structural = False
                elif s == "S_new":  # corrected criterion (A): +311 mains are in no training fold of any own_frac model (check 2)
                    r36 = len(scored_mains)
                    scored_mains = {m for m in run1_mains if m in plus311_mains}
                    scored_note = (f"scored: {len(scored_mains)} run1 mains in no training fold of the scoring models "
                                   f"({r36} r36-mapped, scored by their fold model; {len(scored_mains) - r36} fold −1, five-model mean)")
                    structural = len(scored_mains) > 0
                else:
                    scored_note, structural = f"scored: {len(scored_mains)} r36-mapped run1 mains", len(scored_mains) > 0
            else:
                scored_mains = run1_mains & universe[pool]
                scored_note, structural = f"scored: {len(scored_mains)} own_all universe run1 mains", len(scored_mains) > 0
            elig_scored = sorted(c for c in elig if cell_main[c] in scored_mains)
            pc = {name: pool_check(s, mem, tids, name.startswith("H_A")) for name, mem in pools[s].items()}
            withheld = (not structural) or not all(v["pass"] for v in pc.values())
            n = len({cell_main[c] for c in elig_scored}) if not withheld else 0
            coh_s = coh[s]
            coh_missing = {c: ("no run1 and run2 pair by key" if c not in both else "run1/run2 share a training launch" if c not in elig else "no Sol #36 fold for its main (fold -1 for own_frac; Stage A's fold -1 rule decides scoring); not counted here")
                           for c in coh_s if c not in elig_scored}
            per_set[s] = {
                "attack_scope": "S1-S3 + S5 (batch_of_cell @ membership710); pools whole" if s == "S_newmains" else "all cells of the set",
                "arms": {w: {"runs": sum(len(v) for v in by_cell[w].values()), "cells": len(by_cell[w]), "launch_ids": sorted({r["launch"] for v in by_cell[w].values() for r in v}),
                             "cells_sharing_training_launch": sum(1 for v in by_cell[w].values() if any(r["launch"] in tids for r in v))} for w in by_cell},
                "sim_arm": {"launch_ids": sorted({r["launch"] for v in by_cell["sim"].values() for r in v}), "cells": len(by_cell["sim"]),
                            "note": "the registered sim arm (not a launch-disjointness criterion)"},
                "cells_with_run1_and_run2": len(both), "eligible_cells_launch_disjoint": len(elig),
                "scored": scored_note, "structurally_scored": structural, "eligible_and_scored_cells": elig_scored,
                "calibration_pools": pc, "withheld": withheld, "eligible_mains_counted": n,
                "cohort_check": {"cohort": "pooled455 (cohort_source)" if s != "S_newmains" else "membership710 cohort_without_S4",
                                 "cohort_cells": len(coh_s), "cohort_in_eligible_and_scored": len(coh_s) - len(coh_missing),
                                 "cohort_cells_not_covered": coh_missing,
                                 "statement": (f"own row covers fewer cells than the cohort: {len(coh_s) - len(coh_missing)} of {len(coh_s)}" if coh_missing else "cohort subset of eligible: yes")},
            }
            if probe == "own_frac100_L48":
                per_set[s]["label"] = OWN_FRAC_LABEL
                if s == "S_newmains":
                    per_set[s]["label"] = NM_WITHHELD
                    per_set[s]["panel_note"] = "PANEL's headline cohort is pooled455 = S_old (183) + S_new (272); S_newmains is not in PANEL's headline row"
            total += n
        sets_out[probe] = per_set
        avail[probe] = {"eligible_mains_total": total, "available": total >= 30}
    choice = next((p for p in ("own_all_L48", "own_frac100_L48") if avail[p]["available"]), None)
    for (s, w), rs in sorted(runs.items()):
        prov_out[f"{s}|{w}"] = [{k: r[k] for k in ("key", "launch", "cell", "main", "source")} for r in rs]
    ha25f = {cell: {"eval_sha256": h, "in_upload_manifests": sha_index.get(h, "no"),
                    "by_key_in_inputs": [r["key"] for r in runs[("S_new", "honest_a")] if r["cell"] == cell and "ha25f" in r["source"]]}
             for cell, h in ins["pins"]["ha25f_selected"].items()}

    rule_text = git_bytes(*RULE).decode()
    verbatim = []
    for needle in RULE_NEEDLES:
        line = next((ln for ln in rule_text.splitlines() if needle in ln), None)
        if line is None:
            raise SystemExit(f"RULE @ {RULE[0]}: registered text not found: {needle}")
        verbatim.append(line.strip())
    script_rel = "experiments/2026-10-15_launch_disjoint/make_launch_disjoint_table.py"
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    committed_blob = subprocess.run(["git", "rev-parse", f"HEAD:{script_rel}"], capture_output=True, text=True).stdout.strip()
    disk_blob = subprocess.run(["git", "hash-object", script_rel], capture_output=True, text=True, check=True).stdout.strip()
    if committed_blob != disk_blob:
        raise SystemExit("commit the script before running it (generator must equal HEAD's copy)")
    stamp = datetime.datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {
        "label": "on-policy launch-disjoint eligibility file v5; provenance only; no score read",
        "deciding": DECIDING, "supersedes": SUPERSEDES,
        "stamp (PT)": stamp, "generator": {"path": script_rel, "commit": head, "blob_sha1": disk_blob},
        "rule": {"base": "@ 1ffffca72", "amendment_1": "@ 24fd07866", "clarifications": ["@ 47565ecc2", "@ bf2ab0ade", "@ 4e0c17de7"],
                 "v3_rulings": "09 via 2e: S_newmains withheld for own_frac; choice by availability, provisional; no fallback"},
        "sources": src | {"chain_manifests": {f"{CHAIN_SHA}:{CHAIN_CFG}/{k}": v for k, v in sorted(used_manifests.items())}},
        "registered": {"readouts_and_rates_verbatim": {f"{RULE[0]}:{RULE[1]}": verbatim},
                       "pair_order": "Δ_rr = TPR_run2(τ_A) − TPR_run1(τ_B) (registered order)", "delta_full_real_arms": ["run1", "run2"],
                       "pooling": "equal-main weighting within a set; pooled rows weight mains equally over the counted sets",
                       "tau_real_pool": {"S_old": "B u C", "S_new": "H_B", "S_newmains": "H_B (membership710 pool_honest_b + 4r list)"},
                       "no_fallback": "if the chosen probe's row later fails a registered validity test it is reported unavailable; no fallback"},
        "criterion": CRITERION, "proxy_error_v3_v4": PROXY_ERROR, "checks": checks,
        "cohort_source": cohort_source, "s_new_cells_never_launched": {"cells": never_launched_new, "basis": "in the +311 cell list, no run of any world in the committed S_new manifests or inputs"}, "probes": probes, "sets": sets_out, "availability": avail,
        "global_choice": choice or "none",
        "global_choice_label": (OWN_FRAC_LABEL + "; confirmed only by famexp's by-key read of spec 5e2409a7; if that read shows otherwise the row stops (no fallback)")
                               if choice == "own_frac100_L48" else "",
        "ha25f_selected_honest_a": ha25f,
        "per_key_provenance": prov_out,
    }
    path = HERE / "results" / f"launch_disjoint_eligibility_v5_{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    with path.open("x") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(path)
    for probe, ss in sets_out.items():
        for s, v in ss.items():
            print(probe, s, "| run1&run2", v["cells_with_run1_and_run2"], "| disjoint", v["eligible_cells_launch_disjoint"], "|", v["scored"][:90],
                  "| elig&scored", len(v["eligible_and_scored_cells"]), "| withheld", v["withheld"],
                  {k: (x["pass"], x["members"], len(x["unknown"]), len(x["unknown_not_in_registered_exclusions"]), x["sharing_training_launch"]) for k, x in v["calibration_pools"].items()},
                  "| counted", v["eligible_mains_counted"], "| cohort", v["cohort_check"]["cohort_in_eligible_and_scored"], "/", v["cohort_check"]["cohort_cells"])
    print(json.dumps(avail), "choice:", choice or "none")


if __name__ == "__main__":
    main()
