"""Launch-disjoint ELIGIBILITY FILE for own_all_LOO_L48 (09 inclusion amendment 2 @ 8b9ca793f, item 4; POST HOC).

Same rule and counting basis as eligibility v5 (make_launch_disjoint_table.py @ 1706761b5 on aswin/launch-disjoint-2e; 09's RULE
@ 1ffffca72, amendment 1 @ 24fd07866, clarifications, v3 rulings, corrected criterion (A)), with the LOO probe's per-set scoring model:
- set o (0 S_old, 1 S_new, 2 S_newmains) is scored ONLY by model o, so set o's eligibility uses model o's training launches.
- A cell is eligible iff its run1 AND run2 are both launch-disjoint from model o's training launches; run3 is never substituted.
- A set's row is withheld if ANY contributing H_A or tau_real-pool member shares a training launch of model o, or has unknown
  provenance (fail closed). Pools are whole: S4 is scoped out of the attack cohort only.
- Amendment 2 item 4: an assert by key that no training launch of model o appears in set o's arms or pools, and that no main of
  set o is in model o's training mains. Any violation is a STOP, never a silent withhold.
- Available iff >= 30 counted eligible mains. Choice by availability (amendment 2 item 5): own_all_LOO_L48, then own_all_L48, then
  own_frac100_L48 (those two as in v5 @ 90c0c3148, unchanged here), then none; never by value; no fallback after a validity failure.

Reads ONLY committed fit / score / verify / inputs records, upload manifests, pool-membership and cohort key lists, and cell lists,
from git by sha:path. No score value is read (the score record holds counts and shas only). Writes
results/own_all_loo_eligibility_<stamp PT>.json (create-only).

Usage (repo root, script committed first):
  python3 -I experiments/2026-10-15_own_all_loo/make_loo_eligibility.py <commit> <loo fit record path> <loo score record path> <loo verify record path>
"""

import collections
import datetime
import hashlib
import json
import re
import subprocess
import sys
import zoneinfo
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAM = "experiments/2026-10-14_p2_family_expansion/results"
CHAIN_SHA, CHAIN_CFG = "0abe8f1af", "experiments/2026-10-14_q3_overnight_chain/config"
AMD2 = ("8b9ca793f", "experiments/2026-10-14_priority1_spec_09/RULE.md")
V5 = ("90c0c3148", "experiments/2026-10-15_launch_disjoint/results/launch_disjoint_eligibility_v5_2026-10-15_224056.json")
SRC = {  # name: (commit, path, pinned sha256 or None): eligibility v5's sources (make_launch_disjoint_table.py @ 1706761b5)
    "own_all_inputs": ("924ccd43b", f"{FAM}/own_all_inputs_2026-10-15.json", "26f9d8d1fe08dd2ef430a1bef8af50f336da47234fe364e1a94879bcb54491d1"),
    "inputs_final": ("32abe4120", "experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FINAL_2026-10-15_203213.json", None),
    "final_record": ("3ce6ea55e", "experiments/2026-10-14_xerr_scaleup_09/results/xerr_scaleup_final_2026-10-15_013255.json", "7dceb8bb25752aa3619cfde64add702fd4f7d13a1309a56dbc8880a6b27af9f5"),
    "membership710": ("32abe4120", "experiments/2026-10-15_fpr_sweep_507_prep/config/membership710.json", "6516d15fab77d7f344497e72db10b60c5f709b35c3a1859a2c9e016d2fc6e4ad"),
    "t196_upload": (CHAIN_SHA, f"{CHAIN_CFG}/manifest_q4b_eval_upload_v1.json", None),
    "batch_c_upload": (CHAIN_SHA, f"{CHAIN_CFG}/manifest_q4f_c_eval_upload_v1.json", None),
    "t196_cells": ("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb"),
    "plus311_cells": ("0caadbfb5", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json", "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b"),
    "nm360_cells": ("ef171c813", "experiments/2026-10-14_q3_new_mains_mode/config/new_mains_cells_v1.json", "98bfb97e26e7ec580b08643a3a38a5e66a0f6b63333f4013aa57559383402e8e"),
    "eligibility_v5": V5 + (None,),
    "own_all_score_main": ("924ccd43b", f"{FAM}/own_all_score_2026-10-15_174317.json", None),  # own_all_L48's registered scoring (the coverage basis)
    "own_all_score_r57": ("924ccd43b", f"{FAM}/own_all_score_2026-10-15_175400.json", None),
    "own_all_score_r58": ("924ccd43b", f"{FAM}/own_all_score_2026-10-15_184258.json", None),
    "own_all_score_r59_61": ("924ccd43b", f"{FAM}/own_all_score_2026-10-15_192204.json", None),
}
NEED_WORLDS = {"S_old": {"run1", "run2", "sim", "honest_b", "honest_c", "simh_full"}, "S_new": {"run1", "run2", "sim", "honest_a", "honest_b", "simh_311"},
               "S_newmains": {"run1", "run2", "sim", "honest_a", "honest_b", "simh_nm360"}}  # arms + pools (H_A, tau_real, H_sim) each set's row touches
SET_MANIFESTS = {"S_new": r"^manifest_su311_q[1-4]r?_(main|run2|sim)(_s[0-9]+)?\.json$", "S_newmains": r"^manifest_nm360_s[0-9a-z]+_(main|run2|sim)(_s[0-9]+)?\.json$"}
NM_SCOPE = {"S1", "S2", "S3", "S5"}
ORIGINS = ("S_old", "S_new", "S_newmains")
PROBE = "own_all_LOO_L48"
LABEL = "new probe, trained on the other task sets; POST HOC (09 inclusion amendment 2 @ 8b9ca793f); PRELIMINARY"
CRITERION = ("eligible = all touched real arms and pools launch-disjoint from the scoring model's training launches (model o for set o), "
             "and the scored main in no training fold of that model")


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


def set_of_folder(kind, fo):
    if kind in ("t196", "a25simh"):
        return "S_old"
    if (kind == "scaleup" and fo.startswith("su311_")) or (kind == "simh" and fo.startswith("simh311_")):
        return "S_new"
    if (kind == "scaleup" and fo.startswith("nm360_")) or (kind == "simh" and fo.startswith("simh_nm360_")):
        return "S_newmains"
    return None


def main():
    commit, fit_path, score_path, verify_path = sys.argv[1:5]
    SRC["loo_fit"], SRC["loo_score"], SRC["loo_verify"] = (commit, fit_path, None), (commit, score_path, None), (commit, verify_path, None)
    src, d = {}, {}
    for n in SRC:
        d[n], src[n] = load(n)
    ins, lf, ls, lv = d["own_all_inputs"], d["loo_fit"], d["loo_score"], d["loo_verify"]
    # ---- gates of amendment 2 item 4, by key from the committed records
    g = lf["fit"]["gate"]
    if lf["inputs"]["sha256"] != src["own_all_inputs"]["sha256"] or not (g["refit_match_pass"] and g["all_certified"]) or any(g["dropped_mains_set_o"].values()):
        raise SystemExit("STOP: LOO fit record does not name own_all's inputs, or its refit-match / certification / dropped-mains gate failed")
    if ls["fit_record_sha256"] != lf["fit"]["fit_record_sha256"] or lv.get("pass") is not True:
        raise SystemExit("STOP: the score record is not under this fit, or the verify record did not pass")
    # Codex r2 item 2: the verify record must name THIS score record (path and sha256) and THIS fit
    if {"path": score_path, "sha256": src["loo_score"]["sha256"]} not in lv["score_records"] or \
            not lv["fit_record"].startswith(f"Volume fit record {lf['fit']['fit_record_sha256'][:8]}") or not lv["negative_e2e"].get("failed_closed"):
        raise SystemExit("STOP: the verify record is not bound to this score record and fit, or its end-to-end negative did not fail closed")
    viol = {fo: x["in_fold_violations"] for r in ls["results"].values() for fo, x in r["folders"].items() if x["in_fold_violations"]}
    if viol:
        raise SystemExit(f"STOP: in-fold violations in {len(viol)} folders")
    lo = lf["loo"]
    tids_of = {s: set(lo["training_launches"][str(o)]) for o, s in enumerate(ORIGINS)}
    trm_of = {s: set(lo["training_lists"][str(o)]) for o, s in enumerate(ORIGINS)}
    setm_of = {s: set(lo["set_mains"][str(o)]) for o, s in enumerate(ORIGINS)}
    if any(trm_of[s] & setm_of[s] for s in ORIGINS):
        raise SystemExit("STOP: a main of set o is in model o's training mains")
    stage_a = d["inputs_final"]["pools"]["stage_a"]
    if any(stage_a[s]["H_B"]["sha256"] != src["final_record"]["sha256"] for s in ("S_old", "S_new")) or stage_a["S_newmains"]["H_B"]["sha256"] != src["membership710"]["sha256"]:
        raise SystemExit("INPUTS_FINAL pool refs != pinned membership records")

    cell_main, set_cells = {}, {}
    for n, s in (("t196_cells", "S_old"), ("plus311_cells", "S_new"), ("nm360_cells", "S_newmains")):
        cells = d[n]["cells"]
        if len({c["main"] for c in cells}) != len(cells):
            raise SystemExit(f"{n}: not one cell per main")
        cell_main.update({c["key"]: c["main"] for c in cells})
        set_cells[s] = [c["key"] for c in cells]
    if {s: {cell_main[c] for c in set_cells[s]} for s in ORIGINS} != setm_of:
        raise SystemExit("STOP: the fit's set_mains differ from the committed cell lists")

    # ---- per-run provenance (v5's code, verbatim): (set, world) -> runs {key, launch, cell, main, source}
    runs = collections.defaultdict(list)
    for w, entries in d["t196_upload"]["worlds"].items():
        if w in ("run1", "run2", "honest_a", "honest_b", "sim"):
            for e in entries:
                runs[("S_old", w)].append({"key": e["path"], "launch": e["path"].split("/", 1)[0], "cell": e["task"], "source": "t196_upload.worlds"})
    for f in d["batch_c_upload"]["files"]:
        runs[("S_old", "honest_c")].append({"key": f["path"], "launch": f["launch"], "cell": f["task"], "source": "batch_c_upload.files"})
    chain_files = subprocess.run(["git", "ls-tree", "--name-only", f"{CHAIN_SHA}:{CHAIN_CFG}"], check=True, capture_output=True, text=True).stdout.split()
    used_manifests = {}
    for s, pat in SET_MANIFESTS.items():
        seen = {}
        for fn in sorted(x for x in chain_files if re.match(pat, x)):
            b = git_bytes(CHAIN_SHA, f"{CHAIN_CFG}/{fn}")
            used_manifests[fn] = hashlib.sha256(b).hexdigest()
            for f in json.loads(b)["files"]:
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

    # ---- registered calibration membership (v5's code, verbatim)
    xr = d["final_record"]["result"]["calibration_records"]["primary"]["sol_a|R|0.01|primary|BuC|B"]
    xs = xr["schemes"][xr["primary"]]
    m710 = d["membership710"]
    launched_new = sorted({r["cell"] for (s_, w_), rs in runs.items() if s_ == "S_new" for r in rs})
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

    cohort = d["final_record"]["result"]["cohort"]["cohort_cells"]
    coh = {"S_old": sorted(c for c in cohort if c in set(set_cells["S_old"])), "S_new": sorted(c for c in cohort if c in set(set_cells["S_new"])),
           "S_newmains": sorted(m710["cohort_without_S4"])}
    if (len(coh["S_old"]), len(coh["S_new"])) != (183, 272) or len(coh["S_old"]) + len(coh["S_new"]) != len(cohort):
        raise SystemExit("pooled455 cohort is not 183 + 272")

    # ---- scoring coverage (Codex r2 item 1), counts only: every folder own_all_L48's registered scoring covered (its main run + the r57 / r58 /
    # r59-61 addenda; epoch-2 excluded) must be in the LOO score record with EQUAL rows, runs and runs-without-rows in BOTH readouts; LOO rows
    # of the three sets are all mapped (fold = origin; 0 fold -1); and each set's arms and pools (NEED_WORLDS) have mapped rows in both readouts.
    reg = {}
    for n in ("own_all_score_main", "own_all_score_r57", "own_all_score_r58", "own_all_score_r59_61"):
        for k, r in d[n]["results"].items():
            for fo, x in r["folders"].items():
                if not fo.startswith("ep2_"):
                    reg[(k, fo)] = x
    got = {(k, fo): x for k, r in ls["results"].items() for fo, x in r["folders"].items()}
    cov = {s: {"folders_expected": 0, "folders_equal": 0, "mismatch": [], "fold_minus1_rows": 0, "mapped_by_world": {}} for s in (*ORIGINS, "other (IL, ug)")}
    for (k, fo), x in sorted(reg.items()):  # EVERY registered non-epoch-2 folder (Codex r3); IL / ug folders checked too, under "other"
        s = set_of_folder(x["kind"], fo) or "other (IL, ug)"
        c = cov[s]
        c["folders_expected"] += 1
        y = got.get((k, fo))
        if y is None or any(y["counts"][rd][f] != x["counts"][rd][f] for rd in ("R", "C") for f in ("rows", "runs", "runs_without_rows")):
            c["mismatch"].append(fo)
            continue
        c["folders_equal"] += 1
        c["fold_minus1_rows"] += sum(y["counts"][rd]["rows_fold_minus1"] for rd in ("R", "C"))
        for rd in ("R", "C"):
            for w, m in y["counts"][rd]["by_world"].items():
                c["mapped_by_world"].setdefault(w, {"R": 0, "C": 0})[rd] += m["mapped"]
    fm1_all = sum(y["counts"][rd]["rows_fold_minus1"] for y in got.values() for rd in ("R", "C"))  # Codex r4: over EVERY LOO folder, not only registered keys
    extra = sorted(fo for (k, fo) in set(got) - set(reg))
    global_ok = all(not c["mismatch"] for c in cov.values()) and fm1_all == 0 and set(got) == set(reg)
    cov["other (IL, ug)"]["global"] = {"loo_folders": len(got), "registered_folders": len(reg), "extra_loo_folders": extra, "fold_minus1_rows_all_loo_folders": fm1_all}
    for s in ORIGINS:
        c = cov[s]
        c["missing_worlds"] = sorted(w for w in NEED_WORLDS[s] if not all(c["mapped_by_world"].get(w, {}).get(rd, 0) > 0 for rd in ("R", "C")))
        c["global_coverage_ok"] = global_ok  # any registered folder missing / unequal, or any fold -1 row anywhere, withholds every set
        c["complete"] = global_ok and c["folders_expected"] > 0 and not c["missing_worlds"]

    # ---- eligibility, per set with model o's training launches; amendment 2 item 4's assert by key
    per_set, total, asserts = {}, 0, {}
    for o, s in enumerate(ORIGINS):
        tids = tids_of[s]
        arm_launch = {w: {r["launch"] for r in runs.get((s, w), [])} for w in ("run1", "run2", "sim", "honest_a", "honest_b", "honest_c")}
        pool_launch = {name: set(pool_check(s, mem, set(), name.startswith("H_A"))["launch_ids"]) for name, mem in pools[s].items()}
        hit = {w: sorted(v & tids) for w, v in arm_launch.items() if v & tids} | {n: sorted(v & tids) for n, v in pool_launch.items() if v & tids}
        asserts[s] = {"model": f"o{o} (held out: {s})", "training_launch_ids": sorted(tids), "arm_and_pool_launches_in_training": hit,
                      "set_mains_in_training_mains": len(trm_of[s] & setm_of[s]), "pass": not hit and not (trm_of[s] & setm_of[s])}
        if not asserts[s]["pass"]:
            raise SystemExit(f"STOP: amendment 2 item 4 assert failed for set {s}: {hit}")
        by_cell = {w: collections.defaultdict(list) for w in ("run1", "run2", "sim")}
        for w in by_cell:
            for r in runs.get((s, w), []):
                if in_attack_scope(s, r["cell"]):
                    by_cell[w][r["cell"]].append(r)
        both = sorted(set(by_cell["run1"]) & set(by_cell["run2"]))
        elig = sorted(c for c in both if all(r["launch"] not in tids for w in ("run1", "run2") for r in by_cell[w][c]))
        run1_mains = {cell_main[c] for c in by_cell["run1"]}
        structural = cov[s]["complete"]
        scored_mains = run1_mains & setm_of[s] if structural else set()  # fold = origin: every run1 main of set s is scored by model o
        elig_scored = sorted(c for c in elig if cell_main[c] in scored_mains)
        pc = {name: pool_check(s, mem, tids, name.startswith("H_A")) for name, mem in pools[s].items()}
        withheld = (not structural) or not all(v["pass"] for v in pc.values())
        n = len({cell_main[c] for c in elig_scored}) if not withheld else 0
        coh_missing = {c: ("no run1 and run2 pair by key" if c not in both else "run1/run2 share a training launch" if c not in elig else "not scored")
                       for c in coh[s] if c not in elig_scored}
        per_set[s] = {
            "scoring_model": f"{PROBE} o{o}", "attack_scope": "S1-S3 + S5 (batch_of_cell @ membership710); pools whole" if s == "S_newmains" else "all cells of the set",
            "arms": {w: {"runs": sum(len(v) for v in by_cell[w].values()), "cells": len(by_cell[w]), "launch_ids": sorted({r["launch"] for v in by_cell[w].values() for r in v}),
                         "cells_sharing_training_launch": sum(1 for v in by_cell[w].values() if any(r["launch"] in tids for r in v))} for w in by_cell},
            "cells_with_run1_and_run2": len(both), "eligible_cells_launch_disjoint": len(elig),
            "scored": f"scored: {len(scored_mains)} run1 mains (fold = origin; model o{o} only)", "scoring_coverage": cov[s],
            "structurally_scored": structural, "eligible_and_scored_cells": elig_scored,
            "calibration_pools": pc, "withheld": withheld, "eligible_mains_counted": n,
            "cohort_check": {"cohort": "pooled455 (v5's cohort_source)" if s != "S_newmains" else "membership710 cohort_without_S4",
                             "cohort_cells": len(coh[s]), "cohort_in_eligible_and_scored": len(coh[s]) - len(coh_missing), "cohort_cells_not_covered": coh_missing},
        }
        total += n
    avail = {"eligible_mains_total": total, "available": total >= 30}
    v5 = d["eligibility_v5"]
    script_rel = "experiments/2026-10-15_own_all_loo/make_loo_eligibility.py"
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    committed_blob = subprocess.run(["git", "rev-parse", f"HEAD:{script_rel}"], capture_output=True, text=True, check=False).stdout.strip()
    disk_blob = subprocess.run(["git", "hash-object", script_rel], capture_output=True, text=True, check=True).stdout.strip()
    if committed_blob != disk_blob:
        raise SystemExit("commit the script before running it (generator must equal HEAD's copy)")
    stamp = datetime.datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {
        "label": f"{PROBE} launch-disjoint eligibility file (09 inclusion amendment 2 item 4); provenance only; no score read; {LABEL}",
        "stamp (PT)": stamp, "generator": {"path": script_rel, "commit": head, "blob_sha1": disk_blob, "derived_from": "make_launch_disjoint_table.py @ 1706761b5 (eligibility v5)"},
        "rule": {"amendment_2": f"@ {AMD2[0]}", "base": "@ 1ffffca72", "amendment_1": "@ 24fd07866", "criterion_A": "as v5 @ 90c0c3148"},
        "criterion": CRITERION, "sources": src | {"chain_manifests": {f"{CHAIN_SHA}:{CHAIN_CFG}/{k}": v for k, v in sorted(used_manifests.items())}},
        "gates": {"fit": {"refit_match_pass": g["refit_match_pass"], "all_certified": g["all_certified"], "dropped_mains_set_o": g["dropped_mains_set_o"],
                          "fit_record_sha256": lf["fit"]["fit_record_sha256"], "loo_sha256": lo["_sha256"]},
                  "score": {"in_fold_violations_total": 0, "folders": sum(len(r["folders"]) for r in ls["results"].values())},
                  "verify": {"pass": lv["pass"]}},
        "amendment_2_item_4_asserts": asserts, "sets": {PROBE: per_set}, "availability": {PROBE: avail},
        "scoring_coverage_other_folders": cov["other (IL, ug)"], "scoring_coverage_basis": "own_all_L48's registered score records (main + r57 / r58 / r59-61 addenda; epoch-2 excluded), per folder, both readouts",
        "availability_order": "own_all_LOO_L48, then own_all_L48, then own_frac100_L48, then none (amendment 2 item 5); by availability, never by value",
        "others_unchanged": {"own_all_L48": v5["availability"]["own_all_L48"], "own_frac100_L48": v5["availability"]["own_frac100_L48"], "source": src["eligibility_v5"]},
        "timing_rule": "joins tonight's headline iff this file, with every gate passed, is committed and pushed BEFORE the first Stage C launch; otherwise the second refresh (amendment 2 item 6)",
    }
    path = HERE / "results" / f"own_all_loo_eligibility_{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    with path.open("x") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(path)
    for s, v in per_set.items():
        print(PROBE, s, "| run1&run2", v["cells_with_run1_and_run2"], "| disjoint", v["eligible_cells_launch_disjoint"], "| elig&scored", len(v["eligible_and_scored_cells"]),
              "| withheld", v["withheld"], {k: (x["pass"], x["members"], len(x["unknown_not_in_registered_exclusions"]), x["sharing_training_launch"]) for k, x in v["calibration_pools"].items()},
              "| counted", v["eligible_mains_counted"], "| cohort", v["cohort_check"]["cohort_in_eligible_and_scored"], "/", v["cohort_check"]["cohort_cells"])
    print(json.dumps(avail), json.dumps({s: a["pass"] for s, a in asserts.items()}))


if __name__ == "__main__":
    main()
