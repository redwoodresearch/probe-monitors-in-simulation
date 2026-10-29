"""Did any record relayed on 2026-10-15 use an S4 "orphan" row for the nm360 new-mains tasks? (2e's independent check; read-only, no Modal.)

Orphans: d9's S4 source map (s4_source_map_4r_2026-10-15.json @ b68ab0a53), key outside_source_map: 115 rows from the earlier S4 roots
(nm360_s4_main_v1, nm360_s4_main_s1_v1, nm360_s4_run2_v1, nm360_s4_run2_s1_v1) whose cells 4r owns; 52 are staged. A key string (world|cell) is
the same for an orphan row and its authoritative 4r row, so a record "uses" an orphan row only if it attributes that key to a folder of an orphan
root (or carries the orphan's .eval path or sha256). Both attributions present = double count; the orphan one alone = substitution.

Only metadata, manifests and input fields are read: folder lists, (folder, key) attributions, key lists, cell lists, file paths and shas. No score,
threshold, gap or catch value is read, and no attack text. Every answer is computed here and written to results/.

  python3 experiments/2026-10-15_s4_orphan_check/check_s4_orphans.py
"""

import base64
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ORPH_ROOTS = ("nm360_s4_main_v1", "nm360_s4_main_s1_v1", "nm360_s4_run2_v1", "nm360_s4_run2_s1_v1")
SRC_MAP = ("b68ab0a53", "experiments/2026-10-14_q3_overnight_chain/results/s4_source_map/s4_source_map_4r_2026-10-15.json")
FOURR_MANIFESTS = [("b68ab0a53", "experiments/2026-10-14_q3_overnight_chain/config/manifest_nm360_s4r_main.json"),
                   ("b68ab0a53", "experiments/2026-10-14_q3_overnight_chain/config/manifest_nm360_s4r_run2.json")]
D507 = "experiments/2026-10-15_fpr_sweep_507_prep"


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout


def show(commit, path):
    return git("show", f"{commit}:{path}")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def root_of(folder):
    return folder.split("/")[-1].split("__")[0]


def strings(o):
    """Every string (dict keys and string values) in a metadata subtree."""
    if isinstance(o, dict):
        for k, v in o.items():
            yield str(k)
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)
    elif isinstance(o, str):
        yield o


def main():
    srcb = show(*SRC_MAP)
    src = json.loads(srcb)
    orph = src["outside_source_map"]
    staged = [r for r in orph if r["staging"] == "staged"]
    by_root = {}
    for r in staged:
        by_root[r["root"]] = by_root.get(r["root"], 0) + 1
    o_keys = {r["key"] for r in staged}
    o_cells = {r["cell"] for r in staged}
    o_shas = {r["sha256"] for r in staged}
    o_evals = {Path(r["record"]).name for r in staged}
    o_launches = {r["launch"] for r in staged}
    out = {"label": "S4 orphan-row check (2e, 2026-10-15); POST HOC; read-only; metadata, manifests and input fields only",
           "orphans": {"source": f"{SRC_MAP[1]} @ {SRC_MAP[0]} (sha256 {sha(srcb)})", "outside_source_map": len(orph), "staged": len(staged),
                       "staged_cells": len(o_cells), "staged_by_root": by_root, "staged_by_world": {w: sum(r["world"] == w for r in staged) for w in sorted({r["world"] for r in staged})}},
           "records": {}}

    # same run under two names: staged orphan .eval sha256 among the 4r folders' staging manifests
    hits = []
    for c_, p_ in FOURR_MANIFESTS:
        mb = show(c_, p_)
        for e in json.loads(mb)["files"]:
            if e["sha256"] in o_shas or Path(e["rel"]).name in o_evals:
                hits.append({"manifest": f"{p_} @ {c_}", "rel": e["rel"], "sha256": e["sha256"]})
    out["orphan_eval_sha_in_4r_staging_manifests"] = {"manifests": [f"{p_} @ {c_} (sha256 {sha(show(c_, p_))})" for c_, p_ in FOURR_MANIFESTS],
                                                       "matches": hits, "verdict": "none" if not hits else "FOUND"}

    def ownership_verdict(name, mem, roles_note):
        """membership710: folder -> keys (ownership by record). The producers keep a key's rows only from the folder that owns it."""
        own = {}
        for f, ks in mem["folder_keys"].items():
            for k in ks:
                own.setdefault(k, []).append(f)
        two = sorted(k for k, v in own.items() if len(v) > 1)
        used, per = [], {}
        for r in staged:
            fs = own.get(r["key"], [])
            orph_f = [f for f in fs if root_of(f) in ORPH_ROOTS]
            fourr_f = [f for f in fs if root_of(f).startswith("nm360_s4r")]
            per[r["key"] + " @ " + r["root"]] = {"owners": fs}
            if orph_f:
                used.append({"key": r["key"], "orphan_root": r["root"], "owner": orph_f, "kind": "double count" if fourr_f else "substitution"})
        n_orph_owned = {f: len(ks) for f, ks in mem["folder_keys"].items() if root_of(f) in ORPH_ROOTS}
        return {"verdict": "clean" if not used and not two else "orphan keys used", "orphan_keys_used": used, "keys_owned_by_two_folders": two,
                "staged_orphan_key_owners": per, "keys_owned_by_orphan_root_folders": n_orph_owned,
                "staged_orphan_keys_owned_by_a_4r_folder": sum(any(root_of(f).startswith("nm360_s4r") for f in v["owners"]) for v in per.values()),
                "staged_orphan_keys_with_no_owner": sorted(k for k, v in per.items() if not v["owners"]), "how": roles_note}

    mems = {}
    for c_ in ("0f9d96158", "f3e57dc98"):
        b_ = show(c_, f"{D507}/config/membership710.json")
        mems[sha(b_)] = (c_, json.loads(b_))

    # 1. all-tasks record
    rb = show("6acf23b06", f"{D507}/results/alltasks_CRD_2026-10-15_161306.json")
    rec = json.loads(rb)
    msha = rec["meta"]["alltasks"]["membership710_sha256"]
    c_, mem = mems[msha]
    v = ownership_verdict("alltasks", mem, "S_newmains rows (attack, honest B pool, honest A) are read from membership710's folders and kept only for the keys each "
                                          "folder owns (run_peraction311.row, NM branch @ 6acf23b06)")
    out["records"]["alltasks_CRD_2026-10-15_161306.json @ 6acf23b06"] = {"record_sha256": sha(rb), "membership710": f"{D507}/config/membership710.json @ {c_} (sha256 {msha})"} | v

    # 2. sweep records (shared-pool rows use the same membership and the same row() code); the union Sol/c_asis addendum if landed
    for f in ("sweep_final_C_2026-10-15_160740.json", "sweep_final_R_2026-10-15_160743.json"):
        rb = show("d62cfe39e", f"{D507}/results/{f}")
        rec = json.loads(rb)
        msha = rec["meta"]["sweep"]["membership710_sha256"]
        c_, mem = mems[msha]
        origins = sorted({o for r_ in rec["rows"].values() for o in r_})
        v = ownership_verdict(f, mem, "S_newmains and union rows read membership710's folders, rows kept by folder ownership (run_peraction311.row @ 9e18a3971)")
        out["records"][f"{f} @ d62cfe39e"] = {"record_sha256": sha(rb), "membership710": f"{D507}/config/membership710.json @ {c_} (sha256 {msha})", "origins": origins} | v
    head = git("rev-parse", "origin/aswin/fpr-sweep-507-prep").decode().strip()
    union_files = [p for p in git("ls-tree", "--name-only", "origin/aswin/fpr-sweep-507-prep", f"{D507}/results/").decode().split() if "union" in Path(p).name.lower()]
    out["records"]["union Sol/c_asis addendum"] = {"verdict": "not landed" if not union_files else "present (not checked here)", "searched": f"{D507}/results/ @ {head}",
                                                   "files": union_files}

    # 3. pooled v2: does it touch nm360 at all?
    for f in ("pooled455v2_C_2026-10-15_160734.json", "pooled455v2_R_2026-10-15_160736.json"):
        rb = show("672b1dd1b", f"{D507}/results/{f}")
        rec = json.loads(rb)
        meta_strs = list(strings(rec["meta"]))
        origins = sorted({o for r_ in rec["rows"].values() for o in r_})
        roots = sorted({k for r_ in rec["rows"].values() for o, x in r_.items() if isinstance(x, dict) for k in x.get("n_per_batch", {}).get("H_sim_runs_by_root", {})})
        nm = [s for s in meta_strs if "nm360" in s] + [r_ for r_ in roots if "nm360" in r_]
        out["records"][f"{f} @ 672b1dd1b"] = {"record_sha256": sha(rb), "origins": origins, "membership_in_meta": "membership710_sha256" in json.dumps(rec["meta"]),
                                              "nm360_strings_in_meta_or_roots": nm, "verdict": "clean (does not touch nm360)" if not nm and origins == ["S_new", "S_old"] else "touches nm360"}

    # 4. rmshits S_newmains and its deviation export: (folder, key) per arm and cell, H_B folders and keys, from the config the records pin
    b4 = "origin/aswin/il-z-e2e-2e"
    E4 = "experiments/2026-10-15_il_z_e2e"
    for f in ("rmshits_S_newmains_2026-10-15_145501.json", "rmshits_S_newmains_deviation_2026-10-15_150228.json"):
        rb = show(b4, f"{E4}/results/{f}")
        rec = json.loads(rb)
        cb = show(rec["meta"]["git"], f"{E4}/{rec['meta']['config']}")
        assert sha(cb) == rec["meta"]["config_sha256"], f
        cfg = json.loads(cb)
        att = [(arm, cell, fk[0], fk[1]) for arm, cells in cfg["arm_rows"].items() for cell, fk in cells.items()]
        used = [{"arm": a, "cell": c, "folder": fo, "key": k} for a, c, fo, k in att if root_of(fo) in ORPH_ROOTS]
        hb_orph = [fo for fo in cfg["H_B"]["folders"] if root_of(fo) in ORPH_ROOTS]
        folders_orph = [fo for fo in cfg["folders"] if root_of(fo) in ORPH_ROOTS]
        cells_orph = sorted(set(rec["cells"]) & o_cells)
        hb_keys_orph = sorted(set(cfg["H_B"]["keys"]) & o_keys)
        out["records"][f"{f} @ {git('rev-parse', '--short=9', b4).decode().strip()}"] = {
            "record_sha256": sha(rb), "config": f"{E4}/{rec['meta']['config']} @ {rec['meta']['git'][:9]} (sha256 {rec['meta']['config_sha256']})",
            "attack_rows_from_orphan_root_folders": used, "H_B_folders_from_orphan_roots": hb_orph, "folders_from_orphan_roots": folders_orph,
            "orphan_cells_in_cohort": cells_orph, "orphan_keys_in_H_B_keys": hb_keys_orph,
            "examined": {"arm_row_attributions": len(att), "arm_rows_from_s4r_folders": sum(root_of(fo).startswith("nm360_s4r") for _, _, fo, _ in att),
                         "H_B_folders": len(cfg["H_B"]["folders"]), "H_B_keys": len(cfg["H_B"]["keys"]), "cells": len(rec["cells"]), "folders": len(cfg["folders"])},
            "verdict": "clean" if not (used or hb_orph or folders_orph or cells_orph or hb_keys_orph) else "orphan rows used"}

    # 5. leakage addendum: its file pins and sources (folders it read)
    f5 = "experiments/2026-10-15_il_z_e2e/results/leakage_addendum_2026-10-15_164923.json"
    rb = show("95587b121", f5)
    rec = json.loads(rb)
    pins = list(rec["files"])
    orph_pins = [p for p in pins if root_of(p.split("/")[1]) in ORPH_ROOTS] if pins else []
    src_strs = [s for s in strings(rec["sources"]) if any(r_ in s for r_ in ORPH_ROOTS) or s in o_shas]
    out["records"][f"{Path(f5).name} @ 95587b121"] = {"record_sha256": sha(rb), "file_pins": len(pins), "pins_from_orphan_roots": orph_pins,
                                                      "pinned_folders": sorted({p.split("/")[1] for p in pins}),
                                                      "orphan_identifiers_in_sources": src_strs,
                                                      "what_it_reads": "Sol training-main lists and manifests (no rows); see files",
                                                      "verdict": "clean" if not orph_pins and not src_strs else "orphan roots read"}

    # 6. IL extended S_newmains record: its input folders, files read, keylist, and the parent X_err scope it reads run 1 / run 2 from
    f6 = "experiments/2026-10-15_il_analysis_09/results/il_ext_final_result_il_2026-10-15_134705.json"
    rb = show("4d92cc2ca", f6)
    r6 = json.loads(rb)["result_il"]
    meta6 = {k: r6[k] for k in ("inputs", "inputs_read", "extended_arms_not_read_out_of_parent_scope", "non_authoritative_keys_not_used", "extension_accounting")}
    kl = json.loads(base64.b64decode(r6["record_inputs"]["keylist"]["content_b64"]))
    s6 = list(strings(meta6)) + list(strings(kl))
    hits6 = sorted({s for s in s6 if any(r_ + "__" in s or r_ + "/" in s or s == r_ for r_ in ORPH_ROOTS) or s in o_shas or s in o_evals})
    in_folders6 = sorted(k for k in r6["inputs"] if root_of(k) in ORPH_ROOTS)
    out["records"][f"{Path(f6).name} @ 4d92cc2ca"] = {
        "record_sha256": sha(rb), "input_folders": len(r6["inputs"]), "input_folders_from_orphan_roots": in_folders6, "orphan_identifiers_in_inputs": hits6,
        "examined": {"strings": len(s6), "nm_input_folders": sum(k.startswith("ilnm") for k in r6["inputs"])},
        "parent_scope": r6["extended"]["label"] if isinstance(r6["extended"].get("label"), str) else None,
        "s4_cells_not_analysed": {k: v for k, v in r6["extended"]["accounting"]["not_paired_by_reason"].items() if "S4" in k},
        "paired_cell_list": "not stored (cells_paired_list_sha256 only); run 1 / run 2 for nm cells come from the parent X_err extended record (item 7)",
        "verdict": "clean" if not in_folders6 and not hits6 else "orphan identifiers found"}

    # 7. X_err extended record: scale-up input folders, the S_newmains cohort and pools
    f7 = "experiments/2026-10-14_xerr_scaleup_09/results/xerr_scaleup_final_2026-10-15_113025.json"
    rb = show("1a7e95259", f7)
    r7 = json.loads(rb)["result"]
    fol7 = list(r7["scale_up_inputs"])
    orph_fol7 = [f for f in fol7 if root_of(f) in ORPH_ROOTS]
    coh7 = set(r7["cohort"]["cohort_cells"])
    pool_strs = [s for k_ in ("primary", "extended") for s in strings(r7["calibration_records"][k_]) if "|" in s]
    keys7 = set(pool_strs)
    s7 = list(strings(r7["scale_up_inputs"]))
    hits7 = sorted({s for s in s7 if s in o_shas or s in o_evals or any(r_ + "__" in s or r_ + "/" in s for r_ in ORPH_ROOTS)})
    out["records"][f"{Path(f7).name} @ 1a7e95259"] = {
        "record_sha256": sha(rb), "scale_up_input_folders": len(fol7), "nm360_folders": sorted(f for f in fol7 if f.startswith("nm360")),
        "folders_from_orphan_roots": orph_fol7, "orphan_identifiers_in_scale_up_inputs": hits7,
        "orphan_cells_in_cohort": sorted(coh7 & o_cells), "orphan_keys_in_calibration_records": sorted(keys7 & o_keys),
        "examined": {"cohort_cells": len(coh7), "cohort_nm_cells": sum(c.startswith("nm") for c in coh7), "calibration_key_strings": len(keys7),
                     "calibration_nm_keys": sum(k.split("|", 1)[1].startswith("nm") for k in keys7 if "|" in k), "scale_up_strings": len(s7)},
        "verdict": "clean" if not (orph_fol7 or hits7 or (coh7 & o_cells) or (keys7 & o_keys)) else "orphan rows used"}

    # 8-10. LLM-monitor records (the viewer's additions): only their key, cell, launch, exclusion and source fields are read
    PM = "experiments/2026-10-15_q2_monitor_su311/results"
    ID_FIELDS = {"nm360_s123_analysis_2026-10-15_031836.json": ("3eb8db6b1", None, ("nm360_keys", "cohort_tasks_by_launch", "attempted_all_three", "status_rows",
                                                                                   "dropped_by_key", "excluded_by_key", "no_reference_cells_by_launch", "score_file", "rule")),
                 "nm360_s5_analysis_2026-10-15_044811.json": ("1fae2a855", "S5", ("keys_meta", "attempted_all_three", "status_rows", "dropped_by_key", "excluded_by_key")),
                 "breakdown_POSTHOC_2026-10-15_030445.json": ("7e30115e8", None, ("sources", "cohorts"))}  # sort_fix_check holds synthetic names (nm041_x)
    for f, (c_, sub, fields) in ID_FIELDS.items():
        rb = show(c_, f"{PM}/{f}")
        rec = json.loads(rb)
        part = rec[sub] if sub else rec
        meta = {k: part[k] for k in fields if k in part} | ({"score_files": rec["score_files"]} if "score_files" in rec else {})
        ss = list(strings(meta))
        launches = sorted({x.split("/")[0] for x in ss if x.startswith("2026-10-14_newmains_")})  # launch names (also the first component of .eval paths)
        cu = (part.get("nm360_keys") or part.get("keys_meta") or {}).get("cell_union_per_launch", {})  # the launches whose cells the record reads
        cells = {x for x in ss if x.startswith("nm") and "_" in x and "|" not in x and "/" not in x}
        cells |= {x.split("|", 1)[1] for x in ss if "|" in x and x.split("|", 1)[1].startswith("nm")}
        cells |= {x.split("/")[2].split("_r2")[0].split("_redraw")[0] for x in ss if x.startswith("2026-10-14_newmains_") and x.count("/") >= 3}
        hits = sorted({x for x in ss if x in o_shas or Path(x).name in o_evals or "newmains_s4/" in x or "newmains_s4_" in x or any(r_ + "__" in x for r_ in ORPH_ROOTS)})
        out["records"][f"{f} @ {c_}"] = {
            "record_sha256": sha(rb), "fields_read": list(meta), "launches_named": launches, "cell_union_per_launch_launches": sorted(cu),
            "orphan_launches": sorted(o_launches), "orphan_launch_read": sorted(set(cu) & o_launches) + sorted(set(launches) & o_launches),
            "examined": {"strings": len(ss), "nm_cells_named": len(cells)},
            "orphan_cells_named": sorted(cells & o_cells), "s4_or_orphan_identifiers": hits,
            "reads_nm360": bool(cells) or bool(launches), "cohort_labels": sorted(part["cohorts"]) if "cohorts" in part else None,
            "verdict": (("clean (does not read nm360)" if not cells and not launches else "clean")
                        if not (cells & o_cells) and not hits and not (set(cu) | set(launches)) & o_launches else "orphan cells, launch or identifiers present")}

    out["meta"] = {"git": git("rev-parse", "HEAD").decode().strip(), "script_sha256": sha(Path(__file__).read_bytes()),
                   "stamp_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")}
    out["summary"] = {k: v["verdict"] for k, v in out["records"].items()} | {"orphan .eval sha256 in 4r staging manifests": out["orphan_eval_sha_in_4r_staging_manifests"]["verdict"]}
    dest = HERE / "results" / f"s4_orphan_check_{out['meta']['stamp_pt']}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["summary"], indent=1))
    print(dest)


if __name__ == "__main__":
    main()
