"""Build the evidence rows for the simh_q4_rlr replacements (t361, t411, t413) from the box/diag/hash/structure/census
JSONs in <dir> plus the staging and generation manifests saved there, and the contract census (originals).
Usage: python3 -I build_rows_q4_rlr.py <dir> <contract census json> <out dir>"""
import hashlib, json, os, re, sys

D, CC, OUT = sys.argv[1:4]
L = lambda n: json.load(open(os.path.join(D, n)))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
LAUNCH, ORIG = "2026-10-15_simh_q4_rlr", "2026-10-15_simh_q4"
REFS = {"staging": "experiments/2026-10-14_q3_overnight_chain/config/manifest_simh311_q4_sim_rlr.json @ 1609cb43e",
        "generation": "experiments/2026-10-15_q3_simhonest_311/results/manifest_simh_q4_rlr_2026-10-15_162657.json @ 1e3be03e2"}
LABELS = {"t361": "replacement (rlr) superseding simh_q4 | t361; test of the :313 hypothesis",
          "t411": "replacement (rlr) superseding simh_q4 | t411", "t413": "replacement (rlr) superseding simh_q4 | t413"}
box, dg, cen, hr, ho = (L("box_simh_q4_rlr_b17.json"), L("diag_simh_q4_rlr_t361_t411_t413.json"), L("census_simh_q4_rlr_b17.json"),
                        L("hash_q4_rlr_t361_t411_t413.json"), L("hash_q4_orig_t361_t411_t413.json"))
sm, gm, cc = L("staging_manifest.json"), L("gen_manifest.json"), json.load(open(CC))
inputs = ["box_simh_q4_rlr_b17.json", "diag_simh_q4_rlr_t361_t411_t413.json", "census_simh_q4_rlr_b17.json",
          "hash_q4_rlr_t361_t411_t413.json", "hash_q4_orig_t361_t411_t413.json", "list_simh_q4_rlr.json"]
assert hr["hash_method"] == ho["hash_method"]
assert gm["launch"] == "q4_rlr" and gm["host_box"] == "xbox3" and gm["arm"] == "sim_honest@september"
assert sm["counts"]["files"] == len(sm["files"]) == 3
RUNS = "/home/aswin/phase2/runs/"
assert box["run_dir"] == dg["run_dir"] == RUNS + LAUNCH and cen["box_json_sha256"] == sha(os.path.join(D, "box_simh_q4_rlr_b17.json"))
assert all(x["run_dir"] == RUNS + LAUNCH for x in hr["cells"]) and all(x["run_dir"] == RUNS + ORIG for x in ho["cells"])
rows = {}


def site_of(raise_site):
    m = re.fullmatch(r"contract\.py:([0-9]+) .+", raise_site)
    return int(m.group(1)) if m else None


def one(xs, pred, what):
    m = [x for x in xs if pred(x)]
    assert len(m) == 1, what
    return m[0]


for p in ("t361", "t411", "t413"):
    b = one(box["cells"], lambda c: c["cell"] == p, "box")
    d = one(dg["cells"], lambda c: c["cell"] == p, "diag")
    full = b["run"].split("/", 1)[1]
    c = one(cen["cells"], lambda c: c["cell"] == b["run"], "census")["census"]
    h_r = one(hr["cells"], lambda c: c["cell"] == full, "hash rlr")
    h_o = one(ho["cells"], lambda c: c["cell"] == full, "hash orig")
    st_name = f"structure_ref_q4_rlr_{p}.json"
    st = L(st_name)
    assert st["target"]["cell"] == full and st["reference"]["cell"] == full and st["target"]["run_dir"] == RUNS + LAUNCH
    assert st["control"]["run_dir"] == RUNS + "2026-10-15_simh_q1_rlr" and c["box"] == "xbox3" and c["launch"] == LAUNCH
    # the prose in a_eval_complete / b_misalignment_side holds only if these structure facts hold
    t, k = st["target"], st["control"]
    assert t["has_header"] and t["n_sample_files"] == 1 and t["has_summaries"] and t["log_status"] == "success" and t["sample_has_error"]
    assert t["last_event_type"] == "error" and t["n_error_events"] == 1 and not t["has_reductions"] and not t["has_results_block"]
    assert t["messages_by_role"] == {"user": 1} and k["has_reductions"] and k["has_results_block"] and not k["sample_has_error"]
    ev = one(b["evals"], lambda e: e["final"], "final eval")
    s = ev["samples"][0]
    assert ev["path"] == b["final_eval"] and d["eval_sha256"] == ev["sha256"] == h_r["eval_sha256"]
    att = b["attempts"]
    run_id = f"{LAUNCH}|{b['run']}"
    orig = one(cc["runs"], lambda r: r["run_id"] == f"{ORIG}|{b['run']}", "original")
    sf = one(sm["files"], lambda f: f["launch"] == f["launch_id"] == f["sim_launch"] == LAUNCH and f["arm_dir"] == "sim_honest@september"
             and f["rel"].split("/")[:2] == ["sim_honest@september", full] and os.path.basename(f["path"]) == os.path.basename(ev["path"]), "staging row")
    gc = one(gm["cells"], lambda g: g["task"] == full and g["key"] == f"simh_311|{full}" and g["main"] == d["main"] and g["env"] == d["env"]
             and str(g["reference_source"]).split("/")[-3:-1] == ["honest_a", full], "generation row")
    # fail closed: every invariant the row reports must hold, else no row is written
    assert ev["n_samples"] == 1 and len(ev["samples"]) == 1 and b["run"] == f"sim_honest@september/{full}" == d["run"]
    for hx in (h_r, h_o):
        lj = hx["launch_json_outcome_sample_errors"]
        assert lj and all(v.get("type") == "str" and v.get("sha256") for v in lj) and hx["last_attempt_sample_errors0_equals_eval_message"]
        assert hx["eval_sample_error_message"].get("sha256")
    assert h_r["eval_sample_error_message"]["sha256"] == d["message_sha256"] and h_r["eval_sample_error_message"]["len"] == d["message_len"]
    assert h_o["eval_sample_error_message"]["sha256"] == orig["message_sha256"] and h_o["eval_sample_error_message"]["len"] == orig["message_len"]
    df = d["deepest_frame"]
    classes = {"attempt_record": [x for a in att for x in a["sample_errors"]], "eval_sample": s["error_class"], "diag": d["error_class"],
               "census": [r["class"] for r in c["rows"]]}
    site = df["line"]
    same_site_originals = sorted({r["message_sha256"] for r in cc["runs"] if site_of(r["raise_site"]) == site and not r["is_replacement"]})
    all_hashes = {h_r["eval_sample_error_message"]["sha256"], h_o["eval_sample_error_message"]["sha256"],
                  *[v.get("sha256") for v in h_r["launch_json_outcome_sample_errors"] + h_o["launch_json_outcome_sample_errors"]]}
    assert len({*classes["attempt_record"], classes["eval_sample"], classes["diag"], *classes["census"]}) == 1 and c["rows"]
    assert set(classes["attempt_record"]) == {"OutputContractError"}  # the prose names this class
    assert d["deepest_parsed_frame_verified"] and df["file"].endswith("/sim/contract.py") and df["side"].startswith("simulator")
    assert s["messages_by_role"] == {"user": 1}
    assert sf["sha256"] == ev["sha256"] and gc["eval_sha256"] == st["reference"]["reference_sha256"] and st["reference"]["copy_rehash_equal"]
    row = {
        "label": LABELS[p], "run_id": run_id, "box": f"xbox3 (the generation manifest's host_box: {gm['host_box']})",
        "exit_code": [a["exit_code"] for a in att], "clean": [a["clean"] for a in att], "policy_called": [a["policy_called"] for a in att],
        "n_attempts": b["n_attempts"], "attempt_kinds": [a["kind"] for a in att], "start_end": [[a["start"], a["end"]] for a in att],
        "policy_steps": [q["steps"] for a in att for q in a["policy_call_summaries"]],
        "terminal_class": {**classes, "census_ref": c["ref"], "census_bin": [r["bin"] for r in c["rows"]],
                           "concordant": True},
        "raise_site": f"{df['file']}:{df['line']} {df['func']}", "raise_source_line": d["raise_source_line"].strip(),
        "message_withheld": d["message_withheld"], "message_sha256": d["message_sha256"], "message_len": d["message_len"],
        "original": {"run_id": orig["run_id"], "raise_site": orig["raise_site"], "message_sha256": orig["message_sha256"],
                     "message_len": orig["message_len"], "source": "experiments/2026-10-15_simh_exclusion_evidence/results/contract_census_2026-10-15.json @ b63c73cd7"},
        "raise_site_equals_original": site_of(orig["raise_site"]) == site,
        "message_sha_equals_original": d["message_sha256"] == orig["message_sha256"],
        f"distinct_message_sha256_over_original_{site}_runs": same_site_originals,
        "message_hash": {
            "method": hr["hash_method"],
            "census_field": "contract_census @ b63c73cd7 message_sha256 = simh_err_diag.py's sha256(.eval sample.error.message, UTF-8), i.e. eval_field",
            "replacement": {"launch_json_field": f"{LAUNCH}/launch.json (xbox3) runs[key={b['run']}].attempts[i].outcome.sample_errors[j]",
                            "launch_json_values": h_r["launch_json_outcome_sample_errors"],
                            "eval_field": "final .eval sample error.message (inspect_ai read_eval_log)", "eval_value": h_r["eval_sample_error_message"],
                            "eval_sha256": h_r["eval_sha256"], "launch_json_equals_eval": h_r["last_attempt_sample_errors0_equals_eval_message"]},
            "original": {"launch_json_field": f"{ORIG}/launch.json (xbox1) runs[key={b['run']}].attempts[i].outcome.sample_errors[j]",
                         "launch_json_values": h_o["launch_json_outcome_sample_errors"],
                         "eval_field": "final .eval sample error.message (inspect_ai read_eval_log)", "eval_value": h_o["eval_sample_error_message"],
                         "eval_sha256": h_o["eval_sha256"], "launch_json_equals_eval": h_o["last_attempt_sample_errors0_equals_eval_message"]},
            "same_sha_on_both_fields_both_runs": len(all_hashes) == 1 and None not in all_hashes},
        "traceback_location": "sample.error.traceback in samples/<file>.json, read with inspect_ai read_eval_log (the frames are parsed from it)",
        "sample_error_has_traceback": st["target"]["sample_error_has_traceback"],
        "last_tool_before_error": d["last_tool_before_error"], "tool_name_counts": d["tool_name_counts"],
        "messages_by_role": s["messages_by_role"], "tool_events_final": b["tool_events_final"], "tool_steps_at_failure": d["tool_steps_before_error"],
        "eval": {"path": ev["path"], "sha256": ev["sha256"], "bytes": ev["size"]},
        "manifest_rows": {
            "staging_chain": {"file": REFS["staging"], "launch": sf["launch"], "rel_tail": "/".join(sf["path"].split("/")[-3:]),
                              "sha256": sf["sha256"], "bytes": sf["bytes"], "equals_eval_sha256": sf["sha256"] == ev["sha256"]},
            "generation": {"file": REFS["generation"], "note": "a GENERATION manifest: its eval_sha256 is the fork-source reference, not this run's output",
                           "eval_sha256": gc["eval_sha256"], "selected_attempt_kind": gc["selected_attempt_kind"],
                           "equals_launch_reference": gc["eval_sha256"] == st["reference"]["reference_sha256"]}},
        "fork_source": {k: st["reference"][k] for k in ("reference_sha256", "copy_rehash", "copy_rehash_equal", "references_from", "try")},
        "routing": "not observed: no simh-q4-rlr routing file in chain_run_state/routing at read time (17:2x PT); the staging manifest exists",
        "a_eval_complete": {k: st["target"][k] for k in ("has_header", "n_sample_files", "has_summaries", "has_reductions", "has_results_block",
                                                          "log_status", "sample_has_error", "sample_error_has_traceback", "last_event_type", "n_events", "n_error_events")}
        | {"control_clean_same_box": {"run": "2026-10-15_simh_q1_rlr|sim_honest@september/" + st["control"]["cell"],
                                      **{k: st["control"][k] for k in ("has_reductions", "has_results_block", "sample_has_error", "messages_by_role", "last_event_type")}},
           "reading": f"a finalized errored artifact: header, its one sample (ending on the terminal error event), summaries, log status success; no reductions/results (the clean control on the same box has both). The sample's messages are NOT captured beyond the user prompt ({s['messages_by_role']} against {b['tool_events_final']} tool events); the steps are in the events"},
        "b_misalignment_side": f"supports run side: the run itself errored (OutputContractError raised in the simulator at contract.py:{site}, last tool {d['last_tool_before_error']['tool']}). An independent reader (this one) sees the same mismatch: the errored sample carries only the user message while its events hold {b['tool_events_final']} tool events, which is the expected result of an events-vs-messages check on an errored sample. The staging row sha equals the .eval sha (the stager staged these exact bytes). The routing exclusion itself was not yet observed at read time",
        "inputs_sha256": {n[:-5]: sha(os.path.join(D, n)) for n in inputs + [st_name]},
        "scripts": {"simh_excl_evidence.py": "943b986e9", "simh_err_diag.py": "f53e0e6c6", "simh_census.py": "3d3cddbaa", "simh_eval_structure.py": "bf37cd397"},
    }
    rows[p] = row
    print(json.dumps({"cell": p, "site": site, "concordant": row["terminal_class"]["concordant"], "sha_eq_orig": row["message_sha_equals_original"],
                      "same_all_fields": row["message_hash"]["same_sha_on_both_fields_both_runs"], "staging_eq": row["manifest_rows"]["staging_chain"]["equals_eval_sha256"],
                      "gen_eq_ref": row["manifest_rows"]["generation"]["equals_launch_reference"], "roles": row["messages_by_role"], "tef": row["tool_events_final"],
                      "msg": row["message_sha256"][:16], "len": row["message_len"], "orig_site": row["original"]["raise_site"]}))

# write only after all three rows were built and validated
for p, row in rows.items():
    with open(os.path.join(OUT, f"row_simh_q4_rlr_{p}_2026-10-15.json"), "w") as f:
        json.dump(row, f, indent=1)
        f.write("\n")
