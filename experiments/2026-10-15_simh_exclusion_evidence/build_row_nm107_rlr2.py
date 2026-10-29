import hashlib, json, os, sys
O = sys.argv[1]; CC = sys.argv[2]
L = lambda n: json.load(open(os.path.join(O, n)))
sha = lambda n: hashlib.sha256(open(os.path.join(O, n), "rb").read()).hexdigest()
lst, box, cen, dg, st = L("list_simh_nm_s4d_rlr2.json"), L("box_simh_nm_s4d_rlr2_b16.json"), L("census_simh_nm_s4d_rlr2_b16.json"), L("diag_simh_nm_s4d_rlr2_nm107.json"), L("structure_ref_nm107_rlr2.json")
hr, ho = L("hash_nm107_rlr2.json"), L("hash_nm107_orig.json")
sm, gm, rt, cc = L("staging_manifest.json"), L("gen_manifest.json"), L("routing_simh-nm-s4d-rlr2.json"), json.load(open(CC))
assert len(lst["runs"]) == 1 and len(box["cells"]) == 1 and len(dg["cells"]) == 1 and len(cen["cells"]) == 1
b, d, c = box["cells"][0], dg["cells"][0], cen["cells"][0]["census"]
ev = [e for e in b["evals"] if e["final"]]; assert len(ev) == 1; ev = ev[0]; s = ev["samples"][0]
assert ev["path"] == b["final_eval"] and d["eval_sha256"] == ev["sha256"]
att = b["attempts"]
run_id = f"{os.path.basename(box['run_dir'])}|{b['run']}"
orig = [r for r in cc["runs"] if r["run_id"] == "2026-10-15_simh_nm_s4d|sim_honest@september/nm107_media_processing"]; assert len(orig) == 1; orig = orig[0]
same344 = sorted({r["message_sha256"] for r in cc["runs"] if r["raise_site"].startswith("contract.py:344")})
assert len(sm["files"]) == 1 and len(gm["cells"]) == 1
sf, gc = sm["files"][0], gm["cells"][0]
df = d["deepest_frame"]
classes = {"attempt_record": [x for a in att for x in a["sample_errors"]], "eval_sample": s["error_class"], "census": [r["class"] for r in c["rows"]]}
excl = rt["staging_exclusions"]
row = {
 "label": "replacement (rlr2) superseding simh_nm_s4d | nm107",
 "run_id": run_id, "box": "box1 (the S4d_rlr2 generation manifest's host_box)",
 "exit_code": [a["exit_code"] for a in att], "clean": [a["clean"] for a in att], "policy_called": [a["policy_called"] for a in att],
 "n_attempts": b["n_attempts"], "attempt_kinds": [a["kind"] for a in att], "start_end": [[a["start"], a["end"]] for a in att],
 "policy_steps": [p["steps"] for a in att for p in a["policy_call_summaries"]],
 "terminal_class": {**classes, "census_ref": c["ref"], "census_bin": [r["bin"] for r in c["rows"]],
   "concordant": len({*classes["attempt_record"], classes["eval_sample"], *classes["census"]}) == 1},
 "raise_site": f"{df['file']}:{df['line']} {df['func']}", "raise_source_line": d["raise_source_line"].strip(), "message_withheld": d["message_withheld"],
 "message_sha256": d["message_sha256"], "message_len": d["message_len"],
 "original": {"run_id": orig["run_id"], "raise_site": orig["raise_site"], "message_sha256": orig["message_sha256"], "message_len": orig["message_len"],
   "source": "experiments/2026-10-15_simh_exclusion_evidence/results/contract_census_2026-10-15.json @ b63c73cd7"},
 "message_sha_equals_original": d["message_sha256"] == orig["message_sha256"],
 "distinct_message_sha256_over_5_344_originals": same344,
 "traceback_location": "sample.error.traceback in samples/<file>.json, read with inspect_ai read_eval_log (the frames above are parsed from it)",
 "sample_error_has_traceback": st["target"]["sample_error_has_traceback"],
 "last_tool_before_error": d["last_tool_before_error"], "tool_name_counts": d["tool_name_counts"],
 "messages_by_role": s["messages_by_role"], "tool_events_final": b["tool_events_final"], "tool_steps_at_failure": d["tool_steps_before_error"],
 "eval": {"path": ev["path"], "sha256": ev["sha256"], "bytes": ev["size"]},
 "manifest_rows": {
  "staging_chain": {"file": "experiments/2026-10-14_q3_overnight_chain/config/manifest_simh_nm360_s4d_sim_rlr2.json @ ae15b0d09", "launch": sf["launch"], "rel_tail": "/".join(sf["path"].split("/")[-3:]),
     "sha256": sf["sha256"], "bytes": sf["bytes"], "equals_eval_sha256": sf["sha256"] == ev["sha256"]},
  "generation_21d6f3079": {"file": "experiments/2026-10-15_q3_simhonest_311/results/manifest_simh_nm_s4d_rlr2_2026-10-15_165010.json @ 21d6f3079",
     "note": "a GENERATION manifest: its eval_sha256 is the fork-source reference, not this run's output", "host_box": gm["host_box"],
     "eval_sha256": gc["eval_sha256"], "selected_attempt_kind": gc["selected_attempt_kind"], "equals_launch_reference": gc["eval_sha256"] == st["reference"]["reference_sha256"]}},
 "fork_source": {k: st["reference"][k] for k in ("reference_sha256", "copy_rehash", "copy_rehash_equal", "references_from", "try")},
 "routing": {"file": "chain_run_state/routing/simh-nm-s4d-rlr2__simh_nm360_s4d_sim_rlr2.json (17:09, not in git)", "sha256": sha("routing_simh-nm-s4d-rlr2.json"),
   "n_staging_exclusions": len(excl), "reasons": sorted(set(excl.values())), "key_is_this_eval": [k.split("/")[0] + "|" + "/".join(k.split("/")[1:3]) == run_id and k.split("/")[3].split("#")[0] == os.path.basename(ev["path"]) for k in excl]},
 "message_hash": {
   "method": hr["hash_method"],
   "replacement": {"launch_json_field": "2026-10-15_simh_nm_s4d_rlr2/launch.json runs[key=sim_honest@september/nm107_media_processing].attempts[i].outcome.sample_errors[j]",
     "launch_json_values": hr["cells"][0]["launch_json_outcome_sample_errors"],
     "eval_field": "final .eval sample error.message (inspect_ai read_eval_log)", "eval_value": hr["cells"][0]["eval_sample_error_message"],
     "eval_sha256": hr["cells"][0]["eval_sha256"], "launch_json_equals_eval": hr["cells"][0]["last_attempt_sample_errors0_equals_eval_message"]},
   "original": {"launch_json_field": "2026-10-15_simh_nm_s4d/launch.json (box4) runs[key=sim_honest@september/nm107_media_processing].attempts[i].outcome.sample_errors[j]",
     "launch_json_values": ho["cells"][0]["launch_json_outcome_sample_errors"],
     "eval_field": "final .eval sample error.message (inspect_ai read_eval_log)", "eval_value": ho["cells"][0]["eval_sample_error_message"],
     "eval_sha256": ho["cells"][0]["eval_sha256"], "launch_json_equals_eval": ho["cells"][0]["last_attempt_sample_errors0_equals_eval_message"]},
   "census_field": "contract_census @ b63c73cd7 message_sha256 = simh_err_diag.py's sha256(sample.error.message UTF-8), i.e. the eval_field",
   "same_sha_on_both_fields_both_runs": len({hr["cells"][0]["eval_sample_error_message"]["sha256"], ho["cells"][0]["eval_sample_error_message"]["sha256"],
       *[v["sha256"] for v in hr["cells"][0]["launch_json_outcome_sample_errors"] + ho["cells"][0]["launch_json_outcome_sample_errors"]]}) == 1},
 "a_eval_complete": {k: st["target"][k] for k in ("has_header", "n_sample_files", "has_summaries", "has_reductions", "has_results_block", "log_status", "sample_has_error", "sample_error_has_traceback", "last_event_type", "n_events", "n_error_events")}
   | {"control_clean_same_box": {"run": "2026-10-15_simh_nm_s1|sim_honest@september/" + st["control"]["cell"], **{k: st["control"][k] for k in ("has_reductions", "has_results_block", "sample_has_error", "messages_by_role", "last_event_type")}},
      "reading": "a finalized errored artifact: header, its one sample (ending on the terminal error event), summaries, log status success; no reductions/results (the clean control on the same box has both). The sample's messages are NOT captured beyond the user prompt ({user: 1} against 20 tool events); the steps are in the events"},
 "b_misalignment_side": "run side: the run itself errored (OutputContractError raised in the simulator at contract.py:344, last tool text_editor); the errored sample carries only the user message while its events hold the 20 tool steps, which is what the stager's events-vs-messages check flags; the stager read the file as written (its row sha equals the .eval sha), so not a stager mis-parse",
 "inputs_sha256": {n[:-5]: sha(n) for n in ("list_simh_nm_s4d_rlr2.json", "box_simh_nm_s4d_rlr2_b16.json", "census_simh_nm_s4d_rlr2_b16.json", "diag_simh_nm_s4d_rlr2_nm107.json", "structure_ref_nm107_rlr2.json", "hash_nm107_rlr2.json", "hash_nm107_orig.json")},
 "scripts": {"simh_excl_evidence.py": "943b986e9", "simh_err_diag.py": "f53e0e6c6", "simh_census.py": "3d3cddbaa", "simh_eval_structure.py": "this commit"},
}
print(json.dumps(row, indent=1))
