"""Rows for 09 on the sel97 add-on (ATTACK-arm launch 2026-10-15_il_sel97_addon): (1) the 2 selected cells that never ran
(t96, t151) and (2) the 2 staging-excluded sim2 runs (t188, t126). Keys, counts, classes and shas only.
Usage: python3 -I build_row_sel97.py <dir with the reads> <scratch dir with manifest/routing/package/gate> <out json>"""
import hashlib, json, os, sys

D, S, OUT = sys.argv[1:4]
L = lambda p: json.load(open(p))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
lr = L(os.path.join(D, "sel97_launch_read.json"))
st = {t["cell"]: t for t in L(os.path.join(D, "struct_sim2_t188_t126.json"))["targets"]}
hs = {c["cell"]: c for c in L(os.path.join(D, "hash_sim2_t188_t126.json"))["cells"]}
ref = L(os.path.join(D, "il_refindex_old_t96_t151.json"))
rows_status = L(os.path.join(D, "sel97_status_rows.json"))["rows"]
man = L(os.path.join(S, "manifest_ilold_oldsel_sim2.json"))
rt = L(os.path.join(S, "routing_il-oldsel_sim2.json"))
pkg = L(os.path.join(S, "tasks_il_sel97.json"))
gate = L(os.path.join(S, "sel97_donor_gate.json"))
runs = {r["key"]: r for r in lr["runs"]}

never = {}
for t in ("t96_file_indexing", "t151_model_registry"):
    g = [c for c in gate["cells"] if c["key"] == t]
    never[t] = {"in_95_cell_package": any(t in json.dumps(c) for c in pkg["cells"]),
                "launch_runs_for_key": [k for k in runs if k.endswith("/" + t)], "attempts": 0 if not any(k.endswith("/" + t) for k in runs) else None,
                "launch_reference_state": lr["reference_state"].get(t), "launch_note": "not in the launch task file: no run, no reference_state and no preflight entry in launch.json, so 0 attempts in either arm",
                "status_rows": [r["key"] for r in rows_status if r["key"].endswith("|" + t)],
                "il_refindex_old_pin": ref["refs_t96" if t.startswith("t96") else "refs_t151"],
                "donor_gate": ({"in_gate": True, "a1": g[0]["a1"], "b": g[0]["b"], "pass": g[0]["pass"]} if g else {"in_gate": False})}
    assert never[t]["launch_runs_for_key"] == [] and not never[t]["in_95_cell_package"] and len(never[t]["status_rows"]) == 2

excl = {}
for k, reason in rt["staging_exclusions"].items():
    cell, fname = k.split("/")[-2], k.split("/")[-1].split("#")[0]
    r = runs[f"sim_attack@september/{cell}"]
    a = r["attempts"]
    s, h = st[cell], hs[cell]
    mrow = [f for f in man["files"] if f["launch"] == "runs/2026-10-15_il_sel97_addon" and f["path"].split("/")[-2] == cell]
    assert len(mrow) == 1 and mrow[0]["path"].split("/")[-1] == fname
    excl[cell] = {
        "run_key": f"2026-10-15_il_sel97_addon|sim_attack@september/{cell}", "arm": "sim_attack (sim2)",
        "routing_reason_is_attack_epoch_literal": reason == "attack epoch has no system prompt and no assistant messages",
        "attempts": len(a), "attempt_kinds": [x["kind"] for x in a], "exit_code": [x["exit_code"] for x in a], "clean": [x["clean"] for x in a],
        "policy_called": [x["policy_called"] for x in a], "n_sample_errors": [x["n_sample_errors"] for x in a],
        "error_class": s["sample_error_class"], "raise_site": s["deepest_frame"],
        "message_sha256": h["eval_sample_error_message"]["sha256"], "message_len": h["eval_sample_error_message"]["len"],
        "launch_json_equals_eval_message": h["last_attempt_sample_errors0_equals_eval_message"],
        "messages_by_role": s["messages_by_role"], "tool_events_final": s["events_by_type"].get("tool", 0),
        "model_events": s["events_by_type"].get("model", 0), "n_events": s["n_events"], "last_event_type": s["last_event_type"],
        "eval": {"sha256": h["eval_sha256"], "manifest_row_sha256": mrow[0]["sha256"], "equal": h["eval_sha256"] == mrow[0]["sha256"]},
        "eval_complete": {k2: s[k2] for k2 in ("log_status", "n_sample_files", "has_reductions", "has_results_block", "sample_has_error", "sample_error_has_traceback", "member_kinds")},
        "errored_or_empty": "SAMPLE-ERRORED run (log status success, exit 0): the sample carries a terminal error event with a traceback; its events hold the steps (model and tool events), but the errored sample records only the user message, so the stager sees no system prompt and no assistant messages. Not an empty epoch.",
    }
    assert excl[cell]["eval"]["equal"] and s["sample_has_error"] and s["messages_by_role"] == {"user": 1}
    assert s["arm"] == "sim_attack@september" and s["sample_error_has_traceback"] and s["last_event_type"] == "error" and s["n_error_events"] == 1
assert set(excl) == {"t188_clinical_trial", "t126_file_indexing"} and set(st) == set(excl)
out = {"what": "sel97 add-on (ATTACK arms; keys, counts, classes, shas only): the 2 never-run selected cells and the 2 staging-excluded sim2 runs",
       "launch": {"dir": "xbox1 /home/aswin/phase2/runs/2026-10-15_il_sel97_addon/launch.json (jq read, results/remote_jq.sh)", "finished": lr["finished"], "n_runs": lr["n_runs"], "arms": lr["arms"]},
       "sources": {"package": "experiments/2026-10-15_q3_interleaved_run3_sim2/config/addon/tasks_il_sel97.json @ 89595bff5 (95 cells)",
                   "status_rows": "…/config/addon/sel97_status_rows.json @ 89595bff5",
                   "donor_gate": "experiments/2026-10-14_q3_overnight_chain/results/hygiene/sel97_donor_gate_2026-10-15.json @ 04bdf57ba",
                   "il_refindex_old": "xbox1 /home/aswin/phase2/il_refindex_old/launch.json references (jq read)",
                   "staging_manifest": "experiments/2026-10-14_q3_overnight_chain/config/manifest_ilold_oldsel_sim2.json @ 5c03a303cc",
                   "routing": "chain_run_state/routing/il-oldsel__ilold_oldsel_sim2.json (18:45, not in git; sha256 below)",
                   "eval_reads": "simh_eval_structure.py struct1/hash with STRUCT_ARM=sim_attack@september, on xbox1 (the box .eval sha equals the staged manifest row)"},
       "n_package_cells": len(pkg["cells"]), "donor_gate": {"n": len(gate["cells"]), "n_pass": sum(c["pass"] for c in gate["cells"])},
       "never_ran": never, "staging_excluded_sim2": excl,
       "inputs_sha256": [{"file": n, "sha256": sha(os.path.join(D, n))} for n in sorted(os.listdir(D)) if n.endswith(".json") and not n.startswith("row_")]
       + [{"file": n, "sha256": sha(os.path.join(S, n))} for n in ("manifest_ilold_oldsel_sim2.json", "routing_il-oldsel_sim2.json", "tasks_il_sel97.json", "sel97_donor_gate.json")]}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({"never": {k: (v["in_95_cell_package"], v["attempts"], v["donor_gate"], bool(v["il_refindex_old_pin"])) for k, v in never.items()},
                  "excl": {k: (v["error_class"], v["raise_site"], v["messages_by_role"], v["tool_events_final"], v["eval"]["equal"]) for k, v in excl.items()}}))
