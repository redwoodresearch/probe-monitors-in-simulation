"""One evidence row for a staging-excluded sim-ATTACK run: keys, counts, classes, allowlisted frames and shas only.
Usage: python3 -I build_row_attack_excl.py <reads dir> <manifest json> <manifest ref> <routing json> <launch> <cell> <label> <out json>"""
import hashlib, json, os, sys


class RowError(Exception):
    pass


def need(cond, what):
    if not cond:
        raise RowError(what)

D, MAN, MREF, RT, LAUNCH, CELL, LABEL, OUT = sys.argv[1:9]
L = lambda p: json.load(open(p))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
lr, st, hs, man, rt = (L(os.path.join(D, "gap_launch_read.json")), L(os.path.join(D, "struct_nm242.json")),
                       L(os.path.join(D, "hash_nm242.json")), L(MAN), L(RT))
run = [r for r in lr["runs"] if r["key"] == f"sim_attack@september/{CELL}"]
t = [x for x in st["targets"] if x["cell"] == CELL]
h = [x for x in hs["cells"] if x["cell"] == CELL]
mrow = [f for f in man["files"] if f["launch"] == LAUNCH and f["path"].split("/")[-2] == CELL]
ex = [(k, v) for k, v in rt["staging_exclusions"].items() if k.split("/")[-2] == CELL]
need(len(run) == len(t) == len(h) == len(mrow) == len(ex) == 1, "joins are not exactly one")
need(st["targets"][0]["run_dir"].rstrip("/").endswith("/" + LAUNCH) and h[0]["run_dir"].rstrip("/").endswith("/" + LAUNCH), "run_dir is not the launch")
k0 = ex[0][0]
need(k0.split("/")[-3] == "sim_attack@september" and f"/{LAUNCH}/" in "/" + k0 and k0.endswith("#e0"), "routing key is not this launch/arm/epoch 0")
need(ex[0][1] == "attack epoch has no system prompt and no assistant messages", "unexpected routing reason")
need(h[0]["last_attempt_sample_errors0_equals_eval_message"], "launch.json and .eval error messages differ")
need(MREF.endswith(" @ 5cbd3354c8") or " @ " in MREF, "manifest ref")
run, t, h, mrow, (ekey, ereason) = run[0], t[0], h[0], mrow[0], ex[0]
need(t["arm"] == "sim_attack@september" and ekey.split("/")[-1].split("#")[0] == mrow["path"].split("/")[-1], "arm or file mismatch")
a = run["attempts"]
errored = t["sample_has_error"] and t["sample_error_has_traceback"] and t["last_event_type"] == "error" and t["n_error_events"] == 1
row = {"label": LABEL, "run_id": f"{LAUNCH}|sim_attack@september/{CELL}", "arm": "sim_attack (sim2)",
       "routing_reason_is_attack_epoch_literal": ereason == "attack epoch has no system prompt and no assistant messages",
       "attempts": len(a), "attempt_kinds": [x["kind"] for x in a], "exit_code": [x["exit_code"] for x in a], "clean": [x["clean"] for x in a],
       "policy_called": [x["policy_called"] for x in a], "n_sample_errors": [x["n_sample_errors"] for x in a],
       "reference_state": lr["reference_state"].get(CELL),
       "error_class": t["sample_error_class"], "raise_site": t["deepest_frame"],
       "message_sha256": h["eval_sample_error_message"]["sha256"], "message_len": h["eval_sample_error_message"]["len"],
       "launch_json_equals_eval_message": h["last_attempt_sample_errors0_equals_eval_message"],
       "messages_by_role": t["messages_by_role"], "tool_events_final": t["events_by_type"].get("tool", 0),
       "model_events": t["events_by_type"].get("model", 0), "n_events": t["n_events"], "last_event_type": t["last_event_type"],
       "eval": {"sha256": h["eval_sha256"], "manifest_row_sha256": mrow["sha256"], "equal": h["eval_sha256"] == mrow["sha256"], "manifest": MREF},
       "eval_complete": {k: t[k] for k in ("log_status", "n_sample_files", "has_reductions", "has_results_block", "sample_has_error", "sample_error_has_traceback", "member_kinds")},
       "errored_or_empty": ("SAMPLE-ERRORED run (log status success, exit 0), not an empty epoch: the sample ends on a terminal error event with a traceback and its events hold the steps; the errored sample records only the user message, so the stager sees no system prompt and no assistant messages"
                            if errored and t["messages_by_role"] == {"user": 1} else "NOT the errored pattern: see fields"),
       "inputs_sha256": [{"file": n, "sha256": sha(os.path.join(D, n))} for n in sorted(os.listdir(D)) if not n.startswith("row_")]
       + [{"file": os.path.basename(MAN), "sha256": sha(MAN)}, {"file": os.path.basename(RT), "sha256": sha(RT)}]}
need(row["eval"]["equal"], ".eval sha != manifest row")
json.dump(row, open(OUT, "w"), indent=1)
print(json.dumps({k: row[k] for k in ("attempts", "exit_code", "clean", "error_class", "raise_site", "messages_by_role", "tool_events_final", "errored_or_empty")})[:600])
