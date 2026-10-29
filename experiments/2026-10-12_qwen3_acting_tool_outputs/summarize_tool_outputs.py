"""Print CHECK.md's numbers from a check_tool_outputs.py result file (json.load by key; absent keys are 0).

  uv run --no-project --python 3.14 python -I summarize_tool_outputs.py results/tool_outputs_<stamp>.json
"""

import json
import sys
from collections import Counter

d = json.load(open(sys.argv[1]))
R, P = d["runs_1009"], d["runs_1006_rerendered"]
T, TP = Counter(d["totals_1009"]), Counter(d["totals_1006_rerendered"])
with_msgs = {k: r for k, r in R.items() if r["counts"].get("steps", 0) > 0}
print("10-09 runs:", len(R), "with saved messages:", len(with_msgs), "stopped with no saved messages:", len(R) - len(with_msgs))
for launch in sorted({r["launch"] for r in R.values()}):
    rs = [r for r in R.values() if r["launch"] == launch]
    print(f"  {launch}: runs {len(rs)}, steps {sum(r['counts'].get('steps', 0) for r in rs)}")
for launch in sorted({r["launch"] for r in P.values()}):
    rs = [r for r in P.values() if r["launch"] == launch]
    print(f"  {launch}: runs {len(rs)}, steps {sum(r['counts'].get('steps', 0) for r in rs)}, models {sorted({m for r in rs for m in r['models']})}")
print("10-09 tool outputs in saved messages:", T["tool_outputs"])
for k in ("outputs_found_byte_exact", "outputs_not_found", "outputs_rejected_400", "outputs_rejected_cancelled", "outputs_no_record",
          "outputs_never_in_a_request", "outputs_never_in_a_request_is_last_message"):
    print(f"  {k}: {T[k]}")
print("checks (one per tool message per request with a 200 record, incl. retries and stopped runs):",
      T["checks_found_byte_exact"] + T["checks_not_found"], "found", T["checks_found_byte_exact"], "not found", T["checks_not_found"],
      "negative control found", T["negative_control_perturbed_needle_found"])
print("  sent equals the .eval tool message:", T["checks_sent_equals_eval_tool_message"], "differs:", T["checks_sent_differs_from_eval_tool_message"],
      "no .eval tool message (stopped runs):", T["checks_sent_without_eval_tool_message"])
print("  requests 200 with new tool messages:", T["requests_ok_with_new_tool_messages"], "span count equals new tool messages:",
      T["requests_ok_tool_response_spans_equal_new_tool_messages"])
print("  requests:", T["requests"], "rejected 400:", T["requests_rejected_400"], "cancelled only:", T["requests_rejected_cancelled"],
      "no record:", T["requests_no_record"], "more than one 200 record:", T["requests_with_more_than_one_ok_record"])
print("  empty outputs delivered:", T["outputs_delivered_empty"], "with Inspect's truncation notice:", T["outputs_delivered_with_inspect_truncation_notice"])
print("  tool errors:", T["tool_errors"], {k: v for k, v in T.items() if k.startswith("tool_errors_type")})
print("  str_replace:", T["str_replace_calls"], "with new_str:", T["str_replace_with_new_str"])
print("  str_replace from every model call's output:", T["model_output_str_replace_calls"], "with new_str:", T["model_output_str_replace_with_new_str"])
for k in ("editor_reads_or_edits", "editor_relative_path", "editor_missing_path_error", "editor_missing_path_error_relative_path",
          "editor_on_missing_path_first_time", "editor_on_path_told_missing_before", "editor_on_path_told_missing_before_and_still_missing",
          "editor_on_path_told_missing_by_previous_output"):
    print(f"  {k}: {T[k]}")
print("10-06 re-rendered: tool outputs", TP["tool_outputs"], "rendered", TP["tool_outputs_rendered"], "found", TP["rendered_found_byte_exact"],
      "empty", TP["empty_outputs"], "truncation notice", TP["inspect_truncation_notice"], "tool errors", TP["tool_errors"])
print("  str_replace:", TP["str_replace_calls"], "with new_str:", TP["str_replace_with_new_str"])
print("  str_replace from every model call's output:", TP["model_output_str_replace_calls"], "with new_str:", TP["model_output_str_replace_with_new_str"])
for k in ("editor_reads_or_edits", "editor_missing_path_error", "editor_on_missing_path_first_time", "editor_on_path_told_missing_before",
          "editor_on_path_told_missing_by_previous_output"):
    print(f"  {k}: {TP[k]}")
print("seeded example:", json.dumps(d["seeded_example"]))
print("git:", d["git"], "revision:", d["revision"])
