"""t88_model_registry's preflight evidence for 09 (ruling @ fb3d4649a: "never started"; commit the preflight cause, else
"never started, cause undocumented"). ON q3-xbox3, read-only, any python3; stdout is captured on the laptop:
  DEVBOX=aswin@i-0f2299f29dd296259 bash devbox.sh 'python3 - ' < ug_t88_evidence.py > <out.json>
Machine records and harness code only: the launcher's per-task preflight record (launch.json reference_state), the
preflight result JSON's per-mode fields, the harness line whose assertion fired (code text, with the file's sha256), and
the preflight log's sha256, line count and which lines carry the assertion (booleans only: the log is not copied, since
an attack-mode build can print approach text).
"""

import hashlib
import json
import re

RUN = "/home/aswin/phase2/runs/2026-10-15_ug"
KEY = "t88_model_registry"
HARNESS_TASKS = "/home/aswin/phase2/attack_check_ug/tasks.py"
MSG = "the registry's approach differs from September's"


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


launch = json.load(open(f"{RUN}/launch.json"))
ref = launch["reference_state"][KEY]
pre_rec = ref["preflight"]["september_ungrounded"]
pre_path = pre_rec["result"]
pre = json.load(open(pre_path))
rows = []
for r in pre.get("results", []):
    err = r.get("error") or ""
    rows.append({k: r.get(k) for k in ("mode", "built", "ungrounded", "reasoning_blocks_marked", "tool_outputs_marked",
                                       "output_marker_in_prompt", "reasoning_marker_in_prompt")}
                | {"error_class": (re.match(r"^\s*([A-Za-z_]\w*)", err).group(1) if err else None),
                   "error_is_harness_assertion": err == f'AssertionError("{MSG}")'})
src = open(HARNESS_TASKS).read().splitlines()
hits = [i for i, l in enumerate(src, 1) if MSG in l]
assert len(hits) == 1, "the assertion text is not exactly one harness line"
log = f"{RUN}/logs/preflight_{KEY}_september_ungrounded.log"
lines = open(log, errors="replace").read().splitlines()
run = [r for r in launch["runs"] if r["key"].endswith("/" + KEY)]
out = {
    "cell": KEY,
    "run_record": [{"key": r["key"], "state": r.get("state"), "n_attempts": len(r.get("attempts") or [])} for r in run],
    "reference_state": {"tries": ref.get("tries"), "state": ref.get("state"), "reused_from": ref.get("reused_from")},
    "launcher_preflight_record": {k: pre_rec.get(k) for k in ("state", "start", "end", "exit_code", "result")},
    "preflight_result": {"path": pre_path, "sha256": sha(pre_path), "failed": pre.get("failed"),
                         "simulator": pre.get("simulator"), "sims_src_sha256": pre.get("sims_src_sha256"), "rows": rows},
    "harness_assertion": {"file": HARNESS_TASKS, "sha256": sha(HARNESS_TASKS), "line": hits[0], "code": src[hits[0] - 1].rstrip()},
    "preflight_log": {"path": log, "sha256": sha(log), "lines": len(lines),
                      "lines_with_assertion_and_harness_message": [i for i, l in enumerate(lines, 1)
                                                                   if "AssertionError" in l and MSG in l]},
}
# the evidence must PROVE each claim, or the script fails (Codex r24): fixed messages only, no values
att = [r for r in rows if r["mode"] == "attack"]
hon = [r for r in rows if r["mode"] == "honest"]
assert len(run) == 1 and run[0].get("state") == "not run" and not run[0].get("attempts"), "run record: not exactly one not-run run"
assert pre_rec.get("state") == "failed" and pre_rec.get("exit_code") == 1, "launcher preflight record: not failed with exit 1"
assert len(att) == 1 and att[0]["built"] is False and att[0]["error_is_harness_assertion"], "attack row: not the harness assertion"
assert (len(hon) == 1 and hon[0]["built"] is True and hon[0]["ungrounded"] is True and hon[0]["output_marker_in_prompt"] == 0
        and hon[0]["reasoning_marker_in_prompt"] == 0 and (hon[0]["reasoning_blocks_marked"] or 0) > 0
        and (hon[0]["tool_outputs_marked"] or 0) > 0), "honest row: negative control not shown"
assert out["preflight_log"]["lines_with_assertion_and_harness_message"], "log: no line with the harness assertion"
print(json.dumps(out, indent=1))
