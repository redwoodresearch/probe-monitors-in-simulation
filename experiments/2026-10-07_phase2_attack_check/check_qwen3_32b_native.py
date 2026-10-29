"""Harness checks behind QWEN3_32B_NATIVE_RESULTS.md, from the raw .eval samples of a launch. Read-only, $0.

  uv run --no-project --python 3.14 python -I check_qwen3_32b_native.py <run dir copy> <out json>

Per run: how an out-of-context end happened (the last policy model event's server error: vLLM's 400 text and the prompt
length it states), the error kinds of failed str_replace calls, the longest streak of one identical action, the longest
streak of text_editor inserts, and the scorer's first explanation line. Counts and short error kinds in the output;
the explanation lines are the scorer's, not the model's.
"""
import collections
import itertools
import json
import re
import sys
import zipfile
from pathlib import Path

run_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
launch = json.loads((run_dir / "launch.json").read_text())
rows, kinds = [], collections.Counter()
for r in launch["runs"]:
    counted = [a for a in r["attempts"] if not a.get("redrawn_because")]
    if not counted:  # never started (the first launch's runs after the abort)
        continue
    a = counted[0] if r["arm"] == "honest_a" else counted[-1]
    (ev,) = sorted((run_dir / Path(a["log_dir"]).relative_to(Path(a["log_dir"]).parents[1])).glob("*.eval"))
    with zipfile.ZipFile(ev) as z:
        (n,) = [x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")]
        s = json.loads(z.read(n))
    att = s.get("attachments") or {}
    text = lambda v: att.get(v[13:], v) if isinstance(v, str) and v.startswith("attachment://") else v  # noqa: E731
    policy = [e for e in s.get("events") or [] if e.get("event") == "model" and "Qwen3-32B" in str(e.get("model"))]
    ended = (s.get("store") or {}).get("policy_ended_by")
    server_400 = None
    if ended == "out_of_context" and policy:
        resp = (policy[-1].get("call") or {}).get("response") or {}
        msg = str(text(resp.get("message")) if isinstance(resp, dict) else "")
        m = re.search(r"maximum context length is (\d+) tokens.*?requested (\d+) output tokens.*?at least (\d+) input", msg)
        server_400 = {"type": resp.get("type"), "code": resp.get("code"),
                      "window_reply_prompt": [int(x) for x in m.groups()] if m else None}
    res = {x.get("tool_call_id"): x for x in s["messages"] if x["role"] == "tool"}
    acts, fails = [], collections.Counter()
    for x in s["messages"]:
        for tc in x.get("tool_calls") or []:
            if tc["function"] == "submit":
                continue
            args = tc.get("arguments") or {}
            acts.append((tc["function"], args.get("command"), json.dumps(args, sort_keys=True)))
            err = (res.get(tc.get("id")) or {}).get("error")
            if args.get("command") == "str_replace" and err:
                k = re.sub(r"`.*", "`…", re.sub(r"/\S+", "<path>", (err.get("message") or "")))[:60]
                fails[k] += 1
    kinds.update(fails)

    def streak(pred) -> int:
        best = cur = 0
        for i, x in enumerate(acts):
            cur = cur + 1 if pred(i, x) else 0
            best = max(best, cur)
        return best

    same = max((len(list(g)) for _, g in itertools.groupby(acts)), default=0)
    inserts = streak(lambda i, x: x[1] == "insert")
    expl = str((((s.get("scores") or {}).get("linux_scorer") or {}).get("explanation")) or "").split("\n")
    rows.append({"arm": r["arm"], "task": r["task"], "ended": ended or "not out of context",
                 "server_400": server_400, "str_replace_failed": dict(fails), "longest_identical_streak": same,
                 "longest_insert_streak": inserts, "scorer_first_line": next((l for l in expl[1:] if l.strip()), "")[:120]})
ooc = [x for x in rows if x["ended"] == "out_of_context"]
print(f"run {run_dir.name} | runs {len(rows)}")
print(f"out of context: {len(ooc)}; ended by vLLM's 400: {sum(bool(x['server_400'] and x['server_400']['code'] == 400) for x in ooc)}; "
      f"(window, reply cap, prompt) seen: {sorted({tuple(x['server_400']['window_reply_prompt'] or ()) for x in ooc if x['server_400']})}")
print("failed str_replace error kinds (top 6):", kinds.most_common(6))
print("longest identical-action streaks, top 8:",
      sorted(((x["longest_identical_streak"], f"{x['arm']}/{x['task']}") for x in rows), reverse=True)[:8])
print("longest insert streak over all runs:", max(x["longest_insert_streak"] for x in rows))
out.write_text(json.dumps({"run": run_dir.name, "rows": rows, "str_replace_failure_kinds": dict(kinds)}, indent=1))
print(f"-> {out}")
