"""Per-run report for the Qwen3-32B native-window honest arm (coordinator's spec, 10-09). Read-only, $0.

  uv run --no-project --python 3.14 --with numpy --with boto3 --with pyyaml python -I report_qwen3_32b_native.py \
      <laptop copy of the run dir> <ps-onpol checkout> <out json>

<ps-onpol checkout> is a checkout of branch aswin/phase2-onpolicy-behavior at 99f97a6 (pushed; its PR is not open yet):
this imports experiments/2026-10-09_phase2_onpolicy_behavior/analyze_onpolicy_behavior.py from it, which labels calls
with src/probe_shift/calls.py (labeler fingerprint eb014307fd1aab31, on main) and PR #67's failure rule.

Per run: steps (agent turns with a non-submit call), ending (submitted, step limit, out of context, error...), the
real scorer's main-task score and the counted success (an out-of-context run counts as a failure), file views vs
edits (the labeler's primary category: file_read vs file_write and file_write_tmp), failed str_replace calls and
tool-error replies (PR #67's rule), and replies cut at the 8,192 cap. Then 2 raw lines from 3 runs drawn with seed
20261009. Counts only in the output file; quotes print to the console.
"""
import collections
import json
import random
import sys
import zipfile
from pathlib import Path

run_dir, onpol, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(onpol / "experiments" / "2026-10-09_phase2_onpolicy_behavior"))
import analyze_onpolicy_behavior as B  # noqa: E402  measure(), failure(), the labeler (eb014307)

launch = json.loads((run_dir / "launch.json").read_text())
rows, samples = [], {}
for r in launch["runs"]:
    counted = [a for a in r["attempts"] if not a.get("redrawn_because")]
    if not counted:
        rows.append({"arm": r["arm"], "task": r["task"], "ending": f"not started ({r['state']})"})
        continue
    a = counted[0] if r["arm"] == "honest_a" else counted[-1]
    evals = sorted((run_dir / Path(a["log_dir"]).relative_to(Path(a["log_dir"]).parents[1])).glob("*.eval"))
    if len(evals) != 1:
        rows.append({"arm": r["arm"], "task": r["task"], "ending": f"{len(evals)} .eval files ({r['state']})"})
        continue
    with zipfile.ZipFile(evals[0]) as z:
        (n,) = [x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")]
        s = json.loads(z.read(n))
    m, _, _ = B.measure(s, r)
    pcs = (a.get("outcome") or {}).get("policy_calls") or [{}]
    store = s.get("store") or {}
    ending = ("error: " + (s["error"].get("message") or "")[:60] if s.get("error")
              else "out of context" if store.get("policy_ended_by") == "out_of_context"
              else "submitted" if m["submitted"] else str(pcs[-1].get("ended")))
    score = (((s.get("scores") or {}).get("linux_scorer") or {}).get("value") or {}).get("main_task_success")
    res = {x.get("tool_call_id"): x for x in s["messages"] if x["role"] == "tool"}
    replaces = [(tc, res.get(tc.get("id"))) for x in s["messages"] if x["role"] == "assistant"
                for tc in x.get("tool_calls") or [] if tc["function"] == "text_editor"
                and (tc.get("arguments") or {}).get("command") == "str_replace"]
    cats = m["categories"]
    rows.append({"arm": r["arm"], "task": r["task"], "steps": m["steps"], "calls": m["calls"], "ending": ending,
                 "main_task_raw": score, "success_counted": score == "C" and ending != "out of context",
                 "views": cats.get("file_read", 0), "edits": cats.get("file_write", 0) + cats.get("file_write_tmp", 0),
                 "str_replace": len(replaces),
                 "str_replace_failed": sum(bool(o and B.failure("text_editor", o)) for _, o in replaces),
                 "tool_errors": round(m["fail_share"] * m["calls"]) if m["fail_share"] is not None else None,
                 "tool_error_share": m["fail_share"], "cap_hits": sum(p.get("cap_hits", 0) for p in pcs),
                 "out_of_context_prompt_tokens": store.get("policy_out_of_context_prompt_tokens")})
    samples[(r["arm"], r["task"])] = s

done = [x for x in rows if "steps" in x]
print(f"run {run_dir.name} | runs {len(rows)} | read {len(done)}")
print("endings:", dict(collections.Counter(x["ending"] for x in rows)))
print("main-task score (real scorer, raw):", dict(collections.Counter(str(x.get("main_task_raw")) for x in done)))
print(f"success counted (C and not out of context): {sum(x['success_counted'] for x in done)} of {len(rows)}")
print("runs with a reply cut at 8,192:", sum(bool(x["cap_hits"]) for x in done), "| cut replies:",
      sum(x["cap_hits"] for x in done))
med = lambda f: sorted(x[f] for x in done)[len(done) // 2] if done else None  # noqa: E731
print(f"median steps {med('steps')} | median views {med('views')} | median edits {med('edits')} | str_replace "
      f"{sum(x['str_replace'] for x in done)} (failed {sum(x['str_replace_failed'] for x in done)}) | tool errors "
      f"{sum(x['tool_errors'] or 0 for x in done)} of {sum(x['calls'] for x in done)} calls")
print("\nper run: arm task | steps | ending | score raw / counted | views edits | str_replace (failed) | tool errors "
      "| cap hits | out-of-context prompt")
for x in rows:
    if "steps" not in x:
        print(f"  {x['arm']:8s} {x['task']:22s} | {x['ending']}")
        continue
    print(f"  {x['arm']:8s} {x['task']:22s} | {x['steps']:3d} | {x['ending']:16s} | {x['main_task_raw']} / "
          f"{'C' if x['success_counted'] else '-'} | {x['views']:3d} {x['edits']:3d} | {x['str_replace']:2d} "
          f"({x['str_replace_failed']}) | {x['tool_errors']} | {x['cap_hits']} | {x['out_of_context_prompt_tokens']}")

rng = random.Random(20261009)
print("\nwhat it did: 2 raw lines from each of 3 runs drawn with seed 20261009 (run = arm/task, step = agent turn)")
for arm, task in rng.sample(sorted(samples), min(3, len(samples))):
    s = samples[(arm, task)]
    acts, step = [], 0
    for x in s["messages"]:
        if x["role"] != "assistant":
            continue
        tcs = [tc for tc in x.get("tool_calls") or [] if tc["function"] != "submit"]
        step += bool(tcs)
        for tc in tcs:
            a = tc.get("arguments") or {}
            act = (f"text_editor {a.get('command')} {a.get('path')}" if tc["function"] == "text_editor"
                   else f"{tc['function']}: {a.get('cmd', a.get('command', a))}")
            acts.append((step, " ".join(str(act).split())[:200]))
    for st, act in [acts[0], acts[rng.randrange(1, len(acts))]] if len(acts) > 1 else acts:
        print(f"  [{arm}/{task}, step {st}] {act!r}")
out.write_text(json.dumps({"run": run_dir.name, "rows": rows}, indent=1))
print(f"-> {out}")
