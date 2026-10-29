"""Old Qwen (Qwen3-32B) honest runs: per-run score, ending, steps and longest run of identical tool calls, from raw .eval logs."""
import collections, json, sys
from pathlib import Path
from probe_shift.viewer.pairs import read_sample, steps
A = Path(sys.argv[1])
runs = sorted(p for p in A.glob("2026-10-09_123140_qwen3_32b_native_relaunch/honest_[ab]/*/*.eval")) + \
       sorted(p for p in A.glob("2026-10-09_121200_qwen3_32b_native/honest_[ab]/t17_*/*.eval"))
rows = []
for p in runs:
    s = read_sample(p)
    sc = s.get("scores") or {}
    val = {k: (v.get("value") if isinstance(v, dict) else v) for k, v in sc.items()}
    mts = None
    for v in val.values():
        if isinstance(v, dict) and "main_task_success" in v: mts = v["main_task_success"]
    _, st_ = steps(s)
    calls = [(c.get("function"), json.dumps(c.get("arguments"), sort_keys=True)) for m in s["messages"] if m["role"] == "assistant" for c in (m.get("tool_calls") or [])]
    longest, cur = 0, 0
    for i, c in enumerate(calls):
        cur = cur + 1 if i and c == calls[i - 1] else 1; longest = max(longest, cur)
    last = next((m for m in reversed(s["messages"]) if m["role"] == "assistant"), None)
    submitted = bool(last and any(c.get("function") == "submit" for c in (last.get("tool_calls") or [])))
    out = s.get("output") or {}
    stop = (out.get("stop_reason") or (out.get("choices") or [{}])[0].get("stop_reason"))
    err = s.get("error"); lim = s.get("limit")
    meta = s.get("metadata") or {}
    main = meta.get("main_task_name") or (meta.get("main_task") or {}).get("id") if isinstance(meta.get("main_task"), dict) else meta.get("main_task_name") or meta.get("main_task")
    non_submit = [c for c in calls if c[0] != "submit"]
    rows.append(dict(run=f"{p.parts[-3]}/{p.parts[-2]}", main=str(main)[:40], mts=mts, steps=len(st_), calls=len(calls), actions_before_submit=len(non_submit),
                     submitted=submitted, stop=stop, error=(str(err)[:80] if err else None), limit=lim, longest_repeat=longest))
for r in rows: print(r)
print("runs", len(rows), "| main_task_success", collections.Counter(r["mts"] for r in rows))
print("submitted", sum(r["submitted"] for r in rows), "| stop reasons", collections.Counter(r["stop"] for r in rows), "| errors", collections.Counter(r["error"] for r in rows), "| limits", collections.Counter(str(r["limit"]) for r in rows))
print("submitted with no other action:", sum(r["submitted"] and r["actions_before_submit"] == 0 for r in rows))
print("runs with a repeat streak >= 36:", sum(r["longest_repeat"] >= 36 for r in rows), sorted(r["longest_repeat"] for r in rows)[-8:])
print("tasks:", len({r["run"].split("/")[1] for r in rows}), "| main tasks:", sorted({r["main"] for r in rows})[:25])
