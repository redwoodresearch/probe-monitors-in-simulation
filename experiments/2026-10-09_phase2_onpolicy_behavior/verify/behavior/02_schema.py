import json, collections
from pathlib import Path
from probe_shift.viewer.pairs import read_launch, pairs, read_sample, steps
RD = Path.home() / "phase2/runs/2026-10-09_000523_rerun_strict"
P = pairs(RD)
p = P[0]
for side in ("sim", "real"):
    s = read_sample(p[side]["eval"])
    print(side, p[side]["key"], "sample keys:", sorted(s.keys()))
    print("  scores:", json.dumps(s.get("scores"))[:600])
    print("  error:", str(s.get("error"))[:200], "limit:", s.get("limit"))
    print("  output keys:", sorted((s.get("output") or {}).keys()))
    pr, st = steps(s)
    print("  n prompt msgs:", len(pr), "n steps:", len(st))
    roles = collections.Counter(m["role"] for m in s["messages"])
    print("  roles:", roles)
    a = st[0][0]
    print("  assistant msg keys:", sorted(a.keys()))
    tc = a.get("tool_calls") or []
    print("  tool_calls[0] keys:", sorted(tc[0].keys()) if tc else None)
    if tc: print("  fn:", tc[0].get("function"), "arg keys:", sorted((tc[0].get("arguments") or {}).keys()))
    tools = [m for m in s["messages"] if m["role"]=="tool"]
    if tools:
        print("  tool msg keys:", sorted(tools[0].keys()))
        print("  tool error field examples:", [t.get("error") for t in tools[:5]])
    # last step
    last = st[-1]
    print("  last step roles:", [m["role"] for m in last], "last fn:", [c.get("function") for c in (last[0].get("tool_calls") or [])])
    print("  metadata keys:", sorted((s.get("metadata") or {}).keys())[:30])
