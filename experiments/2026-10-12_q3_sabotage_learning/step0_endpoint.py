"""Step 0 ($0, metadata only): do the sabotage gate's call_end rows read the rule-1 command span's final token?
Joins every staged row of rows_v2.json (x, chunk, row) to the meta call_end entry at that row, then to the same
call's `command` entry (same irid, call_index), and compares token positions (`readout`). No text is read."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

rows_path, metas_root = sys.argv[1], Path(sys.argv[2])
rows = json.loads(Path(rows_path).read_text())["rows"]
idx, cmd = {}, {}
for xdir in sorted(metas_root.iterdir()):
    for f in sorted(xdir.glob("*/*.json")):
        m = json.loads(f.read_text())
        arm = f.parent.name
        for c in m["calls"]:
            k = (arm, c.get("irid"), c.get("call_index"))
            if c.get("kind", "command") == "command":
                cmd[(xdir.name, m["chunk"]) + k] = c
        for c in m["calls"]:
            if c.get("kind") == "call_end" and c.get("row") is not None:
                idx[(xdir.name, m["chunk"], m["row_start"] + c["row"])] = (
                    arm,
                    c,
                    cmd.get((xdir.name, m["chunk"], arm, c.get("irid"), c.get("call_index"))),
                )
out = defaultdict(Counter)
by_name = defaultdict(Counter)
for r in rows:
    hit = idx.get((r["x"], r["chunk"], r["row"]))
    role = r["role"] + ("" if r["y"] is None else f"/y={r['y']}")
    if hit is None:
        out[role]["no_call_end_entry"] += 1
        continue
    arm, ce, co = hit
    if co is None:
        out[role]["no_command_entry"] += 1
        continue
    if co.get("row") is None or co.get("readout") is None:
        out[role]["command_entry_without_row"] += 1
        continue
    same = ce["readout"] == co["readout"]
    after = ce["readout"] > co["readout"]
    out[role]["same_token" if same else ("call_end_after_command_end" if after else "call_end_before_command_end")] += 1
    by_name[(ce.get("name"), ce.get("param"), co.get("param"))]["same" if same else "differs"] += 1
res = {
    "rows": len(rows),
    "by_role": {k: dict(v) for k, v in sorted(out.items())},
    "by_tool_param": {
        f"{a}|call_end_param={b}|command_param={c}": dict(v)
        for (a, b, c), v in sorted(by_name.items(), key=lambda kv: str(kv[0]))
    },
}
print(json.dumps(res, indent=1))
