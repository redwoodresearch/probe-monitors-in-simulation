"""Item E (RULE.md): move the stage-1 gate's staged rows (stage_gate.py output) from each call's `call_end` reading to
the same call's `command` reading, the final token of the rule-1 command parameter. Keys and numbers only, $0, local.

Each staged row (x, chunk, row) joins to its meta call_end entry, then to the command entry of the same call (same
extraction, chunk, arm, irid, call_index); the new row is row_start + command.row. Stops if a row fails to join or a
location repeats. Order, roles, labels, weights, folds and scrambles are unchanged.

  python build_rows_command.py <rows json> <metas root: <x>/<arm>/*.json> <out json>
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def main(rows_path, metas_root, out_path):
    raw = Path(rows_path).read_bytes()
    spec = json.loads(raw)
    ce, cmd = {}, {}
    for f in sorted(Path(metas_root).glob("*/*/*.json")):
        x, arm, m = f.parent.parent.name, f.parent.name, json.loads(f.read_text())
        for c in m["calls"]:
            k = (x, m["chunk"], arm, c.get("irid"), c.get("call_index"))
            if c.get("kind", "command") == "command" and c.get("row") is not None:
                assert k not in cmd, f"repeated command entry {k}"
                cmd[k] = (m["row_start"] + c["row"], c["readout"])
            elif c.get("kind") == "call_end" and c.get("row") is not None:
                loc = (x, m["chunk"], m["row_start"] + c["row"])
                assert loc not in ce, f"repeated call_end location {loc}"
                ce[loc] = (k, c["readout"])
    rows, counts = [], Counter()
    for r in spec["rows"]:
        k, ro = ce[(r["x"], r["chunk"], r["row"])]
        new, cro = cmd[k]
        same = ro == cro
        counts[(r["role"], "same_token" if same else ("call_end_after" if ro > cro else "call_end_before"))] += 1
        rows.append({**r, "row": new, "call_end_row": r["row"], "same_token": same})
    locs = [(r["x"], r["chunk"], r["row"]) for r in rows]
    assert len(set(locs)) == len(locs), "two staged calls map to one command reading"
    out = {
        **spec,
        "rows": rows,
        "endpoint": "command",
        "source_rows_sha256": hashlib.sha256(raw).hexdigest(),
        "remap": {f"{a}|{b}": n for (a, b), n in sorted(counts.items())},
    }
    Path(out_path).write_text(json.dumps(out))
    print(json.dumps(out["remap"], indent=1), "\nsame-token rows", sum(r["same_token"] for r in rows), "of", len(rows))


if __name__ == "__main__":
    main(*sys.argv[1:])
