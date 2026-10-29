"""Deception commands, review check (local, free; Codex on #456): every honest calibration row's readout position in
the segment-end dump's meta, compared with q3's staging record, by (irid, call_index). Rows as the scoring took them:
counted attempts (no redrawn_because, attempt kind not "rerun"), kind "command", rule label impactful, a stored row.
Inputs are the dump's meta files (keys and positions only, copied from the Volume with `modal volume get`) and the
staging record; no transcript or activation is read.

  python3 experiments/2026-10-09_q2_deception_commands/check_honest_readouts.py <meta folder> <staging json>
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def main(meta: Path, staging: Path):
    st = {}
    for r in json.loads(staging.read_text())["calls"]:
        if r.get("kind") == "command":
            st.setdefault((r["irid"], r["call_index"]), []).append(r["readout"])
    n, per_arm = Counter(), Counter()
    for arm in ("honest_a", "honest_b"):
        for f in sorted((meta / arm).glob("*.json")):
            m = json.loads(f.read_text())
            if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
                continue
            for c in m["calls"]:
                if c.get("kind") != "command" or c.get("label") != "impactful" or c.get("row") is None:
                    continue
                s = st.get((c["irid"], c["call_index"]))
                k = "missing from staging" if s is None else "several staging rows" if len(s) != 1 else "matched" if s[0] == c["readout"] else "unequal"
                n[k] += 1; per_arm[f"{arm} | {k}"] += 1
    out = {"staging sha256": hashlib.sha256(staging.read_bytes()).hexdigest(), "honest rows": sum(n.values()), "counts": dict(n), "by arm": dict(per_arm)}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    res = main(Path(sys.argv[1]), Path(sys.argv[2]))
    if len(sys.argv) > 3:
        with open(sys.argv[3], "x") as fh:
            fh.write(json.dumps(res, indent=1) + "\n")
