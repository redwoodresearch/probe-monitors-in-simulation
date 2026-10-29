"""Share of old Qwen's file edits (str_replace/insert) on a path it never viewed or cat'ed earlier in the run; same for new Qwen's rerun is not available locally."""
import json, re, sys
from pathlib import Path
from probe_shift.viewer.pairs import read_sample
A = Path(sys.argv[1])
runs = sorted(A.glob("2026-10-09_123140_qwen3_32b_native_relaunch/honest_[ab]/*/*.eval")) + sorted(A.glob("2026-10-09_121200_qwen3_32b_native/honest_[ab]/t17_*/*.eval"))
tot = blind = runs_with_edits = runs_all_blind = 0
for p in runs:
    seen, e, b = set(), 0, 0
    for m in read_sample(p)["messages"]:
        for c in (m.get("tool_calls") or []) if m["role"] == "assistant" else []:
            a = c.get("arguments") or {}
            if c.get("function") == "text_editor":
                path, cmd = a.get("path"), a.get("command")
                if cmd == "view": seen.add(path)
                elif cmd in ("str_replace", "insert"):
                    e += 1; b += path not in seen
                elif cmd == "create": seen.add(path)
            elif c.get("function") == "bash":
                for tok in re.findall(r"(/[\w./-]+)", str(a.get("cmd", ""))): seen.add(tok)
    tot += e; blind += b; runs_with_edits += e > 0; runs_all_blind += (e > 0 and b == e)
print(f"edits {tot}; on a never-viewed path {blind} ({blind/max(tot,1):.0%}); runs with edits {runs_with_edits}; runs where every edit was blind {runs_all_blind}")
