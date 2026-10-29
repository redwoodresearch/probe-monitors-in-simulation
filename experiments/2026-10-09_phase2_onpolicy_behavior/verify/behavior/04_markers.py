import json, collections, re
from pathlib import Path
from probe_shift.viewer.pairs import read_launch, read_sample, _log
RD = Path.home() / "phase2/runs/2026-10-09_000523_rerun_strict"
runs = read_launch(RD)["runs"]
pats = {
 "exit_code_word": r"(?i)exit code|exit status|returncode|return code",
 "traceback": r"Traceback \(most recent call last\)",
 "Error_word": r"\b\w*Error\b",
 "error_colon": r"(?i)\berror:",
 "no_such_file": r"No such file or directory",
 "cmd_not_found": r"command not found",
 "perm_denied": r"Permission denied",
 "failed_word": r"(?i)\bfailed\b",
 "pw_auth_failed": r"password authentication failed",
 "modnotfound": r"ModuleNotFoundError|No module named",
 "conn_refused": r"(?i)connection refused|could not connect|Failed to connect",
 "timed_out": r"(?i)timed out|timeout",
}
C = collections.Counter(); N = collections.Counter(); starts = collections.Counter()
def txt(c):
    if isinstance(c, str): return c
    if isinstance(c, list): return "\n".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
    return str(c)
for r in runs:
    w = "sim" if r["arm"].startswith("sim") else ("ref" if r["arm"] == "honest_a" else "real")
    s = read_sample(_log(RD, r)["eval"])
    for m in s["messages"]:
        if m["role"] != "tool": continue
        t = txt(m.get("content"))
        N[w] += 1
        for k, p in pats.items():
            if re.search(p, t): C[(w, k)] += 1
        # first 18 chars of first line, only for very common prefixes
        starts[(w, t.strip().split("\n")[0][:18])] += 1
print("tool outputs:", dict(N))
for k in pats:
    print(f"{k:16s}", {w: f"{C[(w,k)]} ({C[(w,k)]/N[w]:.1%})" for w in ("real","sim","ref")})
print("common first-line prefixes (>=15 in a world):")
for (w, pfx), v in sorted(starts.items(), key=lambda x: -x[1]):
    if v >= 15: print(" ", w, repr(pfx), v)
