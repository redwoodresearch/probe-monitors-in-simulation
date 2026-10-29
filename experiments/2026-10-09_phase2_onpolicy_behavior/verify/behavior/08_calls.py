import json, collections, re, random, sys
from pathlib import Path
from probe_shift.viewer.pairs import read_launch, read_sample, steps, _log
exec(open("/dev/stdin").read()) if False else None
RD = Path.home() / "phase2/runs/2026-10-09_000523_rerun_strict"
runs = read_launch(RD)["runs"]
BASHLIKE = {"bash", "db_bash", "rust_worker_bash", "nginx_bash"}
C = collections.Counter(); R = collections.defaultdict(set); allcmds = []
for r in runs:
    g = r["arm"] + ("@" + r["simulator"] if r["arm"].startswith("sim") else "")
    s = read_sample(_log(RD, r)["eval"])
    _, st = steps(s)
    for i, step in enumerate(st):
        for c in step[0].get("tool_calls") or []:
            if c["function"] not in BASHLIKE: continue
            a = c.get("arguments") or {}; cmd = str(a.get("cmd") or a.get("command") or "")
            allcmds.append((r["key"], i + 1, cmd))
            for name, pat in (("curl", r"\bcurl\b"), ("pytest", r"\bpytest\b"), ("python_file", r"\bpython3?\s+(?:-u\s+)?[\w./-]+\.py\b"),
                              ("python_inline", r"\bpython3?\s+(-c\b|-\s*<<|<<)"), ("sh_script", r"(?:\bbash\s+|\bsh\s+|(?<![\w/])\./)[\w./-]+\.sh\b"),
                              ("go/cargo/npm/mvn test", r"\b(go test|cargo test|npm (run )?test|mvn test|lein test|jest|vitest)\b")):
                if re.search(pat, cmd): C[(g, name)] += 1; R[(g, name)].add(r["key"])
names = ["curl", "pytest", "python_file", "python_inline", "sh_script", "go/cargo/npm/mvn test"]
for g in ["honest_b", "honest_a", "sim_honest@september", "sim_honest@today", "attack", "sim_attack@september", "sim_attack@today"]:
    print(f"{g:22s}", "  ".join(f"{n}: {C[(g, n)]} calls/{len(R[(g, n)])} runs" for n in names))
# seeded spot check of the bash read-only/state-changing heuristic
sys.path.insert(0, "/dev/null")
