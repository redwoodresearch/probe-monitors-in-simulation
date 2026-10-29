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
EDIT = {'create','str_replace','insert'}
RO_VERBS = {"cat","ls","grep","egrep","rg","find","head","tail","wc","pwd","which","echo","ps","env","printenv","stat","file",
            "tree","du","df","diff","less","more","sort","uniq","cut","id","whoami","uname","date","jq","nl","realpath",
            "readlink","basename","dirname","type","hostname","test","[","true","sleep","cd","printf","xxd","md5sum","sha256sum","column","tr","awk","sed"}
SC_VERBS = {"rm","mv","cp","mkdir","touch","chmod","chown","pip","pip3","npm","yarn","pnpm","apt","apt-get","tee","kill","pkill",
            "killall","systemctl","service","make","cargo","mvn","gradle","patch","ln","rmdir","truncate","install","supervisorctl",
            "nohup","lein","go","dotnet","javac","gcc","tar","unzip","git","createdb","dropdb","redis-cli"}
_RE_FAIL = re.compile(r"Traceback \(most recent call last\)|\b\w+Error\b|\berror:|No such file or directory|command not found|"
                     r"Permission denied|(?i:connection refused)|(?i:exit code [1-9])|(?i:\bfailed\b)", )
def txt(c):
    if isinstance(c, str): return c
    if isinstance(c, list): return "\n".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
    return "" if c is None else str(c)
def bash_cmd(c):
    a = c.get("arguments") or {}
    return str(a.get("cmd") or a.get("command") or "")
def classify_bash(cmd):
    redir = re.sub(r"\d?>\s*/dev/null|2>&1|&>\s*/dev/null|>&2", "", cmd)
    if re.search(r"(^|[^<>=-])>{1,2}(?!=)", redir) or re.search(r"\bsed\s+(-\w*\s+)*-i", cmd) or "<<" in cmd and re.search(r"\bcat\b[^|;&]*>", cmd):
        return "sc"
    parts = [p.strip() for p in re.split(r"&&|\|\||;|\||\n", cmd) if p.strip()]
    verbs = []
    for p in parts:
        w = p.split()[0] if p.split() else ""
        if w in ("sudo", "timeout") and len(p.split()) > 1:
            w = p.split()[1] if w == "sudo" else (p.split()[2] if len(p.split()) > 2 else "")
        verbs.append(w)
    if any(v in SC_VERBS for v in verbs): return "sc"
    if verbs and all(v in RO_VERBS for v in verbs): return "ro"
    return "unc"

rng = random.Random(20261009)
for key, step, cmd in rng.sample(allcmds, 12):
    print("SPOT", classify_bash(cmd), key, step, repr(cmd[:90]))
