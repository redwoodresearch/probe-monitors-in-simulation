"""Per-run numeric behavior records (no model text) for every run in the rerun_strict folder, plus short quotes."""
import json, collections, re, shlex, sys
from pathlib import Path
from probe_shift.viewer.pairs import read_launch, read_sample, steps, _log, pairs
RD = Path.home() / "phase2/runs/2026-10-09_000523_rerun_strict"
runs = read_launch(RD)["runs"]
BASHLIKE = {"bash", "db_bash", "rust_worker_bash", "nginx_bash"}
EDIT = {"create", "str_replace", "insert"}
RO_VERBS = {"cat","ls","grep","egrep","rg","find","head","tail","wc","pwd","which","echo","ps","env","printenv","stat","file",
            "tree","du","df","diff","less","more","sort","uniq","cut","id","whoami","uname","date","jq","nl","realpath",
            "readlink","basename","dirname","type","hostname","test","[","true","sleep","cd","printf","xxd","md5sum","sha256sum","column","tr","awk","sed"}
SC_VERBS = {"rm","mv","cp","mkdir","touch","chmod","chown","pip","pip3","npm","yarn","pnpm","apt","apt-get","tee","kill","pkill",
            "killall","systemctl","service","make","cargo","mvn","gradle","patch","ln","rmdir","truncate","install","supervisorctl",
            "nohup","lein","go","dotnet","javac","gcc","tar","unzip","git","createdb","dropdb","redis-cli"}
RE_FAIL = re.compile(r"Traceback \(most recent call last\)|\b\w+Error\b|\berror:|No such file or directory|command not found|"
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
records, quotes = [], collections.defaultdict(list)
for r in runs:
    lg = _log(RD, r)
    arm = r["arm"]
    world = "sim" if arm.startswith("sim") else "real"
    kind = {"attack": "attack", "sim_attack": "attack", "honest_b": "honest", "sim_honest": "honest", "honest_a": "reference"}[arm]
    s = read_sample(lg["eval"])
    prompt, st = steps(s)
    calls = []  # (step_idx, call, output_text, output_error)
    for i, step in enumerate(st):
        outs = {m.get("tool_call_id"): m for m in step if m["role"] == "tool"}
        for c in step[0].get("tool_calls") or []:
            o = outs.get(c["id"])
            calls.append((i + 1, c, txt(o.get("content")) if o else None, (o or {}).get("error")))
    fn = collections.Counter(); te = collections.Counter(); cls = collections.Counter()
    curl = pytest = pyscript = shscript = 0
    for _, c, _, _ in calls:
        f = c["function"]
        if f in BASHLIKE:
            fn["bash"] += 1
            cmd = bash_cmd(c)
            cls[classify_bash(cmd)] += 1
            curl += len(re.findall(r"\bcurl\b", cmd))
            pytest += len(re.findall(r"\bpytest\b", cmd))
            pyscript += len(re.findall(r"\bpython3?\s+(?:-u\s+)?[\w./-]+\.py\b", cmd))
            shscript += len(re.findall(r"(?:\bbash\s+|\bsh\s+|(?<![\w/])\./)[\w./-]+\.sh\b", cmd))
        elif f == "text_editor":
            cm = (c.get("arguments") or {}).get("command")
            te[cm] += 1; fn["text_editor"] += 1
            cls["ro" if cm == "view" else "sc" if cm in EDIT else "unc"] += 1
        elif f == "submit":
            fn["submit"] += 1
        else:
            fn["other:" + f] += 1; cls["unc"] += 1
    # error replies
    outs = [(o, e) for _, _, o, e in calls if o is not None]
    n_out = len(outs)
    n_errfield = sum(e is not None for _, e in outs)
    n_failtext = sum((e is not None) or bool(RE_FAIL.search(o)) for o, e in outs)
    # re-reads right after an edit
    edits = rereads = 0
    for j, (_, c, _, _) in enumerate(calls):
        if c["function"] == "text_editor" and (c.get("arguments") or {}).get("command") in EDIT:
            edits += 1
            p = str((c.get("arguments") or {}).get("path") or "")
            if j + 1 < len(calls) and p:
                n = calls[j + 1][1]
                if n["function"] == "text_editor" and (n.get("arguments") or {}).get("command") == "view" and (n.get("arguments") or {}).get("path") == p:
                    rereads += 1
                elif n["function"] in BASHLIKE:
                    cmd = bash_cmd(n)
                    if re.search(r"\b(cat|head|tail|nl|less|more)\b|\bsed\s+-n", cmd) and (p in cmd or Path(p).name in cmd):
                        rereads += 1
    # ending
    last_fn = calls[-1][1]["function"] if calls else None
    lim = s.get("limit"); err = s.get("error")
    end = "error" if err else (f"limit:{(lim or {}).get('type') if isinstance(lim, dict) else lim}" if lim else ("submit" if last_fn == "submit" else f"other(last={last_fn})"))
    # live-env problem patterns
    pw = mod = 0
    for si, c, o, e in calls:
        if o is None: continue
        for name, pat in (("pw", r"password authentication failed"), ("mod", r"ModuleNotFoundError|No module named")):
            hits = [ln for ln in o.split("\n") if re.search(pat, ln)]
            if hits:
                if name == "pw": pw += 1
                else: mod += 1
                if len(quotes[(world, kind, name)]) < 3:
                    quotes[(world, kind, name)].append((r["key"], si, hits[0].strip()[:140]))
    sc = s.get("scores") or {}
    score_vals = {f"{k}.{vk}": vv for k, v in sc.items() for vk, vv in ((v.get("value") or {}).items() if isinstance(v.get("value"), dict) else [("_", v.get("value"))])}
    records.append(dict(key=r["key"], arm=arm, world=world, kind=kind, simulator=r.get("simulator") if world == "sim" else None,
        state=r.get("state"), attempt=lg["attempt"], of=lg["of"], n_steps=len(st), n_calls=len(calls),
        n_bash=fn["bash"], n_te=fn["text_editor"], n_view=te["view"], n_create=te["create"], n_str_replace=te["str_replace"],
        n_insert=te["insert"], n_submit=fn["submit"], n_other=sum(v for k, v in fn.items() if k.startswith("other:")),
        other_fns=sorted(k for k in fn if k.startswith("other:")),
        n_ro=cls["ro"], n_sc=cls["sc"], n_unc=cls["unc"], n_out=n_out, n_errfield=n_errfield, n_failtext=n_failtext,
        edits=edits, rereads=rereads, end=end, sample_error=bool(err), limit=json.dumps(lim)[:80] if lim else None,
        curl=curl, pytest=pytest, pyscript=pyscript, shscript=shscript, pw_hits=pw, mod_hits=mod,
        score_keys=sorted(sc.keys()), score_vals=score_vals))
P = pairs(RD)
pair_list = [dict(task=p["task"], arm=p["arm"], simulator=p["simulator"], sim=p["sim"]["key"], real=p["real"]["key"]) for p in P]
print(json.dumps({"records": records, "pairs": pair_list, "quotes": {"|".join(k): v for k, v in quotes.items()}}))
