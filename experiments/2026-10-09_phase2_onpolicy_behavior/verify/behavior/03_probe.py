import json, collections, re
from pathlib import Path
from probe_shift.viewer.pairs import read_launch, pairs, read_sample, steps, _log
RD = Path.home() / "phase2/runs/2026-10-09_000523_rerun_strict"
runs = read_launch(RD)["runs"]
fnargs = collections.Counter(); errfields = collections.Counter(); multi = collections.Counter()
meta = collections.Counter(); tcmds = collections.Counter(); errtypes = collections.Counter()
unread = []
for r in runs:
    lg = _log(RD, r)
    try:
        s = read_sample(lg["eval"])
    except Exception as e:
        unread.append((r["key"], repr(e)[:100])); continue
    md = s.get("metadata") or {}
    w = "sim" if r["arm"].startswith("sim") else "real"
    meta[(w, "fork", json.dumps(md.get("fork"))[:80])] += 1
    meta[(w, "seeded_prefix", json.dumps(md.get("seeded_prefix"))[:80])] += 1
    for m in s["messages"]:
        if m["role"] == "assistant":
            tc = m.get("tool_calls") or []
            multi[(w, len(tc))] += 1
            for c in tc:
                fnargs[(w, c["function"], tuple(sorted((c.get("arguments") or {}).keys())))] += 1
                if c["function"] == "text_editor":
                    tcmds[(w, (c.get("arguments") or {}).get("command"))] += 1
        if m["role"] == "tool":
            e = m.get("error")
            errfields[(w, e is not None)] += 1
            if e is not None:
                errtypes[(w, e.get("type") if isinstance(e, dict) else type(e).__name__)] += 1
print("unreadable:", unread)
for k, v in sorted(meta.items(), key=str): print("META", k, v)
for k, v in sorted(multi.items(), key=str): print("NTOOLCALLS/assistant", k, v)
for k, v in sorted(fnargs.items(), key=str): print("FN", k, v)
for k, v in sorted(tcmds.items(), key=str): print("TE cmd", k, v)
for k, v in sorted(errfields.items(), key=str): print("tool.error set", k, v)
for k, v in sorted(errtypes.items(), key=str): print("tool.error type", k, v)
