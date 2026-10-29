# outcomes_all.py <run dir>: per cell, the counted (last) attempt from launch.json: start/end clock times, steps, how it ended,
# usable, attempts, error classes (first token only), sample limits. Then totals. Keys and counters only, no message text.
import collections, json, re, sys
d = json.load(open(sys.argv[1] + "/launch.json"))
tot = collections.Counter()
for r in d["runs"]:
    a = r["attempts"][-1] if r["attempts"] else {}
    o = a.get("outcome") or {}
    pc = (o.get("policy_calls") or [{}])[0]
    cls = [(re.match(r"^([A-Za-z_]\w*)", str(e)) or [None, "?"])[1] for e in o.get("sample_errors", [])]
    t = lambda s: (s or "")[11:19]
    print(r["task"], t(a.get("start")), t(a.get("end")), "steps", pc.get("steps"), "ended", pc.get("ended"), "usable", pc.get("usable"),
          "attempts", len(r["attempts"]), "state", r["state"], "exit", a.get("exit_code"), "errors", cls or "none",
          "sample_limits", o.get("sample_limits"), "failed_calls", pc.get("failed_calls"), "timeouts", pc.get("timeouts"))
    tot["cells"] += 1; tot["usable"] += pc.get("usable") or 0; tot["with_errors"] += bool(cls)
print("TOTALS", dict(tot), "finished", d.get("finished"), "spend_usd", d["spend"][-1]["actual_usd"])
