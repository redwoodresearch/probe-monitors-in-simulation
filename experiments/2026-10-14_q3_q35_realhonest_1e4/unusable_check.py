"""Why relaunch runs are unusable (Aswin's question, 2026-10-14 ~17:00 PT). On the devbox, reading only launch.json fields
(states, servers, timestamps, the launcher's outcome counters; never a transcript):
    ~/sims/simulated-control-evals-sept/.venv/bin/python unusable_check.py <launch.json> <relaunch_selection json>
For every ended run (state "done" or "no policy call", gate1_relaunch.py's ENDED) whose usable count is 0 (the gate's own
count: the policy calls' `usable` over attempts that ran and weren't redrawn): key, replicate, environment, state, exit code,
the policy call's ended/failed_after_start/steps/timeouts/cap_hits, duration, server index, whether it is one of the 74
re-run keys. Plus totals and shared causes. Prints JSON."""
import json
import sys
from collections import Counter
from datetime import datetime

d = json.load(open(sys.argv[1]))
affected = set(json.load(open(sys.argv[2]))["affected"])
T = lambda s: datetime.fromisoformat(s).timestamp()  # noqa: E731
srv = {u: i for i, u in enumerate(d["servers"])}
ended, rows = 0, []
for r in d["runs"]:
    if r["state"] not in ("done", "no policy call"):
        continue
    ended += 1
    finals = [a for a in r["attempts"] if a.get("outcome") and not a.get("redrawn_because")]
    if sum(p["usable"] for a in finals for p in a["outcome"]["policy_calls"]):
        continue
    a = r["attempts"][-1]
    pc = ((a.get("outcome") or {}).get("policy_calls") or [{}])[0]
    rows.append({"key": r["task"], "replicate": r["task"].rsplit("_", 1)[1], "env": r["task"].split("_", 1)[1].rsplit("_", 1)[0],
                 "state": r["state"], "exit_code": a.get("exit_code"), "clean": (a.get("outcome") or {}).get("clean"),
                 "policy_called": a.get("policy_called"), "ended": pc.get("ended"), "failed_after_start": pc.get("failed_after_start"),
                 "steps": pc.get("steps"), "timeouts": pc.get("timeouts"), "cap_hits": pc.get("cap_hits"),
                 "samples": (a.get("outcome") or {}).get("samples"), "attempts": len(r["attempts"]),
                 "redraws": sum(x.get("kind") == "redraw" for x in r["attempts"]),
                 "minutes": round((T(a["end"]) - T(a["start"])) / 60, 1) if a.get("end") else None,
                 "end": a.get("end", "")[11:19], "server": srv.get(r["server"]), "rerun_key": r["task"] in affected})
usable = ended - len(rows)
out = {"ended": ended, "usable": usable, "unusable": len(rows), "rows": rows,
       "shared": {"by_env": dict(Counter(x["env"] for x in rows)), "by_server": dict(Counter(x["server"] for x in rows)),
                  "by_ended": dict(Counter(str(x["ended"]) for x in rows)), "rerun_keys": sum(x["rerun_key"] for x in rows)},
       "ended_by_server": dict(Counter(srv.get(r["server"]) for r in d["runs"] if r["state"] in ("done", "no policy call"))),
       "ended_rerun_keys": sum(r["task"] in affected for r in d["runs"] if r["state"] in ("done", "no policy call"))}
print(json.dumps(out, indent=1))
