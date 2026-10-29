"""Append 35 item 4 (09 @ d99ce7997), ON THE BOX, metadata only: the frozen length-selection check for one new-mains
simulated arm (sim_attack@september of a follow-up launch), committed before any affected score.
  Proxy per cell: its run1 = the main launch's real attack/<cell> (r1, last attempt): policy steps (launch.json policy_calls
  steps) and policy output tokens (the run1 .eval's model events for the policy model: output_tokens summed); missing
  proxies listed. Quartile cut points over the ORIGINAL started simulated cells (a simulated run with >= 1 attempt), ties
  to the lower quartile. Groups: rate-limit failures (any attempt's terminal error RateLimitError), usable, other failures.
  Reported: proxy quantiles per group, failure share by proxy quartile, failure timestamps, simulator-call exposure
  (simulator model calls before the failure, from the failed attempt's .eval events). No p-value.
  <sept venv python> length_check_nm.py <main run dir> <follow-up run dir>"""
import glob, json, re, statistics, sys
from inspect_ai.event import ModelEvent
from inspect_ai.log import read_eval_log
main, fu = sys.argv[1], sys.argv[2]
M, F = json.load(open(main + "/launch.json")), json.load(open(fu + "/launch.json"))
policy = M.get("policy")
def task(r): return r["task"] if isinstance(r["task"], str) else r["task"]["key"]
def evfile(a):
    ev = sorted(glob.glob(a["log_dir"] + "/*.eval")); return ev[0] if len(ev) == 1 else None
def usable(r):
    finals = [a for a in r["attempts"] if a.get("outcome") and not a.get("redrawn_because")]
    calls = finals[-1]["outcome"]["policy_calls"] if finals else []
    return bool(calls) and all(c["usable"] for c in calls)
run1 = {task(r): r for r in M["runs"] if r["real"] and r["key"].startswith("attack/") and not task(r).endswith("_r2")}
rows = []
for r in F["runs"]:
    if r["real"] or not r["attempts"]:
        continue
    c = task(r); p = run1.get(c); steps = tokens = None
    if p and p["attempts"]:
        a = p["attempts"][-1]; steps = sum(x.get("steps", 0) for x in (a.get("outcome") or {}).get("policy_calls") or [])
        e = evfile(a)
        if e:
            s = read_eval_log(e).samples[0]
            tokens = sum(ev.output.usage.output_tokens for ev in s.events if isinstance(ev, ModelEvent) and ev.model == policy and ev.output and ev.output.usage)
    errs = [re.search(r"raised (\w+)", str(x)) for a in r["attempts"] for x in ((a.get("outcome") or {}).get("sample_errors") or [])]
    rl = any(m and m.group(1) == "RateLimitError" for m in errs)
    row = {"cell": c, "run_key": r["key"], "state": r["state"], "run1_steps": steps, "run1_policy_output_tokens": tokens,
           "group": "rate_limit" if rl else "usable" if usable(r) else ("other_failure" if r["state"] in ("done", "no policy call") else r["state"])}
    if rl:
        a = r["attempts"][-1]; e = evfile(a)
        row["failure_end"] = a.get("end")
        if e:
            s = read_eval_log(e).samples[0]
            row["simulator_calls_before_failure"] = sum(1 for ev in s.events if isinstance(ev, ModelEvent) and ev.model != policy)
    rows.append(row)
have = sorted(r["run1_steps"] for r in rows if r["run1_steps"] is not None)
q = statistics.quantiles(have, n=4, method="inclusive") if len(have) >= 4 else []
def quart(x):
    if x is None or not q: return None
    return 1 + sum(x > c for c in q)  # ties go to the lower quartile
for r in rows: r["run1_steps_quartile"] = quart(r["run1_steps"])
groups = {}
for g in ("rate_limit", "usable", "other_failure"):
    xs = sorted(r["run1_steps"] for r in rows if r["group"] == g and r["run1_steps"] is not None)
    groups[g] = {"n": sum(r["group"] == g for r in rows), "run1_steps_quantiles": statistics.quantiles(xs, n=4, method="inclusive") if len(xs) >= 4 else xs}
share = {k: {"cells": sum(r["run1_steps_quartile"] == k for r in rows), "rate_limit": sum(r["run1_steps_quartile"] == k and r["group"] == "rate_limit" for r in rows)} for k in (1, 2, 3, 4)}
print(json.dumps({"main": main, "follow_up": fu, "policy": policy, "started_cells": len(rows), "quartile_cuts_run1_steps": q,
                  "missing_proxy": [r["cell"] for r in rows if r["run1_steps"] is None], "groups": groups,
                  "failure_share_by_run1_steps_quartile": share, "rows": rows}))
