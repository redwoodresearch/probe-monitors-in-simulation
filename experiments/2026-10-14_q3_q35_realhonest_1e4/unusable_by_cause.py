"""Unusable runs of both launches, by cause (0d via d9, for the end-of-launch report). A Modal CPU function reads only the two
final launch.json files on this job's Volume (evals_v1/<run>/launch.json: states, servers, timestamps and the launcher's
outcome counters; never a transcript) and the committed re-run selection. A run is unusable as gate 1 counts it: its
attempts that ran and weren't redrawn have no `usable` policy call (run_attack_check.py @ c42666d7 lines 178-179: a
non-submit tool call and no sample error). Ended runs ("done", "no policy call") and runs the launcher stopped are classed:
  infra: server 0 down       launch 1, server 0, last attempt running at or after 15:19:51 PT (the selection's rule (a))
  cut by the launcher's stop state "stopped" (launch 1: gate 1's Ctrl+C at 15:26:10; relaunch: a drain or a stop)
  environment start failure  state "no policy call" (pre-start failures, redraws spent)
  error after start          ended "error" (the sample ended in an error after the policy was called)
  a limit or timeout         the policy call's `ended` as run_attack_check.py line 164 sets it ("time limit", "<type> limit",
                             "policy_timeout", "out_of_context", "step limit or no submit")
  1-step submit              ended "submit" with at most one step (no non-submit tool call)
  other                      anything else, with its `ended` value
Per class: count, and per run its key, launch, replicate, server slot, steps, minutes, and whether it is a re-run key.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run experiments/2026-10-14_q3_q35_realhonest_1e4/unusable_by_cause.py
"""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-q3realhonest_unusable_kill-2026-11-14"
VOL = "probe-shift_aswin-surya_sub-q3realhonest_data_kill-2026-11-14"
RUNS = ["2026-10-14_135334_honest_rh980", "2026-10-14_160335_honest_rh_relaunch"]
STOP_SERVER0 = "2026-10-14T15:19:51-07:00"
app = modal.App(APP)


@app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=2, memory=8192, timeout=900, retries=0,
              volumes={"/ev": modal.Volume.from_name(VOL, version=2).read_only()})
def classify(affected: list) -> dict:
    from collections import Counter

    T = lambda s: datetime.fromisoformat(s).timestamp()  # noqa: E731
    affected, stop0 = set(affected), T(STOP_SERVER0)
    rows, totals = [], {}
    for run in RUNS:
        d = json.loads((Path("/ev/evals_v1") / run / "launch.json").read_text())
        srv = {u: i for i, u in enumerate(d["servers"])}
        c = Counter()
        for r in d["runs"]:
            if r["state"] not in ("done", "no policy call", "stopped"):
                c[f"state {r['state']}"] += 1
                continue
            c[f"state {r['state']}"] += 1
            finals = [a for a in r["attempts"] if a.get("outcome") and not a.get("redrawn_because")]
            usable = sum(p["usable"] for a in finals for p in a["outcome"]["policy_calls"])
            c["usable"] += bool(usable)
            if usable:
                continue
            a = r["attempts"][-1]
            pc = ((a.get("outcome") or {}).get("policy_calls") or [{}])[0]
            ended, steps = pc.get("ended"), pc.get("steps")
            end = T(a["end"]) if a.get("end") else None
            if run == RUNS[0] and srv.get(r["server"]) == 0 and (end is None or end >= stop0):
                cls = "infra: server 0 down"
            elif r["state"] == "stopped":
                cls = "cut by the launcher's stop"
            elif r["state"] == "no policy call":
                cls = "environment start failure"
            elif ended == "error":
                cls = "error after start"
            elif ended in ("policy_timeout", "out_of_context", "step limit or no submit") or str(ended).endswith(" limit"):
                cls = {"policy_timeout": "policy timeout", "out_of_context": "out of context"}.get(ended, ended)  # run_attack_check.py:164
            elif ended == "submit" and (steps or 0) <= 1:
                cls = "1-step submit"
            else:
                cls = f"other (ended {ended})"
            rows.append({"class": cls, "key": r["task"], "launch": run, "replicate": r["task"].rsplit("_", 1)[1],
                         "server": srv.get(r["server"]), "state": r["state"], "ended": ended, "steps": steps,
                         "minutes": round((end - T(a["start"])) / 60, 1) if end and a.get("start") else None,
                         "rerun_key": r["task"] in affected})
        totals[run] = dict(c)
        last = {}  # per server, the latest attempt end (the relaunch's monitor pause at 21:59: idle exits or failures?)
        for r in d["runs"]:
            for a in r["attempts"]:
                if a.get("end") and r.get("server") in srv:
                    last[srv[r["server"]]] = max(last.get(srv[r["server"]], ""), a["end"])
        totals[run + " last_end_by_server"] = dict(sorted(last.items()))
    by = {}
    for x in rows:
        by.setdefault(x["class"], []).append(x)
    return {"totals": totals, "classes": {k: {"count": len(v), "runs": v} for k, v in sorted(by.items())}}


@app.local_entrypoint()
def main():
    sel = json.loads((HERE / "results" / "relaunch_selection_2026-10-14.json").read_text())
    r = classify.remote(sel["affected"])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"unusable_by_cause_{stamp}.json"
    dest.open("x").write(json.dumps(r, indent=1))
    print(json.dumps(r["totals"]))
    for k, v in r["classes"].items():
        print(f"{k}: {v['count']}: " + ", ".join(f"{x['key']}({x['launch'][-8:]},s{x['server']}{',rerun' if x['rerun_key'] else ''})" for x in v["runs"]))
    print("->", dest)
