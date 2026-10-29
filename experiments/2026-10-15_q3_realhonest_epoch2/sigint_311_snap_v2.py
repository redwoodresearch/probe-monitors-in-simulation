"""ON q3-scaleup4: v2 of sigint_311_snap.py for the amended stop condition (pending 09; d9 ~17:13): a run at 0-based index < P
(265) counts as done for the stop once its COUNTED attempt has ended: the first try's redraw chain before any reference rerun, the
attempt without redrawn_because (gate_ep2_arms.py counted(); Append 38's counting). Reference reruns (and redraws of a rerun) never
count, so a below-P run may still be running a rerun at the stop; that rerun is stopped and listed. The snapshot keeps, per
below-P run, the counted attempt's fields (kind, start, end, exit_code) separately from the run state, so the after-check can
require 0 change in COUNTED attempts while listing rerun stops. Keys, states and times only.
    python3 sigint_311_snap_v2.py <out json> [--check]"""
import hashlib
import json
import sys

raw = open("/home/aswin/phase2/ep2_311/pilot_tasks_ep2_311_309.json", "rb").read()
assert hashlib.sha256(raw).hexdigest() == "3b645674937459b697a9fec427aa703b322082260b34f0485a22eb6ac37a8a84"
idx = {c["key"]: i for i, c in enumerate(json.loads(raw)["cells"])}
d = json.load(open("/home/aswin/phase2/runs_ep2/2026-10-15_122516_ep2_311/launch.json"))
P = 265
key = lambda r: f"{r['task']}/{r['arm']}"


sys.path.insert(0, "/home/aswin/phase2/ep2_311")  # the staged gate (sha256 af3d3876..., checked below): one definition of "counted"
assert hashlib.sha256(open("/home/aswin/phase2/ep2_311/gate_ep2_arms.py", "rb").read()).hexdigest().startswith("af3d387669740bc1")
from gate_ep2_arms import counted as _counted  # noqa: E402  (returns [] or [the counted attempt]; asserts at most one)


def counted(r):
    c = _counted(r)
    return c[0] if c else None


below, in_rerun = {}, []
for r in d["runs"]:
    if idx[r["task"]] >= P:
        continue
    c = counted(r)
    below[key(r)] = {"state": r["state"], "attempts": len(r["attempts"]),
                     "counted": ({k: c.get(k) for k in ("kind", "start", "end", "exit_code")} if c else None)}
    if r["state"] == "running" and c is not None and c.get("end") and r["attempts"][-1] is not c:
        in_rerun.append(key(r))
running = sorted(key(r) for r in d["runs"] if r["state"] == "running")
counted_open = sorted(k for k, v in below.items() if not (v["counted"] and v["counted"]["end"]))
# Codex r77: a below-P run waiting to start anything (a pending redraw of the counted chain, or a pending reference rerun) could
# start between the check and the signal, so it holds the stop; only terminal runs and runs already executing a rerun may remain
pending_below = sorted(k for k, v in below.items() if v["state"] in ("ready", "waiting"))
not_terminal_or_rerun = sorted(k for k, v in below.items() if v["state"] not in ("done", "no policy call") and k not in in_rerun)
running_below_not_rerun = [k for k in running if idx[k.split("/")[0]] < P and k not in in_rerun]
stopped = sorted(key(r) for r in d["runs"] if r["attempts"] and r["attempts"][-1].get("stopped_by_launcher"))
snap = {"finished": d.get("finished"), "stopped_by_signal": d.get("stopped_by_signal"), "below_P": below, "running": running,
        "below_P_in_reference_rerun": sorted(in_rerun), "counted_not_ended_below_P": counted_open,
        "pending_below_P": pending_below, "below_P_neither_terminal_nor_in_rerun": not_terminal_or_rerun,
        "in_rerun_label": "reference rerun, not counted, interrupted by SIGINT",
        "running_below_P_not_rerun": running_below_not_rerun, "stopped_by_launcher": stopped,
        "stopped_below_P": [k for k in stopped if idx[k.split("/")[0]] < P]}
open(sys.argv[1], "w").write(json.dumps(snap))
ok = not snap["finished"] and not counted_open and not running_below_not_rerun and not pending_below and not not_terminal_or_rerun
print("CONDITION", "PASS" if ok else "FAIL", f"| runs below P {len(below)}, counted attempt not ended {len(counted_open)}, pending {len(pending_below)} | running {len(running)},"
      f" below P not in a rerun {len(running_below_not_rerun)}, below P in a reference rerun {len(in_rerun)} | stopped_by_launcher {len(stopped)}")
print("counted not ended below P (first 20):", counted_open[:20])
print("below P pending (ready/waiting):", pending_below[:20], "| neither terminal nor in a rerun:", not_terminal_or_rerun[:20])
print("below P in a reference rerun (would be stopped; 'reference rerun, not counted, interrupted by SIGINT'):", sorted(in_rerun))
if "--check" in sys.argv[2:] and not ok:
    sys.exit(3)
