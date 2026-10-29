"""ON q3-scaleup4 (sigint_311_v2.sh copies it to /tmp): a by-key snapshot of +311 launch 1 (RULE.md amendment 3): every run at
0-based index < P (265) in launch 1's tasks file (sha 3b645674..., asserted): state, attempt count, last attempt's end; plus the
running runs, the runs whose last attempt was stopped by the launcher, and the stop condition. Keys and states only.
    python3 sigint_311_snap.py <out json> [--check]   (--check: exit 3 unless the condition holds)"""
import hashlib
import json
import sys

raw = open("/home/aswin/phase2/ep2_311/pilot_tasks_ep2_311_309.json", "rb").read()
assert hashlib.sha256(raw).hexdigest() == "3b645674937459b697a9fec427aa703b322082260b34f0485a22eb6ac37a8a84"
idx = {c["key"]: i for i, c in enumerate(json.loads(raw)["cells"])}
d = json.load(open("/home/aswin/phase2/runs_ep2/2026-10-15_122516_ep2_311/launch.json"))
P = 265
key = lambda r: f"{r['task']}/{r['arm']}"
below = {key(r): {"state": r["state"], "attempts": len(r["attempts"]),
                  "last_end": (r["attempts"][-1].get("end") if r["attempts"] else None)} for r in d["runs"] if idx[r["task"]] < P}
running = sorted(key(r) for r in d["runs"] if r["state"] == "running")
running_below = [k for k in running if idx[k.split("/")[0]] < P]
not_terminal = sorted(k for k, v in below.items() if v["state"] not in ("done", "no policy call") or not v["last_end"])
stopped = sorted(key(r) for r in d["runs"] if r["attempts"] and r["attempts"][-1].get("stopped_by_launcher"))
snap = {"finished": d.get("finished"), "stopped_by_signal": d.get("stopped_by_signal"), "below_P": below, "running": running,
        "running_below_P": running_below, "not_terminal_below_P": not_terminal, "stopped_by_launcher": stopped,
        "stopped_below_P": [k for k in stopped if idx[k.split("/")[0]] < P]}
open(sys.argv[1], "w").write(json.dumps(snap))
ok = not snap["finished"] and not running_below and not not_terminal
print("CONDITION", "PASS" if ok else "FAIL", f"| runs below P {len(below)}, not terminal {len(not_terminal)} | running {len(running)},"
      f" below P {len(running_below)} | stopped_by_launcher {len(stopped)}")
print("not terminal below P (first 20):", not_terminal[:20])
print("running at this read:", running)
if "--check" in sys.argv[2:] and not ok:
    sys.exit(3)
