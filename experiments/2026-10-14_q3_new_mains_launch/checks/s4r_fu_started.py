"""ON BOX 4, read-only (status for the ungrounded fork, 10-15): (1) the tmux -L nm session names, read positively (Codex
r36 item 3: a tmux error other than "no server" raises, so the publisher marks the line unreadable); (2) 4r's main real
runs by state (ready/waiting, running, terminal); (3) how many simulated runs of 4r's follow-up have started (an attempt
with a start time), of how many, or "absent" before its launch.json exists. One JSON line."""
import json
import re
import subprocess
from pathlib import Path

p = subprocess.run(["tmux", "-L", "nm", "list-sessions", "-F", "#{session_name}"], capture_output=True, text=True)
if p.returncode == 0:
    sessions = sorted(p.stdout.split())
elif re.match(r"(no server running on |error connecting to .*\(No such file or directory\))", p.stderr.strip()):
    sessions = []
else:
    raise SystemExit(f"tmux -L nm unreadable: rc {p.returncode}")
m = json.loads(Path("/home/aswin/phase2/nm/runs/2026-10-14_newmains_s4r/launch.json").read_text())
real = [r for r in m["runs"] if r["real"]]
out = {"nm_sessions": sessions, "main_real_total": len(real),
       "main_real_not_started": sum(r["state"] in ("ready", "waiting") for r in real),
       "main_real_running": sum(r["state"] == "running" for r in real)}
f = Path("/home/aswin/phase2/nm/runs/2026-10-14_newmains_s4r_fu/launch.json")
if not f.exists():
    out["fu_launch_json"] = "absent"
else:
    sims = [r for r in json.loads(f.read_text())["runs"] if not r["real"]]
    out.update({"fu_launch_json": "present", "fu_sims_total": len(sims),
                "fu_sims_started": sum(1 for r in sims if any(a.get("start") for a in r["attempts"]))})
print(json.dumps(out))
