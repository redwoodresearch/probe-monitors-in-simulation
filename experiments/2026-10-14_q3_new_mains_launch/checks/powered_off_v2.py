"""Laptop, read-only (0d's request via d9, 10-15 10:09): the census's powered-off rule. For every box whose census file this round
is empty (no data), a fresh `aws ec2 describe-instances --profile redwood-sandbox` (60 s bound) and the box's launches in its last
committed, clean census that HAS data for it. The box is recorded "powered off, 0 launches" only if the EC2 state is "stopped" or
"terminated" AND every launch there was finished (launch.json "finished" set). Anything else (another state, a failed or timed-out
read, an unfinished launch, no earlier data): not recorded, so the box stays "no data". Prints {"<box>": {...}} (possibly {}).
  python3 powered_off_v2.py <census dir> <worktree> <results dir relative to the worktree>"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BOXES = {"box1": "i-002bbb22c185c0f07", "box2": "i-0c02a38abcde32bfb", "box3": "i-0cc7555acb95c694b", "box4": "i-07bce857a818b6f69",
         "xbox1": "i-04331f272d6778df0", "xbox2": "i-00db08777bb29d430", "xbox3": "i-0f2299f29dd296259", "simsastra": "i-05b1d910fd70016e6"}
cdir, wt, rel = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
PT = ZoneInfo("America/Los_Angeles")


def committed_clean(f: str) -> bool:  # Codex r48: both git calls bounded, and status must succeed before "clean" counts
    try:
        if subprocess.run(["git", "-C", str(wt), "ls-files", "--error-unmatch", f], capture_output=True, timeout=30).returncode != 0:
            return False
        st = subprocess.run(["git", "-C", str(wt), "status", "--porcelain", "--", f], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return False
    return st.returncode == 0 and not st.stdout.strip()


def last_seen(box: str):
    for f in sorted((wt / rel).glob("sim_error_census_2026-10-15_*.json"), reverse=True):
        r = str(f.relative_to(wt))
        if not committed_clean(r):
            continue
        try:
            v = json.loads(f.read_text())["per_box"].get(box)
        except (OSError, ValueError, KeyError):
            continue
        if isinstance(v, dict) and v:
            return f.name, v
    return None, None


out = {}
for box, iid in BOXES.items():
    p = cdir / f"{box}.json"
    if p.exists() and p.read_text().strip():
        continue  # the box returned data this round
    read_pt = datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S")
    try:
        aw = subprocess.run(["aws", "ec2", "describe-instances", "--profile", "redwood-sandbox", "--instance-ids", iid,
                             "--query", "Reservations[0].Instances[0].State.Name", "--output", "text"],
                            capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        continue
    if aw.returncode != 0:  # Codex r48: a failed read never counts
        continue
    state = aw.stdout.strip()
    if state not in ("stopped", "terminated"):
        continue
    name, launches = last_seen(box)
    if not launches or any(not isinstance(x, dict) or "error" in x or x.get("stall", {}).get("state") != "stopped" for x in launches.values()):
        continue
    out[box] = {"instance": iid, "state": state, "read_pt": read_pt, "running_launches": 0, "region": "default",
                "launch_end_times": {l: x.get("finished") or "finished set (time not recorded in that census)" for l, x in sorted(launches.items())},
                "last_activity": {l: x.get("stall", {}).get("last_attempt_activity") for l, x in sorted(launches.items())},
                "last_seen_census": name}
print(json.dumps(out))
