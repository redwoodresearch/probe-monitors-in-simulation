"""Modal Pacific-day guard for the scale-up (RULE.md, Stops; 0d's levels via d9). Laptop side and light: per poll
(every POLL_S >= 120 s) ONE `modal billing report` and ONE `modal app list`, nothing else.

Two estimates of today's (Pacific) Modal cost of every app whose name contains "aswin"; the guard uses the LARGER:
  - billing view: billed cost (intervals >= today's 00:00 PT, hourly, `--start <UTC date> --end <UTC date + 2>` so the
    current partial hour is included) + for each live app, its hourly rate x hours since the latest billed interval's
    start, minus what that interval already billed;
  - accrued view: the first poll's billing view, then at every poll the larger of (the previous accrued view + each live
    app's hourly rate x the time since the previous poll) and the new billing view. Usage of an app that stops before
    billing catches up stays counted, and so does an upward billing correction (Codex rounds 1 and 2); it never falls.
An app's hourly rate is the larger of its largest hourly cost billed today and its live containers x RATE ($5.30, an
H200's; conservative for CPU apps and for partly billed hours).
Levels: total >= NO_NEW ($1,700): status NO_NEW (launch_311.sh starts no server and releases nothing); total >= STOP
($1,800), or FAIL_MAX (5, about 10 minutes) polls in a row without a reading or a readable launch list (fail closed): status STOP, LATCHED for the rest of the Pacific
day; every poll from then on stops each scale-up box whose stop is not yet confirmed, each box on its own (one box's
failure never skips another; stop_group.sh exit 0 or 2 with app ids present = confirmed). The guard exits only once every
box is confirmed. One line per poll to the log; the latest to STATUS_FILE
("<level> <total> <billing view> <accrued view> <live containers> <epoch>"). A new Pacific day starts at $0.
ONCE=1: print one reading and exit (no status file, no stop).

  python3 modal_day_guard.py <launches json: [{"name","devbox","session","app_ids_file","harness"}], one per quarter
  launched, re-read every poll> <status file>
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
NO_NEW, STOP = float(os.environ.get("NO_NEW_USD", 1700)), float(os.environ.get("STOP_USD", 1800))
RATE, POLL, FAIL_MAX = float(os.environ.get("RATE_USD_H", 5.3)), max(120, int(os.environ.get("POLL_S", 120))), 5  # ~10 min
MODAL = ["/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal"]  # the pinned modal 1.6.1 binary (laptop load, d9 15:20)
if os.environ.get("TEST_MODAL_CMD"):  # negative tests only (negative_tests_guard.sh): a failing Modal CLI, fast polls
    MODAL, POLL = os.environ["TEST_MODAL_CMD"].split(), int(os.environ.get("TEST_POLL_S", 1))
HERE = Path(__file__).resolve().parent
boxes_file, status_file = Path(sys.argv[1]), Path(sys.argv[2])  # re-read every poll: quarters join as they launch


def log(msg: str) -> None:
    print(f"{datetime.now(PT):%F %T} PT {msg}", flush=True)


def call(args: list, timeout: int) -> str:
    return subprocess.run(MODAL + args, capture_output=True, text=True, timeout=timeout, check=True).stdout


def reading() -> dict:
    now = datetime.now(timezone.utc)
    d0 = datetime.now(PT).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    rows = json.loads(call(["billing", "report", "--start", f"{d0:%Y-%m-%d}", "--end",
                            f"{d0 + timedelta(days=2):%Y-%m-%d}", "--resolution", "h", "--json"], 180))
    ts = lambda r: datetime.fromisoformat(r["interval_start"]).replace(tzinfo=timezone.utc)  # noqa: E731
    mine = [r for r in rows if "aswin" in str(r.get("description", "")).lower() and ts(r) >= d0]
    billed = sum(float(r["cost"]) for r in mine)
    last = max((ts(r) for r in mine), default=d0)
    in_last = sum(float(r["cost"]) for r in mine if ts(r) == last)
    hourly = {}
    for r in mine:
        hourly[r["object_id"]] = max(hourly.get(r["object_id"], 0.0), float(r["cost"]))
    apps = json.loads(call(["app", "list", "--json"], 120))
    live = [a for a in apps if "aswin" in str(a.get("description", "")).lower()
            and str(a.get("state", "")).lower() not in ("stopped", "stopping") and int(a.get("tasks") or 0) > 0]
    # $/h now; a partly billed hour can understate an app's rate, so never below its containers x RATE (conservative)
    rate = sum(max(hourly.get(a.get("app_id"), 0.0), int(a.get("tasks") or 0) * RATE) for a in live)
    unbilled = max(0.0, rate * (now - last).total_seconds() / 3600 - in_last)
    return {"day": f"{d0:%F}", "billing_view": billed + unbilled, "rate": rate, "t": time.time(),
            "containers": sum(int(a.get("tasks") or 0) for a in live)}


def stop_box(b: dict) -> bool:
    """One box: its launcher pane (if found), then its apps (FORCE_APPS=1). True only when its apps are confirmed."""
    try:
        f = Path(b["app_ids_file"])
        apps = f.read_text().strip().replace("\n", ",") if f.exists() else ""
        q = (f"tmux -L phase2 list-panes -a -F '#{{session_name}} #{{pane_id}}:#{{pane_pid}}' 2>/dev/null "
             f"| awk '$1==\"{b['session']}\" {{print $2}}' | paste -sd, -")
        try:
            panes = subprocess.run(["perl", "-e", "alarm 60; exec @ARGV", "bash", f"{b['harness']}/devbox.sh", q],
                                   capture_output=True, text=True, timeout=90,
                                   env=os.environ | {"DEVBOX": b["devbox"]}).stdout.strip()
        except Exception as e:  # noqa: BLE001 - unreachable devbox: the apps are still stopped below
            panes = ""
            log(f"{b['name']}: pane lookup failed ({e!r})")
        env = os.environ | {"H": b["harness"], "DEVBOX": b["devbox"], "FORCE_APPS": "1", "APP_IDS": apps}
        r = subprocess.run(["bash", str(HERE / "stop_group.sh"), panes], env=env, timeout=900)
        log(f"{b['name']}: stop_group.sh exit {r.returncode} (panes {panes or 'none found'}, apps {apps or 'none yet'})")
        return r.returncode in (0, 2) and bool(apps)
    except Exception as e:  # noqa: BLE001 - logged; retried next poll; the other boxes are still stopped
        log(f"{b['name']}: stop failed ({e!r}); retrying next poll")
        return False


if os.environ.get("ONCE"):
    log("ONCE: " + json.dumps(reading()))
    sys.exit(0)
fails, accrued, prev, latched, confirmed = 0, 0.0, None, None, set()
boxes = json.loads(boxes_file.read_text())
while True:
    try:  # the launch list, every poll (Codex round 3, item 7): an unreadable list counts as a failed poll
        boxes = json.loads(boxes_file.read_text())
        rd = reading()
        fails = 0
    except Exception as e:  # noqa: BLE001 - every failure counts toward the fail-closed stop, and is logged
        rd = None
        fails += 1
        log(f"no reading ({type(e).__name__}: {str(e)[:200]}), {fails} in a row")
    if rd is not None:
        if prev is None or prev["day"] != rd["day"]:  # first poll, or a new Pacific day: restart from the billing view
            accrued = rd["billing_view"]
            if latched is not None and latched != rd["day"]:
                latched, confirmed = None, set()
        else:  # never below the billing view: an upward billing correction stays in (Codex round 2, item 3)
            accrued = max(accrued + prev["rate"] * (rd["t"] - prev["t"]) / 3600, rd["billing_view"])
        prev = rd
        total = accrued
        if total >= STOP:
            latched = rd["day"]
        level = "STOP" if latched == rd["day"] else "NO_NEW" if total >= NO_NEW else "OK"
        status_file.write_text(f"{level} {total:.2f} {rd['billing_view']:.2f} {accrued:.2f} {rd['containers']} {int(time.time())}\n")
        log(f"{level}: Pacific-day Modal ${total:.2f} (billing view ${rd['billing_view']:.2f}, accrued ${accrued:.2f}, "
            f"{rd['containers']} live containers)")
    else:  # Amendment 3 (unwatched): every failed poll writes UNREAD (no new launch passes on an old OK); it never stops anything
        status_file.write_text(f"UNREAD nan nan nan nan {int(time.time())}\n")
        log(f"no Modal reading ({fails} in a row): status UNREAD, NOT a stop (only a fresh reading over $1,800 stops)")
    if latched is not None:
        for b in boxes:
            if b["name"] not in confirmed and stop_box(b):
                confirmed.add(b["name"])
        if rd is not None and fails == 0 and {b["name"] for b in boxes} <= confirmed:  # only on a list read this poll
            log("every scale-up box's apps confirmed stopped; exiting (status stays STOP)")
            sys.exit(2)
    time.sleep(POLL)
