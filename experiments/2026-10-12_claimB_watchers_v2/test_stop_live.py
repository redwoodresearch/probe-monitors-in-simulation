"""Live stop test for the watchers (v2.3, 10-13; 0d's request after d9 saw every `modal app stop <name>` fail with rc 1 on
ephemeral `modal run` apps). modal pinned to 1.6.1 here and in watch_cost.py. Stages, strictly one launch at a time:
four variants (stop by name or by app ID, with or without --yes; stdin closed, so "without --yes" is the non-interactive
case), then watch_cost.py v2.3 stopping one launch itself. Each stage launches the dummy app with `modal run --detach`,
takes the app ID from that launch's own output, checks the app list shows it under the dummy's name, runs its stop and
records rc, stdout and stderr verbatim and the state 15 s later; then the stage's app is stopped by ID with --yes and
confirmed stopped before the next launch (an unresolved cleanup ends the test). The watcher stage passes only with a logged
cost stop (rc 0), a normal watcher exit, and the app listed stopped before the dummy's 300 s sleep ends. A final pass
terminates every launcher, then stops anything still listed under the dummy's name and created since the start. The
dummy's own 420 s timeout bounds any leftover (five full timeouts are about $0.005). Results are saved after every stage
and on any exception, to results/stop_live_<stamp>.json, and the test passes only if every stage completed.

  uv run --no-project --python 3.12 python test_stop_live.py
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

HERE = Path(__file__).parent
MODAL = ["uv", "run", "--no-project", "--python", "3.12", "--with", "modal==1.6.1", "modal"]
NAME = "probe-shift_aswin-surya_sub-claimb-wtest_kill-2026-11-12"  # used only by this test (b7)
PT = ZoneInfo("America/Los_Angeles")
STAMP = datetime.now(PT).strftime("%Y-%m-%d_%H%M%S")
OUT = HERE / "results" / f"stop_live_{STAMP}.json"
T_START = datetime.now(PT).isoformat(sep=" ", timespec="seconds")  # same form as app list created_at


def m(*args, timeout=120):
    try:
        r = subprocess.run([*MODAL, *args], capture_output=True, text=True, timeout=timeout, cwd=HERE, stdin=subprocess.DEVNULL)
        return {"cmd": ["modal", *args], "rc": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    except Exception as e:  # a timeout or a launch failure is recorded, never raised (cleanup must reach every ID)
        return {"cmd": ["modal", *args], "rc": f"error: {e!r}", "stdout": "", "stderr": ""}


def listed(timeout=120):
    r = m("app", "list", "--json", timeout=timeout)
    if r["rc"] != 0:
        raise RuntimeError(f"app list rc {r['rc']}")
    rows = json.loads(r["stdout"])
    return rows if isinstance(rows, list) else rows["data"]


def state(app_id):
    try:
        return next((x.get("state") for x in listed() if x.get("app_id") == app_id), "not listed")
    except Exception as e:
        return f"lookup failed: {e!r}"


def stop_confirmed(app_id):
    """Stop by ID with --yes until the list shows it stopped (3 tries); never raises."""
    rec = {"app_id": app_id, "attempts": []}
    for _ in range(3):
        if state(app_id) == "stopped":
            break
        rec["attempts"].append(m("app", "stop", app_id, "--yes", timeout=60))
        time.sleep(10)
    rec["final_state"] = state(app_id)
    return rec


def launch(tag, procs):
    log = HERE / "results" / f"stop_live_launch_{STAMP}_{tag}.log"
    with open(log, "w") as fh:
        p = subprocess.Popen([*MODAL, "run", "--detach", "dummy_sleep_modal.py"], cwd=HERE, stdout=fh, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    procs.append(p)
    deadline = time.monotonic() + 120  # v2.4: a real 120 s window, with each lookup capped by the time left
    while time.monotonic() < deadline:
        time.sleep(2)
        ids = set(re.findall(r"ap-[A-Za-z0-9]{22}", log.read_text()))
        if len(ids) > 1:
            raise RuntimeError(f"{tag}: launch output names several app IDs {sorted(ids)}")
        if ids:
            app_id = ids.pop()
            row = next((x for x in listed(timeout=max(5, deadline - time.monotonic())) if x.get("app_id") == app_id), None)
            if row is None:  # v2.4: the app list can lag the launch output; keep looking within the 120 s window
                continue
            if row.get("description") != NAME:
                raise RuntimeError(f"{tag}: {app_id} is listed under {row.get('description')!r}, not {NAME}")
            return app_id, row.get("state"), log.name
    raise RuntimeError(f"{tag}: no listed app ID from the launch output within 120 s")


def main():
    (HERE / "results").mkdir(exist_ok=True)
    out = {"modal_version": m("--version")["stdout"].strip(), "app_name": NAME, "variants": [], "watcher": None, "cleanup": [],
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
           "stages_completed": 0, "pass": False}
    save = lambda: OUT.write_text(json.dumps(out, indent=1) + "\n")  # noqa: E731
    launched, procs = [], []
    try:
        if any(x.get("description") == NAME and x.get("state") == "deployed" for x in listed()):
            raise RuntimeError(f"a deployed app named {NAME} exists; a stop by name could hit it")
        for by, yes in (("name", True), ("name", False), ("id", True), ("id", False)):
            app_id, st0, log = launch(f"{by}_{'yes' if yes else 'noyes'}", procs)
            launched.append(app_id)
            time.sleep(20)  # let the container start
            res = m("app", "stop", NAME if by == "name" else app_id, *(["--yes"] if yes else []), timeout=60)
            time.sleep(15)
            rec = {"by": by, "yes": yes, "stdin": "closed", "app_id": app_id, "launch_log": log, "state_at_launch": st0,
                   "stop": res, "state_15s_after": state(app_id)}
            rec["cleanup"] = stop_confirmed(app_id)  # this stage's app is gone before the next launch
            out["variants"].append(rec)
            save()
            print(json.dumps({"by": by, "yes": yes, "rc": res["rc"], "state_15s_after": rec["state_15s_after"],
                              "after_cleanup": rec["cleanup"]["final_state"]}), flush=True)
            if rec["cleanup"]["final_state"] != "stopped":
                raise RuntimeError(f"{app_id} not confirmed stopped after cleanup: {rec['cleanup']['final_state']}")
            out["stages_completed"] += 1
        t0 = time.time()  # before the launch, so the 300 s bound counts from the start of this stage
        app_id, st0, log = launch("watcher", procs)
        launched.append(app_id)
        wlog = HERE / "results" / f"stop_live_watcher_{STAMP}.log"
        w = subprocess.run([sys.executable, str(HERE / "watch_cost.py"), "--app-id", app_id, "--stop", "0.01", "--rate", "1.5",
                            "--cap-min", "6", "--log", str(wlog)], capture_output=True, text=True, timeout=420, stdin=subprocess.DEVNULL)
        t1 = time.time()
        time.sleep(10)
        lines = [json.loads(x) for x in wlog.read_text().splitlines()] if wlog.exists() else []
        cost_stops = [x for x in lines if x.get("stop_attempted") and x.get("reason") == "cost" and x.get("returncode") == 0]
        wrec = {"app_id": app_id, "launch_log": log, "rc": w.returncode, "stdout": w.stdout, "stderr": w.stderr, "log": wlog.name,
                "log_lines": lines, "seconds_stage_start_to_watcher_exit": round(t1 - t0, 1), "state_after": state(app_id)}
        wrec["pass"] = bool(w.returncode == 0 and cost_stops and wrec["state_after"] == "stopped" and t1 - t0 < 300)
        wrec["cleanup"] = stop_confirmed(app_id)
        out["watcher"] = wrec
        out["stages_completed"] += 1
        save()
    except Exception:
        out["error"] = traceback.format_exc()
        save()
    finally:
        for p in procs:  # launchers first, so no new app can appear after the scan below; SIGKILL if SIGTERM is not enough
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(p.pid, sig)
                except ProcessLookupError:
                    pass
                except Exception as e:
                    out.setdefault("launcher_errors", []).append(repr(e))
                try:
                    p.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if p.poll() is None:
                out.setdefault("launchers_still_running", []).append(p.pid)
        try:
            extra = [x["app_id"] for x in listed() if x.get("description") == NAME and x.get("app_id") not in launched
                     and x.get("state") != "stopped" and str(x.get("created_at", "")) >= T_START]
        except Exception as e:
            extra, out["cleanup_scan_error"] = [], repr(e)
        for app_id in launched + extra:
            out["cleanup"].append(stop_confirmed(app_id))
        out["pass"] = bool(out["stages_completed"] == 5 and out["watcher"] and out["watcher"]["pass"] and "cleanup_scan_error" not in out
                           and "launchers_still_running" not in out
                           and all(r["final_state"] == "stopped" for r in out["cleanup"]))
        save()
        print(OUT.name, "pass:", out["pass"], "cleanup:", [r["final_state"] for r in out["cleanup"]], flush=True)
    if not out["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
