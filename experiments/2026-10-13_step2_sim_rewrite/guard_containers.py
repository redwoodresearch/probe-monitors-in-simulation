"""Container-count guard (RULE.md, this folder), beside the v2.1 watchers: every 30 s it lists the app's live containers
(`modal container list --json`, by app ID). It stops the app (`modal app stop <id> --yes`, retried until the app lists
as stopped) when more than --max containers are live, or on the first failed or malformed poll (fail closed). It exits
0 once the app lists as stopped; a stop it made exits 3.

  uv run --no-project --python 3.12 --with "modal==1.6.1" python guard_containers.py --app-id ap-... --max 8 --log PATH
"""
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone

CLI = ["uv", "run", "--no-project", "--python", "3.12", "--with", "modal==1.6.1", "modal"]


def sh(*args, timeout=90):
    try:
        return subprocess.run(CLI + list(args), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 124, "", "timeout")
    except OSError:
        return subprocess.CompletedProcess(args, 125, "", "launch failed")


def state(app_id):
    r = sh("app", "list", "--json")
    if r.returncode:
        raise RuntimeError(f"app list rc {r.returncode}")
    rows = [a for a in json.loads(r.stdout) if a.get("app_id") == app_id]
    if len(rows) != 1:
        raise RuntimeError("app missing from the app list")
    return rows[0]["state"]


def live(app_id):
    r = sh("container", "list", "--json")
    if r.returncode:
        raise RuntimeError(f"container list rc {r.returncode}")
    rows = json.loads(r.stdout)
    if not isinstance(rows, list) or not all(isinstance(c, dict) and isinstance(c.get("app_id"), str) and c.get("container_id") for c in rows):
        raise RuntimeError("container list is malformed")  # fail closed: a row without an app ID or container ID stops the app
    return sum(1 for c in rows if c["app_id"] == app_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-id", required=True)
    ap.add_argument("--max", type=int, required=True)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()

    def log(msg):  # diagnostics are never fatal: a logging failure can't stop the guard or its stop
        try:
            with open(a.log, "a") as fh:
                fh.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
        except Exception:
            pass
    log(f"guard start: app {a.app_id}, max {a.max} live containers")
    reason = None
    while reason is None:
        try:
            st = state(a.app_id)
            if st == "stopped":
                log("app stopped; guard exits")
                return 0
            n = live(a.app_id)
            log(f"state {st}, live containers {n}")
            if n > a.max:
                reason = f"{n} live containers > max {a.max}"
        except Exception as e:  # fail closed: any failed or malformed poll stops the app
            reason = f"failed poll ({type(e).__name__}: {e})"
        if reason is None:
            time.sleep(30)
    log(f"STOP: {reason}")
    while True:
        r = sh("app", "stop", a.app_id, "--yes")
        log(f"app stop rc {r.returncode}")
        try:
            if state(a.app_id) == "stopped":
                log("stop confirmed")
                return 3
        except Exception as e:
            log(f"state after stop failed ({type(e).__name__}); retrying")
        time.sleep(20)


if __name__ == "__main__":
    raise SystemExit(main())
