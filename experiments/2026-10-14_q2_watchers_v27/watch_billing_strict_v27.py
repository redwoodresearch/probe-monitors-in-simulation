"""Billing watcher v2.7 (q2, 10-14, after the 15:19 incident: v2.6b's "no good poll for 300 s -> stop the app" rule
killed a HEALTHY server while the laptop's network was failing). Same CLI as v2.6b; the change is the stop rule:
- the app is stopped ONLY on a FRESH bill reading at or over the stop: a `billing report` read in THIS poll, parsed, and
  summed over this app ID's rows. Nothing else stops it: failed, stale, slow or unparseable polls are logged (ALERT) and
  polling continues with backoff; the deadline only ends the watcher; an idle or finished app is never stopped.
- stops are by app ID only, retried until the app shows as stopped (as v2.6b).
This watcher is a CONVENIENCE. A job's spend is bounded by its function `timeout=` (about expected + 25%, with the worst
case at timeout <= its cap) and by the gates inside the job; a laptop that sleeps or goes offline must never stop a job.
usage: python3 watch_billing_strict_v27.py <app id> <stop at $, whole app> <deadline, epoch s>
       python3 watch_billing_strict_v27.py --selftest   (reads the row 11 pilot app's settled $3.50; needs the network)
The modal CLI is $WATCH_MODAL_CLI if set (e.g. "modal"), else `uv run --no-project --with modal==1.6.1 modal`."""
import json
import math
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
CLI = shlex.split(os.environ.get("WATCH_MODAL_CLI", "uv run --no-project --with modal==1.6.1 modal"))
POLL_S, MAX_BACKOFF_S, FRESH_MAX_S = 120.0, 600.0, 150.0  # v2.7: poll 120 s; back off to 600 s; a poll whose
# reads took longer than 150 s in all (e.g. a laptop suspended mid-poll) is stale and can never stop the app


def say(msg):
    try:
        print(f"{datetime.now(PT).strftime('%Y-%m-%d %H:%M:%S %Z')} {msg}", flush=True)
    except Exception:  # diagnostics are never fatal
        pass


def cli(*args, timeout=60):
    """One CLI call in its own process group; on timeout the whole group is killed, then TimeoutExpired is raised."""
    import signal
    p = subprocess.Popen(CLI + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.communicate()
        raise
    if p.returncode:
        raise subprocess.CalledProcessError(p.returncode, args, out, err)
    return out


def parse_bill(text: str, app: str) -> float:
    """Sum this app ID's rows of one `billing report --json` output; any non-finite cost or total raises (a failed read)."""
    rows = json.loads(text)
    rows = rows if isinstance(rows, list) else rows.get("data", rows)
    costs = [float(r["cost"]) for r in rows if r.get("object_id") == app]
    total = sum(costs)
    if not all(math.isfinite(c) for c in costs) or not math.isfinite(total):
        raise ValueError(f"non-finite cost in the billing report for {app}")
    return total


def bill(app: str) -> float:
    """This app ID's total bill from one `billing report` read (UTC dates; end at UTC date + 2 so every hour is inside)."""
    end = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%d")
    return parse_bill(cli("billing", "report", "--start", "2026-10-11", "--end", end, "--resolution", "h", "--json"), app)


class AppMissing(Exception):
    """The app is not in `modal app list`: a failed lookup, never read as "gone"."""


def state(app: str) -> tuple[str, str]:
    rows = json.loads(cli("app", "list", "--json"))
    a = next((a for a in (rows if isinstance(rows, list) else rows.get("data", rows)) if a.get("app_id") == app), None)
    if a is None:
        raise AppMissing(f"{app} not in app list")
    return (a["state"], str(a["tasks"] if "tasks" in a else a["running_tasks"]))


def running(st) -> bool:
    return isinstance(st, str) and (st == "deployed" or st.startswith("ephemeral"))


def ended(st) -> bool:
    return isinstance(st, str) and (st == "gone" or st.startswith("stopped"))


def stop(app: str, why: str, recheck=None, *, stop_cmd=None, state_fn=None, sleep_fn=time.sleep, max_attempts=None):
    """Stop by app ID until the app shows as stopped (then exit 3). Before every RETRY, `recheck()` must return True: a
    fresh reading that still meets the stop condition. False abandons the stop (returns "abandoned"); None (no fresh
    reading, e.g. offline) waits and asks again, so a retry never fires on stale evidence."""
    stop_cmd = stop_cmd or (lambda a: cli("app", "stop", a, "--yes", timeout=120))
    state_fn = state_fn or state
    say(f"STOP: {why}")
    attempt, issued = 0, False
    while max_attempts is None or attempt < max_attempts:
        if attempt and recheck is not None:
            try:
                still = recheck()
            except Exception as e:
                still = None
                say(f"stop recheck failed ({type(e).__name__}); waiting for a fresh reading")
            if still is False:
                say("stop abandoned: a fresh reading no longer meets the stop condition")
                return "abandoned"
            if still is None:
                sleep_fn(60)
                continue
        attempt += 1
        try:
            stop_cmd(app)
            issued = True
        except Exception as e:  # Modal errors on an already-stopped app, so a failed stop still checks state
            say(f"stop attempt {attempt}: app stop failed ({type(e).__name__}); checking state")
        try:
            st = state_fn(app)[0]
        except AppMissing:
            st = "gone" if issued else None
        except Exception as e:
            st = None
            say(f"stop attempt {attempt}: state check failed ({type(e).__name__})")
        if ended(st):
            sys.exit(3)
        sleep_fn(20)
    return "max_attempts"


def main(app, cap, deadline, *, bill_fn=bill, state_fn=state, stop_fn=stop, sleep_fn=time.sleep, now_fn=time.time,
         max_polls=None):
    """Poll until the app ends or the deadline passes. The only stop is a fresh bill reading >= cap (see the docstring)."""
    cap, deadline, fails, polls, ready = float(cap), float(deadline), 0, 0, False
    while max_polls is None or polls < max_polls:
        polls += 1
        if now_fn() >= deadline:
            say(f"deadline reached; the watcher exits WITHOUT stopping {app} (its timeout= bounds the job)")
            return "deadline"
        t0 = now_fn()
        try:
            b = bill_fn(app)  # fresh: read in this poll, parsed, summed over this app ID's rows
        except Exception as e:
            b = None
            say(f"ALERT: bill read failed ({type(e).__name__}); no stop on a missing reading; polling again")
        try:
            st, tasks = state_fn(app)
        except Exception as e:
            st = tasks = None
            say(f"ALERT: state read failed ({type(e).__name__}); polling again")
        took = now_fn() - t0
        if b is not None and b >= cap:  # stop only on a fresh bill AND a fresh running state, read within FRESH_MAX_S
            if running(st) and took <= FRESH_MAX_S and now_fn() < deadline:
                say(f"ALARM: fresh bill {b:.3f} >= stop {cap}, app {st}, read in {took:.0f} s; stopping {app}")

                def recheck():  # each stop retry needs a new fresh over-cap reading of a running app
                    r0 = now_fn()
                    b2, st2 = bill_fn(app), state_fn(app)[0]
                    return None if now_fn() - r0 > FRESH_MAX_S else (b2 >= cap and running(st2))
                if stop_fn(app, f"fresh bill {b:.3f} >= stop {cap}", recheck) != "abandoned":
                    return "stopped"
                say(f"stop abandoned (a fresh reading no longer met the stop); the watcher keeps polling {app}")
            else:
                say(f"ALERT: bill {b:.3f} >= stop {cap} but state {st!r}, reads took {took:.0f} s: not fresh; no stop")
        fails = fails + 1 if (b is None or st is None) else 0
        bill_txt = "n/a" if b is None else f"{b:.3f}"
        say(f"state={st} tasks={tasks} bill={bill_txt} stop={cap} left={int(deadline - now_fn())}s fails={fails}")
        if not ready and running(st):
            ready = True
            say("READY")
        if ended(st):
            say(f"{app} has ended ({st}); the watcher exits")
            return "ended"
        sleep_fn(min(POLL_S * 2 ** min(fails, 8), max(POLL_S, MAX_BACKOFF_S)))
    return "max_polls"


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest-tolerance"]:  # v2.6-compatible flag: the offline tests (cheap; run them on Modal)
        sys.exit(subprocess.run([sys.executable, str(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                "test_watchers_v27.py"))]).returncode)
    if sys.argv[1:] == ["--selftest"]:
        b = bill("ap-PcL8X4bXVggJqkQMhDdQg7")  # the row 11 pilot, billed $3.50 on 10-11 20:49-22:30 PT
        print(f"selftest bill {b:.4f}")
        sys.exit(0 if abs(b - 3.5008) < 0.01 else 5)
    r = main(*sys.argv[1:4])
    sys.exit(0 if r in ("ended", "deadline") else 4)
