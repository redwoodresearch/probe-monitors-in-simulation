"""Billing watcher, strict, v2.6: a copy of v2.5 (main @ c44b5b46, sha256 1c5631c1...) whose ONLY change is the failed-poll
rule (0d, 10-13 about 18:27 PT, after a laptop DNS drop at 18:09-18:11 made v2.5 stop healthy apps on one failed poll):
a failed or erroring poll is tolerated, with an ALERT line, while the last good bill is under 80% of the stop AND fewer than
TOLERATE (3) polls in a row have failed AND the last good poll is under MAX_GAP_S (300 s) old; the next failed poll after
that (the 4th in a row), any failed poll at >= 80% of the stop, or a 300 s gap stops the app as v2.5 does. A good poll resets
the count; a good poll at or over the stop, or the deadline, stops at once as before.
usage: uv run --no-project --python 3.12 python watch_billing_strict_v26.py <app id> <stop at $, whole app> <deadline, epoch s>
       uv run --no-project --python 3.12 python watch_billing_strict_v26.py --selftest   (must read the row 11 pilot app's settled $3.50 before any launch)
       python3 watch_billing_strict_v26.py --selftest-tolerance   (offline: the failed-poll rule's cases, PASS/FAIL each)
Every 60 s: the app's bill (modal 1.6.1 `billing report`, hourly, UTC-safe) and its state. Stops the app when the bill
reaches the threshold, at the deadline, or on a failed poll the rule above doesn't tolerate; a stop is retried until the app
shows as stopped. Exits once the app is idle for 3 polls after work."""
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")  # v2.4: stamps in Pacific Time (watchers run on the laptop only)

CLI = ["uv", "run", "--no-project", "--with", "modal==1.6.1", "modal"]


def say(msg):
    try:
        print(msg, flush=True)
    except Exception:  # v2 (10-12): diagnostics are never fatal
        pass


def cli(*args, timeout=60):
    """One CLI call in its own process group: on timeout the whole group (uv and the modal process under it) is killed,
    so a grandchild holding the pipes open can't hang the watcher; then TimeoutExpired is raised."""
    import os
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


def bill(app: str) -> float:
    # the report reads dates in UTC: end at the UTC date + 2 so every hour so far is inside (local + 1 cut off 17:00 PT)
    end = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%d")
    rows = json.loads(cli("billing", "report", "--start", "2026-10-11", "--end", end, "--resolution", "h", "--json"))
    return sum(float(r["cost"]) for r in (rows if isinstance(rows, list) else rows.get("data", rows)) if r.get("object_id") == app)


class AppMissing(Exception):
    """v2 (10-12): the app is not in `modal app list` (a dedicated type, so a KeyError is never read as "gone")."""


def state(app: str) -> tuple[str, str]:
    rows = json.loads(cli("app", "list", "--json"))
    a = next((a for a in (rows if isinstance(rows, list) else rows.get("data", rows)) if a.get("app_id") == app), None)
    if a is None:  # v2 (10-12): an app missing from the list is a failed lookup, never "gone" (it used to exit without a stop)
        raise AppMissing(f"{app} not in app list")
    return (a["state"], str(a["tasks"] if "tasks" in a else a["running_tasks"]))


def running(st) -> bool:
    """v2.6b (d9, 10-13 18:39): detached apps report "ephemeral (detached)", so any state starting with "ephemeral" counts,
    plus "deployed". Used for every running-state comparison."""
    return isinstance(st, str) and (st == "deployed" or st.startswith("ephemeral"))


def ended(st) -> bool:
    """v2.6b: "gone", or any state starting with "stopped". Used for every ended-state comparison."""
    return isinstance(st, str) and (st == "gone" or st.startswith("stopped"))


def stop(app: str, why: str):
    say(f"{datetime.now(PT).strftime('%H:%M:%S')} STOP: {why}")
    attempt, issued = 0, False
    while True:  # never gives up: retried until the app shows as stopped
        attempt += 1
        try:
            cli("app", "stop", app, "--yes", timeout=120)
            issued = True  # v2.2: a stop that succeeded once stays known across later retries
        except Exception as e:  # v2.2 (10-13): Modal errors on an already-stopped app, so a failed stop still checks state
            say(f"stop attempt {attempt}: app stop failed ({type(e).__name__}); checking state")
        try:
            st = state(app)[0]
        except AppMissing:  # v2.2: missing counts as gone only after a stop call succeeded; else it is a failed lookup
            st = "gone" if issued else None
            if not issued:
                say(f"stop attempt {attempt}: app not listed and no stop has succeeded yet; retrying")
        except Exception as e:  # fail closed: logged, then retried (sys.exit's SystemExit is not an Exception)
            st = None
            say(f"stop attempt {attempt}: state check failed ({type(e).__name__})")
        if ended(st):
            sys.exit(3)
        time.sleep(20)


TOLERATE, FRACTION, MAX_GAP_S = 3, 0.8, 300.0  # v2.6: failed polls tolerated in a row, bill share, longest gap


def tolerate_failure(fails: int, last_good: float, cap: float, gap_s) -> tuple[bool, str]:
    """v2.6: after `fails` failed polls in a row (this one included), with the last good bill and the seconds since the last
    good poll (None before the first good poll): (True, why) to keep polling, (False, why) to stop. Up to TOLERATE failures
    in a row are tolerated; the next stops. No failure is tolerated before the first good poll (as v2.5)."""
    if gap_s is None:
        return False, "no good poll yet"
    if last_good >= FRACTION * cap:
        return False, f"last good bill {last_good:.3f} >= {FRACTION:.0%} of the stop {cap}"
    if fails > TOLERATE:
        return False, f"{fails} failed polls in a row (tolerates {TOLERATE})"
    if gap_s > MAX_GAP_S:
        return False, f"{gap_s:.0f} s since the last good poll (> {MAX_GAP_S:.0f})"
    return True, f"failure {fails} of {TOLERATE} tolerated; last good bill {last_good:.3f} < {FRACTION:.0%} of {cap}"


def main(app: str, cap: str, deadline: str):
    cap, deadline, last, seen, idle, ready = float(cap), float(deadline), 0.0, False, 0, False
    fails, good_t = 0, None  # v2.6: failed polls in a row, time of the last good poll (None before the first)
    while True:
        try:
            last = bill(app)  # v2.6 (Codex): kept even if state() then fails, so the 80% rule sees a good bill read at once
            st, tasks = state(app)
            fails, good_t = 0, time.time()
        except Exception as e:  # v2.6: a failed or erroring poll (a timeout included) is tolerated or stops, per the rule
            fails += 1
            keep, why = tolerate_failure(fails, last, cap, None if good_t is None else time.time() - good_t)
            if keep and time.time() < deadline:
                say(f"{datetime.now(PT).strftime('%H:%M:%S')} ALERT: poll failed ({type(e).__name__}); {why}; polling again")
                time.sleep(60)
                continue
            say(f"{datetime.now(PT).strftime('%H:%M:%S')} ALARM: poll failed ({type(e).__name__}); {why}; stopping the app")
            stop(app, f"failed poll ({type(e).__name__}); {why}; last good bill {last:.3f}")
        say(f"{datetime.now(PT).strftime('%H:%M:%S')} state={st} tasks={tasks} bill={last:.3f} left={int(deadline - time.time())}s")
        if not ready and running(st) and last < cap and time.time() < deadline:  # modal run apps are ephemeral, detached ones "ephemeral (detached)"
            ready = True
            say("READY")
        if last >= cap or time.time() >= deadline:
            say(f"{datetime.now(PT).strftime('%H:%M:%S')} ALARM: bill {last:.3f} (stop {cap}) or deadline reached; stopping the app")
            stop(app, f"bill {last:.3f} (stop {cap}), deadline passed {time.time() >= deadline}")
        if ended(st):
            sys.exit(0)
        seen, idle = (True, 0) if tasks != "0" else ((seen, idle + 1) if seen else (seen, 0))
        if idle >= 3:  # v2 (10-12): finished work still ends in a stop, so an app that respawns is never left unwatched
            stop(app, "no tasks for 3 polls: finished")
        time.sleep(60)


def selftest_tolerance() -> bool:
    """v2.6, offline: the failed-poll rule on fixed cases; prints PASS/FAIL per case."""
    cases = [  # (name, fails, last good bill, stop, gap s, expected keep)
        ("2 failures under 80%: keep", 2, 10.0, 100.0, 120.0, True),
        ("3 failures under 80%: keep (the 3rd is tolerated)", 3, 10.0, 100.0, 180.0, True),
        ("4th failure in a row under 80%: stop", 4, 10.0, 100.0, 240.0, False),
        ("1 failure at 80% of the stop: stop", 1, 80.0, 100.0, 60.0, False),
        ("1 failure at 95% of the stop: stop", 1, 95.0, 100.0, 60.0, False),
        ("1 failure, 301 s since the last good poll: stop", 1, 10.0, 100.0, 301.0, False),
        ("1 failure before any good poll: stop", 1, 0.0, 100.0, None, False),
        ("bill read $110 then state() failed (cap $100): stop", 1, 110.0, 100.0, 60.0, False),
    ]
    ok = True
    for name, f, lg, c, g, want in cases:
        got = tolerate_failure(f, lg, c, g)[0]
        ok &= got == want
        print(f"{'PASS' if got == want else 'FAIL'}: {name}")
    # a good poll at or over the stop stops at once: main() checks `last >= cap` on every good poll (unchanged from v2.5);
    # a good poll resets the failure count, so 2 failures, a good poll, then 2 more never reach the 4th in a row
    seq, fails, stops = ["F", "F", "G", "F", "F"], 0, False
    for ev in seq:
        fails = 0 if ev == "G" else fails + 1
        stops |= ev == "F" and not tolerate_failure(fails, 10.0, 100.0, 60.0)[0]
    ok &= not stops
    print(f"{'PASS' if not stops else 'FAIL'}: F,F,good,F,F under 80%: keep (a good poll resets the count)")
    for s, want_run, want_end in [("ephemeral (detached)", True, False), ("ephemeral", True, False), ("deployed", True, False),
                                  ("stopped", False, True), ("gone", False, True), ("stopping", False, False), (None, False, False)]:
        good = running(s) == want_run and ended(s) == want_end
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}: state {s!r}: running={running(s)} ended={ended(s)}")
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest-tolerance"]:
        sys.exit(0 if selftest_tolerance() else 6)
    if sys.argv[1:] == ["--selftest"]:
        b = bill("ap-PcL8X4bXVggJqkQMhDdQg7")  # the row 11 pilot, billed $3.50 on 10-11 20:49-22:30 PT
        print(f"selftest bill {b:.4f}")
        sys.exit(0 if abs(b - 3.5008) < 0.01 else 5)
    main(*sys.argv[1:4])
