"""SET-WIDE billing guard for the pooled set (0d, 10-16 ~17:3x). Every POLL_S: ONE `billing report --start/--end` read (watch_billing_strict_v27's
cli; window UTC date - 3 .. + 2, inside Modal's 7-day hourly limit), summed over ALL the set's app ids (the ids file is re-read every poll, so
apps launched later join; '#' lines are comments, a line must be exactly one app id (ap-<letters and digits>), repeats count once, any other
line is an ALERT, as is a file with no ids). When a FRESH total reaches the cap, EVERY listed app that has not ended is stopped (by id, v27's
stop, bounded, never exiting the loop): one stop command per app first, then up to STOP_ROUNDS bounded re-stops of the apps not yet ended; a
missing id or an unreadable state still gets the bounded stops (an app that was stopped and has since dropped out of `app list` reads as
missing: it gets the stops and exit 4, a false alarm in the safe direction). Then the ids file is re-read and any id added meanwhile is stopped
the same way, for up to NEW_ID_PASSES passes; ids still arriving after that, an ids file unreadable after the stops, a line that is not an
app id (its app is unwatched), or any app not confirmed ended is an ALERT (exit 4), else exit 3.
A failed or stale read never stops anything (v27 rule) and is an ALERT; an id with no billing rows (it reads $0) is an ALERT. A read missing
some hours of a listed app cannot be detected (it under-counts). The guard works only while its host is awake: each app's timeout= is the hard
spend bound (v27).
usage: python3 watch_set.py <ids file: one app id per line> <cap $> <deadline epoch s>
       python3 watch_set.py --selftest   (reads the first catalogue attempt's settled bill through the set path; needs the network)"""
import importlib.util
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

V27 = Path(__file__).resolve().parents[2] / "2026-10-14_q2_watchers_v27/watch_billing_strict_v27.py"
spec = importlib.util.spec_from_file_location("v27", V27)
v27 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v27)
POLL_S, STOP_ATTEMPTS, STOP_ROUNDS, NEW_ID_PASSES = 120.0, 5, 2, 3


def say(msg):  # v27.say stamps the time
    v27.say(msg)


def read_ids(ids_file):
    """(the app ids in the file, in order; the non-comment lines that are not app ids). ALERTs on such a line and on a file with no ids (Claude
    r7 NB 1); the caller counts the bad lines as unstopped apps (Claude r9 NB 2)."""
    lines = [x.strip() for x in Path(ids_file).read_text().splitlines() if x.strip() and not x.strip().startswith("#")]
    ok = [x for x in lines if re.fullmatch(r"ap-[A-Za-z0-9]+", x)]  # r8 NB-A1: a whole line is one id; r8 NB-A2: repeats count once
    ids, bad = list(dict.fromkeys(ok)), [x for x in lines if x not in ok]
    if bad:
        say(f"ALERT: {len(bad)} line(s) in {ids_file} are not app ids (their spend is not watched): {bad[:4]}")
    if not ids:
        say(f"ALERT: no app ids in {ids_file}; the set bill reads $0")
    return ids, bad


def total(ids):
    """{app id: (bill $, billing rows)} from ONE billing report read; a non-finite cost raises (a failed read)."""
    now = datetime.now(timezone.utc)
    end, start = (now + timedelta(days=2)).strftime("%Y-%m-%d"), (now - timedelta(days=3)).strftime("%Y-%m-%d")
    txt = v27.cli("billing", "report", "--start", start, "--end", end, "--resolution", "h", "--json", timeout=90)
    rows = json.loads(txt)
    rows = rows if isinstance(rows, list) else rows.get("data", rows)
    return {a: (v27.parse_bill(txt, a), sum(1 for r in rows if r.get("object_id") == a)) for a in ids}


def stop_all(ids, why, *, state_fn=None, stop_fn=None):
    """Stop EVERY listed app that has not ended (Claude r6 BLOCKING 1; r7 NBs 2-4): ended apps are skipped; a missing id or an unreadable
    state is ALERTed and still gets the bounded stops (v27: a missing app is a failed lookup, never read as gone). One stop command goes to
    every live app first (no waits), then up to STOP_ROUNDS rounds of bounded v27 stops for the apps not yet ended; each stop's sys.exit (the
    app showed ended) is caught. No recheck: a bill is CUMULATIVE (never decreases), so a fresh over-cap reading stays true for every later
    stop and retry; v27's recheck guards per-app caps against stale readings, and here it could only delay a stop. Returns the apps not
    confirmed ended."""
    state_fn = state_fn or v27.state
    stop_fn = stop_fn or v27.stop

    def live(apps, alert):  # the apps not showing ended; a missing or unreadable one counts as live
        out = []
        for a in apps:
            try:
                st = state_fn(a)[0]
            except v27.AppMissing:
                st = "missing from the app list"
            except Exception as e:  # noqa: BLE001
                st = f"unreadable ({type(e).__name__})"
            if not v27.ended(st):
                if alert or not v27.running(st):
                    say(f"ALERT: {a} is not confirmed stopped (state {st})" if alert else f"ALERT: {a}: state {st}; stopping it anyway")
                out.append(a)
        return out

    def stop_pass(apps, n, **kw):
        for a in apps:
            try:
                stop_fn(a, why, max_attempts=n, **kw)
            except SystemExit:  # v27.stop exits once the app shows ended; go on to the next app
                pass

    left = live(ids, False)
    stop_pass(left, 1, sleep_fn=lambda s: None)  # every stop command goes out before any retry waits (NB 4)
    for _ in range(STOP_ROUNDS):  # bounded re-stops of the apps not yet ended (NB 3)
        left = live(left, False)
        if not left:
            break
        stop_pass(left, STOP_ATTEMPTS)
    return live(ids, True)


def main(ids_file, cap, deadline, *, total_fn=total, stop_all_fn=stop_all, now_fn=time.time, sleep_fn=time.sleep):
    while now_fn() < deadline:
        try:
            ids, bad = read_ids(ids_file)
            t0 = now_fn()
            per = total_fn(ids)
            fresh = now_fn() - t0 < v27.FRESH_MAX_S
        except Exception as e:  # noqa: BLE001  a failed read (or ids file) never stops anything (v27)
            say(f"ALERT: set bill read failed ({type(e).__name__}); polling again")
            sleep_fn(POLL_S)
            continue
        s = sum(b for b, _ in per.values())
        say(f"set bill={s:.3f} cap={cap} apps={len(ids)} fresh={fresh} per={json.dumps({k: round(b, 3) for k, (b, _) in per.items()})}")
        nobill_ = sorted(a for a, (_, n) in per.items() if n == 0)
        if nobill_:
            say(f"ALERT: no billing rows yet for {nobill_} (they read $0)")
        if s >= cap and not fresh:
            say(f"ALERT: a STALE read is over the cap ({s:.2f} >= {cap}); no stop on stale evidence, polling again")
        if s >= cap and fresh:
            why, done, left = f"set-wide bill {s:.2f} >= {cap}", [], []
            todo = ids
            for _ in range(NEW_ID_PASSES):  # ids added to the file while stopping are stopped too (507-prep, r7)
                left += stop_all_fn(todo, why)
                done += todo
                try:
                    ids2, bad2 = read_ids(ids_file)
                    todo = [a for a in ids2 if a not in done]
                    bad += [b for b in bad2 if b not in bad]
                except Exception as e:  # noqa: BLE001
                    say(f"ALERT: ids file unreadable after the stops ({type(e).__name__}); new ids may be unstopped")
                    left.append("<ids file unreadable after the stops>")  # r8 NB-A3: fail closed (exit 4)
                    todo = []
                if not todo:
                    break
            if todo:
                say(f"ALERT: ids still being added after {NEW_ID_PASSES} stop passes; not stopped: {todo}")
                left += todo
            left += [f"<not an app id, unwatched and unstopped: {b}>" for b in bad]  # Claude r9 NB 2: a bad line is never 'stopped' (exit 4)
            say("every listed live app is stopped; the watcher exits" if not left else f"ALERT: apps not confirmed stopped: {left}")
            return 3 if not left else 4
        sleep_fn(POLL_S)
    say("deadline reached; the watcher exits (it never stops on a deadline)")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        r = total(["ap-A3SanChZKF4pO2H395LaNk"])["ap-A3SanChZKF4pO2H395LaNk"]  # the first catalogue attempt, settled at $0.333
        print(f"selftest set bill {r[0]:.4f} rows {r[1]}")
        sys.exit(0 if abs(r[0] - 0.333) < 0.01 and r[1] > 0 else 1)
    sys.exit(main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3])))
