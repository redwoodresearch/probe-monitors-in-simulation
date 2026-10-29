"""Mocked test of watch_set.py's stop path (no Modal call, no real app is touched). Stops go through the REAL v27.stop (retry bound and
"never ends" path) with a fake stop command, a fake app list and no sleeping. Covers: running apps stopped and ended ones skipped whatever the
order (a missing id in the middle never blocks later apps); a missing id, an app that never ends and an unreadable state each get exactly the
bounded stop commands and end in exit 4; exit 3 only when every live app is confirmed stopped; ids added to the file during the stops are
stopped; a stale over-cap read and failed reads stop nothing; bad lines and an empty ids file are ALERTs. Usage: python3 test_watch_set.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import watch_set as W  # noqa: E402

said = []
W.v27.say = said.append
BOUND = 1 + W.STOP_ROUNDS * W.STOP_ATTEMPTS  # stop commands an app that never ends receives
ok, states, cmds, never, unread, late, gone = [], {}, [], set(), set(), {}, set()
f = Path(tempfile.mkdtemp()) / "ids.txt"


def state_fn(a):
    if a in unread:
        raise RuntimeError("app list timed out")
    if a not in states:
        raise W.v27.AppMissing(a)
    return (states[a], "0")


def cmd(a):  # fake `modal app stop`: errors on an unknown id; an app in `never` keeps running
    cmds.append(a)
    if a in late:  # an app launched while the guard is stopping: its id lands in the ids file
        f.write_text(f.read_text() + late.pop(a) + "\n")
    if a in gone:  # the ids file becomes unreadable during the stops
        gone.discard(a)
        f.unlink()
    if a not in states and a not in unread:
        raise RuntimeError("no such app")
    if a not in never:
        states[a] = "stopped"


def stop_fn(a, why, max_attempts=None, **kw):
    return W.v27.stop(a, why, stop_cmd=cmd, state_fn=state_fn, sleep_fn=lambda s: None, max_attempts=max_attempts)


def run(ids_text, st, bill=500.0, deadline=1e9, clock=None, total_fn=None):
    states.clear()
    states.update(st)
    cmds.clear()
    f.write_text(ids_text)
    clk = iter(clock or range(0, 10_000, 5))
    return W.main(str(f), 400.0, deadline, total_fn=total_fn or (lambda ids_: {a: (bill, 1) for a in ids_}),
                  stop_all_fn=lambda ids_, why: W.stop_all(ids_, why, state_fn=state_fn, stop_fn=stop_fn), now_fn=lambda: next(clk), sleep_fn=lambda s: None)


R, E = "ephemeral (detached)", "stopped"
rc = run("ap-ended1\nap-missing\nap-run1\nap-ended2\nap-run2\n", {"ap-ended1": E, "ap-ended2": E, "ap-run1": R, "ap-run2": R})
ok.append(("missing id in the middle: both running apps stopped, ended ones never sent a stop",
           states["ap-run1"] == E and states["ap-run2"] == E and "ap-ended1" not in cmds and "ap-ended2" not in cmds))
ok.append((f"missing id: exactly the bounded {BOUND} stop commands, then exit 4", cmds.count("ap-missing") == BOUND and rc == 4))
ok.append(("every stop command goes out before any retry (first commands in id order)", cmds[:3] == ["ap-missing", "ap-run1", "ap-run2"]))
never.add("ap-stuck")
rc = run("ap-stuck\nap-run1\n", {"ap-stuck": R, "ap-run1": R})
never.clear()
ok.append((f"an app that never ends: {BOUND} stop commands (real v27 retry bound), later app still stopped, exit 4",
           cmds.count("ap-stuck") == BOUND and states["ap-run1"] == E and rc == 4))
unread.add("ap-unread")
rc = run("ap-unread\nap-run1\n", {"ap-run1": R})
unread.clear()
ok.append(("unreadable state: still sent the bounded stops, not confirmed, exit 4", cmds.count("ap-unread") == BOUND and rc == 4))
rc = run("# the pooled set\nap-ended1\nap-run1\nap-run2\n", {"ap-ended1": E, "ap-run1": R, "ap-run2": R})
ok.append(("exit 3 when every live app is confirmed stopped (comment line ignored)", rc == 3 and sorted(cmds) == ["ap-run1", "ap-run2"]))
late["ap-run1"] = "ap-late"
rc = run("ap-run1\n", {"ap-run1": R, "ap-late": R})
ok.append(("an id added to the file during the stops is stopped too, exit 3", states["ap-late"] == E and rc == 3))
rc = run("ap-run1\n", {"ap-run1": R}, deadline=1000, clock=[0, 0, 200, 2000])  # one poll whose read takes 200 s (stale), then the deadline
ok.append(("a stale over-cap read stops nothing", rc == 0 and cmds == []))


def boom(ids_):
    raise RuntimeError("billing report failed")


said.clear()
rc = run("ap-run1\n", {"ap-run1": R}, deadline=12, total_fn=boom)
ok.append(("failed reads stop nothing and ALERT", rc == 0 and cmds == [] and any("read failed" in m for m in said)))
said.clear()
f.write_text("ap-run1\napp-typo\n\n# note\n")
ids, bad = W.read_ids(str(f))
ok.append(("a line that is not an app id is an ALERT", ids == ["ap-run1"] and bad == ["app-typo"] and any("not app ids" in m and "app-typo" in m for m in said)))
said.clear()
f.write_text("# nothing yet\n")
ok.append(("an ids file with no ids is an ALERT", W.read_ids(str(f)) == ([], []) and any("no app ids" in m for m in said)))
said.clear()
f.write_text("ap-run1\nap-x1 # note\nap-run1\nap-run2\n")
ids, bad = W.read_ids(str(f))
ok.append(("r8 A1/A2: an id line with trailing text is an ALERT, repeated ids count once",
           ids == ["ap-run1", "ap-run2"] and bad == ["ap-x1 # note"] and any("not app ids" in m and "ap-x1 # note" in m for m in said)))
rc = run("ap-run1\nap-x1 # note\n", {"ap-run1": R})
ok.append(("Claude r9 NB 2: over the cap with a bad line, the listed app is stopped and the guard exits 4 (the bad line's app is unstopped)",
           states["ap-run1"] == E and rc == 4))
gone.add("ap-run1")
rc = run("ap-run1\n", {"ap-run1": R})
ok.append(("r8 A3: an ids file unreadable after the stops exits 4 (fail closed)", states["ap-run1"] == E and rc == 4))
for n, v in ok:
    print("PASS" if v else "FAIL", n)
print("ALL PASS" if all(v for _, v in ok) else "FAILURES")
sys.exit(0 if all(v for _, v in ok) else 1)
