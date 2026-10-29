"""Real honest epoch 2 (experiments/2026-10-15_q3_realhonest_epoch2/RULE.md), the +311 and nm360 launches: gate_ep2.py (T196) with
the groups being the ARMS honest_a and honest_b (each run's launch.json "arm"; the task keys carry no batch suffix) in place of the
batches e2a/e2b; everything else as gate_ep2.py: the expected count per arm read from the launch (its distinct tasks), an arm
passing with >= 90% usable of its runs, and the watch ALERT ONLY.
ALERT ONLY (d9 10-15 ~11:45, Aswin's standing rule: in-job gates pause, never kill; a real-only launch has no pause): the
watch never sends Ctrl+C or any signal. Below the alert line it writes <run>/GATE1_ALERT.txt once; at the deadline it writes
<run>/DEADLINE_ALERT.txt once; the laptop reads both and tells d9. The alert line is the 1e-4 job's measured usable rate minus
3 standard errors at 100 runs: 753 usable of 772 ended in its relaunch and 188 of 192 in its first launch (97.5%,
results/unusable_by_cause_2026-10-14_234651.json and gate1_final_2026-10-14_152712.json there), so 0.975 - 3 x 0.0156 = 0.93.
The runs are bounded by the servers' Modal timeout and the box's poweroff, not by this script. On the devbox, with the
September venv's python.

  watch <run dir> <launcher tmux session> <stop at, epoch s>: in its own tmux session (gaterh: no session name may start
      with the launcher's, since tmux matches name prefixes). Every 120 s, while the launcher's session exists (exact
      match) and launch.json isn't finished, it reads <run>/launch.json (rewritten after every run start and end).
      Once >= 100 runs have ended, if fewer than 93% of them are usable it writes <run>/GATE1_ALERT.txt once; at <stop at>
      it writes <run>/DEADLINE_ALERT.txt once (checked every pass, whether or not launch.json reads whole). No signal is
      ever sent: alert only. One line per check goes to <run>/gate1_watch.log (stdout until the
      run folder exists). Exits once the session is gone or the launch has finished.
  final <run dir>: per-batch counts from the final launch.json -> <run>/gate1_final.json (a batch passes with all its tasks
      present and >= 90% of them usable), then the upload manifests, from every attempt directory the
      launcher recorded (redraws and failed attempts included): <run>/upload_manifest_<batch>.tsv (that batch's .eval
      files) and <run>/upload_manifest_common.tsv (launch.json, gate1_final.json); one line per file:
      <path relative to the run>\t<bytes>\t<sha256>. Exits 1 unless both batches pass.

"Usable" is the policy calls' `usable` (run_pilot.py's summarize()) of each run's COUNTED attempt only (the first try's redraw chain
before any reference rerun, the attempt without redrawn_because; at most one, asserted): unlike gate_ep2.py's honest_b-only T196 launch, honest_a here makes reference reruns
after an unusable counted attempt, and those never count (Codex r64). "Ended" is the run's state "done" or "no policy call"
(so its reruns have finished too). Upload manifests keep every attempt."""
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ENDED = ("done", "no policy call")


BATCHES = ("honest_a", "honest_b")


def group(r: dict) -> str:
    assert r["arm"] in BATCHES and r["real"], (r["key"], r["arm"])
    return r["arm"]


def counted(r: dict) -> list:
    """The run's counted attempt (RULE.md, Counting; the counting note): the first try's redraw chain, i.e. the attempts before
    the first reference rerun (kind "first", then "redraw"s), and of those the one without redrawn_because. At most one.
    honest_a's reference reruns (kind "rerun", after a counted attempt that wasn't usable) never count (Codex r64), nor do the
    redraws of a rerun (kind "redraw" after a "rerun": run_pilot.py start()/reap(); Codex r65)."""
    chain = []
    for a in r["attempts"]:
        if a.get("kind") == "rerun":
            break
        chain.append(a)
    assert [a.get("kind") for a in chain[:1]] in ([], ["first"]) and all(a.get("kind") == "redraw" for a in chain[1:]), r["key"]
    c = [a for a in chain if not a.get("redrawn_because")]
    assert len(c) <= 1, (r["key"], len(c))
    return c


def counts(launch: dict) -> dict:
    """Per arm: runs, ended (the run's state, so honest_a's reruns have finished too), usable (of the counted attempt only),
    reference reruns, and the states."""
    out = {}
    for r in launch["runs"]:
        c = out.setdefault(group(r), Counter())
        c["runs"] += 1
        c[f"state {r['state']}"] += 1
        c["ended"] += r["state"] in ENDED
        c["usable"] += sum(p["usable"] for a in counted(r) if a.get("outcome") for p in a["outcome"]["policy_calls"])
        c["reference_reruns"] += sum(a.get("kind") == "rerun" for a in r["attempts"])
    return {k: dict(v) for k, v in sorted(out.items())}


def stamp() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT")


def alive(session: str) -> bool:
    return subprocess.run(["tmux", "-L", "phase2", "has-session", "-t", f"={session}"],
                          stderr=subprocess.DEVNULL).returncode == 0


ALERT_RATE = 0.93  # the 1e-4 job's measured 97.5% usable minus 3 standard errors at 100 runs (module docstring)


def interrupt(run: Path, session: str, name: str, why: str, note) -> None:
    """ALERT ONLY: no Ctrl+C, no signal of any kind (d9, Aswin's standing rule); a marker file and a log line."""
    note(f"{why}; ALERT ONLY: no signal sent to {session}")
    try:
        (run / name).write_text(f"{stamp()} {why}\n")
    except OSError as e:
        note(f"could not write {name}: {e!r}")


def watch(run: Path, session: str, stop_at: float) -> None:
    log, stopped, deadline_noted = None, False, False

    def note(msg: str) -> None:
        nonlocal log
        if log is None and run.is_dir():
            log = open(run / "gate1_watch.log", "a")
        print(f"{stamp()} {msg}", file=log or sys.stdout, flush=True)

    while alive(session):
        launch = None
        try:
            launch = json.loads((run / "launch.json").read_text())
        except FileNotFoundError:  # before the launcher's first save
            note("no launch.json yet")
        except json.JSONDecodeError:  # caught mid-rewrite: the next check reads it whole
            note("launch.json unreadable (mid-write?); next check")
        if launch is not None:
            if launch.get("finished"):
                note("launch finished")
                return
            c = counts(launch)
            ended, usable = sum(v["ended"] for v in c.values()), sum(v["usable"] for v in c.values())
            note(f"ended {ended}, usable {usable}; " +
                 "; ".join(f"{k}: ended {v['ended']} usable {v['usable']}" for k, v in c.items()))
            if not stopped and ended >= 100 and usable < ALERT_RATE * ended:
                interrupt(run, session, "GATE1_ALERT.txt", f"GATE 1 ALERT: {usable} usable of {ended} ended runs (< {ALERT_RATE:.0%})",
                          note)
                stopped = True
        if not deadline_noted and time.time() >= stop_at:  # whatever launch.json's state: the deadline never waits on it
            interrupt(run, session, "DEADLINE_ALERT.txt", f"DEADLINE ALERT at {stop_at:.0f} (epoch s)", note)
            deadline_noted = True
        time.sleep(120 if launch is not None else 30)
    note(f"session {session} gone")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def final(run: Path) -> bool:
    launch = json.loads((run / "launch.json").read_text())
    c = counts(launch)
    want = len({r["task"] for r in launch["runs"]})
    for v in c.values():
        v["pass"] = v["runs"] == want and v["usable"] >= 0.9 * want
    res = {"made": stamp(), "launch_finished": launch.get("finished"), "not_run": launch.get("not_run"),
           "stopped_by_signal": launch.get("stopped_by_signal"), "by_arm": launch.get("by_arm"), "replicates": c,
           "tasks_per_batch": want, "all_pass": len(c) == len(BATCHES) and all(v["pass"] for v in c.values())}
    (run / "gate1_final.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["replicates"]), "all_pass", res["all_pass"], flush=True)
    files = {b: set() for b in BATCHES}
    for r in launch["runs"]:
        for a in r["attempts"]:
            if a.get("log_dir"):
                files[group(r)] |= {p for p in Path(a["log_dir"]).rglob("*.eval") if p.is_file()}
    files["common"] = {run / "launch.json", run / "gate1_final.json"}
    for name, paths in files.items():
        lines = [f"{p.relative_to(run)}\t{p.stat().st_size}\t{sha256(p)}\n" for p in sorted(paths)]
        (run / f"upload_manifest_{name}.tsv").write_text("".join(lines))
        print(f"upload_manifest_{name}.tsv: {len(lines)} files", flush=True)
    return res["all_pass"]


if __name__ == "__main__":
    if sys.argv[1] == "watch":
        watch(Path(sys.argv[2]), sys.argv[3], float(sys.argv[4]))
    elif sys.argv[1] == "final":
        sys.exit(0 if final(Path(sys.argv[2])) else 1)
    else:
        sys.exit(f"unknown mode {sys.argv[1]}")
