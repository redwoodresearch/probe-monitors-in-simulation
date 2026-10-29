"""RELAUNCH copy (RULE.md, Amendment 9) of gate1.py: the same live gate and graceful deadline, counting ONLY the relaunch's runs;
`final` reports per-replicate counts without the 196-run pass test (the relaunch holds 7 to 196 runs per replicate; the
per-replicate gate is computed across both launches at analysis). gate1.py's docstring follows.

Gate 1 and the graceful deadline of the 980-run launch (RULE.md, Amendments 2-3). On the devbox, with the September
venv's python.

  watch <run dir> <launcher tmux session> <stop at, epoch s>: in its own tmux session (gaterh: no session name may start
      with the launcher's, since tmux matches name prefixes). Every 120 s, while the launcher's session exists (exact
      match) and launch.json isn't finished, it reads <run>/launch.json (rewritten after every run start and end).
      Once >= 100 runs have ended, if fewer than 90% of them are usable it writes <run>/GATE1_STOP.txt and sends
      Ctrl+C to the launcher once (the launcher interrupts and records its runs). At <stop at> it does the same with
      <run>/DEADLINE_STOP.txt, so the launcher ends and writes its records before the laptop watcher's later deadline;
      the deadline is checked every pass, whether or not launch.json exists or reads whole. Ctrl+C goes first, so a
      marker that can't be written never blocks it. One line per check goes to <run>/gate1_watch.log (stdout until the
      run folder exists). Exits once the session is gone or the launch has finished.
  final <run dir>: per-replicate counts from the final launch.json -> <run>/gate1_final.json (a replicate passes with
      >= 177 usable of its 196 runs; batch C: 192 of 196), then the upload manifests, from every attempt directory the
      launcher recorded (redraws and failed attempts included): <run>/upload_manifest_r<i>.tsv (that replicate's .eval
      files) and <run>/upload_manifest_common.tsv (launch.json, gate1_final.json); one line per file:
      <path relative to the run>\t<bytes>\t<sha256>. Exits 1 unless all five replicates pass.

"Usable" is the launcher's own count (run_pilot.py's summarize(): the policy calls' `usable` of every attempt that ran
and wasn't redrawn); "ended" is the state "done" or "no policy call"."""
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


def replicate(task_key: str) -> str:
    rep = task_key.rsplit("_", 1)[1]
    assert rep in {f"r{i}" for i in range(1, 6)}, task_key
    return rep


def counts(launch: dict) -> dict:
    """Per replicate: runs, ended, usable, and the states."""
    out = {}
    for r in launch["runs"]:
        c = out.setdefault(replicate(r["task"]), Counter())
        c["runs"] += 1
        c[f"state {r['state']}"] += 1
        c["ended"] += r["state"] in ENDED
        c["usable"] += sum(p["usable"] for a in r["attempts"] if a.get("outcome") and not a.get("redrawn_because")
                           for p in a["outcome"]["policy_calls"])
    return {k: dict(v) for k, v in sorted(out.items())}


def stamp() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT")


def alive(session: str) -> bool:
    return subprocess.run(["tmux", "-L", "phase2", "has-session", "-t", f"={session}"],
                          stderr=subprocess.DEVNULL).returncode == 0


def interrupt(run: Path, session: str, name: str, why: str, note) -> None:
    """Ctrl+C first, so a marker that can't be written never blocks the stop."""
    subprocess.run(["tmux", "-L", "phase2", "send-keys", "-t", f"={session}:", "C-c"], check=True)
    note(f"{why}; Ctrl+C sent to {session}")
    try:
        (run / name).write_text(f"{stamp()} {why}\n")
    except OSError as e:
        note(f"could not write {name}: {e!r}")


def watch(run: Path, session: str, stop_at: float) -> None:
    log, stopped = None, False

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
            if not stopped and ended >= 100 and usable < 0.9 * ended:
                interrupt(run, session, "GATE1_STOP.txt", f"GATE 1 STOP: {usable} usable of {ended} ended runs (< 90%)",
                          note)
                stopped = True
        if not stopped and time.time() >= stop_at:  # whatever launch.json's state: the deadline never waits on it
            interrupt(run, session, "DEADLINE_STOP.txt", f"DEADLINE STOP at {stop_at:.0f} (epoch s)", note)
            stopped = True
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
    for v in c.values():
        v["pass"] = None  # across both launches, at analysis
    res = {"made": stamp(), "launch_finished": launch.get("finished"), "not_run": launch.get("not_run"),
           "stopped_by_signal": launch.get("stopped_by_signal"), "by_arm": launch.get("by_arm"), "replicates": c,
           "all_pass": None, "usable_share": sum(v["usable"] for v in c.values()) / max(1, sum(v["ended"] for v in c.values()))}
    (run / "gate1_final.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["replicates"]), "usable share", res["usable_share"], flush=True)
    files = {f"r{i}": set() for i in range(1, 6)}
    for r in launch["runs"]:
        for a in r["attempts"]:
            if a.get("log_dir"):
                files[replicate(r["task"])] |= {p for p in Path(a["log_dir"]).rglob("*.eval") if p.is_file()}
    files["common"] = {run / "launch.json", run / "gate1_final.json"}
    for name, paths in files.items():
        lines = [f"{p.relative_to(run)}\t{p.stat().st_size}\t{sha256(p)}\n" for p in sorted(paths)]
        (run / f"upload_manifest_{name}.tsv").write_text("".join(lines))
        print(f"upload_manifest_{name}.tsv: {len(lines)} files", flush=True)
    return True


if __name__ == "__main__":
    if sys.argv[1] == "watch":
        watch(Path(sys.argv[2]), sys.argv[3], float(sys.argv[4]))
    elif sys.argv[1] == "final":
        sys.exit(0 if final(Path(sys.argv[2])) else 1)
    else:
        sys.exit(f"unknown mode {sys.argv[1]}")
