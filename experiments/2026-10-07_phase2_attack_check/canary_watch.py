"""Aswin's canary for the strict-tools rerun (10-08), and a guard for the whole launch. On the devbox, beside the
launcher, with the tree's venv:

  canary_watch.py <run dir> <canary task key> [--every 30]

Every `every` seconds it reads every real attempt of the launch (its finished .eval, read once; else Inspect's live
sample buffer) and judges every text_editor str_replace tool event, i.e. the call as executed:
- one without a non-empty new_str, anywhere and at any time: writes <run dir>/CANARY_ABORT, which run_pilot.py checks
  every loop and stops every run on (the file is the stop; a Ctrl+C to tmux -L phase2 session `run` is only a second
  try), and exits 2;
- the first completed one of the canary task with a non-empty new_str, no tool error, and its first changed line
  (new_str's first non-blank line not in old_str) in the result: writes <run dir>/CANARY_OK, which opens
  run_pilot.py's gate. An edit with no new line (indentation or a reorder) neither opens nor aborts. It keeps
  watching.
It exits 0 once the launch has finished (launch.json's `finished`) and every real attempt's log was read, and 3 if
every canary real run finished, all read, with no passing str_replace (the gate stays closed: ask Aswin). A log it
can't read yet is retried, never skipped.

With --open-on-clean-finish (the Qwen3-32B native-window arm, coordinator 10-09: old Qwen may never call str_replace),
the gate also opens once every canary real run finished and was read with no str_replace missing new_str, including
none at all; CANARY_OK then records the canary's str_replace count. Aborting is unchanged.
"""

import argparse
import glob
import json
import sqlite3
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from inspect_ai.log import read_eval_log

ABORT_MISSING_ONLY = False  # --abort-missing-only: an empty new_str is the model's own deletion, counted, not an abort

BUFFERS = Path.home() / ".local" / "share" / "inspect_ai" / "samplebuffer"


def buffer_events(log_dir: Path) -> list[dict]:
    """Tool events from a running attempt's sample buffer, attachments resolved (empty if none is readable yet)."""
    events = []
    for db in glob.glob(str(BUFFERS / "*" / "*.db")):
        if not any(Path(db).name.startswith(p.stem) for p in log_dir.glob("*.eval")) and str(log_dir) not in db:
            continue
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:  # an event is rewritten as it completes: keep each event_id's latest row
                rows = con.execute("select data from events where id in (select max(id) from events group by event_id)"
                                   ).fetchall()
                attachments = dict(con.execute("select hash, content from attachments").fetchall())
            finally:
                con.close()
        except sqlite3.Error:
            continue
        for (data,) in rows:
            try:
                d = json.loads(data)
            except (TypeError, ValueError):
                continue  # a row cut short mid-write: read again next round
            if d.get("event") == "tool":
                events.append(resolve(d, attachments))
    return events


def resolve(value, attachments: dict):  # noqa: ANN001, ANN201
    if isinstance(value, str) and value.startswith("attachment://"):
        return attachments.get(value.removeprefix("attachment://"), value)
    if isinstance(value, dict):
        return {k: resolve(v, attachments) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, attachments) for v in value]
    return value


def tool_events(log_dir: Path, cache: dict[str, list[dict]], prestart: str | None) -> tuple[list[dict], bool]:
    """(tool events of every .eval in the attempt's folder, whether all of them are final). A final .eval (with its
    sample, or a final status and none when the launcher recorded a definite pre-start failure for the attempt) is
    read once and cached by its path, so a file that appears later is still read; anything else (a live log, or a
    final one with no sample and no proven pre-start failure) also reads Inspect's buffer and stays unfinished."""
    events, all_final, any_file = [], True, False
    for path in sorted(log_dir.glob("*.eval")):
        any_file = True
        key = str(path)
        if key not in cache:
            try:
                log = read_eval_log(path, resolve_attachments=True)
            except Exception:  # noqa: BLE001 — still being written: read again next round
                all_final = False
                continue
            if log.samples:
                cache[key] = [e.model_dump() for e in log.samples[0].events if getattr(e, "event", "") == "tool"]
            elif log.status in ("success", "error", "cancelled") and prestart and not prestart.startswith("unknown"):
                cache[key] = []  # a final log with no sample and a proven pre-start failure: no call to judge
            else:
                all_final = False  # live, or ended without a sample and no proven pre-start failure
                continue
        events += cache[key]
    if not any_file or not all_final:
        events += buffer_events(log_dir)
    return events, any_file and all_final


def now() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds")


def verdict(event: dict) -> str | None:
    """'abort', 'pass', or None (not a str_replace, or its result isn't in yet)."""
    args = event.get("arguments") or {}
    if event.get("function") != "text_editor" or args.get("command") != "str_replace":
        return None
    if not isinstance(args.get("new_str"), str):
        return "abort"  # missing: the strict-tools bug
    if not args["new_str"].strip():
        return "empty" if ABORT_MISSING_ONLY else "abort"  # empty: the bug's deletion, or the model's own
    if event.get("pending") or not event.get("completed"):
        return None  # still running: its result isn't final
    if event.get("error") or event.get("result") is None:
        return None  # a tool error (e.g. old_str not found) says nothing about new_str; wait for another
    old = {line.strip() for line in str(args.get("old_str") or "").splitlines()}
    new = [line.strip() for line in args["new_str"].splitlines()]
    first = next((line for line in new if line and line not in old), "")
    if not first:
        return None  # indentation-only or a reorder: not the bug, but its edit can't be confirmed in the result
    return "pass" if first in str(event.get("result")) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("task")
    ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--open-on-clean-finish", action="store_true")
    ap.add_argument("--abort-missing-only", action="store_true")
    args = ap.parse_args()
    global ABORT_MISSING_ONLY
    ABORT_MISSING_ONLY = args.abort_missing_only
    cache: dict[str, list[dict]] = {}  # each final .eval's tool events, read once
    gate_open = (args.run_dir / "CANARY_OK").exists()
    while True:
        try:
            launch = json.loads((args.run_dir / "launch.json").read_text())
        except (OSError, json.JSONDecodeError):  # not written yet, or caught mid-rewrite: read again next round
            time.sleep(args.every)
            continue
        real = [r for r in launch["runs"] if r["real"]]
        canary = [r for r in real if r["task"] == args.task]  # launch.json keeps the task key
        assert canary, f"{args.task} has no real runs in this launch"
        aborts, passes, unread, canary_replaces, empties = [], [], 0, 0, []
        for r in real:
            for attempt in r["attempts"]:
                log_dir = attempt["log_dir"]
                events, done = tool_events(Path(log_dir), cache, attempt.get("prestart"))
                terminal = r["state"] not in ("running", "ready", "waiting") or attempt is not r["attempts"][-1]
                unread += terminal and not done and not attempt.get("redrawn_because")  # finished, log not read yet
                for e in events:
                    v = verdict(e)
                    canary_replaces += r["task"] == args.task and e.get("function") == "text_editor" and (
                        e.get("arguments") or {}).get("command") == "str_replace"
                    record = {"arm": r["arm"], "task": r["task"], "attempt": log_dir, "time": now(),
                              "keys": list((e.get("arguments") or {}))}
                    if v == "abort":
                        aborts.append(record)
                    elif v == "empty":
                        empties.append(record)
                    elif v == "pass" and r["task"] == args.task:
                        passes.append(record)
        if aborts:
            (args.run_dir / "CANARY_ABORT").write_text("".join(json.dumps(a) + "\n" for a in aborts))
            tried = subprocess.run(["tmux", "-L", "phase2", "send-keys", "-t", "run", "C-c"], capture_output=True,
                                   text=True)
            print(f"ABORT: {len(aborts)} real str_replace without a non-empty new_str, e.g. {aborts[0]}; "
                  f"run_pilot.py stops on CANARY_ABORT (Ctrl+C too: exit {tried.returncode} {tried.stderr.strip()})",
                  flush=True)
            raise SystemExit(2)
        if passes and not gate_open:
            (args.run_dir / "CANARY_OK").write_text(json.dumps(passes[0]) + "\n")
            gate_open = True
            print(f"{now()} OK, gate open: a real str_replace carried new_str and its edit shows: {passes[0]}; "
                  "still watching every real run", flush=True)
        canary_done = all(r["state"] in ("done", "no policy call", "stopped", "not run") for r in canary)
        if not gate_open and canary_done and not unread and args.open_on_clean_finish:
            record = {"opened_by": "every canary real run finished, read, no str_replace without new_str",
                      "canary_str_replace": canary_replaces, "time": now()}
            (args.run_dir / "CANARY_OK").write_text(json.dumps(record) + "\n")
            gate_open = True
            print(f"{now()} OK, gate open: {record}; still watching every real run", flush=True)
        print(f"{now()} final logs read: {len(cache)}; str_replace with an empty new_str (not aborts): "
              f"{len(empties)}; unread finished: {unread}; gate "
              f"{'open' if gate_open else 'closed'}; launch {'finished' if launch.get('finished') else 'running'}",
              flush=True)
        if launch.get("finished") and not unread:
            if not gate_open:
                print("launch finished, every real attempt read, no str_replace without new_str, but the gate never "
                      "opened: no confirmed real str_replace; ask Aswin")
                raise SystemExit(3)
            print("launch finished and every real attempt read: no str_replace without new_str")
            raise SystemExit(0)
        if not gate_open and canary_done and not unread:
            print("every canary real run finished and was read, with no passing str_replace: gate stays closed; "
                  "ask Aswin")
            raise SystemExit(3)
        time.sleep(args.every)


if __name__ == "__main__":
    main()
