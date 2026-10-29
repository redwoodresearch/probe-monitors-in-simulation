"""Price one run_pilot.py launch's API policy spend (zoo C's real Sol runs, config/pilot_zoo_c_oct11c.yaml) for its
spend stop. Runs ON THE DEVBOX beside the launcher, with the September snapshot's Python (its sim.prices):

  ~/sims/simulated-control-evals-sept/.venv/bin/python policy_spend_watch.py --run <run dir> --launcher-pid <pid> \\
      --stop-usd 120

Every --interval seconds (30), each attempt folder under <run>/honest_b/ is priced from its policy calls (run_pilot.run_spend
with policy_calls: a finished log, or Inspect's buffer while the sample runs; the last reading is kept when neither can
be read). worst = the larger of --worst-per-run and the costliest finished attempt; committed = actual + each running
attempt's remaining worst, over the attempt folders listed in `attempts_seen`. The reading goes to <run>/policy_spend.json (replaced atomically), which run_pilot.py reads
before every new run (pilot `policy_stop_usd`: no fresh reading, no new run), and is appended to policy_spend.jsonl. If
actual spend alone reaches the stop, the launcher gets one SIGTERM, which stops every run. It ends when the launcher (the
pid, still a run_pilot.py process for this run folder) exits, after a final reading.
"""

import argparse
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from run_pilot import BufferGone, run_spend


def finished(log_dir: Path) -> bool:
    from inspect_ai.log import read_eval_log

    for p in log_dir.glob("*.eval"):
        try:
            if read_eval_log(str(p), header_only=True).status != "started":
                return True
        except Exception:  # noqa: BLE001 — still being written
            pass
    return False


def launcher_alive(pid: int, run: Path) -> bool:
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return any(a.endswith(b"run_pilot.py") for a in argv) and str(run).encode() in argv


def main(run: Path, pid: int, stop: float, worst_min: float, interval: int, policy: str):
    from sim.prices import load_prices

    prices = load_prices()
    assert policy in prices, f"{policy} has no price row: its spend can't be counted"
    assert launcher_alive(pid, run), f"{pid} is not run_pilot.py with --out {run}"
    for _ in range(300):  # the launcher creates the run folder (it refuses an existing one): never create it here
        if run.is_dir():
            break
        time.sleep(1)
    else:
        raise SystemExit(f"{run} never appeared: the launcher didn't start")
    last, done, signalled, read_errors, fails = {}, {}, False, 0, {}
    while True:
        alive = launcher_alive(pid, run)
        dirs = sorted((run / "honest_b").glob("*")) if (run / "honest_b").is_dir() else []
        try:  # attempts the launcher recorded as ended (process gone), and those it saw make no policy call
            lj = json.loads((run / "launch.json").read_text())
            ended_by_launcher = {Path(a["log_dir"]).name for r in lj["runs"] for a in r["attempts"] if "end" in a}
            no_call = {Path(a["log_dir"]).name for r in lj["runs"] for a in r["attempts"]
                       if "end" in a and a.get("policy_called") is False}
        except (OSError, ValueError, KeyError):
            ended_by_launcher, no_call = set(), set()
        for d in dirs:
            if d.name in done:
                continue
            ended = finished(d)
            try:
                usd = run_spend(d, policy, prices, policy_calls=True)
            except BufferGone:  # the sample just finished: read from its log next time
                usd = None
            except Exception:  # noqa: BLE001 — a half-written log or odd usage: keep the last reading, as spend() does
                usd, read_errors = None, read_errors + 1
            if usd is not None:
                last[d.name] = usd
            if ended and usd is not None:  # priced once, from its final log
                done[d.name] = usd
            elif d.name in no_call:  # no policy call at all: nothing more can be spent
                done[d.name] = last.get(d.name, 0.0)
            elif d.name in ended_by_launcher:  # gone, but its final log isn't read yet: retry, and after 5 failed
                fails[d.name] = fails.get(d.name, 0) + 1  # polls price it at the worst case (never below its reading)
                if fails[d.name] >= 5:
                    done[d.name] = max(worst_min, last.get(d.name, 0.0))
        last.update(done)  # a final or forced price (never below the live reading) counts in actual spend too
        actual = sum(last.values())
        worst = max(worst_min, max(done.values(), default=0.0))
        running = [d.name for d in dirs if d.name not in done]
        committed = actual + sum(max(0.0, worst - last.get(k, 0.0)) for k in running)
        rec = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "actual_usd": round(actual, 3),
               "finished": len(done), "running": len(running), "worst_per_run_usd": round(worst, 3),
               "committed_usd": round(committed, 3), "stop_usd": stop, "attempts_seen": [d.name for d in dirs],
               "read_errors_so_far": read_errors,
               "mean_finished_usd": round(sum(done.values()) / len(done), 3) if done else None}
        if actual >= stop and alive and not signalled:  # once: a second SIGTERM could kill the launcher's own wrap-up
            os.kill(pid, signal.SIGTERM)
            signalled = True
            rec["action"] = "launcher sent SIGTERM"
        tmp = run / "policy_spend.json.tmp"
        tmp.write_text(json.dumps(rec))
        tmp.replace(run / "policy_spend.json")
        with open(run / "policy_spend.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        if not alive:
            break
        time.sleep(interval)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--launcher-pid", type=int, required=True)
    ap.add_argument("--stop-usd", type=float, required=True)
    ap.add_argument("--worst-per-run", type=float, default=2.0)
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--policy", default="openai/gpt-5.6-sol")
    a = ap.parse_args()
    main(a.run.resolve(), a.launcher_pid, a.stop_usd, a.worst_per_run, a.interval, a.policy)
