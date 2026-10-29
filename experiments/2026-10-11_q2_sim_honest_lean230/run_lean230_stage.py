"""Run one stage of q2's lean230 simulated honest runs on the devbox, with a cost stop checked before each sub-batch.

Calls the n437 batch launcher (`run_sim_honest.py` at 902a57f, sha256 4e890036…ce99e, checked, with its
config/astra_honest.json beside it, checked) once per sub-batch with `--mains <sub-batch> --workers <size>` into one
--out folder, so the launcher's protocol and per-task inputs checks hold across sub-batches and a rerun resumes.

Before anything: the sims tree must be n437's (HEAD 3e0878a6, src diff sha256 491906a5…, `.settings` present), the
stage's references.json must equal the one in --refs, and every reference file must match its recorded sha256.
Before each sub-batch: spent = API cost of every .eval in --out (all attempts, from each header's model_usage at sce's
listed rates; a log with no header or no usage counts at WORST_USD, an unpriced model stops the stage) + devbox time
since this driver started (DEVBOX_USD_H); next = sum over the sub-batch's unfinished tasks of max(n437 cost, MEAN_USD) x
1.5. The sub-batch runs only if spent + next <= --stop. The stop is enforced between sub-batches only: inside one
sub-batch of --batch tasks, spend can run past the projection (n437's largest single run was about $34).
A nonzero launcher exit stops the stage (exit 1). SIGTERM or Ctrl+C to this driver is passed to the launcher as
SIGTERM, which stops its running evals (SIGINT, then a kill after 60 s), so nothing keeps spending.
Progress (keys and numbers only) goes to <out>/stage_progress.jsonl and stdout. Keys come from `rr-env run`.

  RR_OWNER_ID=aswin-surya rr-env run -- ~/simulated-control-evals-sept/.venv/bin/python run_lean230_stage.py \\
      --refs lean230_refs_stage1_<stamp>.json --refs-dir refs --out runs_stage1 --stop 511 --batch 10 \\
      --launcher launcher/run_sim_honest.py --sims-tree ~/simulated-control-evals-sept
(the September tree's Python, because the launcher imports its inspect_ai).
"""

import argparse
import hashlib
import json
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
LAUNCHER_SHA = "4e890036a7a86f910556e7eac2ea8287e7ccfa39b42f705fb101081931ace99e"  # the n437 batch launcher (902a57f = 0c9f6e0)
CONFIG_SHA = "cfc9a532adc5d08908d4b0108b95824e8c2f551c0156f2f79d2026d8562f744a"  # config/astra_honest.json beside it
# the n437 batch's assembled September tree, as its manifest records (protocol.sims_commit, sims_src_diff_sha256)
SIMS_COMMIT, SIMS_DIFF = "3e0878a608f4e75aff97d2b5dc3fb445cd5be4c0", "491906a57e9bdc4ce6875fc305c9bc4ae1cd9642dd5777a022e9ef9a0884fbd4"
RATES = {"openai/gpt-6-astra": (10.0, 50.0, 1.0), "openai/gpt-5.6-sol": (5.0, 30.0, 0.5), "openai/gpt-5.6-terra": (2.0, 12.0, 0.2),
         "openai/gpt-5-nano-2025-08-07": (0.05, 0.4, 0.005)}  # input, output, cached input per million (sce prices.py)
MEAN_USD, WORST_USD, DEVBOX_USD_H = 7.86, 35.0, 1.70  # m7i.8xlarge on demand, rounded up
CHILD: list[subprocess.Popen] = []
STATE: dict = {}  # out, t0, before: so a signal can save devbox hours too


def log_usd(path: Path) -> tuple[float, bool]:
    """(API cost, known), from the log header's model_usage via inspect_ai's own reader (the logs are zstd-compressed,
    which this Python's zipfile can't read). A log whose header or usage can't be read counts at WORST_USD."""
    from inspect_ai.log import read_eval_log

    try:
        stats = read_eval_log(str(path), header_only=True).stats
        usage = {m: u.model_dump() for m, u in (stats.model_usage or {}).items()} if stats else None
    except Exception:  # noqa: BLE001  (counted at the worst case and reported as logs_without_usage)
        usage = None
    if not usage:
        return WORST_USD, False
    total = 0.0
    for m, u in usage.items():
        if m not in RATES:
            raise SystemExit(f"unpriced model in {path.name}; stage stopped")
        r = RATES[m]
        total += ((u.get("input_tokens") or 0) * r[0] + (u.get("output_tokens") or 0) * r[1] + (u.get("input_tokens_cache_read") or 0) * r[2]) / 1e6
    return total, True


def save_hours(out: Path, t0: float, before: float) -> None:
    (out / "stage_time.json").write_text(json.dumps({"hours_before": before + (time.time() - t0) / 3600}))


def spent(out: Path, t0: float, before: float) -> tuple[float, int]:
    logs = [log_usd(p) for p in (out / "runs").glob("*/*.eval")] if (out / "runs").exists() else []
    return sum(c for c, _ in logs) + DEVBOX_USD_H * (before + (time.time() - t0) / 3600), sum(1 for _, k in logs if not k)


def finished_mains(out: Path) -> set:
    path = out / "progress.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return {r["main"] for r in rows if r["status"] in ("finished", "skipped (finished)")}


def note(out: Path, row: dict) -> None:
    row = {"at": datetime.now(PT).isoformat(timespec="seconds")} | row
    with open(out / "stage_progress.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def forward(signum, _frame):
    """Stop once: ignore further signals, pass SIGTERM on (a terminal Ctrl+C already reaches the launcher, which shares
    the process group), and wait for the launcher's own cleanup before exiting."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for proc in CHILD:
        if proc.poll() is None:
            if signum == signal.SIGTERM:
                proc.send_signal(signal.SIGTERM)  # the launcher's own handler stops its evals
            try:
                proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                proc.kill()
    if STATE:
        save_hours(STATE["out"], STATE["t0"], STATE["before"])
    raise SystemExit(f"stopped by signal {signum}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--refs", type=Path, required=True, help="make_stage_refs.py output")
    p.add_argument("--refs-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stop", type=float, required=True)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--launcher", type=Path, required=True)
    p.add_argument("--sims-tree", type=Path, required=True)
    args = p.parse_args()
    t0 = time.time()
    hours_path = args.out / "stage_time.json"
    before = json.loads(hours_path.read_text())["hours_before"] if hours_path.exists() else 0.0
    assert hashlib.sha256(args.launcher.read_bytes()).hexdigest() == LAUNCHER_SHA, "launcher differs from n437's"
    assert hashlib.sha256((args.launcher.parent / "config" / "astra_honest.json").read_bytes()).hexdigest() == CONFIG_SHA, "config differs from n437's"
    commit = subprocess.run(["git", "-C", str(args.sims_tree), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    diff = subprocess.run(["git", "-C", str(args.sims_tree), "diff", "HEAD", "--", "src"], capture_output=True, text=True, check=True).stdout
    assert commit == SIMS_COMMIT and hashlib.sha256(diff.encode()).hexdigest() == SIMS_DIFF, "the sims tree is not n437's"
    assert (args.sims_tree / ".settings").is_dir(), "the env checkouts (.settings) are missing"
    stage = json.loads(args.refs.read_text())
    tasks = stage["tasks"]
    for t in tasks:  # the file the launcher will open (from references[main]) is the one whose hash was checked
        r = stage["references"][t["main"]]
        assert t["reference"] == f"{r['environment']}/{r['fork_run']}/{r['source_sample'].replace(':', '_')}.eval", f"{t['main']}: reference path"
        assert hashlib.sha256((args.refs_dir / t["reference"]).read_bytes()).hexdigest() == t["reference_sha256"], f"{t['main']}: reference sha256"
    args.out.mkdir(parents=True, exist_ok=True)
    refs_file = args.out / "references.json"  # the launcher's --references input: this stage's tasks only
    if refs_file.exists():
        assert json.loads(refs_file.read_text()) == {"references": stage["references"]}, "references.json differs from --refs"
    else:
        refs_file.write_text(json.dumps({"references": stage["references"]}))
    STATE.update(out=args.out, t0=t0, before=before)
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    for i in range(0, len(tasks), args.batch):
        done = finished_mains(args.out)
        sub = [t for t in tasks[i:i + args.batch] if t["main"] not in done]
        if not sub:
            continue
        s, unknown = spent(args.out, t0, before)
        nxt = sum(max(t["n437_usd"], MEAN_USD) * 1.5 for t in sub)
        row = {"sub_batch": i // args.batch, "tasks": len(sub), "spent_usd": round(s, 2), "logs_without_usage": unknown,
               "next_projected_usd": round(nxt, 2), "stop_usd": args.stop}
        if s + nxt > args.stop:
            note(args.out, {"event": "stopped before sub-batch"} | row)
            save_hours(args.out, t0, before)
            raise SystemExit(2)
        note(args.out, {"event": "sub-batch start", "positions": [sub[0]["position"], sub[-1]["position"]]} | row)
        proc = subprocess.Popen([sys.executable, str(args.launcher), "--references", str(refs_file), "--refs-dir", str(args.refs_dir),
                                 "--out", str(args.out), "--mains", ",".join(t["main"] for t in sub), "--workers", str(len(sub)),
                                 "--sims-tree", str(args.sims_tree)])
        CHILD.append(proc)
        code = proc.wait()
        save_hours(args.out, t0, before)
        before, t0 = json.loads(hours_path.read_text())["hours_before"], time.time()
        STATE.update(t0=t0, before=before)
        s, unknown = spent(args.out, t0, before)
        note(args.out, {"event": "sub-batch end", "sub_batch": i // args.batch, "launcher_exit": code, "spent_usd": round(s, 2),
                        "logs_without_usage": unknown})
        if code != 0:
            note(args.out, {"event": "stage stopped: the launcher reported failures", "sub_batch": i // args.batch})
            raise SystemExit(1)
    s, unknown = spent(args.out, t0, before)
    save_hours(args.out, t0, before)
    note(args.out, {"event": "stage done", "spent_usd": round(s, 2), "logs_without_usage": unknown, "stop_usd": args.stop,
                    "finished": len(finished_mains(args.out)), "tasks": len(tasks)})


if __name__ == "__main__":
    main()
