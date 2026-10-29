"""How long are Qwen3.5-27B's turns, and which calls timed out? Laptop, no spend; numbers only.

  uv run python experiments/2026-10-07_phase2_attack_check/analyze_turn_lengths.py <run dir> [<run dir> ...]

Every model call in every .eval under the given folders: output tokens (reasoning plus content), reasoning tokens,
stop reason, seconds, the tool it called and its argument sizes. Calls that failed (timeouts) have no output; they are
counted separately with their error type and Inspect's retry count. Inspect records no working time for a failed call,
so its seconds are null, not 0. Logs are read with attachments resolved: unresolved, every long string (reasoning, a
written file) reads as a 45-character `attachment://` reference. Writes results/turn_lengths_<stamp>.json.
"""

import argparse
import glob
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from inspect_ai.event import ModelEvent
from inspect_ai.log import read_eval_log

HERE = Path(__file__).parent


POLICY = "vllm/Qwen/Qwen3.5-27B"  # config/run.yaml `policy`; a simulated run's sample also holds the simulator's calls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", type=Path, nargs="+")
    args = ap.parse_args()
    turns, failed, no_usage = [], [], []
    arms = {}  # log path -> arm (None: infer it from the sample)
    for d in args.run_dirs:
        launch = json.loads((d / "launch.json").read_text()) if (d / "launch.json").exists() else {}
        assert launch.get("config", {}).get("policy", POLICY) == POLICY, f"{d}: the launch's policy is not {POLICY}"
        if "runs" in launch:  # run_attack_check.py's runs: each run's final attempt, with its own arm
            for run in launch["runs"]:
                log_dir = Path(run.get("log_dir") or d / run["arm"] / run["env"])
                log_dir = d / log_dir.relative_to(log_dir.parents[1]) if not log_dir.exists() else log_dir
                arms |= {str(p.resolve()): run["arm"] for p in log_dir.glob("*.eval")}
        else:  # other runs (the 10-06 pilot's launch.json lists "tasks"): every log, arm inferred from the sample
            for p in glob.glob(str(d / "**" / "*.eval"), recursive=True):
                if "references" in Path(p).relative_to(d).parts:  # a sim_attack run's copied fork sources, not runs
                    continue
                arms.setdefault(str(Path(p).resolve()), None)  # never overwrite an arm a launcher recorded
    paths = sorted(arms)
    for path in paths:  # deduplicated, so overlapping run dirs can't count a log twice
        for sample in read_eval_log(path, resolve_attachments=True).samples or []:
            env = (sample.metadata or {}).get("environment", {}).get("id")
            arm = arms[path] or ("attack" if (sample.metadata or {}).get("real_side_task_id") else "honest")
            key = {"log": path, "sample": sample.id, "epoch": sample.epoch, "arm": arm, "env": env}
            for i, e in enumerate(x for x in sample.events if isinstance(x, ModelEvent) and x.model == POLICY):
                # A call still pending when the sample ended (the time limit) was cancelled, output or not.
                if e.error is not None or e.pending or e.output is None or not e.output.choices:
                    reason = str(e.error)[:80] if e.error is not None else "cancelled (pending at sample end)"
                    failed.append(key | {"call": i, "error": reason, "retries": e.retries, "seconds": e.working_time})
                    continue
                if e.output.usage is None:  # completed, but token counts unknown: neither a failure nor a turn length
                    no_usage.append(key | {"call": i})
                    continue
                u = e.output.usage
                msg = e.output.message
                turns.append(key | {"call": i, "output_tokens": u.output_tokens,
                                    "reasoning_tokens": u.reasoning_tokens or 0, "stop_reason": e.output.stop_reason,
                                    "seconds": round(e.working_time or 0, 1),
                                    "tools": [tc.function for tc in msg.tool_calls or []],
                                    "arg_keys": [sorted(tc.arguments) for tc in msg.tool_calls or []],
                                    "arg_chars": sum(len(json.dumps(tc.arguments)) for tc in msg.tool_calls or [])})
    out_tok = np.array([t["output_tokens"] for t in turns])
    stats = {"turns": len(turns), "no_usage": len(no_usage)} | ({} if not turns else {
        "median": float(np.median(out_tok)), "p90": float(np.percentile(out_tok, 90)),
        "p99": float(np.percentile(out_tok, 99)), "max": int(out_tok.max()),
        "over_8k": int((out_tok > 8000).sum()), "over_16k": int((out_tok > 16000).sum()),
        "over_24k": int((out_tok > 24000).sum()), "stop_reasons": dict(Counter(t["stop_reason"] for t in turns))})
    longest = sorted(turns, key=lambda t: -t["output_tokens"])[:8]
    fails = {"calls": len(failed), "calls_without_seconds": sum(f["seconds"] is None for f in failed),
             "by_error": dict(Counter(f["error"].split("(")[0] for f in failed)),
             "by_log": dict(Counter(f"{f['arm']}/{f['env']}/{Path(f['log']).name}" for f in failed))}
    print(json.dumps({"stats": stats, "longest": longest, "failed": fails}, indent=1))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"turn_lengths_{stamp}.json"
    with open(out, "x") as f:
        json.dump({"run_dirs": [str(d) for d in args.run_dirs], "logs": paths, "stats": stats, "longest": longest,
                   "failed_summary": fails, "failed": failed, "no_usage": no_usage, "turns": turns}, f, indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
