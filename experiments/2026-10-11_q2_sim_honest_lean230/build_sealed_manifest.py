"""Sealed manifest for q2's lean230 stage 1 (Aswin, via 0d at 19:21:54 PT 10-11: "the lean230 runs (positions 1–50) are
a never-used confirmation set ... Record only: cost; run IDs; run status and errors; sha256s and sizes"). Reads only
each .eval log's HEADER through inspect_ai (`read_eval_log(header_only=True)`: status, eval/run ids, dataset sample ids,
model usage, log-level error type); never opens samples or transcript content, and computes nothing about the runs'
content or behaviour (no steps, lengths, times or scores). Run on the devbox with the September tree's Python.

Writes <out>/manifest_sealed.json:
  note: the seal and the end of the purchase;
  per task: position, main, environment, reference (path, sha256), the launcher's own status for the task, and for
  every log in runs/<main>/: file, sha256, bytes, header status, eval_id, run_id, sample ids, log-level error type (or
  null), API cost at sce's listed rates (all attempts);
  totals: tasks done / failed, logs, cost.

  ~/sims/simulated-control-evals-sept/.venv/bin/python build_sealed_manifest.py --out runs_stage1 --refs lean230_refs_stage1.json \\
      --prefix s3://rr-sce-sim-transcripts/transcripts/<name>/
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from inspect_ai.log import read_eval_log

RATES = {"openai/gpt-6-astra": (10.0, 50.0, 1.0), "openai/gpt-5.6-sol": (5.0, 30.0, 0.5), "openai/gpt-5.6-terra": (2.0, 12.0, 0.2),
         "openai/gpt-5-nano-2025-08-07": (0.05, 0.4, 0.005)}  # input, output, cached input per million (sce prices.py)
NOTE = ("SEALED (Aswin 19:21:54 PT 10-11 via 0d): confirmation set, not analysed. Purchase ended at stage 1 (Aswin, 19:26:13 PT "
        "10-11): positions 51-230 not bought. Batch started 18:03:07 PT while 2 of 2e's zoo C runs (p029_r2, p038_r2, "
        "media_processing) were still running; 2e's last run ended 18:29:01 PT, and 2e found none of its containers left at 18:31.")


def usd(usage) -> float:
    total = 0.0
    for m, u in (usage or {}).items():
        r = RATES[m]  # an unpriced model raises
        total += ((u.input_tokens or 0) * r[0] + (u.output_tokens or 0) * r[1] + (u.input_tokens_cache_read or 0) * r[2]) / 1e6
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--refs", type=Path, required=True)
    p.add_argument("--prefix", required=True)
    args = p.parse_args()
    stage = json.loads(args.refs.read_text())
    status = {}
    for line in (args.out / "progress.jsonl").read_text().splitlines():
        r = json.loads(line)
        status[r["main"]] = r["status"]  # the launcher's last word on each task
    tasks, total, n_logs = [], 0.0, 0
    for t in stage["tasks"]:
        logs = []
        for f in sorted((args.out / "runs" / t["main"]).glob("*.eval")):
            h = read_eval_log(str(f), header_only=True)
            cost = usd(h.stats.model_usage if h.stats else None)
            logs.append({"file": f.name, "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "bytes": f.stat().st_size, "status": h.status,
                         "eval_id": h.eval.eval_id, "run_id": h.eval.run_id, "sample_ids": [str(s) for s in (h.eval.dataset.sample_ids or [])],
                         "error_type": (h.error.message or "").split("(")[0].split(":")[0].strip()[:80] if h.error else None, "usd": round(cost, 4)})
            total += cost
            n_logs += 1
        tasks.append({"position": t["position"], "main": t["main"], "environment": t["environment"],
                      "reference": {"file": f"references/{t['reference']}", "sha256": t["reference_sha256"]},
                      "launcher_status": status.get(t["main"], "not run"), "logs": logs})
    done = sum(1 for t in tasks if t["launcher_status"] in ("finished", "skipped (finished)"))
    man = {"note": NOTE, "prefix": args.prefix, "created": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
           "recipe": "n437 (see PLAN.md): launcher 902a57f sha256 4e890036...ce99e, config astra_honest.json sha256 cfc9a532..., "
                     "September tree 3e0878a6 (src diff 491906a5...)",
           "stage_refs_sha256": hashlib.sha256(args.refs.read_bytes()).hexdigest(),
           "tasks_requested": len(tasks), "tasks_done": done, "tasks_not_done": len(tasks) - done, "logs": n_logs,
           "logs_with_error": sum(1 for t in tasks for g in t["logs"] if g["error_type"]),
           "usd_all_attempts": round(total, 2), "tasks": tasks,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dst = args.out / "manifest_sealed.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(man, indent=1))
    print(json.dumps({k: man[k] for k in ("tasks_requested", "tasks_done", "tasks_not_done", "logs", "logs_with_error", "usd_all_attempts")}))


if __name__ == "__main__":
    main()
