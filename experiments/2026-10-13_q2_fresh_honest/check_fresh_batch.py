"""Fresh-task honest runs: settings check of a whole batch (check_fresh_smoke.py's per-run checks over every run). For each
honest_b/<run id>/*.eval of a transferred run folder: the plan and eval config against the batch's first run, reasoning
effort on every policy call, the sample's status, error and limit; then the watcher's final priced spend and the cost per
run. Counts and settings only, no message text. $0, local (the run folder is about 7 MB of .eval logs).

  uv run --no-project --python 3.12 --with zstandard python experiments/2026-10-13_q2_fresh_honest/check_fresh_batch.py <run folder>
"""
import argparse
import hashlib
import json
import subprocess
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import evalzip  # noqa: F401  (zstd members)

HERE = Path(__file__).parent
POLICY = "openai/gpt-5.6-sol"


def main(rd: Path):
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    evals = sorted((rd / "honest_b").glob("*/*.eval"))
    plans, cfgs, efforts, status, calls, with_calls = Counter(), Counter(), Counter(), Counter(), [], 0
    for p in evals:
        with zipfile.ZipFile(p) as z:
            h = json.loads(z.read("header.json"))
            names = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
            if len(names) > 1:
                raise SystemExit(f"{p.parent.name}: an .eval with {len(names)} samples")
            s = json.loads(z.read(names[0])) if names else {"events": [], "error": "no sample (setup failed)"}
        plans[json.dumps([(x["solver"], x.get("params", {})) for x in h["plan"]["steps"]], sort_keys=True)] += 1
        cfgs[json.dumps({k: h["eval"]["config"].get(k) for k in ("time_limit", "max_samples", "epochs", "retry_on_error")}, sort_keys=True)] += 1
        c = [e for e in s["events"] if e.get("event") == "model" and e.get("model") == POLICY]
        calls.append(len(c))
        efforts.update(str((e.get("config") or {}).get("reasoning_effort")) for e in c)
        status[(h.get("status"), bool(s.get("error")), (s.get("limit") or {}).get("type"))] += 1
        with_calls += bool(c)
    fin = json.loads((rd / "policy_spend.json").read_text())
    out = {"label": "fresh-task honest batch settings check", "git": git, "run folder": rd.name,
           "evals sha256 (sorted, joined)": hashlib.sha256("".join(hashlib.sha256(p.read_bytes()).hexdigest() for p in evals).encode()).hexdigest(),
           "distinct plans": len(plans), "distinct eval configs": len(cfgs), "eval configs": {k: v for k, v in cfgs.items()},
           "policy calls": {"total": sum(calls), "min per run": min(calls), "max per run": max(calls)},
           "reasoning effort counts": dict(efforts), "effort medium on every call": set(efforts) == {"medium"},
           "status, error, limit counts": {json.dumps(k): v for k, v in status.items()},
           "watcher final": {k: fin.get(k) for k in ("time", "actual_usd", "finished", "running", "stop_usd")},
           "attempts (.eval files, pre-start redraws included)": len(evals), "attempts with a policy call": with_calls,
           "API $ per attempt with a policy call": round(fin["actual_usd"] / with_calls, 4) if with_calls else None}
    # settings: identical plan and config, medium effort on every call, nothing running. Errored attempts are reported in the
    # status counts (the run list decides which runs are errored), not failed here.
    out["checks passed"] = bool(len(plans) == 1 and len(cfgs) == 1 and out["effort medium on every call"] and fin.get("running") == 0)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dst := HERE / "results" / f"batch_check_{rd.name}_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_folder", type=Path)
    main(ap.parse_args().run_folder)
