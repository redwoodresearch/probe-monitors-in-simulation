"""Pricing numbers for one agent-pilot model, from its raw files. The Seed pilot's pricing_stats.py (83465c61) with the
model picked by --model. Read-only, $0.
Numbers only: no model text is read or written.

  uv run --no-project --python 3.14 python pricing_stats_pilots.py --model {gptoss,qcoder} --runs <smoke run dir> <main run dir> \\
      --server-logs <server_launch_*.log ...> --app-logs <saved `modal app logs` text, one per app ...> \\
      --billing <`modal billing report --json` output> --apps <app id,...> --devbox-usd <job watcher's devbox $>

Per honest run, from its .eval and launch.json:
- tokens summed over the run's policy calls: input, output, reasoning, and total (input + output);
- policy calls (turns);
- wall-clock (launch.json process_seconds, the Inspect process start to exit);
- the ending (launch.json and the sample error).
Per request, from the proxy's request logs: status, prompt and completion tokens, seconds. The logs carry no
cached-token count (prompt_tokens_details is null), so the input after prefix-cache hits is not available per request.
vLLM's prefix-cache hit rate is read from the servers' logged "Engine 000" lines, which `modal app logs` keeps only for
the last part of each app's life (partial). Server sizing comes from the launch logs (KV cache size) and the same lines.
Billing is Modal's report by app ID (hourly; it may still settle). Writes results/pricing_stats_<stamp>.json.
"""

import argparse
import gzip
import json
import re
import statistics
import subprocess
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
# policy name, reply cap (max_tokens of the runs' calls), what, server shape (RULE.md's table)
MODELS = {"gptoss": ("vllm/openai/gpt-oss-120b", 16384, "gpt-oss-120b @ b5c939de",
                     "H200 x1 per server, vLLM 0.30.0, --max-model-len 131072"),
          "qcoder": ("vllm/Qwen/Qwen3-Coder-Next", 32768, "Qwen3-Coder-Next @ a7fbcb5c",
                     "H200:2 per server (tensor parallel 2), vLLM 0.30.0, --max-model-len 262144")}
ENGINE = re.compile(r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, "
                    r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%")
PROGRESS = re.compile(r"running=(\d+) waiting=(\d+) succeeded=(\d+) generated=(\d+) tok/s kv=([\d.]+) gpu=(\d+)%")
KV_SIZE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")


def q(xs: list, p: float):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else None


def summary(xs: list) -> dict:
    return {"n": len(xs), "median": statistics.median(xs) if xs else None, "p90": q(xs, 0.9), "max": max(xs) if xs else None,
            "sum": sum(xs)}


def sample(ev: Path) -> dict:
    with zipfile.ZipFile(ev) as z:
        (n,) = [x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")]
        return json.loads(z.read(n))


def counted_dir(run_dir: Path, r: dict) -> Path:
    """The counted attempt's log dir, as report_qwen3_32b_native.py picks it (honest_a the first un-redrawn attempt,
    honest_b the last, e.g. a redraw after a setup failure)."""
    counted = [a for a in r["attempts"] if not a.get("redrawn_because")]
    a = counted[0] if r["arm"] == "honest_a" else counted[-1]
    return run_dir / Path(a["log_dir"]).relative_to(Path(a["log_dir"]).parents[1])


def main(model: str, runs: list[str], server_logs: list[str], app_logs: list[str], billing: str, apps: str, devbox_usd: float):
    POLICY, CAP, WHAT, GPU = MODELS[model]
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    rows, reqs = [], []
    for run in map(Path, runs):
        launch = json.loads((run / "launch.json").read_text())
        for r in launch["runs"]:
            counted = [x for x in r["attempts"] if not x.get("redrawn_because")]
            a = counted[0] if r["arm"] == "honest_a" else counted[-1]
            evs = sorted(counted_dir(run, r).glob("*.eval"))
            s = sample(evs[-1]) if evs else {}
            u = (s.get("model_usage") or {}).get(POLICY) or {}
            calls = sum(1 for e in s.get("events", []) if e.get("event") == "model" and e.get("model") == POLICY)
            err = str((s.get("error") or {}).get("message") or "")
            rows.append({"launch": run.name, "arm": r["arm"], "task": r["task"], "policy_calls": calls,
                         "input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0),
                         "reasoning_tokens": u.get("reasoning_tokens", 0),
                         "total_tokens": u.get("input_tokens", 0) + u.get("output_tokens", 0),
                         "process_seconds": a.get("process_seconds"), "sample_total_time_s": s.get("total_time"),
                         "error_kind": err.split("(")[0] if err else None,
                         "scored": (((s.get("scores") or {}).get("linux_scorer") or {}).get("value") or {}).get("main_task_success")})
        for f in sorted((run.parent / f"{run.name}.request_logs").glob("requests_*.jsonl.gz")):
            with gzip.open(f, "rt") as fh:
                for line in fh:
                    x = json.loads(line)
                    if x.get("kind") == "header":
                        continue
                    u = x.get("usage") or {}
                    reqs.append({"status": x.get("status"), "max_tokens": (x.get("params") or {}).get("max_tokens"),
                                 "prompt": u.get("prompt_tokens"), "completion": u.get("completion_tokens"),
                                 "cached": ((u.get("prompt_tokens_details") or {}) or {}).get("cached_tokens"), "seconds": x.get("seconds")})
    policy_reqs = [x for x in reqs if x["status"] == 200 and x["max_tokens"] == CAP]  # the runs' calls, not the server checks
    kv_sizes = [int(m.group(1).replace(",", "")) for f in server_logs for m in KV_SIZE.finditer(Path(f).read_text(errors="replace"))]
    eng, prog = [], []
    for f in app_logs:
        t = Path(f).read_text(errors="replace")
        eng += [tuple(map(float, m.groups())) for m in ENGINE.finditer(t)]
        prog += [tuple(map(float, m.groups())) for m in PROGRESS.finditer(t)]
    active = [e for e in eng if e[2] > 0]
    bill = json.loads(Path(billing).read_text())
    want = apps.split(",")
    by_app = {a: round(sum(float(r["cost"]) for r in bill if r.get("object_id") == a), 4) for a in want}
    endings = Counter(r["error_kind"] or "no error" for r in rows)
    out = {
        "git": git, "what": f"{WHAT} as the acting policy, 40 honest runs (20 tasks x honest_a, honest_b)", "model": model,
        "runs": len(rows),
        "per_run": {k: summary([r[k] for r in rows]) for k in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens",
                                                               "policy_calls", "process_seconds")},
        "per_run_rows": rows,
        "requests": {"policy_200": len(policy_reqs),
                     "prompt_tokens": summary([x["prompt"] for x in policy_reqs if x["prompt"] is not None]),
                     "completion_tokens": summary([x["completion"] for x in policy_reqs if x["completion"] is not None]),
                     "seconds": summary([x["seconds"] for x in policy_reqs if x["seconds"] is not None]),
                     "status_counts": dict(Counter(str(x["status"]) for x in reqs)),
                     "cached_tokens_reported": sum(x["cached"] is not None for x in policy_reqs)},
        "input_tokens_after_prefix_cache": None,
        "input_tokens_after_prefix_cache_note": "not in the logs: the proxy's usage has no cached-token count; the servers' "
                                                "prefix-cache hit rate (below, partial) is the only measure",
        "servers": {"gpu": GPU,
                    "kv_cache_tokens_per_server": sorted(set(kv_sizes)),
                    "smoke": {"servers": 1, "runs": 2}, "main": {"servers": 1, "runs": 38, "max_runs_per_server": 38},
                    "engine_lines_partial": len(eng), "engine_lines_with_running_requests": len(active),
                    "generation_tokens_per_s_per_server_when_busy": summary([e[1] for e in active]),
                    "prompt_tokens_per_s_per_server_when_busy": summary([e[0] for e in active]),
                    "running_requests_per_server": summary([e[2] for e in active]),
                    "kv_cache_usage_pct": summary([e[4] for e in active]),
                    "prefix_cache_hit_rate_pct": summary([e[5] for e in eng]),
                    "progress_lines_partial": len(prog), "kv_fraction_peak_in_progress_lines": max((p[4] for p in prog), default=None),
                    "gpu_util_pct_in_progress_lines": summary([p[5] for p in prog])},
        "billing_usd_by_app": by_app, "billing_usd_total": round(sum(by_app.values()), 4),
        "devbox_usd": devbox_usd,
        "endings": dict(endings), "submit_rate_note": "submitted runs are in combined_40 (ending 'submitted')",
    }
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"pricing_stats_{model}_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_run_rows"}, indent=1))
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--server-logs", nargs="+", required=True)
    ap.add_argument("--app-logs", nargs="+", required=True)
    ap.add_argument("--billing", required=True)
    ap.add_argument("--apps", required=True)
    ap.add_argument("--devbox-usd", type=float, required=True)
    main(**{k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()})
