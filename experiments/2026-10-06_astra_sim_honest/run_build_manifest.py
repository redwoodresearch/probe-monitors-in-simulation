"""Build the cohort's S3 layout and manifest from a finished run folder (pulled from the devbox).

Writes `<stage>/logs/<main>.eval` (the attempt that is the run: a successful log with an error-free sample, else the
latest), `<stage>/failed_attempts/<main>/*.eval` (any other logs of the task, kept), `<stage>/references/...` (the
reference each run was given), and `<stage>/manifest.json`:
  - cohort: the launch protocol (every setting, Inspect arguments, sims code), the config file, code provenance
    (archive path and sha256, base commit, patch sha256, the devbox commit of the assembled tree), Python version,
    machine, counts by outcome, total cost, and the reference-selection file;
  - per task: environment, cell, log file, sample id, outcome (`valid` or the invalidity reason), submit calls, actions,
    time, cost of the run and of all attempts (with the minimum number of model calls no log holds usage for, and
    whether `usd_all_attempts` is therefore only a lower bound), the reference (its file, fork run, source sample, the September
    sample it came from) and environment digests (September's, the one stamped, whether stamped locally), and retries:
    every Inspect retry with its error message, and whether the log's total token usage equals the final sample's
    (true means the failed attempts made no model call, so nothing outcome-dependent was redrawn).
Uploading `<stage>` to the cohort's S3 prefix is a separate, explicit step (`aws s3 sync <stage> <prefix>`).

  uv run python experiments/2026-10-06_astra_sim_honest/run_build_manifest.py --runs <pulled run folder> \\
    --refs-dir <references the runs used> --references <results/references_*.json> --config <config json> \\
    --mode honest --prefix s3://rr-sce-sim-transcripts/transcripts/<name>/ --stage <empty local folder> \\
    --machine "<one line>" --python "<version>" --launcher-source "<commit or bundle>"
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from sim.prices import load_prices

from probe_shift.transcripts import invalidity, submit_calls

HERE = Path(__file__).parent
PT = ZoneInfo("America/Los_Angeles")


def read(path: Path) -> tuple[dict | None, list[dict]]:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        header = json.loads(z.read("header.json")) if "header.json" in names else None
        samples = [json.loads(z.read(n)) for n in names if n.startswith("samples/") and n.endswith(".json")]
    return header, samples


def usd(usage: dict | None, prices) -> float:
    return sum(prices[m].usd(input_tokens=u["input_tokens"], output_tokens=u["output_tokens"],
                             cache_read_tokens=u.get("input_tokens_cache_read") or 0) for m, u in (usage or {}).items())


def min_unrecorded_calls(sample: dict | None, retries: list[dict]) -> int:
    """At least this many model calls have no usage in any log, though the provider may have billed them: calls that
    ended in an error (a timeout the caller then retried) or never finished (`pending`, e.g. cut off by the time
    limit), Inspect's API retries inside a call, and the in-flight call of each sample retry caused by a model-call
    timeout (`AttemptTimeoutError`; a sandbox `TimeoutError` involves no model call). A sample retry keeps only the
    error message, not the failed attempt's calls, so it counts as 1 here although the call may have been tried
    several times."""
    events = [e for e in (sample["events"] if sample else []) if e.get("event") == "model"]
    return (sum(bool(e.get("error") or e.get("pending")) + (e.get("retries") or 0) for e in events)
            + sum("AttemptTimeoutError" in (r.get("message") or "") for r in retries))


def tokens(usage: dict | None) -> dict:
    return {m: (u["input_tokens"], u["output_tokens"], u.get("input_tokens_cache_read") or 0) for m, u in (usage or {}).items()}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True, help="the launcher's --out folder")
    p.add_argument("--refs-dir", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--mode", choices=("honest", "attack"), required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--stage", type=Path, required=True)
    p.add_argument("--machine", required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--launcher-source", required=True, help="where the launcher that made the runs came from")
    args = p.parse_args()
    assert not args.stage.exists() or not any(args.stage.iterdir()), f"{args.stage} is not empty"
    prices, refs = load_prices(), json.loads(args.references.read_text())
    config = json.loads(args.config.read_text())
    protocol = json.loads((args.runs / "protocol.json").read_text())
    requested = sorted({m for line in (args.runs / "requested.jsonl").read_text().splitlines()
                        for m in json.loads(line)["mains"]})

    tasks, outcomes, total, total_all = [], Counter(), 0.0, 0.0
    for main in requested:
        ref = refs["references"][main]
        run_dir = args.runs / "runs" / main
        logs = sorted(run_dir.glob("*.eval")) if run_dir.exists() else []
        attempts = [(log, *read(log)) for log in logs]
        good = [a for a in attempts if a[1] and a[1].get("status") == "success" and a[2] and not a[2][0].get("error")]
        chosen = (good or attempts or [None])[-1]
        # An honest cell is its main task; an attack cell is main:side (the design cell the run was given).
        row = {"main": main, "environment": ref["environment"],
               "cell": ref["cell"] if args.mode == "attack" else main, "n_logs": len(logs)}
        if chosen is None:
            row["outcome"] = "never_started"
        else:
            log, header, samples = chosen
            s = samples[0] if samples else None
            status = header.get("status") if header else None
            row["outcome"] = ("no_sample" if s is None else invalidity(s) or
                              (None if status == "success" else f"log_{status or 'unfinished'}") or "valid")
            row |= {"log": f"logs/{main}.eval", "source_log_name": log.name, "log_status": status,
                    "usd": round(usd(s.get("model_usage") if s else None, prices), 4),
                    "usd_all_attempts": round(sum(usd((h or {}).get("stats", {}).get("model_usage"), prices)
                                                  for _, h, _ in attempts), 4),
                    "spend_unknown_logs": [l.name for l, h, _ in attempts if h is None]}
            # Every attempt of the task: other logs (earlier launches) and Inspect's retries inside each log. A
            # redraw is harmless only if the failed attempt made no model call before it failed.
            audit = []
            for other, h, ss in attempts:
                log_usage = (h or {}).get("stats", {}).get("model_usage")
                final = ss[0] if ss else None
                internal = (final or {}).get("error_retries") or []
                audit.append({
                    "log": other.name, "chosen": other == log, "status": (h or {}).get("status"),
                    "sample_error": ((final or {}).get("error") or {}).get("message", "")[:300] or None,
                    "internal_retries": [(r.get("message") or "")[:300] for r in internal],
                    "usd": round(usd(log_usage, prices), 4) if h else None,
                    # internal retries of this log spent nothing iff the log's usage equals its final sample's
                    "internal_retries_made_no_model_call": (tokens(log_usage) == tokens((final or {}).get("model_usage"))
                                                            if internal and h else None),
                    "min_unrecorded_model_calls": min_unrecorded_calls(final, internal),
                })
            row["attempts"] = audit
            row["min_unrecorded_model_calls"] = sum(a["min_unrecorded_model_calls"] for a in audit)
            # Spend may be missing if any call went unrecorded, or a sample retry followed model calls: the failed
            # attempt's events are gone, so a call in flight when it failed can't be ruled out.
            row["usd_all_attempts_is_lower_bound"] = bool(
                row["spend_unknown_logs"] or row["min_unrecorded_model_calls"]
                or any(a["internal_retries_made_no_model_call"] is False for a in audit))
            row["redrawn_after_model_calls"] = any(
                (not a["chosen"] and (a["usd"] is None or a["usd"] > 0)) or a["internal_retries_made_no_model_call"] is False
                for a in audit)
            if s is not None:
                md = s["metadata"]
                row |= {"sample_id": s["id"], "submit_calls": submit_calls(s),
                        "actions": sum(len(m.get("tool_calls") or []) for m in s["messages"] if m["role"] == "assistant"),
                        "total_time_s": s.get("total_time"), "working_time_s": s.get("working_time"),
                        "contract": {"stamped": md["arena_reference_contract"]["contents_sha256"],
                                     "september": ref["sept_contract"]["contents_sha256"]},
                        "contract_stamped_locally": ref["environment"] in config.get(
                            "september_digest_not_reproducible", ["media_processing", "model_registry"])}
            total += row["usd"]
            total_all += row["usd_all_attempts"]
            dest = args.stage / "logs" / f"{main}.eval"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log, dest)
            for other, _, _ in attempts:
                if other != log:
                    (args.stage / "failed_attempts" / main).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(other, args.stage / "failed_attempts" / main / other.name)
        reference = args.refs_dir / ref["environment"] / ref["fork_run"] / f"{ref['source_sample'].replace(':', '_')}.eval"
        recorded = run_dir / "inputs.json"
        if recorded.exists():  # the reference archived must be the one the run was launched with
            launched = json.loads(recorded.read_text())["reference_sha256"]
            assert sha256(reference) == launched, f"{main}: {reference} is not the reference this run was given"
        ref_dest = args.stage / "references" / reference.relative_to(args.refs_dir)
        ref_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reference, ref_dest)
        row["reference"] = {"file": f"references/{reference.relative_to(args.refs_dir)}", "sha256": sha256(reference),
                            "fork_run": ref["fork_run"], "source_sample": ref["source_sample"], "from_cell": ref["cell"],
                            "from_sim_uri": ref["sim_uri"], "from_sim_sample_id": ref["sim_sample_id"]}
        outcomes[row["outcome"]] += 1
        tasks.append(row)

    manifest = {
        "created": datetime.now(PT).isoformat(timespec="seconds"), "prefix": args.prefix, "mode": args.mode,
        "n_requested": len(requested), "outcomes": dict(outcomes),
        "usd_runs": round(total, 2), "usd_all_attempts": round(total_all, 2),
        "retried_tasks": sum(len(t.get("attempts", [])) > 1 or any(a["internal_retries"] for a in t.get("attempts", []))
                             for t in tasks),
        "redrawn_after_model_calls": [t["main"] for t in tasks if t.get("redrawn_after_model_calls")],
        "spend_unknown_logs": sum(len(t.get("spend_unknown_logs", [])) for t in tasks),
        "min_unrecorded_model_calls": sum(t.get("min_unrecorded_model_calls", 0) for t in tasks),
        "usd_all_attempts_is_lower_bound": any(t.get("usd_all_attempts_is_lower_bound") for t in tasks),
        "protocol": protocol, "config_file": args.config.name, "config": config,
        "reference_selection": {"file": args.references.name, "rule": refs.get("rule"), "sha256": sha256(args.references)},
        "machine": args.machine, "python": args.python,
        "probe_shift_commit": subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True,
                                             text=True).stdout.strip(),
        "launcher_source": args.launcher_source,
        "tasks": tasks,
    }
    (args.stage / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("n_requested", "outcomes", "usd_runs", "usd_all_attempts",
                                               "min_unrecorded_model_calls", "usd_all_attempts_is_lower_bound", "retried_tasks",
                                               "redrawn_after_model_calls")}))


if __name__ == "__main__":
    main()
