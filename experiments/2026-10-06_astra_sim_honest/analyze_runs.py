"""Check finished Astra simulated honest runs: validity, settings, cost, time, and that our reader loads them.

Per task folder `<out>/runs/<main>/` (one log per attempt; a resumed task can hold failed attempts too). The run is
the attempt whose sample has no error, else the latest; every attempt is listed and its cost counted:
  - validity by our rules (`probe_shift.transcripts.invalidity`: errored, stalled, no_actions) and submit count;
  - the settings the sample recorded, asserted equal to September's (simulator, fallback, judges, reference mode and
    cap, codebase read, no exec, no notes, no critic) and the reference it was given equal to the built one;
  - cost per model from `model_usage` at sce `sim.prices` (an unpriced model fails loudly), and wall-clock time;
  - `reader_messages` renders it (the Q1/Q2 input), with tool-output and tool-error counts.
Writes results/runs_<stamp>.jsonl and prints a summary.

  uv run python experiments/2026-10-06_astra_sim_honest/analyze_runs.py --out <run dir>
"""

import argparse
import json
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from sim.prices import load_prices

from probe_shift.transcripts import invalidity, reader_messages, submit_calls, system_prompt

HERE = Path(__file__).parent
EXPECTED = {
    "simulator_model": "openai/gpt-6-astra", "simulator_fallback_model": "openai/gpt-5.6-sol",
    "simulator_refusal_judge_model": "openai/gpt-5.6-terra", "simulator_router_judge_model": "openai/gpt-5.6-terra",
    "orig_trajectory_mode": "dump", "orig_trajectory_out_cap": 32768, "codebase_exec": False,
    "session_notes": False, "self_critique": None,
}


def samples(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as z:
        return [json.loads(z.read(n)) for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]


def log_status(path: Path) -> str | None:
    """The log's final status (`success`, `error`, `cancelled`), or None for a log that never finished."""
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("header.json")).get("status") if "header.json" in z.namelist() else None


UNKNOWN_USAGE: list[str] = []  # logs with no final header (a crashed eval): their spend is reported as unknown


def log_usage(path: Path) -> dict:
    """The log's aggregate model usage, which includes Inspect's internal retries (a sample's own usage does not).
    A log without its final header (the eval crashed) is listed in UNKNOWN_USAGE and counted as unknown, not zero."""
    with zipfile.ZipFile(path) as z:
        if "header.json" not in z.namelist():
            UNKNOWN_USAGE.append(str(path))
            return {}
        return json.loads(z.read("header.json"))["stats"]["model_usage"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=("honest", "attack"), default="honest")
    args = p.parse_args()
    prices = load_prices()
    rows = []
    runs = args.out / "runs"
    for run_dir in sorted(runs.iterdir()) if runs.exists() else []:  # a batch stopped before any task started
        logs = sorted(run_dir.glob("*.eval"))
        attempts = [(log, s) for log in logs for s in samples(log)]
        if not attempts:  # failed before writing a sample (or never started): counted, not fatal
            unknown_before = len(UNKNOWN_USAGE)
            rows.append({"main": run_dir.name, "invalid": "no_sample", "logs": [str(l) for l in logs],
                         "usd": 0.0, "usd_all_attempts": round(sum(sum(
                             prices[m].usd(input_tokens=u["input_tokens"], output_tokens=u["output_tokens"],
                                           cache_read_tokens=u.get("input_tokens_cache_read") or 0)
                             for m, u in log_usage(l).items()) for l in logs), 3)})
            rows[-1]["usage_unknown_logs"] = UNKNOWN_USAGE[unknown_before:]
            continue
        # A completed attempt is the launcher's: a successful log whose sample has no error. An interrupted
        # attempt can hold an error-free sample without a successful log; it counts for cost, not as the run.
        clean = [a for a in attempts if not a[1].get("error") and log_status(a[0]) == "success"]
        assert len(clean) <= 1, f"{run_dir}: {len(clean)} clean attempts; expected at most one"
        log, s = clean[0] if clean else attempts[-1]
        md = s["metadata"]
        recorded = {k: md.get(k) for k in EXPECTED}
        assert recorded == EXPECTED, f"{run_dir.name}: settings differ from September: {recorded}"
        assert md.get("codebase_root"), f"{run_dir.name}: no codebase_root, so codebase read had nothing to read"
        if args.mode == "honest":
            assert md.get("side_task") is None and "attack_approach" not in md, f"{run_dir.name}: not an honest sample"
        else:
            assert md.get("real_side_task_id") and (md.get("attack_approach") or "").strip(), \
                f"{run_dir.name}: an attack sample needs its side task and the September attack_approach"
        for key, want in (("simulator_router_mode", "narrow"), ("environment_briefing", None)):
            assert md.get(key, want) == want, f"{run_dir.name}: {key}={md.get(key)!r}, September had {want!r}"
        def usd(usage):
            return {m: prices[m].usd(input_tokens=u["input_tokens"], output_tokens=u["output_tokens"],
                                     cache_read_tokens=u.get("input_tokens_cache_read") or 0)
                    for m, u in (usage or {}).items()}
        cost = usd(s.get("model_usage"))
        unknown_before = len(UNKNOWN_USAGE)
        msgs = reader_messages(s, system_prompt(s))
        rows.append({
            "main": run_dir.name, "environment": md["env"], "log": str(log), "sample_id": s["id"],
            "attempts": len(attempts) + sum(len(a.get("error_retries") or []) for _, a in attempts),
            "usd_all_attempts": round(sum(sum(usd(log_usage(l)).values()) for l in logs), 3),
            "usage_unknown_logs": UNKNOWN_USAGE[unknown_before:],  # their spend is not in usd_all_attempts
            "invalid": invalidity(s) or (None if log_status(log) == "success" else f"log_{log_status(log) or 'unfinished'}"), "error": (s.get("error") or {}).get("message"), "submit_calls": submit_calls(s),
            "actions": sum(len(m.get("tool_calls") or []) for m in s["messages"] if m["role"] == "assistant"),
            "reference_actions": len(md["orig_trajectory"]), "fork": md["fork"],
            "tool_errors": sum(1 for m in s["messages"] if m["role"] == "tool" and m.get("error")),
            "reader_messages": len(msgs), "reader_tool_outputs": sum(m["role"] == "tool" for m in msgs),
            "total_time_s": s.get("total_time"), "working_time_s": s.get("working_time"),
            "usd": round(sum(cost.values()), 3), "usd_by_model": {m: round(c, 3) for m, c in cost.items()},
            "model_usage": s.get("model_usage"),
        })
    requested_path = args.out / "requested.jsonl"
    requested = sorted({m for line in requested_path.read_text().splitlines() for m in json.loads(line)["mains"]}
                       if requested_path.exists() else set())
    for main in sorted(set(requested) - {r["main"] for r in rows}):  # requested but never started
        rows.append({"main": main, "invalid": "never_started", "usd": 0.0, "usd_all_attempts": 0.0,
                     "usage_unknown_logs": []})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    out = HERE / "results" / f"runs_{stamp}.jsonl"
    with open(out, "x") as f:  # results/ is append-only
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        if r["invalid"] == "never_started":
            print(f"{r['main']:45s} NEVER STARTED")
            continue
        if r["invalid"] == "no_sample":
            print(f"{r['main']:45s} NO SAMPLE ({len(r['logs'])} log(s)) ${r['usd_all_attempts']:.2f}")
            continue
        print(f"{r['main']:45s} invalid={r['invalid']} actions={r['actions']} (reference {r['reference_actions']}) "
              f"submit={r['submit_calls']} ${r['usd']:.2f} {'?' if r['total_time_s'] is None else round(r['total_time_s'])}s {r['usd_by_model']}")
    lower_bound = any(r["usage_unknown_logs"] for r in rows)
    reasons = {}
    for r in rows:
        reasons[r["invalid"] or "valid"] = reasons.get(r["invalid"] or "valid", 0) + 1
    print(f"{len(rows)} runs by outcome {reasons}, "
          f"${sum(r['usd_all_attempts'] for r in rows):.2f} total over all attempts"
          f"{' (LOWER BOUND: some spend unknown, see usage_unknown_logs)' if lower_bound else ''}; wrote {out}")
    if UNKNOWN_USAGE:
        print(f"WARNING: {len(UNKNOWN_USAGE)} log(s) have no final header, so their spend is unknown: {UNKNOWN_USAGE[:5]}")


if __name__ == "__main__":
    main()
