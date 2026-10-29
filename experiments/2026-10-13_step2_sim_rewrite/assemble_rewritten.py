"""SUPERSEDED (10-13 17:30 PT; Aswin: nothing runs locally): stage_simrewrite.py now runs these gates on Modal CPU. Kept
because it made the two dry runs in results/partial/. Do not run it.

Step 2 pilot: assemble each finished run's rewritten transcript and check it, then report cost per run.

A rewritten transcript is the real sample with every non-submit tool message's content replaced by the simulator's
reply for that call as the reader sees it (`reader_output`: Inspect's 16 KiB truncation, as every arm's logs carry), and
its `error` cleared (the simulator returns text). Checks per run (PLAN.md):
  1. one ok simulated reply per real non-submit tool reply, matched by tool-call id (the counts are equal);
  2. every other message, and every tool message's other fields, byte-identical to the real sample (sha256 of each);
  3. 0 `attachment://` strings in the rewritten messages (saved alone: the sample's events hold Inspect's placeholders
     by design and are not part of the transcript).
A run with a missing, failed-only or extra reply is not assembled and is listed. Cost per run is every attempt's priced
provider usage (sim_results.jsonl), split by model, with the cached share of input tokens. Rewritten samples go to
<cache>/rewritten/ (outside the repo); results/ gets keys and numbers only.

  PYTHONPATH=src:$HOME/simulated-control-evals/src uv run --no-project --python 3.14 --with boto3 --with pyyaml python \\
      experiments/2026-10-13_step2_sim_rewrite/assemble_rewritten.py --cache <dir>
"""

import argparse
import gzip
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from probe_shift.transcripts import RUNNER_SUBMIT

HERE = Path(__file__).parent
HONEST = "la_public10_real_honest_n1000"
sha = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()  # noqa: E731


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    a = p.parse_args()
    runs = [json.loads(l) for l in (a.cache / "runs.jsonl").read_text().splitlines()]
    rows = [json.loads(l) for l in (a.cache / "sim_results.jsonl").read_text().splitlines()]
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run"]].append(r)
    out_dir = a.cache / "rewritten"
    out_dir.mkdir(exist_ok=True)
    report, problems = [], []
    for run in runs:
        i, rs = run["run"], by_run.get(run["run"], [])
        usd = defaultdict(float)
        tok = defaultdict(lambda: defaultdict(int))
        for r in rs:
            for m, u in r["usage"].items():
                for f in ("input_tokens", "input_tokens_cache_read", "output_tokens"):
                    tok[m][f] += u.get(f, 0) or 0
            usd["all"] += r["cost_usd"]
        ok, dup = {}, False
        for r in rs:
            if r["status"] == "ok":
                if r["k"] in ok:
                    problems.append({"run": i, "problem": f"two ok replies at k={r['k']}"})
                    dup = True
                ok[r["k"]] = r
        entry = {"run": i, "key": run["key"], "environment": run["environment"], "n_actions": run["n_actions"],
                 "ok_replies": len(ok), "attempts": len(rs), "failed_attempts": sum(r["status"] == "failed" for r in rs),
                 "usd": round(usd["all"], 4), "usd_per_call": round(usd["all"] / max(1, len(rs)), 4),
                 "tokens_by_model": {m: dict(v) for m, v in tok.items()},
                 "substitutions": {k: sum(bool(r["result"].get(k)) for r in ok.values())
                                   for k in ("fallback_substitutions", "refusal_substitutions", "block_substitutions", "net_substitutions")},
                 "trusted_prefix_calls": sum(bool(r["result"].get("trusted_prefix")) for r in ok.values()),
                 "assembled": False}
        report.append(entry)
        if dup or len(ok) != run["n_actions"]:
            continue  # duplicated, unfinished or failed: never assembled, listed by its counts
        calls = [json.loads(l) for l in (a.cache / "calls.jsonl").read_text().splitlines() if l.startswith(f'{{"run": {i}, ')]
        id_to_k = {c["step"]["id"]: c["k"] for c in calls}
        samples = json.loads(gzip.decompress((a.cache / "samples" / HONEST / f"{run['file']}.json.gz").read_bytes()))
        s = next(x for x in samples if x["id"] == run["sample_id"] and x["epoch"] == run["epoch"])
        submit_ids = {tc.get("id") for m in s["messages"] if m["role"] == "assistant" for tc in m.get("tool_calls") or []
                      if tc["function"] == RUNNER_SUBMIT}
        new_msgs, replaced, same, errors_cleared = [], set(), 0, 0
        for m in s["messages"]:
            if m["role"] == "tool" and m.get("tool_call_id") not in submit_ids and m.get("function") != RUNNER_SUBMIT:
                k = id_to_k[m["tool_call_id"]]  # KeyError = a tool reply with no simulated call
                assert k not in replaced, f"run {i}: two tool messages for k={k}"
                replaced.add(k)
                errors_cleared += bool(m.get("error"))
                new = dict(m, content=ok[k]["result"]["reader_output"], error=None)
                assert {f: v for f, v in new.items() if f not in ("content", "error")} == {f: v for f, v in m.items() if f not in ("content", "error")}
                new_msgs.append(new)
            else:
                new_msgs.append(m)
                same += 1
        checks = {"replies_equal_calls": replaced == set(range(run["n_actions"])),
                  "other_messages_identical": [sha(m) for m in new_msgs if m["role"] != "tool" or m.get("tool_call_id") in submit_ids]
                  == [sha(m) for m in s["messages"] if m["role"] != "tool" or m.get("tool_call_id") in submit_ids],
                  "message_count_equal": len(new_msgs) == len(s["messages"])}
        # The transcript is the messages (what the reader renders). Inspect keeps attachment:// placeholders in the
        # sample's events by design, so the saved copy holds the messages only and the check covers all of it.
        rewritten = {"key": run["key"], "id": s["id"], "epoch": s["epoch"], "messages": new_msgs}
        checks["no_attachment_strings"] = "attachment://" not in json.dumps(rewritten)
        entry |= {"checks": checks, "errors_cleared": errors_cleared, "unchanged_messages": same}
        if not all(checks.values()):
            problems.append({"run": i, "problem": "a check failed", "checks": checks})
            continue
        with open(out_dir / f"run_{i:02d}.json.gz", "wb") as fh:
            fh.write(gzip.compress(json.dumps(rewritten).encode()))
        entry["assembled"] = True
        entry["rewritten_sha256"] = sha(rewritten)
    done = [e for e in report if e["assembled"]]
    tot = sum(e["usd"] for e in report)
    res = {"label": "Step 2 pilot: assembly checks and cost per run (PRELIMINARY; keys and numbers only)",
           "runs_assembled": len(done), "runs_total": len(runs), "problems": problems,
           "api_usd_all_attempts": round(tot, 2),
           "usd_per_finished_run_mean": round(sum(e["usd"] for e in done) / len(done), 3) if done else None,
           "usd_per_call_mean_finished": round(sum(e["usd"] for e in done) / sum(e["attempts"] for e in done), 4) if done else None,
           "runs": report, "sim_results_sha256": hashlib.sha256((a.cache / "sim_results.jsonl").read_bytes()).hexdigest(),
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"assembly_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(res, indent=1))
    print({k: res[k] for k in ("runs_assembled", "runs_total", "api_usd_all_attempts", "usd_per_finished_run_mean", "usd_per_call_mean_finished")},
          "problems", len(problems), "->", dst)


if __name__ == "__main__":
    main()
