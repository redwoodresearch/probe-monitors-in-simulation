"""Meaning judge, cost basis ($0, numbers only; b7's request after 0d found judges.gemini.cost_usd 2.3169 vs the $1.99
billed). The result files' cost_usd sums over job records, so a request served from the cache to several jobs (identical
replies in both orders, retests, resumes) is counted once per job. The billed cost counts each request once.

Per judge, by provider pin and status: records, record-sum cost, unique requests (request_sha256 for Gemini, cache_id for
Sol) and unique-request cost (each request's cost once). Then each judge's unique-request total, over all statuses and
pins, against its ledger. v2: the total counts each request once per pin and status, i.e. once per send: a request that
an upstream 429 rejected ($0) and that was later re-sent and answered keeps its billed answer (v1 kept the first, $0) (data/gem_cache/spend.json, data/sol_cache/spend_batch.json), to the cent.

  python3 experiments/2026-10-12_claimB_meaning_judge/analyze_cost_basis.py
"""

import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
DATA = HERE / "data"
JUDGES = {"sol": ("meaning_sol_calls.jsonl", "cache_id", "sol_cache/spend_batch.json"),
          "gemini": ("meaning_gemini_calls.jsonl", "request_sha256", "gem_cache/spend.json")}


def main():
    out = {"label": "Meaning judge cost basis: record sums vs unique-request (billed) sums", "judges": {}}
    for name, (calls, rid, ledger) in JUDGES.items():
        recs = [json.loads(x) for x in (DATA / calls).read_text().splitlines()]
        groups, uniq_all = defaultdict(lambda: {"records": 0, "record_sum_usd": 0.0, "unique": {}}), {}
        for r in recs:
            g = groups[f"{r.get('provider_pin') or 'openai-batch'}|{r['status']}"]
            g["records"] += 1
            g["record_sum_usd"] += r.get("cost_usd") or 0.0
            g["unique"].setdefault(r[rid], r.get("cost_usd") or 0.0)
            uniq_all.setdefault((r[rid], r.get("provider_pin"), r["status"]), r.get("cost_usd") or 0.0)  # v2: one per send
        led = json.loads((DATA / ledger).read_text())["spent"]
        out["judges"][name] = {
            "groups": {k: {"records": v["records"], "record_sum_usd": round(v["record_sum_usd"], 6), "unique_requests": len(v["unique"]),
                           "unique_request_usd": round(sum(v["unique"].values()), 6)} for k, v in sorted(groups.items())},
            "records": len(recs), "record_sum_usd": round(sum(r.get("cost_usd") or 0.0 for r in recs), 6),
            "unique_requests": len(uniq_all), "unique_request_usd": round(sum(uniq_all.values()), 6),
            "ledger_spent_usd": round(led, 6), "unique_matches_ledger_to_the_cent": abs(sum(uniq_all.values()) - led) < 0.005}
    out |= {"inputs": {f: hashlib.sha256((DATA / f).read_bytes()).hexdigest() for f in
                       ("meaning_sol_calls.jsonl", "meaning_gemini_calls.jsonl", "sol_cache/spend_batch.json", "gem_cache/spend.json")},
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"cost_basis_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "inputs"}, indent=1))


if __name__ == "__main__":
    main()
