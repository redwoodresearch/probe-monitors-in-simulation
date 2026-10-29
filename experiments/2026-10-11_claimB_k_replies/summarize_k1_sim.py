"""k = 1 stage: the simulator ledger in numbers (from <cache>/sim_results.jsonl and sim_starts.jsonl; no text): calls,
statuses, spend by model (sim.prices at the snapshot, as recorded per call), substitutions, trusted prefixes, and the
build fingerprint shared by every row.

  uv run --no-project --python 3.12 python experiments/2026-10-11_claimB_k_replies/summarize_k1_sim.py --cache <dir>
"""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    a = p.parse_args()
    raw = (a.cache / "sim_results.jsonl").read_bytes()
    rows = [json.loads(l) for l in raw.decode().splitlines()]
    starts = (a.cache / "sim_starts.jsonl").read_text().splitlines()
    by_model = Counter()
    for r in rows:
        for m, u in r["usage"].items():
            by_model[m] += 0  # keys only; spend per row is recorded as cost_usd
    builds = {json.dumps(r["build"], sort_keys=True) for r in rows}
    costs = [r["cost_usd"] for r in rows]
    out = {"label": "k = 1 simulator ledger (numbers only)", "calls": len(rows), "sites": len({r["site"] for r in rows}), "starts": len(starts),
           "statuses": dict(Counter(r["status"] for r in rows)), "spent_usd": sum(costs), "mean_usd": sum(costs) / len(costs), "max_usd": max(costs),
           "models": sorted({m for r in rows for m in r["usage"]}),
           "substitutions": {k: sum(1 for r in rows if (r["result"].get(k) or 0) > 0) for k in
                             ("fallback_substitutions", "refusal_substitutions", "block_substitutions", "net_substitutions")},
           "trusted_prefix": sum(1 for r in rows if r["result"].get("trusted_prefix")), "builds": len(builds),
           "build": json.loads(next(iter(builds))) if len(builds) == 1 else None, "results_sha256": hashlib.sha256(raw).hexdigest(),
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"k1_sim_ledger_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("calls", "sites", "starts", "statuses", "spent_usd", "mean_usd", "max_usd", "substitutions", "trusted_prefix", "builds")}))
    print("->", dst)


if __name__ == "__main__":
    main()
