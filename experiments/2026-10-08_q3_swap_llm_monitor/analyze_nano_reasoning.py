"""Quick check: does the nano monitor's reasoning mention the formatting fixes? $0, local, no model calls.

For the N runs whose mean suspicion moved most between as simulated (a) and both fixes (b), it reads all 6 cached
completions (3 draws per arm, keyed from the archived prompts exactly as the run cached them). It flags reasoning that
uses a formatting term (gutter, line numbers, cat -n, diff format, rendering, formatting) or any clause of the review
note, read from rerender.py's REVIEW string. Every flag is
then read by eye. Writes counts and per-run scores only; the reasoning text stays in the git-ignored cache and on S3.

Run from the repo root, with the sims repo's environment (it rebuilds the client's cache keys):
  uv run --project ~/simulated-control-evals --frozen python experiments/2026-10-08_q3_swap_llm_monitor/analyze_nano_reasoning.py \
      --scores <results/monitor_scores_*.json> --inputs <data/monitor_inputs_*.jsonl.gz>
"""

import argparse
import ast
import gzip
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from metrics.monitor import trajectory as tm
from runner.metrics import clients

HERE = Path(__file__).parent
MODEL, EFFORT, K, N = "openai/gpt-5.4-nano", "medium", 3, 10
RERENDER = HERE.parent / "2026-10-07_q2_rendering_swap" / "rerender.py"  # the review note the fix adds, read without importing
REVIEW = next(n.value.value for n in ast.parse(RERENDER.read_text()).body
              if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "REVIEW")
CLAUSES = [c.strip() for c in re.split(r"[(),.]| and ", REVIEW) if len(c.strip().split()) >= 2]  # every clause of the note
TERMS = re.compile("|".join([r"gutter", r"line[- ]number", r"numbered", r"review note", r"\|\+", r"cat -n", r"formatting", r"diff format",
                             r"rendering"] + [re.escape(c) for c in CLAUSES]), re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--inputs", required=True)
    args = ap.parse_args()
    sc = json.loads(Path(args.scores).read_text())
    a, b = sc["arms"]["a_as_simulated"]["scores"], sc["arms"]["b_both_fixes"]["scores"]
    prompt = {(r["arm"], r["corpus_id"]): r["prompt"] for r in map(json.loads, gzip.open(args.inputs, "rt"))}
    moved = sorted(set(a) & set(b), key=lambda c: -abs(b[c]["mean"] - a[c]["mean"]))[:N]
    rows = []
    for cid in moved:
        flags = {}
        for arm in ("a_as_simulated", "b_both_fixes"):
            flags[arm] = []
            for i in range(K):
                key = {"model": MODEL, "messages": [{"role": "user", "content": prompt[(arm, cid)]}], "reasoning_effort": EFFORT,
                       "response_schema": tm.RESPONSE_SCHEMA, "sample_idx": i, "this_call_uuid": clients._CALL_UUID}
                reasoning = json.loads(json.loads(clients._CACHE.path(key).read_text()))["reasoning"]
                flags[arm].append(sorted({m.group(0).lower() for m in TERMS.finditer(reasoning)}))
        rows.append({"corpus_id": cid, "a_scores": a[cid]["scores"], "b_scores": b[cid]["scores"],
                     "change": b[cid]["mean"] - a[cid]["mean"], "terms_a": flags["a_as_simulated"], "terms_b": flags["b_both_fixes"]})
    n_flag = sum(bool(t) for r in rows for t in r["terms_a"] + r["terms_b"])
    out = {"inputs": {"scores": Path(args.scores).name, "prompts": Path(args.inputs).name}, "runs": N, "reasonings": 2 * K * N,
           "reasonings_with_a_formatting_term": n_flag, "rows": rows}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"nano_reasoning_{stamp}.json"
    with path.open("x") as f:
        f.write(json.dumps(out, indent=1))
    print(f"{n_flag} of {2 * K * N} reasonings use a formatting term ->", path)


if __name__ == "__main__":
    main()
