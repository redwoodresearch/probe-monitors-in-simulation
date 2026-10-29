"""Deception black-box, step 2 ($0; before any judge call): each run's full judge prompt counted with the o200k_base
tokenizer (the GPT-5 family's), against INPUT_LIMIT. The endpoint reports no context length, so the limit is the GPT-5
family's documented 272,000 input tokens. A run over it is dropped for both scorers and counted (Aswin's option A, about
23:33 PT 10-11: "any run too long for the judge is dropped for both scorers and counted"). Also the dry-run cost:
counted input at $5 per million, plus output and reasoning at 300 (typical) and 4,000 (the per-call maximum) tokens at
$30 per million. Writes run keys and counts only.

  uv run --no-project --with tiktoken python experiments/2026-10-11_q2_deception_blackbox/check_tokens.py <judge inputs jsonl>
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import tiktoken

from judge_common import HERE, fill, prompts

INPUT_LIMIT, PRICE_IN, PRICE_OUT, OUT_TYPICAL, OUT_MAX, STOP = 272_000, 5.0, 30.0, 300, 4000, 90.0


def main(inp: Path):
    for f in (Path(__file__).resolve(), HERE / "judge_common.py", HERE / "judge_prompt.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    enc = tiktoken.get_encoding("o200k_base"); system, user = prompts()
    runs = [json.loads(line) for line in inp.open()]
    n = {r["key"]: len(enc.encode(system)) + len(enc.encode(fill(user, r))) + 10 for r in runs}  # + chat framing
    over = sorted(k for k, v in n.items() if v > INPUT_LIMIT)
    kept = [v for k, v in n.items() if k not in over]; tin = sum(kept)
    cost = lambda out: tin * PRICE_IN / 1e6 + len(kept) * out * PRICE_OUT / 1e6
    s = sorted(n.values())
    out = {"plan": "PLAN.md (this folder)", "git": head, "inputs": inp.name, "inputs sha256": hashlib.sha256(inp.read_bytes()).hexdigest(),
           "prompt sha256": hashlib.sha256((HERE / "judge_prompt.md").read_bytes()).hexdigest(),
           "judge_common sha256": hashlib.sha256((HERE / "judge_common.py").read_bytes()).hexdigest(), "tokenizer": "o200k_base", "input limit": INPUT_LIMIT, "runs": len(runs), "runs over the limit (dropped for both scorers)": over,
           "input tokens per run: min / median / 90th / max": [s[0], s[len(s) // 2], s[int(0.9 * len(s))], s[-1]],
           "input tokens, kept runs": tin, "dry-run cost (USD), typical output": round(cost(OUT_TYPICAL), 2),
           "dry-run cost (USD), every call at the output maximum": round(cost(OUT_MAX), 2), "stop": STOP,
           "projected over the stop": cost(OUT_TYPICAL) > STOP, "tokens per run": n}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"token_check_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, separators=(",", ":")) + "\n")  # one line: 683 per-run counts
    print(json.dumps({k: v for k, v in out.items() if k != "tokens per run"}, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
