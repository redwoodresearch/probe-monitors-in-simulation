"""POST HOC (a Bugbot review thread on the KV-cache crop): do part 1's two-token answer spellings depend on the cropped
cache? For each such spelling, the continuation log-probability (its log-probability minus its first token's, where the
first token is in the row's stored top-5). "simulated" runs on the fresh prompt cache; "Simulated" and " Simulated" run
after cache.crop(L). Decides nothing.

  uv run --no-project --with numpy --with transformers==5.19.0 python experiments/2026-10-11_is_this_a_test/analyze_part1_crop_check_POSTHOC.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
ROWS = ROOT / ".run_cache" / "is_this_a_test" / "part1_rows_2026-10-11_225143.json"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", revision="9216db5781bf21249d130ec9da846c4624c16137")
rows = json.loads(ROWS.read_text())
out = {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
       "rows sha256": hashlib.sha256(ROWS.read_bytes()).hexdigest()}
for v in ("simulated", "Simulated", " Simulated"):
    ids = tok.encode(v, add_special_tokens=False)
    c = np.array([r["logprobs"][v] - dict(zip(r["top5"], r["top5_logprobs"]))[ids[0]] for r in rows if ids[0] in r["top5"]])
    out[repr(v)] = {"tokens": tok.convert_ids_to_tokens(ids), "rows": len(c), "mean": float(c.mean()), "min": float(c.min()), "share below -0.1": float(np.mean(c < -0.1))}
path = HERE / "results" / f"part1_crop_check_POSTHOC_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
if path.exists():
    raise FileExistsError(path)
path.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1), f"\n-> {path}")
