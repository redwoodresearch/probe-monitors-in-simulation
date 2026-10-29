"""PLAN.md's A-C and D numbers recomputed from the result files (review on #521: threads L9ts, L9t0, L9tx), local and free.
(1) A-C: the fixed-sim runs' top-3 slip share and its base rate (both means over runs of per-run shares), beside the
pooled slip-carrying share. (2) D: the gap shares by segment type, the ones PLAN.md lists and the remainder, for all
tokens before the final tokens and for content tokens only, and each run's top-20 tokens by where they sit in both
scopes. Content tokens hold none of the final tokens, so D's own content-token numbers must equal the tail recheck's.

  python3 experiments/2026-10-10_claimB_probe_attribution/analyze_plan_numbers.py <out dir> <attribution_abc json> \
      <recheck_d_tail json> <attribution_tokens (D) json>
"""
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

abc, tail, d = (json.loads(Path(p).read_text()) for p in sys.argv[2:5])
LISTED = {"before_final_tokens": ["bash", "text_editor view", "text_editor str_replace", "call text_editor str_replace",
                                  "text_editor insert", "rust_worker_bash", "agent text or template",
                                  "call text_editor create"],
          "content_tokens": ["bash", "text_editor view", "text_editor str_replace"]}
c = abc["output_counts"]["sim-attack-fixed"]
out = {"A-C top-3": abc["sim_top3_slip_share"] | {"pooled": c["slip_carrying"] / c["outputs"]} | c}
for scope, listed in LISTED.items():
    s = tail[scope]["gap_share_by_category"]
    assert math.isclose(sum(s.values()), 1, abs_tol=1e-9), scope
    out[scope] = {"gap": tail[scope]["mean_projection_gap"], "listed": {k: s[k] for k in listed},
                  "remainder": 1 - sum(s[k] for k in listed),
                  "remainder_parts": dict(sorted(((k, v) for k, v in s.items() if k not in listed), key=lambda kv: -kv[1])),
                  "top20_by_category": tail[scope]["top20_tokens_by_category"],
                  "top20_in_slip_outputs": tail[scope]["top20_tokens_in_slip_outputs"]}
    print(scope, json.dumps(out[scope], indent=1))
assert all(tail["content_tokens"][k] == v for k, v in d["D"]["content_tokens"].items()), "content scope differs"
print("A-C top-3", out["A-C top-3"])
here = Path(__file__)
out |= {"inputs_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sys.argv[2:5]},
        "script_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
        "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=here.parent).stdout.strip()}
stamp = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')
dst = Path(sys.argv[1]) / f"plan_numbers_{stamp}.json"
with open(dst, "x") as fh:
    fh.write(json.dumps(out, indent=1))
print("->", dst)
