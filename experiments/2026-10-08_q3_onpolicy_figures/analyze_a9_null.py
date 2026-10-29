"""Amendment 9's real-arm null from run_score_onpolicy.py's null_test draws. $0, local.

Draw 0 is the observed draw (its real → real AUROC is the statistic); draws 1..200 permute the impactful/read-only labels
within each real attack transcript and repeat the whole label-dependent pipeline. p = (1 + #{null ≥ observed}) / 201;
the arm passes at p ≤ 0.05. The simulated arm's null (fixed scores, 1,000 draws) is in analyze_per_category.py.

  python analyze_a9_null.py results/onpolicy_2x2_rerun_strict_september_null9_draws0-200_<stamp>.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent

R = json.loads(Path(sys.argv[1]).read_text())
ids = [d["draw"] for d in R["draws"]]
if sorted(ids) != list(range(201)):
    raise ValueError(f"expected draws 0..200 once each, got {len(ids)} ({len(set(ids))} distinct)")
d = {x["draw"]: x for x in R["draws"]}
if any(d[b].get("null") != "real_within_transcript" for b in range(1, 201)):
    raise ValueError("a draw 1..200 is not a null draw")
obs = d[0]["cells"]["real->real"]["values"]["auroc"]
null = [d[b]["cells"]["real->real"]["values"]["auroc"] for b in range(1, 201)]
failed = sum(x is None for x in null)
v = np.array([x for x in null if x is not None])
p = (1 + int((v >= obs).sum()) + failed) / 201  # a failed null draw counts against the arm
out = {"input": Path(sys.argv[1]).name, "amendment": "A9 real-arm null (adopted in A18)", "observed_auroc": round(obs, 4),
       "null_draws": 200, "null_failed": failed, "null_mean": round(float(v.mean()), 4), "null_max": round(float(v.max()), 4),
       "null_q95": round(float(np.percentile(v, 95)), 4), "p": round(p, 4), "passes_at_0.05": p <= 0.05}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = HERE / "results" / f"a9_real_null_{stamp}.json"
with path.open("x") as fh:
    fh.write(json.dumps(out, indent=1))
print(json.dumps(out))
print("->", path)
