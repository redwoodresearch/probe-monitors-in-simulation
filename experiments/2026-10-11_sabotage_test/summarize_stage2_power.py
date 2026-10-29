"""One row per simulate_stage2_power.py result file: margin, rho, catch grid, the cheapest qualifying budget (or "not
reachable") with its labelled rows and worst probability (screening and confirmation), and the best worst probability
any candidate reached. Numbers only.

  python summarize_stage2_power.py <result json> ... > results/stage2_power_summary_<stamp>.json
"""

import hashlib
import json
import sys
from pathlib import Path

rows = []
for f in sys.argv[1:]:
    d = json.loads(Path(f).read_text())
    b = d["budget"]
    rows.append({"file": Path(f).name, "sha256": hashlib.sha256(Path(f).read_bytes()).hexdigest(), "margin": d["margin"], "rho": d["rho"],
                 "p_grid": d["p_grid"], "code": d["provenance"]["git"]["commit"][:8], "runs_per_task": d["provenance"].get("runs_per_task", 1),
                 "rates_file": d["provenance"]["rates_file"],
                 "budget": "not reachable" if b.get("reachable") is False else {"k_per_run_all_sets": b["k"], "k_per_task": b.get("k_per_task", b["k"]), "audit_rate_unflagged_impactful": b["audit_rate_unflagged_impactful"],
                                                                                "rows_three_sets": b["rows"], "worst_probability": round(b["worst_probability"], 4),
                                                                                "confirmation_worst": round(b["confirmation_worst"], 4)},
                 "best_worst_any_candidate": round(max(t["worst_probability"] for t in d["tried"]), 4)})
print(json.dumps(rows, indent=1))
