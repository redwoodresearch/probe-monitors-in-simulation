"""Main tasks, runs and calls per origin in the refit's analyzed rows (local), for figure 2's captions. Inputs: each
reader's rows.json from Volume probe-shift_aswin-surya_sub-q2-refit_boot_kill-2026-11-08 at
2026-10-08_impactful_refit_bootstrap_v1/<reader>/rows.json (`modal volume get`; metadata only, no activations).

  python3 experiments/2026-10-08_q2_figure_pack/analyze_refit_counts.py <qwen3.5 rows.json> <qwen3-32b rows.json>
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

out = {}
for reader, path in zip(("qwen3.5", "qwen3-32b"), sys.argv[1:3]):
    raw = Path(path).read_bytes()
    rows = json.loads(raw)
    out[reader] = {"rows_sha256": hashlib.sha256(raw).hexdigest()}
    for arm in sorted({r["arm"] for r in rows}):
        a = [r for r in rows if r["arm"] == arm]
        out[reader][arm] = {"main tasks": len({r["main_task"] for r in a}), "runs": len({r["key"] for r in a}),
                            "impactful calls": sum(r["y"] == 1 for r in a), "read-only calls": sum(r["y"] == 0 for r in a)}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
p = Path(__file__).parent / "results" / f"refit_counts_{stamp}.json"
p.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
print(p)
