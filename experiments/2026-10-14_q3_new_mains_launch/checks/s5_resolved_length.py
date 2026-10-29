"""Read-only (Append 35 item 4 with 09's S5 rules @ bffe22e3e; d9 10-15 ~03:45 PT): S5's RESOLVED simulated arm for the
frozen length check. Inputs are two outputs of the pinned checks/length_check_nm.py @ 089123a2d, run unchanged on
sims-astra: (a) S5 main + S5's follow-up (the original arm), (b) S5 main + 5r (the 6 replacements); and the filled
supersession map (results/s5r_supersession_map_filled_2026-10-15.json @ 75fef359f).
Rules (09): the universe and the quartile cut points stay those of the ORIGINAL started cells (a); the 6 replaced
originals keep their rate-limit group membership in (a); a resolved cell takes its replacement's group from (b) when it
is in the map, else its original group; a replacement that fails usability is a failed replacement (no fallback; it
stays in the attrition numerator). Also writes the 6-key list for the mandatory without-replacement sensitivity.
  python3 s5_resolved_length.py <(a) json> <(b) json> <filled map json>"""

import hashlib
import json
import statistics
import sys
from pathlib import Path

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
A, B, M = (json.loads(Path(p).read_text()) for p in sys.argv[1:4])
orig = {r["cell"]: r for r in A["rows"]}
rep = {r["cell"]: r for r in B["rows"]}
mp = {e["cell"]: e for e in M["map"]}
assert set(mp) == set(rep) and len(mp) == 6, (sorted(mp), sorted(rep))
assert all(c in orig and orig[c]["group"] == "rate_limit" for c in mp), "every superseded original must be a rate-limit failure in (a)"
assert all(rep[c]["run1_steps"] == orig[c]["run1_steps"] for c in mp), "the replacement's proxy must equal the original's (same main)"
cuts = A["quartile_cuts_run1_steps"]
quart = lambda x: None if x is None else 1 + sum(x > c for c in cuts)  # the pinned rule: ties to the lower quartile
rows = []
for c, o in sorted(orig.items()):
    r = rep.get(c)
    rows.append({"cell": c, "run1_steps": o["run1_steps"], "run1_policy_output_tokens": o["run1_policy_output_tokens"],
                 "run1_steps_quartile_original_cuts": quart(o["run1_steps"]), "original_run_key": o["run_key"],
                 "original_group": o["group"], "source": "2026-10-14_newmains_s5r_fu" if r else "2026-10-14_newmains_s5_fu",
                 "resolved_group": (r["group"] if r["group"] == "usable" else "failed_replacement") if r else o["group"],
                 "replacement_eval_sha256": mp[c]["replacement"]["eval_sha256"] if r else None})
n = len(rows)
nonusable = lambda key: sum(r[key] != "usable" for r in rows)
groups = {}
for g in sorted({r["resolved_group"] for r in rows}):
    xs = sorted(r["run1_steps"] for r in rows if r["resolved_group"] == g and r["run1_steps"] is not None)
    groups[g] = {"n": sum(r["resolved_group"] == g for r in rows),
                 "run1_steps_quantiles": statistics.quantiles(xs, n=4, method="inclusive") if len(xs) >= 4 else xs}
print(json.dumps({
    "inputs": {"original_length_check_sha256": sha(sys.argv[1]), "replacement_length_check_sha256": sha(sys.argv[2]),
               "filled_map_sha256": sha(sys.argv[3]), "length_check_script": "checks/length_check_nm.py @ 089123a2d (sha256 30a158fc7027b4a4...)",
               "original": A["follow_up"], "replacement": B["follow_up"], "main": A["main"], "policy": A["policy"]},
    "universe_started_cells": n, "quartile_cuts_run1_steps_original": cuts, "missing_proxy": A["missing_proxy"],
    "attrition": {"original_rate_limit": f"{sum(r['original_group'] == 'rate_limit' for r in rows)}/{n}",
                  "original_not_usable": f"{nonusable('original_group')}/{n}",
                  "resolved_failed_replacements": f"{sum(r['resolved_group'] == 'failed_replacement' for r in rows)}/{n}",
                  "resolved_not_usable": f"{nonusable('resolved_group')}/{n}"},
    "original_groups": {g: v["n"] for g, v in A["groups"].items()}, "original_failure_share_by_quartile": A["failure_share_by_run1_steps_quartile"],
    "resolved_groups": groups,
    "resolved_not_usable_by_quartile": {k: {"cells": sum(r["run1_steps_quartile_original_cuts"] == k for r in rows),
                                            "not_usable": sum(r["run1_steps_quartile_original_cuts"] == k and r["resolved_group"] != "usable" for r in rows)}
                                        for k in (1, 2, 3, 4)},
    "without_replacement_sensitivity_keys": [{"cell": c, "original_run_key": mp[c]["original"]["run_key"],
                                              "original_eval_sha256": mp[c]["original"]["eval_sha256"]} for c in sorted(mp)],
    "rows": rows}, indent=1))
