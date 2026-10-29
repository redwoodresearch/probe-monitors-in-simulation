"""The on-policy pairs' scatter inputs (RULE.md): x = −Δ = catch(unplanted) − catch(planted) on the forest's real run
(run 1), in points, with its interval negated; Δ on the second real run; Δ(run 1) − Δ(run 2); the drift gate; read by
key from the score files. Prints and writes results/summary_onpolicy_<stamp>.json.

  python summarize_onpolicy.py results/score_p1_<stamp>.json results/score_p78_<stamp>.json results/splice_gate_and_reach_<stamp>.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

p1, p78, reach = (json.loads(Path(f).read_text()) for f in sys.argv[1:4])


def row(r1: dict, r2: dict, diff: dict, units: str) -> dict:
    return {"units": units, "x_points": -r1["delta_points"], "x_ci95_points": [-r1["ci95_points"][1], -r1["ci95_points"][0]],
            "run1": {k: r1[k] for k in ("catch_unplanted", "catch_planted", "delta_points", "ci95_points", "units", "tasks",
                                        "units_flag_changed", "units_score_moved")},
            "run2_control": {k: r2[k] for k in ("catch_unplanted", "catch_planted", "delta_points", "ci95_points", "units", "tasks",
                                                "units_flag_changed", "units_score_moved")},
            "delta_run1_minus_run2": diff}


out = {"inputs": sys.argv[1:4],
       "P1 (A3)": row(p1["real->real"]["primary"], p1["real->run2"]["primary"], p1["diff_run1_minus_run2"], "impactful commands") | {
           "secondary_after_first_planted_output": {w: p1[w]["secondary_after_first_planted_output"] for w in ("real->real", "real->run2")},
           "drift_gate": {w: p1[w]["drift_gate"] for w in ("real->real", "real->run2")}},
       "P7 (A7, command end)": row(p78["command_end"]["run1"], p78["command_end"]["run2"], p78["command_end"]["diff_run1_minus_run2"], "attack runs") | {
           "drift_gate": {w: p78["command_end"][w]["drift_gate"] for w in ("run1", "run2")}},
       "P8 (B4, reply end)": row(p78["reply_end"]["run1"], p78["reply_end"]["run2"], p78["reply_end"]["diff_run1_minus_run2"], "attack runs") | {
           "drift_gate": {w: p78["reply_end"][w]["drift_gate"] for w in ("run1", "run2")}},
       "reach": reach["reach"], "newline_reaches_ids": {k: {w: v[w]["newline_reaches_ids"] for w in v} for k, v in reach["reach"].items()}}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
dst = Path(__file__).parent / "results" / f"summary_onpolicy_{stamp}.json"
with dst.open("x") as fh:
    fh.write(json.dumps(out, indent=1))
for k in ("P1 (A3)", "P7 (A7, command end)", "P8 (B4, reply end)"):
    r = out[k]
    print(f"{k}: x = {r['x_points']:+.2f} [{r['x_ci95_points'][0]:+.2f}, {r['x_ci95_points'][1]:+.2f}]; run 1 catch {r['run1']['catch_unplanted']:.4f} -> "
          f"{r['run1']['catch_planted']:.4f} (flags changed {r['run1']['units_flag_changed']}, scores moved {r['run1']['units_score_moved']}); run 2 Δ "
          f"{r['run2_control']['delta_points']:+.2f} [{r['run2_control']['ci95_points'][0]:+.2f}, {r['run2_control']['ci95_points'][1]:+.2f}]; "
          f"Δ1−Δ2 {r['delta_run1_minus_run2']['diff_points']:+.2f} [{r['delta_run1_minus_run2']['ci95_points'][0]:+.2f}, {r['delta_run1_minus_run2']['ci95_points'][1]:+.2f}]")
print("->", dst)
