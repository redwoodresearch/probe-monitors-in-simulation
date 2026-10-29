"""Price more attack runs per task for a +-15 real-vs-simulated comparison on the 116 stage-2 tasks (pricing only, $0;
Aswin will ask James). Reads the runs-per-task sizing summary (summarize_stage2_power.py over simulate_stage2_power.py
--runs-per-task r at +-15) and applies measured unit costs:
- real Qwen run: A2's honest batch C, servers $35.91 + devbox $4.32 for 196 runs (origin/main
  experiments/2026-10-08_q3_onpolicy_figures/SPEND_scaleup.md, A2 row, backfilled by PR #748 at 7f82f386);
- simulated run: simulator API $1,222.79 for the strict rerun's 80 simulated runs (SPEND.md, phase2 strict-tools pilot
  rerun), plus the policy's share at the real-run rate;
- replay extraction: honest C, $8.36 for 196 runs (origin/main SPEND_honest_c_extraction.md, PR #736);
- labelling: the judge-panel envelope, $98.371665 reported + $11.934693 reserved = $110.306358 (PR #788's SPEND.md entry,
  aswin/sabotage-stage1-judge-labels at 9398c063), shared by PRs #788 and #789; divided over #788's 1,457 rows it is an
  UPPER bound per row (the sabotage share is not recorded);
- scoring: about $3 of Modal CPU (frozen probe, planted and scramble checks).
Each task already has one run per arm, so new runs = 116 x (r - 1) per arm, for 3 arms (real run 1, real run 2, simulated).

  python price_more_runs.py <runs summary json>
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

TASKS = 116
REAL = (35.91 + 4.32) / 196
SIM = 1222.79 / 80 + REAL
EXTRACT = 8.36 / 196
LABEL = (98.371665 + 11.934693) / 1457
SCORE = 3.0


def main(summary: str):
    rows = []
    for r in json.loads(Path(summary).read_text()):
        b = r["budget"]
        if isinstance(b, str):
            continue
        n = r["runs_per_task"]
        new_real, new_sim = 2 * TASKS * (n - 1), TASKS * (n - 1)
        cost = {"real_runs": new_real * REAL, "sim_runs": new_sim * SIM, "extraction": (new_real + new_sim) * EXTRACT,
                "labelling_upper": b["rows_three_sets"] * LABEL, "scoring": SCORE}
        raw = sum(cost.values())
        rows.append({"runs_per_task_per_arm": n, "yield": "measured 2.03/run" if r["rates_file"].endswith("stage1.json") else "low 1.49/run",
                     "runs_per_arm": TASKS * n, "new_runs": new_real + new_sim, "k_flagged_per_run": b["k_per_run_all_sets"],
                     "labels_rows": b["rows_three_sets"], "worst_probability": b["worst_probability"], "confirmation_worst": b["confirmation_worst"],
                     "cost": {k: round(v, 2) for k, v in cost.items()}, "raw_usd": round(raw, 2), "with_30pct_usd": round(1.3 * raw, 2)})
    out = {"unit_costs": {"real_run": round(REAL, 4), "sim_run": round(SIM, 4), "extraction_per_run": round(EXTRACT, 4), "label_row_upper": round(LABEL, 4),
                          "scoring": SCORE}, "rows": sorted(rows, key=lambda x: (x["runs_per_task_per_arm"], x["yield"])), "source": Path(summary).name,
           "written": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")}
    dst = Path(__file__).parent / "results" / f"price_more_runs_m015_{out['written']}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    for x in out["rows"]:
        print(x["runs_per_task_per_arm"], x["yield"], x["new_runs"], x["k_flagged_per_run"], x["labels_rows"], x["worst_probability"], x["raw_usd"], x["with_30pct_usd"])
    print("->", dst)


if __name__ == "__main__":
    main(sys.argv[1])
