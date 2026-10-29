"""Second real-honest epoch quote (QUOTE ONLY, no launch; d9 for Aswin via 0d, 10-15 ~11:05). Inputs: counts_2026-10-15.json
(quote_epoch2_counts.py) and these recorded rates and figures, each cited:
  - H200 vLLM policy server $5.30/h (new-mains SPEND row and the 1e-4 job's SPEND.md);
  - the 1e-4 real honest job (aswin/q3-q35-realhonest-1e4, experiments/2026-10-14_q3_q35_realhonest_1e4/SPEND.md): 980 real honest runs
    in one launch, 6 H200 x ~7.2 h (servers ~$229), devbox m7i.16xlarge ~$28, staging ~$1, extraction (all 64 layers, H200) ~$41:
    ~$300 raw for 980 runs (its Actual column is "(to fill)": estimates, not billed actuals);
  - the stagger floor (frozen launcher: an environment's launches 90 s apart, strictly ordered; one launcher per box), scaled by the
    1e-4 job's saved scheduler replay: floor 5.47 h, median completion 6.82 h, p90 7.21 h; its tail (the last runs' durations, the
    30 s poll and strict order) adds +1.35 h (median) and +1.74 h (p90) to the floor, applied here as absolute hours.
Writes quote_2026-10-15.json and prints the headline.  python3 quote_epoch2.py <counts json> <out json>"""
import json
import math
import sys
from pathlib import Path

c = json.loads(Path(sys.argv[1]).read_text())
p = c["pooled_real_honest"]
H200, BOX_H = 5.30, 28 / 9.0  # $/h: server; devbox m7i.16xlarge, from the 1e-4 ledger (~$28 for ~9 h up)
RH = {"runs": 980, "servers": 6, "hours": 7.2, "server_usd": 229, "box_usd": 28, "staging_usd": 1, "extract_usd": 41}
per_run = {"gpu_server_h": RH["servers"] * RH["hours"] / RH["runs"], "extract_usd": RH["extract_usd"] / RH["runs"],
           "staging_usd": RH["staging_usd"] / RH["runs"]}
current = c["totals"]["final_plus_nm360"]
import collections
usable = sum(n for k, n in c["nm360_by_launch"].items() if k.endswith("|pooled"))  # Codex r58: from the saved categories
excluded = sum(n for k, n in c["nm360_by_launch"].items() if k.endswith("|excluded"))
tl = c["task_lists"]
tasks = {k: len(v) for k, v in tl.items()}  # Codex r57: distinct tasks with a pooled honest run (A or B), not the cohort size
env = collections.Counter(t.split("_", 1)[1] for v in tl.values() for t in v)
new_runs = 2 * sum(tasks.values())  # one more honest_a and honest_b per task with a pooled honest run
MED, P90 = 6.82 - 5.47, 7.21 - 5.47
designs = {}
for name, launches, servers in (("A: 3 boxes x 1 launch, 4 H200 each", 3, 4), ("B: 2 boxes x 1 launch, 6 H200 each", 2, 6)):
    per_launch = math.ceil(new_runs / launches)
    largest_env = max(math.ceil(2 * n / launches) for n in env.values())  # each environment's runs split evenly over the launches
    floor_h = (largest_env - 1) * 90 / 3600
    wall_h, wall_p90 = floor_h + MED, floor_h + P90  # Codex r57: the 1e-4 replay's completion minus floor, as absolute tail hours
    gpu_h_floor = launches * servers * wall_p90
    gpu_h_ratio = per_run["gpu_server_h"] * new_runs  # the 1e-4 job's measured server-h per run
    gpu_lo, gpu_hi = H200 * min(gpu_h_floor, gpu_h_ratio), H200 * max(gpu_h_floor, gpu_h_ratio)
    box = launches * (wall_h + 1.5) * BOX_H  # +1.5 h setup and staging on each box
    extract, staging = per_run["extract_usd"] * new_runs, per_run["staging_usd"] * new_runs
    raw_lo, raw_hi = gpu_lo + box + extract + staging, gpu_hi + box + extract + staging
    modal_lo, modal_hi = gpu_lo + extract + staging, gpu_hi + extract + staging
    designs[name] = {"launches": launches, "servers_per_launch": servers, "runs_per_launch": per_launch, "largest_env_entries_est": largest_env,
                     "stagger_floor_h": round(floor_h, 2), "wall_h_est": round(wall_h, 1), "wall_h_p90": round(wall_p90, 1), "server_h": [round(min(gpu_h_floor, gpu_h_ratio), 1), round(max(gpu_h_floor, gpu_h_ratio), 1)],
                     "usd": {"gpu_servers": [round(gpu_lo), round(gpu_hi)], "boxes_aws": round(box), "extraction_modal": round(extract), "staging_modal": round(staging, 1),
                             "raw": [round(raw_lo), round(raw_hi)], "with_30pct": [round(raw_lo * 1.3), round(raw_hi * 1.3)],
                             "modal_share_raw": [round(modal_lo), round(modal_hi)], "modal_share_with_30pct": [round(modal_lo * 1.3), round(modal_hi * 1.3)]}}
doc = {"quote_only": True, "question": "a second epoch of real honest runs on every usable task to reach >= 3,000 pooled real honest runs",
       "current": {"by_origin": p, "total_final_pools_plus_nm360": current, "rh1e4_separate": c["rh1e4_separate"],
                   "total_if_rh1e4_counted": c["totals"]["with_rh1e4_staged"]},
       "design": {"proposal": "one more honest_a and one more honest_b run per task that has a pooled honest run (same task files, harness, policy "
                              "Qwen3.5-27B on H200 vLLM, fresh draws), as replicate 2 of each arm; real only, no simulator",
                  "tasks": tasks, "runs_by_environment": {k: 2 * n for k, n in sorted(env.items())}, "new_runs": new_runs,
                  "pooled_after_if_all_usable": current + new_runs,
                  "pooled_after_at_nm360_usable_rate": current + round(new_runs * usable / (usable + excluded)),
                  "usable_note": f"new runs count only after staging: nm360's in-map honest runs passed at {usable} of {usable + excluded} ({excluded} excluded at staging)",
                  "cheaper_alternative": {"if_rh1e4_counts_toward_3000": "the 963 staged 1e-4 runs (5 replicates of T196 honest_b) already bring the "
                                          "pool to 2,666; one honest_b run per S_new and nm360 task (" + str(tasks["S_new_plus311"] + tasks["S_newmains_nm360"]) +
                                          " runs) reaches, if all those runs are usable, " + str(c["totals"]["with_rh1e4_staged"] + tasks["S_new_plus311"] + tasks["S_newmains_nm360"]) +
                                          ", but T196 then weighs (384 + 963) / 196 = 6.87 runs per task on average: a design call for 09/Aswin"}},
       "per_run_from_1e4_job": {k: round(v, 4) for k, v in per_run.items()} | {"all_in_raw_usd": round(300 / 980, 3)},
       "designs": designs,
       "simulator": "real runs make no simulator (gpt-6-astra) calls: no effect on the < 90 / < 115 simulated-running census gates",
       "not_included": "downstream scoring (Sol, frozen probes, monitors) on the new runs; box procurement time",
       "rates_cited": {"h200_per_h": H200, "devbox_per_h": round(BOX_H, 2), "rh1e4": RH}}
Path(sys.argv[2]).write_text(json.dumps(doc, indent=1) + "\n")
print("current", current, "| tasks", tasks, "| new runs", new_runs, "| pooled after", current + new_runs, "(all usable)")
for n, d in designs.items():
    print(n, "| floor", d["stagger_floor_h"], "h | wall ~", d["wall_h_est"], "h (p90", d["wall_h_p90"], ") | raw $", d["usd"]["raw"], "| +30% $", d["usd"]["with_30pct"], "| Modal share +30% $", d["usd"]["modal_share_with_30pct"])
