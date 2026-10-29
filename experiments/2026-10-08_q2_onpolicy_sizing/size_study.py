"""Size and cost the full on-policy study (Aswin's decision F), from two inputs copied into inputs/:
- amendment 40's precision at N = 60 tasks, one cell per task (a40_pertask_first_…json, from commit 3d80227 on
  aswin/phase2-fold-calibration-a40-results): half-widths of the real-vs-simulated gap for standardized partial AUROC
  0–5% and catch at 1% (4 folds cross-fitted). Half-widths are scaled by sqrt(60 / N).
- a1's interim pilot costs (pilot_costs_a1_interim_…json; the pilot is invalid for behavior, its costs usable): the
  September simulator's $ per simulated run, and the billed H200 total spread over every run that called the policy
  (real and simulated alike; a lower bound).
Per task, as in the pilot design: 3 real runs (honest A, honest B, attack) and 2 simulated runs (honest, attack) per
simulator. Today's simulator's cost was not measured (its runs died at start): two planning scenarios price it at the
September mean and at the pilot guard's $25 per run (a prior estimate, not a bound).

  python3 experiments/2026-10-08_q2_onpolicy_sizing/size_study.py
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).parent
a40 = json.loads((HERE / "inputs/a40_pertask_first_2026-10-08_164346.json").read_text())["set 2 comparison at N = 60"]
pilot = json.loads((HERE / "inputs/pilot_costs_a1_interim_2026-10-08_v2.json").read_text())
HW = {"pAUC 0-5%": a40["partial AUROC 0-5%"]["half-width"], "catch@1%": a40["catch@1%, 4 folds cross-fitted"]["half-width"]}
sim_mean, sim_median = pilot["1_simulator_usd"]["september"]["sim_usd_per_run"]["mean"], pilot["1_simulator_usd"]["september"]["sim_usd_per_run"]["median"]
guard = pilot["1_spend_last_entry"]["per_run_estimate_usd"]
gpu_per_run = pilot["2_gpu_usd_per_attempt_with_policy_calls"]  # the billed GPU total over every run that called the policy (real and simulated)
print(f"inputs: pAUC half-width at N=60 {HW['pAUC 0-5%']:.4f}, catch@1% {HW['catch@1%']:.4f}; September sim $/run mean {sim_mean} "
      f"(median {sim_median}); guard ${guard}/run; GPU >= ${gpu_per_run}/policy-calling run (one allocation of the billed total)")
print(f"tasks for a pAUC half-width of 0.03: {60 * (HW['pAUC 0-5%'] / 0.03) ** 2:.1f}")
for sims, label in ((1, "September simulator only"), (2, "both simulators, two scenarios for today's unmeasured cost: Sept mean | $25")):
    print(f"\n{label}")
    print("tasks | pAUC half-width | catch@1% half-width | simulator $ (scenarios) | GPU $ (lower bound) | total $ (scenarios)")
    for n in (60, 80, 100, 125):
        s = math.sqrt(60 / n)
        sim_lo = n * 2 * (sim_mean + (sims - 1) * sim_mean)
        sim_hi = n * 2 * (sim_mean + (sims - 1) * guard)
        gpu = n * (3 + 2 * sims) * gpu_per_run
        print(f"{n} | {HW['pAUC 0-5%'] * s:.4f} | {HW['catch@1%'] * s:.4f} | {sim_lo:,.0f}-{sim_hi:,.0f} | {gpu:,.0f} | {sim_lo + gpu:,.0f}-{sim_hi + gpu:,.0f}")
