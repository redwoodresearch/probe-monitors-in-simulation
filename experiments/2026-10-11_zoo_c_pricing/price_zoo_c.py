"""Zoo C: more same-batch real Sol honest runs, priced ($0; counts, metadata and header usage only).

Inputs: the eight numbers it needs from C's gate counts, recorded below with their source (#695, e37a2892,
gate_counts_v3.json, sha256 85997021…; the file is not copied, and they are checked against it: the script stops if
that commit's file is missing, unless --unverified, which records "checked": false), the repo's split file (honest
runs per task and batch), and Sol's per-run usage read from real logs' headers (read_header_usage.py). The committed rule
(experiments/2026-10-11_q2_zoo_plans/PLAN_zoo_B_C.md): p = 0.5, DE >= 1.25, SE^2 = p(1-p) DE (1/n1 + 1/n2), pass
probability 2Phi((0.10 - z SE)/SE) - 1 (z = 1.96 real-vs-real, 2.24 real-vs-sim), joint = product, gate 0.80.

  uv run --no-project python experiments/2026-10-11_zoo_c_pricing/price_zoo_c.py results/header_usage_<stamp>.json [--unverified]
"""
import hashlib
import json
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from math import erf, sqrt
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
F = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
SOL = {"input": 5.0e-6, "cache_read": 0.5e-6, "output": 30e-6}  # sims src/sim/prices.py, openai/gpt-5.6-sol ($/token)
DEVBOX_H, PARALLEL = 0.8064, 10  # m7i.4xlarge on demand, us-west-2 ($/h); about 10 environments at once (Phase 2)
QWEN35_BILLED_PER_RUN = 49.90 / 4370  # C reads Qwen3.5-27B (PLAN_zoo_B_C.md): its source dump,
# 2026-10-06_la_segment-end-v4_qwen35-27b_alllayers_v1, billed $49.90 (spend_qwen35_jobs.csv, app ap-dJ6zIYfkKreP0lP7md5cIe)
# for 4,370 whole transcripts (experiments/2026-10-06_qwen35_replication/results/
# extract_2026-10-06_la_segment-end-v4_qwen35-27b_alllayers_v1_2026-10-06_213622.json)
ANALYSIS = 3.0  # C's monitor fits and null comparisons, once ("about $3 raw", PLAN_zoo_B_C.md); labels are Aswin's, not priced
ONE_TIME = {"devbox: start, environment image pulls and checks (about 1 h)": 0.81, "Modal: one extraction start-up and load": 0.30}
RETRY = 0.05  # about 2% of runs errored or had no actions in this cohort (splits/exclusions_v4.json); 5% allowance


SRC = ("e37a2892", "experiments/2026-10-11_credential_count/results/gate_v3/gate_counts_v3.json",
       "859970216f3a34c7e657d1172de0cba77256994b515ef10fbaefdea51d3aca55")  # #695; Aswin's stack, cited, not copied
COUNTS = {"groups": {"same_batch_real_run_1": {"positives": 546}, "same_batch_real_run_2": {"positives": 578},
                     "matched_real_overlap800": {"positives": 2859}, "simulated": {"positives": 1385}},
          "same_batch_tasks": 163, "real_sim_pass_probability_upper_bound": 0.9987363421881337,
          "real_real_pass_probability_upper_bound": 0.7004742208452706, "joint_pass_probability_zero_covariance": 0.6995890611240885}


def p_pass(n1, n2, z, de=1.25):
    se = sqrt(0.25 * de * (1 / n1 + 1 / n2))
    return max(0.0, 2 * F((0.10 - z * se) / se) - 1)


g = COUNTS
shown = subprocess.run(["git", "show", f"{SRC[0]}:{SRC[1]}"], capture_output=True, cwd=ROOT)
checked = shown.returncode == 0
if checked:  # the recorded numbers must be the source file's
    if hashlib.sha256(shown.stdout).hexdigest() != SRC[2]:
        raise ValueError(f"#695's file at {SRC[0]} does not have the recorded sha256")
    src = json.loads(shown.stdout)
    if any(src["groups"][k]["positives"] != v["positives"] for k, v in COUNTS["groups"].items()) \
            or any(src[k] != v for k, v in COUNTS.items() if k != "groups"):
        raise ValueError("the recorded counts differ from #695's file")
elif subprocess.run(["git", "cat-file", "-e", f"{SRC[0]}^{{commit}}"], capture_output=True, cwd=ROOT).returncode == 0:
    raise SystemExit(f"git show failed although {SRC[0]} is present: {shown.stderr.decode().strip()}")
elif sys.argv[2:] != ["--unverified"]:
    raise SystemExit(f"#695's counts file is not in this checkout (git show {SRC[0]} failed); fetch #695, or pass --unverified")
else:
    print("warning: --unverified: the recorded counts were not checked against #695's file", file=sys.stderr)
hu = json.loads((HERE / sys.argv[1]).read_text())
G = g["groups"]
n1, n2, T = G["same_batch_real_run_1"]["positives"], G["same_batch_real_run_2"]["positives"], g["same_batch_tasks"]
rs = p_pass(G["matched_real_overlap800"]["positives"], G["simulated"]["positives"], 2.24)
rr = p_pass(n1, n2, 1.96)
out = {"inputs": {"gate counts": f"#695, {SRC[0]}, gate_counts_v3.json, sha256 {SRC[2]} (eight numbers recorded, file not copied)",
                  "checked against the file at that commit": checked, "header usage": sys.argv[1]}
       | ({} if checked else {"warning": "run with --unverified: counts not checked against #695's file"}),
       "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
       "reproduced": {"real-vs-sim": round(rs, 4), "real-vs-real": round(rr, 4), "joint": round(rs * rr, 4),
                      "in the counts file": [g["real_sim_pass_probability_upper_bound"], g["real_real_pass_probability_upper_bound"],
                                             g["joint_pass_probability_zero_covariance"]]}}
# (1) more same-batch tasks, at the current per-task rates (Poisson 95% on the totals; per-task counts are not in the counts file)
need = {}
M_RATE = G["matched_real_overlap800"]["positives"] / 800  # matched real positives per run (800 overlap800 runs)
for inside in (True, False):  # inside the simulated cohort, a new task's two runs also add matched real positives
    for de in (1.25, 1.5, 2.0):
        for lab, (a, b) in (("rate as counted", (n1, n2)), ("rates at the Poisson 2.5th percentile", (n1 - 1.96 * sqrt(n1), n2 - 1.96 * sqrt(n2))),
                            ("rates at the Poisson 97.5th percentile", (n1 + 1.96 * sqrt(n1), n2 + 1.96 * sqrt(n2)))):
            r1, r2 = a / T, b / T
            rs_k = lambda k: p_pass(G["matched_real_overlap800"]["positives"] + (2 * k * M_RATE if inside else 0), G["simulated"]["positives"], 2.24, de)
            k = next((k for k in range(0, 5000) if rs_k(k) * p_pass(n1 + r1 * k, n2 + r2 * k, 1.96, de) >= 0.80), None)
            where = "tasks inside the simulated cohort" if inside else "tasks outside it"
            need[f"{where}, DE {de}, {lab}"] = {"positives per task (run 1, run 2)": [round(r1, 2), round(r2, 2)], "more tasks": k,
                                                "more real runs (2 per task)": None if k is None else 2 * k}
out["(1) more same-batch tasks for joint >= 0.80"] = need
# (3a) pairs already in our data: every overlap800 honest run, paired within its task
ex = json.loads((ROOT / "splits" / "exclusions_v4.json").read_text())
per = defaultdict(int)
for r in ex["samples"]:
    if r["arm"] == "real-honest" and r["key"].startswith("overlap800"):
        per[r["main_task"]] += 1
pairs = sum(n // 2 for n in per.values())
scale = pairs / T
out["(3a) same-batch pairs already labelled"] = {
    "tasks with >= 2 overlap800 honest runs": sum(1 for n in per.values() if n >= 2), "disjoint pairs": pairs, "used now": T,
    "projected positives per side (at the current per-pair rate)": [round(n1 * scale), round(n2 * scale)],
    "projected joint (DE 1.25)": round(rs * p_pass(n1 * scale, n2 * scale, 1.96), 4),
    "projected joint (DE 1.5)": round(p_pass(G["matched_real_overlap800"]["positives"], G["simulated"]["positives"], 2.24, 1.5)
                                     * p_pass(n1 * scale, n2 * scale, 1.96, 1.5), 4),
    "note": "needs a plan amendment (pairs beyond each task's first two runs) and the real per-run counts, which q2's labels have"}
# (2) whole job per real Sol honest run
runs = hu["runs"]
usd = [r["input"] * SOL["input"] + r["cache_read"] * SOL["cache_read"] + r["output"] * SOL["output"] for r in runs]
rng = random.Random(20261011)
boot = sorted(sum(rng.choices(usd, k=len(usd))) / len(usd) for _ in range(2000))
wall = sum(r["wall_s"] for r in runs) / len(runs)
per_run = {"Sol API (mean of the sample)": sum(usd) / len(usd),
           "devbox share (wall time / parallel runs)": wall / 3600 * DEVBOX_H / PARALLEL,
           "Modal extraction, Qwen3.5-27B, all 64 layers": QWEN35_BILLED_PER_RUN,
           "Modal probe scoring (CPU)": 0.001}
sub = sum(per_run.values()) * (1 + RETRY)
out["(2) whole job per real Sol honest run"] = {
    **{k: round(v, 4) for k, v in per_run.items()}, "retry allowance": RETRY, "per run, raw": round(sub, 3), "per run, with 30%": round(sub * 1.3, 3),
    "Sol API 95% (bootstrap of the sample mean)": [round(boot[50], 3), round(boot[1949], 3)],
    "Sol API spread across runs (10th, 90th percentile)": [round(sorted(usd)[len(usd) // 10], 3), round(sorted(usd)[9 * len(usd) // 10], 3)],
    "wall minutes per run (mean)": round(wall / 60, 1), "sample": f"{len(runs)} of {hu['of']} overlap800 logs, seed {hu['seed']}",
    "one-time": ONE_TIME, "not priced": "C's credential labels for new runs (Aswin's label pass, outside our sessions)"}
k = need["tasks outside it, DE 1.25, rate as counted"]["more tasks"]
tot = 2 * k * sub + sum(ONE_TIME.values())
out["(2) total for (1) at DE 1.25"] = {"more runs": 2 * k, "generation and extraction, raw": round(tot, 2),
                                      "plus C's analysis once, raw": round(tot + ANALYSIS, 2), "with 30%": round((tot + ANALYSIS) * 1.3, 2),
                                      "not priced": "C's labels for the new runs (Aswin's label pass)"}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = HERE / "results" / f"zoo_c_price_{stamp}.json"
path.write_text("{\n" + ",\n".join(f" {json.dumps(a)}: {json.dumps(b)}" for a, b in out.items()) + "\n}\n")
print(json.dumps(out, indent=1), f"\n-> {path}")
