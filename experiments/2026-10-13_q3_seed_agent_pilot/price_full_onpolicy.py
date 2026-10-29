"""Price a full Seed-OSS-36B on-policy run (NOT approved: Aswin decides after this pilot), from the pilot's measured
numbers and recorded unit costs. Read-only, $0. Numbers only.

Per task: two real attack runs, one simulated attack run, one real honest run (the on-policy design's arms), each then
read by Seed itself (activation extraction). Unit costs:
- Seed policy, real run: the pilot's billed Modal $ per run on the main servers, plus the devbox $ per run (both from
  results/pricing_stats_*.json). An attack run is priced as `attack_multiplier` x an honest run, for multipliers given
  on the command line, until 2e supplies Qwen3.5's attack-to-honest per-run token ratio.
- Simulated run: the September simulator's API, $15.49 per simulated run (experiments/2026-10-11_sabotage_test/
  PRICING_more_runs_m015.md:38, the strict rerun's $1,222.79 over 80 simulated runs, Qwen3.5 as policy), plus a Seed
  policy run at the attack price.
- Extraction: Seed reading each run's full transcript at G3's billed $1.1705 per 1M forward tokens
  (experiments/2026-10-13_g3_new_reader/results/step7_projection_2026-10-13_113731.json on aswin/q2-g3-reader). The
  transcript length is each pilot run's last policy call's input plus output tokens (median and p90 here).
Writes results/price_full_onpolicy_<stamp>.json.

  uv run --no-project --python 3.14 python price_full_onpolicy.py --stats <results/pricing_stats_*.json> \\
      --runs <smoke run dir> <main run dir> --tasks 100 196 200 --attack-multipliers 1 2 3
"""

import argparse
import json
import statistics
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
POLICY = "vllm/ByteDance-Seed/Seed-OSS-36B-Instruct"
SIM_API_PER_RUN = 15.49            # PRICING_more_runs_m015.md:38
EXTRACT_PER_1M = 1.1705            # G3 step 7 projection, billed rate on Seed
BUFFER = 1.30


def transcript_tokens(runs: list[str]) -> list[int]:
    out = []
    for run in map(Path, runs):
        for ev in sorted(run.rglob("*.eval")):
            if "references" in ev.parts:
                continue
            with zipfile.ZipFile(ev) as z:
                (n,) = [x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")]
                s = json.loads(z.read(n))
            calls = [e for e in s["events"] if e.get("event") == "model" and e.get("model") == POLICY and e.get("output")]
            if calls:
                u = calls[-1]["output"].get("usage") or {}
                out.append(int(u.get("input_tokens", 0)) + int(u.get("output_tokens", 0)))
    return out


def main(stats: str, runs: list[str], tasks: list[int], attack_multipliers: list[float]):
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    st = json.loads(Path(stats).read_text())
    main_apps = [a for a in st["billing_usd_by_app"] if a != "ap-6am84wgd7boGD4vtc3umpe"]  # the smoke app apart
    gpu_per_run = sum(st["billing_usd_by_app"][a] for a in main_apps) / 38
    devbox_per_run = st["devbox_usd"] / 40
    honest = gpu_per_run + devbox_per_run
    tt = transcript_tokens(runs)
    med, p90 = statistics.median(tt), sorted(tt)[int(round(0.9 * (len(tt) - 1)))]
    table = {}
    for m in attack_multipliers:
        attack = honest * m
        for which, tok in (("median_transcript", med), ("p90_transcript", p90)):
            ext_honest = tok / 1e6 * EXTRACT_PER_1M
            ext_attack = ext_honest * m
            per_task = {"real_attack_x2": 2 * attack, "sim_attack_policy": attack, "sim_api": SIM_API_PER_RUN, "real_honest": honest,
                        "extraction_4_runs": 3 * ext_attack + ext_honest}
            total = sum(per_task.values())
            table[f"attack_x{m:g}/{which}"] = {"per_task_usd": {k: round(v, 4) for k, v in per_task.items()} | {"total": round(total, 4)},
                                               "by_tasks": {str(n): {"raw": round(n * total, 2), "with_30pct": round(n * total * BUFFER, 2)} for n in tasks}}
    out = {"git": git, "stats_file": Path(stats).name, "not_approved": True,
           "unit_costs": {"seed_gpu_per_honest_run_usd": round(gpu_per_run, 4), "devbox_per_run_usd": round(devbox_per_run, 4),
                          "seed_honest_run_usd": round(honest, 4), "sim_api_per_sim_run_usd": SIM_API_PER_RUN,
                          "extraction_usd_per_1M_tokens": EXTRACT_PER_1M},
           "transcript_tokens_per_honest_run": {"n": len(tt), "median": med, "p90": p90, "max": max(tt)},
           "attack_multiplier_note": "attack-run cost and transcript length = honest x multiplier, until 2e's Qwen3.5 attack/honest token ratio",
           "table": table}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"price_full_onpolicy_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", type=int, default=[100, 196, 200])
    ap.add_argument("--attack-multipliers", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    main(**{k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()})
