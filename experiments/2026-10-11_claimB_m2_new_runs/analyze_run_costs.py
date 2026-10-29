"""Per-run API cost of the arms new runs would repeat, from the token usage in stored Inspect logs (header.json
model_usage only; no transcript text is read or kept), at sce's listed rates (src/sim/prices.py at e561f936).
Public-10 arms: 10 seeded logs each (random.Random(20261011) over the sorted .eval keys). la18: its cost_estimation
shard (11 runs per arm). Downloads go to a temporary folder outside the repo and are deleted. $0 (S3 reads from the
laptop, profile redwood-sandbox).

  uv run --no-project --python 3.14 python experiments/2026-10-11_claimB_m2_new_runs/analyze_run_costs.py
"""

import hashlib
import json
import random
import subprocess
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
BUCKET, T = "rr-sce-sim-transcripts", "transcripts/"
RATES = {"openai/gpt-6-astra": (10.0, 50.0, 1.0), "openai/gpt-5.6-sol": (5.0, 30.0, 0.5), "openai/gpt-5.6-terra": (2.0, 12.0, 0.2),
         "openai/gpt-5-nano-2025-08-07": (0.05, 0.4, 0.005)}  # input, output, cached input per million (sce prices.py)
ARMS = {"public10_real_attack_e1": "2026-09-18_la-public10_real-attack-solmedium-step150-guidancematched-epoch1-n1000",
        "public10_real_honest": "2026-09-18_la-public10_real-honest-solmedium-step150-overlapaware-n1000",
        "public10_sim_astra": "2026-09-18_la-public10_sim-astra-medium-attack-origtraj-read-nocritic-fork0-overlapaware-n1000",
        "la18_real_attack": "2026-09-18_la18_real-attack-solmedium-step150-attacker0828-contractv3-n200",
        "la18_sim_astra_r1": "2026-09-18_la18_sim-astra-medium-attack-aff-origtraj-fork0-solmedium-step150-n200"}


def aws(*a):
    return subprocess.run(["aws", "--profile", "redwood-sandbox", *a], capture_output=True, text=True, check=True).stdout


def main():
    out = {"label": "per-run API cost from Inspect log token usage at sce's listed rates", "rates": RATES, "arms": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for arm, prefix in ARMS.items():
            keys = sorted(l.split()[-1] for l in aws("s3", "ls", "--recursive", f"s3://{BUCKET}/{T}{prefix}/logs/").splitlines() if l.endswith(".eval"))
            pick = [k for k in keys if "cost-estimation" in k][:1] if arm.startswith("la18") else random.Random(20261011).sample(keys, 10)
            cost, n, by = 0.0, 0, {}
            for i, k in enumerate(pick):
                f = Path(tmp) / f"{arm}_{i}.eval"
                aws("s3", "cp", "--quiet", f"s3://{BUCKET}/{k}", str(f))
                z = zipfile.ZipFile(f)
                h = json.loads(z.read("header.json"))
                n += sum(1 for x in z.namelist() if x.startswith("samples/"))
                for m, u in (h.get("stats", {}).get("model_usage") or {}).items():
                    r = RATES[m]
                    c = ((u.get("input_tokens") or 0) * r[0] + (u.get("output_tokens") or 0) * r[1] + (u.get("input_tokens_cache_read") or 0) * r[2]) / 1e6
                    by[m] = by.get(m, 0.0) + c
                    cost += c
                f.unlink()
            out["arms"][arm] = {"logs": len(pick), "runs": n, "usd_total": cost, "usd_per_run": cost / n, "per_run_by_model": {m: v / n for m, v in by.items()},
                                "log_list_sha256": hashlib.sha256("\n".join(pick).encode()).hexdigest()}
            print(arm, n, round(cost / n, 3), {m.split("/")[1]: round(v / n, 3) for m, v in by.items()})
    out |= {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"run_costs_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
