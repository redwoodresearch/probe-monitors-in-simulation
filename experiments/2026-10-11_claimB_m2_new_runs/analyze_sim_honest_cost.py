"""Per-run API cost of a simulated HONEST run (for q2's sizing of new simulated honest runs; 0d's request about 15:2x PT
10-11, $0): the Astra honest batch of 2026-10-06 (437 runs), from the token usage in its stored Inspect logs
(header.json model_usage only; no transcript text read or kept), at sce's listed rates, by analyze_run_costs.py's method
(RATES and the aws helper imported unchanged). 20 seeded logs (random.Random(20261011)); the 20-run simcode pilot
batch is reported beside it. Downloads go to a temporary folder outside the repo and are deleted.

  uv run --no-project --python 3.14 python experiments/2026-10-11_claimB_m2_new_runs/analyze_sim_honest_cost.py
"""

import hashlib
import json
import random
import statistics
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import analyze_run_costs as A  # noqa: E402

ARMS = {"sim_astra_honest_n437": "2026-10-06_la-public10_sim-astra-medium-honest-origtraj-read-nocritic-fork0",
        "sim_astra_honest_simcode_n20": "2026-10-06_la-public10_sim-astra-medium-honest-origtraj-read-nocritic-fork0-simcode-e561f936-n20"}


def main():
    out = {"label": "per-run API cost of a simulated honest run (Astra) from Inspect log token usage at sce's listed rates", "rates": A.RATES, "arms": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for arm, prefix in ARMS.items():
            keys = sorted(l.split()[-1] for l in A.aws("s3", "ls", "--recursive", f"s3://{A.BUCKET}/{A.T}{prefix}/logs/").splitlines() if l.endswith(".eval"))
            pick = random.Random(20261011).sample(keys, min(20, len(keys)))
            per_run, by, n = [], {}, 0
            for i, k in enumerate(pick):
                f = Path(tmp) / f"{arm}_{i}.eval"
                A.aws("s3", "cp", "--quiet", f"s3://{A.BUCKET}/{k}", str(f))
                z = zipfile.ZipFile(f)
                h = json.loads(z.read("header.json"))
                ns = sum(1 for x in z.namelist() if x.startswith("samples/"))
                c_log = 0.0
                for m, u in (h.get("stats", {}).get("model_usage") or {}).items():
                    r = A.RATES[m]  # an unpriced model raises
                    c = ((u.get("input_tokens") or 0) * r[0] + (u.get("output_tokens") or 0) * r[1] + (u.get("input_tokens_cache_read") or 0) * r[2]) / 1e6
                    by[m] = by.get(m, 0.0) + c
                    c_log += c
                if ns:
                    per_run.append(c_log / ns)
                n += ns
                f.unlink()
            tot = sum(by.values())
            out["arms"][arm] = {"logs_in_batch": len(keys), "logs_sampled": len(pick), "runs": n, "usd_total": tot, "usd_per_run": tot / n,
                                "per_run_by_model": {m: v / n for m, v in by.items()},
                                "per_log_per_run_median": statistics.median(per_run), "per_log_per_run_range": [min(per_run), max(per_run)],
                                "log_list_sha256": hashlib.sha256("\n".join(pick).encode()).hexdigest()}
            print(arm, len(keys), n, round(tot / n, 3), {m.split("/")[1]: round(v / n, 3) for m, v in by.items()})
    out |= {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"sim_honest_cost_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
