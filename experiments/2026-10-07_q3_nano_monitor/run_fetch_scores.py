"""Download the sims team's saved gpt-5-nano monitor scores for LinuxArena to data/ (git-ignored; laptop, read-only
S3 with the redwood-sandbox profile) and record each file's sha256 in results/fetch_<stamp>.json. No model calls.

  uv run --no-project --with boto3 python experiments/2026-10-07_q3_nano_monitor/run_fetch_scores.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
BUCKET = "s3://rr-sce-sim-transcripts/metrics"
# Whole-trajectory monitor (gpt-5-nano, k = 3 rescores, mean), and the per-step monitor that exists for real attack only.
SOURCES = {
    "traj_e1": "gpt5nano-monitor-la/2026-09-21_la_trajectory_monitor_real-attack-guidance-matched-epoch1-n1000-gpt5nano-k3-v2",
    "traj_e2": "gpt5nano-monitor-la/2026-09-21_la_trajectory_monitor_real-attack-guidance-matched-epoch2-n1000-gpt5nano-k3-v2",
    "traj_honest": "gpt5nano-monitor-la/2026-09-21_la_trajectory_monitor_real-honest-overlapaware-n1000-gpt5nano-k3-v2",
    "traj_sim": "gpt5nano-monitor-la/2026-09-21_la_trajectory_monitor_sim-astra-attack-n1000-gpt5nano-k3-v2",
    "per_action_e1_successful": "paper/2026-09-17_la_monitor_real-attack-n1000-nano-v1",
    "per_step_e1": "paper/2026-09-17_la_monitor-observations_real-attack-n1000-nano-medium-v1/monitor",
}


def main():
    out = {}
    for name, prefix in SOURCES.items():
        dest = HERE / "data" / name
        subprocess.run(["aws", "s3", "sync", "--quiet", f"{BUCKET}/{prefix}/", str(dest), "--profile", "redwood-sandbox"], check=True)
        files = sorted(p for p in dest.rglob("*") if p.is_file())
        digest = hashlib.sha256(b"".join(hashlib.sha256(p.read_bytes()).digest() for p in files)).hexdigest()
        out[name] = {"s3": f"{BUCKET}/{prefix}/", "files": len(files), "sha256_of_file_hashes": digest}
        print(name, len(files), "files", flush=True)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with (HERE / "results" / f"fetch_{stamp}.json").open("x") as f:
        f.write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
