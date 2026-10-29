"""Status and fetch for this track's remote Q1 analyses (2026-10-06), in one local process.

For every analysis id in results/q1/runs_registry_2026-10-06.json that is not void: how many of its 64 per-layer
results are on the Volume, and, once its orchestrator has written results.json, a copy of it as
results/q1/q1_gpu_<id>.json (the same file the main session's `fetch` entrypoint writes) and of its run scores as
.run_cache/q1_run_scores/q1_gpu_run_scores_<id>.jsonl (large; kept out of git). Each file is fetched when missing,
independently, and existing local copies are never overwritten. The registry records each analysis's extraction run,
so no config checkout is needed. Reads only; uses Modal's Python API, so one process replaces a `modal volume ls` per
analysis.

  uv run python experiments/2026-10-06_qwen35_replication/fetch_q1_qwen35.py
"""

import json
from pathlib import Path

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
VOL = "probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06"


def main():
    vol = modal.Volume.from_name(VOL, version=2)
    registry = json.loads((HERE / "results" / "q1" / "runs_registry_2026-10-06.json").read_text())
    scores = ROOT / ".run_cache" / "q1_run_scores"
    scores.mkdir(parents=True, exist_ok=True)
    pending, fetched = [], []
    for r in registry:
        if r["status"].startswith("void"):
            continue
        a = r["analysis_id"]
        base = f"{r['extraction_run']}__q1gpu/{a}"
        names = {Path(e.path).name for e in vol.listdir(base)}
        layers = sum(n.startswith("L") and n.endswith(".json") and n[1:-5].isdigit() for n in names)
        out, run_scores = HERE / "results" / "q1" / f"q1_gpu_{a}.json", scores / f"q1_gpu_run_scores_{a}.jsonl"
        if "results.json" in names:
            got = []
            for name, dst in (("results.json", out), ("run_scores.jsonl", run_scores)):
                if name in names and not dst.exists():  # each fetched only when missing; never overwritten
                    dst.write_bytes(b"".join(vol.read_file(f"{base}/{name}")))
                    got.append(name)
            if got:
                fetched.append(f"{a} ({', '.join(got)})")
        elif r["status"].startswith("launched"):
            pending.append((a, layers))
        # a finished local-client run has no results.json on the Volume: its files are in results/q1/ and on S3
    for a in fetched:
        print(f"fetched {a}")
    for a, n in pending:
        print(f"pending {n:2}/64 {a}")
    print(f"{len(fetched)} fetched now; {len(pending)} pending")


if __name__ == "__main__":
    main()
