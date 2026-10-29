"""Item D, before its rule: the pool of honest LinuxArena main tasks per environment (Modal CPU, counts only). Aswin's go
for D, about 15:53 PT 2026-10-12, relayed by 0d, verbatim: "Buy D (~$56)".

Per environment, from the v4 staging's real honest runs: main tasks with a run of at most 40k staged tokens, and how
many of those were used before (the eval-awareness probe's 99 runs and checker draw, and the cue pilot's 120 tasks).
Only environment names and counts leave.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal modal run experiments/2026-10-12_matched_training_d/count_d_pool.py
"""

import json
from pathlib import Path

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
app = modal.App("probe-shift_aswin-surya_sub-matched-d-count_kill-2026-11-12")
la = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2).read_only()


@app.function(image=modal.Image.debian_slim(python_version="3.12"), volumes={"/in": la}, cpu=(2, 2), memory=(8192, 8192),
              timeout=600, startup_timeout=300, retries=0)
def count(used_runs: list, used_tasks: list) -> dict:
    from collections import Counter
    env, task_of, small = {}, {}, set()
    for p in sorted(Path("/in/2026-10-06_la_segment-end-v4_qwen3-32b_v1/inputs").glob("shard_*.jsonl")):
        for it in map(json.loads, p.read_text().splitlines()):
            if it["arm"] == "real-honest":
                env[it["main_task"]] = it["environment"]
                task_of[it["key"]] = it["main_task"]
                if it["tokens"] <= 40_000:
                    small.add(it["main_task"])
    used = {task_of[k] for k in used_runs if k in task_of} | set(used_tasks)
    out = {}
    for e in sorted(set(env.values())):
        tasks = {t for t, x in env.items() if x == e}
        out[e] = {"tasks": len(tasks), "with a run <= 40k": len(tasks & small), "of those, used before": len(tasks & small & used)}
    return {"by environment": out, "used tasks found": len(used), "unknown used runs": sum(k not in task_of for k in used_runs)}


@app.local_entrypoint()
def main():
    ex = json.loads((ROOT / "experiments/2026-10-10_eval_awareness_probe/results/retrain_frozen_inputs.json").read_text())
    dry = json.loads((ROOT / "experiments/2026-10-12_cue_pilot/results/dry_2026-10-12_150605.json").read_text())["result"]
    out = count.remote(ex["runs"] + (ex.get("checker runs") or []), [t for s in dry["tasks"].values() for t in s])
    path = HERE / "results" / "pool_counts.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
