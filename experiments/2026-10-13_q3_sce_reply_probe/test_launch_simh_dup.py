"""Tests for launch_simh.py's duplicate-writer check as refined by 0d (two live writers stop; a preemption restart is an alert to
confirm) and for check_one_result_per_input.py. Synthetic lines and timings, today's Amendment 13 run.logs, and today's records.
$0, local.

  uv run --no-project --python 3.12 python experiments/2026-10-13_q3_sce_reply_probe/test_launch_simh_dup.py <scratch dir with today's job dirs>
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from check_one_result_per_input import check  # noqa: E402
from launch_simh import classify, confirm_gone  # noqa: E402

TODAY_1447 = ("Container terminated due to preemption. Your Function will be restarted with the same input. For more details, "
              "see https://modal.com/docs/guide/preemption")
ALERT = [TODAY_1447, "Retrying input in-abc123 after a worker failure", "Container restarted (exit code 137)", "input in-xyz was re-sent",
         "Preempted: rescheduling", "Resending input", "Retrying..."]
STOP = ["two live containers are running the same input in-abc123", "duplicate writer on shard 3",
        "input in-abc is already being processed by another container"]
NONE = ["Warning: received a 429 for FunctionGetOutputs (attempt 1). Retrying...", "shard 3: 25/28", "✓ Created Function extract_shard.",
        "shard 2: dropped incomplete chunk shard002_chunk001 before resuming", "Stopping app - local entrypoint completed."]
for l in ALERT:
    assert classify(l) == "alert", l
for l in STOP:
    assert classify(l) == "stop", l
for l in NONE:
    assert classify(l) is None, l
assert classify(TODAY_1447) == "alert"  # today's 14:47 preemption: an alert now, not a stop
assert confirm_gone(1000.0, [950.0]) and confirm_gone(1000.0, [1090.0])  # a container died within 120 s: benign
assert not confirm_gone(1000.0, []) and not confirm_gone(1000.0, [700.0, 1300.0])  # no death near the restart: stop
print(f"classify: {len(ALERT)} alerts, {len(STOP)} stops, {len(NONE)} ignored; confirm_gone: 4 cases")

S = Path(sys.argv[1])
hits = {}
for d in sorted(S.glob("simh_job_*")) + sorted(S.glob("seedgrid_job_*")):
    f = d / "run.log"
    if f.exists():
        k = [classify(l) for l in f.read_text(errors="replace").splitlines()]
        hits[d.name] = {"stop": k.count("stop"), "alert": k.count("alert")}
print("today's run.logs:", hits)
assert all(v["stop"] == 0 for v in hits.values()) and {n for n, v in hits.items() if v["alert"]} == {"seedgrid_job_fit5"}

R = HERE / "results"
for name, ok in (("sce_seed_grid_run_2026-10-13_144706.json", True), ("sce_seed_grid_run_2026-10-13_143615.json", True),
                 ("extract_summary_2026-10-13_g3_simhonest_seed-oss-36b_v1_2026-10-13_142022.json", True)):
    if (R / name).exists():
        assert (not check(json.loads((R / name).read_text()))) == ok, name
rec = json.loads((R / "sce_seed_grid_run_2026-10-13_144706.json").read_text())
dup = rec | {"fit": rec["fit"] + [rec["fit"][-1]]}  # synthetic: one cell's fit returned twice
assert check(dup), "a doubled fit must fail"
print("check_one_result_per_input: today's records pass (the 14:47 grid fit: one completed call per input); a doubled fit fails")
