"""Coverage check for the SCE recipe with Qwen3.5-27B as the reader (RULE.md Amendment 3): which arms exist, at
layer 63, in the stored Qwen3.5 dumps, the fix939 copies, and the committed reply detector's work Volume. Listings
only (`modal volume ls --json`), $0; no file is read. Prints and writes results/coverage_qwen35_<stamp>.json.

  uv run --no-project --python 3.12 --with modal==1.6.1 python experiments/2026-10-13_q3_sce_reply_probe/coverage_qwen35.py
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
CLI = ["uv", "run", "--no-project", "--python", "3.12", "--with", "modal==1.6.1", "modal", "volume", "ls"]
Q35 = "probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06"
FIX = "probe-shift_aswin-surya_sub-q2-q35fix_acts_kill-2026-11-13"
WORK = "probe-shift_aswin-surya_sub-q2-qwen35-reply_work_kill-2026-11-12"
DUMPS = {"base": (Q35, "2026-10-06_la_segment-end_qwen35-27b_alllayers_v1"),
         "v4": (Q35, "2026-10-06_la_segment-end-v4_qwen35-27b_alllayers_v1"),
         "fix939": (FIX, "2026-10-13_q2_q35fix_repaired_qwen35-27b_l63_v1")}


def ls(vol, path):
    r = subprocess.run(CLI + [vol, path, "--json"], capture_output=True, text=True, timeout=300)
    if r.returncode:
        return None  # not there (or the listing failed): reported as missing
    return [(e["filename"].rsplit("/", 1)[-1], e["type"], e["size"]) for e in json.loads(r.stdout)]


out = {"written": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"), "dumps": {}}
for name, (vol, run) in DUMPS.items():
    arms = ls(vol, f"{run}/meta") or []
    out["dumps"][name] = {"volume": vol, "run": run,
                          "metas_per_arm": {a: len([e for e in (ls(vol, f"{run}/meta/{a}") or []) if e[0].endswith(".json")])
                                            for a, t, _ in arms if t == "dir"},
                          "L63_chunk_files": len([e for e in (ls(vol, f"{run}/by_layer/L63") or []) if e[0].endswith(".npy")])}
runs = ls(WORK, "runs") or []
out["reply_detector_work"] = {"volume": WORK, "runs": sorted(r for r, t, _ in runs if t == "dir"),
                              "run_2026-10-12_011944": sorted(f"{n} {s}" for n, _, s in (ls(WORK, "runs/2026-10-12_011944") or [])),
                              "data": {d: sorted(f"{n} {s}" for n, _, s in (ls(WORK, f"data/{d}/packed") or []))
                                       for d, t, _ in (ls(WORK, "data") or []) if t == "dir"}}
p = HERE / "results" / f"coverage_qwen35_{out['written']}.json"
p.parent.mkdir(exist_ok=True)
p.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
