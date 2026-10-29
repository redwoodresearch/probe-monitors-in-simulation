"""Q4 (B): sha256, size and modification time of every file the cells and the label-shuffle null left on the Volume, computed on
Modal CPU (one container per directory), for a manifest of Volume-only artifacts. The S3 copy waits for an approved
Modal-to-S3 route. Directories: q4b_cells_work/<cell> (the packed inputs run_q4b_cells.py wrote and run_q4b_cells_null.py
read) and q4b_null_work/<stamp>/<cell> (the null's per-fit scores and results).

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-13_q4_why_weaker_POSTHOC/hash_q4b_volume_artifacts.py
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-q4own_hash_kill-2026-11-13"
VOL = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"
CELLS = ("q35_sol", "q3_sol", "q3_own", "q35_own", "q35_sol_cut32k")
DIRS = ([f"q4b_cells_work/{c}" for c in CELLS] + ["q4b_null_work/2026-10-14_022926/q35_own"]
        + [f"q4b_null_work/2026-10-14_023056/{c}" for c in ("q35_sol", "q3_sol", "q3_own", "q35_sol_cut32k")])
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, cpu=2, memory=4096, timeout=1800, retries=0, volumes={"/v": modal.Volume.from_name(VOL, version=2).read_only()})
def hash_dir(d: str) -> list:
    import hashlib

    out = []
    for p in sorted(Path("/v", d).iterdir()):
        h = hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        mtime = datetime.fromtimestamp(p.stat().st_mtime, ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT")
        out.append({"path": f"{d}/{p.name}", "bytes": p.stat().st_size, "sha256": h.hexdigest(), "modified": mtime})
    return out


@app.local_entrypoint()
def main():
    files = [r for rows in hash_dir.map(DIRS) for r in rows]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M PT")
    note = (f"Q4 (B) Volume-only artifacts, hashed on Modal CPU ({stamp}): the packed inputs of the five cells (run_q4b_cells.py) and "
            "the label-shuffle null's per-fit scores and results (run_q4b_cells_null.py, RULE.md Amendment 7). S3 copy pending the "
            "approved Modal-to-S3 route (#92). The Volume's kill date is 2026-11-13.")
    lines = ["{", f' "note": {json.dumps(note)},', f' "volume": {json.dumps(VOL)},', ' "kill_date": "2026-11-13",', ' "s3": null,', ' "files": [']
    lines += ["  " + json.dumps(r) + ("," if i < len(files) - 1 else "") for i, r in enumerate(files)]
    lines += [" ]", "}"]
    dest = HERE / "results" / "manifest_q4b_volume_v1.json"
    dest.open("x").write("\n".join(lines) + "\n")
    print(len(files), "files,", round(sum(r["bytes"] for r in files) / 1e9, 2), "GB")
    for r in files:
        print(r["path"], r["bytes"], r["sha256"][:12], r["modified"])
