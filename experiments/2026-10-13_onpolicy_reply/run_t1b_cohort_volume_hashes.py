"""T1b cohort refit: sha256, size and modification time of every grid record run_t1b_cohort.py left on the score Volume
(onpolreply_t1bcoh_644e4f57904e/), computed on Modal CPU, for a manifest of Volume-only artifacts. The S3 copy waits for an
approved Modal-to-S3 route (#92). Codex on #1488: "Archive the cohort-refit grid outputs".

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run \\
      experiments/2026-10-13_onpolicy_reply/run_t1b_cohort_volume_hashes.py
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-onpol-reply_t1bcohhash_kill-2026-11-08"
VOL = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08"
ROOT = "onpolreply_t1bcoh_644e4f57904e"
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, cpu=2, memory=4096, timeout=900, retries=0, volumes={"/v": modal.Volume.from_name(VOL, version=2).read_only()})
def hash_tree(root: str) -> list:
    import hashlib

    out = []
    for p in sorted(q for q in Path("/v", root).rglob("*") if q.is_file()):
        h = hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        mtime = datetime.fromtimestamp(p.stat().st_mtime, ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT")
        out.append({"path": str(p.relative_to("/v")), "bytes": p.stat().st_size, "sha256": h.hexdigest(), "modified": mtime})
    return out


@app.local_entrypoint()
def main():
    assert not subprocess.run(["git", "status", "--porcelain", "--", str(HERE)], capture_output=True, text=True, check=True).stdout, "dirty"
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    files = hash_tree.remote(ROOT)
    assert files, f"no files under {ROOT}"
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M PT")
    note = (f"T1b cohort-refit grid records (run_t1b_cohort.py, RULE_T1B_COHORT.md), Volume-only, hashed on Modal CPU ({stamp}). "
            "S3 copy pending the approved Modal-to-S3 route (#92). The Volume's kill date is 2026-11-08.")
    lines = ["{", f' "note": {json.dumps(note)},', f' "git": "{git}",', f' "volume": {json.dumps(VOL)},',
             ' "kill_date": "2026-11-08",', ' "s3": null,', ' "files": [']
    lines += ["  " + json.dumps(r) + ("," if i < len(files) - 1 else "") for i, r in enumerate(files)]
    lines += [" ]", "}"]
    dest = HERE / "results" / "manifest_t1b_cohort_volume_v1.json"
    dest.open("x").write("\n".join(lines) + "\n")
    print(len(files), "files,", round(sum(r["bytes"] for r in files) / 1e9, 3), "GB")
