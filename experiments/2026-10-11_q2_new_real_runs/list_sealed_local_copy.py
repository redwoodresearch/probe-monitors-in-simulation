"""Seal record: the laptop copy of zoo C stage 1's 100 oct11c .eval files, made for the pre-seal labels check. Hashes each
file (bytes only; nothing is parsed or read for content) and compares it with 2e's recorded eval_sha256, so the copy is
auditable. Local, $0.

  python3 experiments/2026-10-11_q2_new_real_runs/list_sealed_local_copy.py <2e's run list json> <the local honest_b folder>
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
RUN_LIST_SHA = "c0b270be308d437b6f641a640347ea232b4ab5a7a6e19c724f79ec498c6b4965"


def main(run_list: Path, folder: Path):
    if hashlib.sha256(run_list.read_bytes()).hexdigest() != RUN_LIST_SHA:
        raise SystemExit("run list differs from 2e's committed file")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(Path(__file__).resolve())], cwd=HERE).returncode:
        raise SystemExit("the script differs from HEAD: commit it before running")
    rows = json.loads(run_list.read_text())["rows"]
    files = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
    listed = {f"{r['key']}/{r['eval_file']}": r["eval_sha256"] for r in rows}
    entries = [{"file": f, "bytes": (folder / f).stat().st_size, "sha256": hashlib.sha256((folder / f).read_bytes()).hexdigest()} for f in files]
    out = {"git": head, "folder": str(folder), "files": len(entries), "files not in 2e's list": sorted(set(files) - set(listed)),
           "listed files missing": sorted(set(listed) - set(files)),
           "sha256 matching 2e's eval_sha256": sum(e["sha256"] == listed.get(e["file"]) for e in entries), "entries": entries}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"sealed_local_copy_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "entries"}, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
