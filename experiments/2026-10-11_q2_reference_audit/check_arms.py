"""Reference audit, part 2: which arms the Q2.3 threshold-in-sim inputs hold, off- and on-policy. Reads arm and world labels
only. Local, $0.

  python3 experiments/2026-10-11_q2_reference_audit/check_arms.py <refit_qwen35_rows.json> <onpolicy_2x2_rerun_strict_september_percall json>
"""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
SHA = {"refit rows": "41a68be0b1d7ae045d573dce15ce2a4b3cfa349f915cc6babc7fd478f28ed8e4",  # the TPR-at-1% manifest (#308)
       "on-policy per-call": "31f459580c74bb017186aca56291857f294ad0e1cdb71c84b2fc72b72a350c76"}


def main(rows_p: Path, percall_p: Path):
    for p, k in ((rows_p, "refit rows"), (percall_p, "on-policy per-call")):
        if hashlib.sha256(p.read_bytes()).hexdigest() != SHA[k]:
            raise SystemExit(f"{k} differs from the manifest's sha256")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(Path(__file__).resolve())], cwd=HERE).returncode:
        raise SystemExit("the script differs from HEAD: commit it before running")
    rows = json.loads(rows_p.read_text()); pc = json.loads(percall_p.read_text())
    out = {"git": head, "inputs sha256": SHA,
           "off-policy refit rows (threshold-in-sim off-policy, per-task false alarms, TPR at 1%)": dict(Counter(r["arm"] for r in rows)),
           "on-policy per-call (threshold-in-sim on-policy)": {"config worlds": pc["config"]["worlds"], "calls by world": dict(Counter(m["world"] for m in pc["meta"]))}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"arms_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
