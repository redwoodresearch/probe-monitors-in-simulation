"""Zoo C stage 1: check 2e's run list against the batch spec and write Aswin's label sheet (IDs only; AMENDMENT_C_GUARD.md
"stage 1 = the first 50 tasks, then Aswin's labels, then the credential-rate check"). Reads run IDs, tasks and statuses
only; never opens a transcript. The sheet has runs_phase1_honest.csv's columns. Local, $0.

  python3 experiments/2026-10-11_q2_new_real_runs/make_stage1_label_sheet.py <zoo_c_runs_oct11c_stage1_2026-10-11_183251.json>
"""
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
RUN_LIST_SHA = "c0b270be308d437b6f641a640347ea232b4ab5a7a6e19c724f79ec498c6b4965"  # 2e's file, aswin/zoo-c-stage1 @ cc70fcf0
SPEC = HERE / "results/batch_spec_2026-10-11_151207.json"
ORDER_SHA = "b804d4d44f1ea8d716ab30b0bb099e7d9e59ef908d427de013cf3d36b0d00c8b"
S3 = "s3://individual-aswin-surya-probe-shift-kill-2027-04-06/phase2/runs"
COLUMNS = ["run_id", "arm", "main_task", "real_run_order (1 first, 2 second, 0 other or single)", "dump"]


def main(run_list: Path):
    if hashlib.sha256(run_list.read_bytes()).hexdigest() != RUN_LIST_SHA:
        raise SystemExit("run list differs from 2e's committed file")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    me = Path(__file__).resolve()
    if (subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode):
        raise SystemExit("the script differs from HEAD: commit it before running")
    spec, d = json.loads(SPEC.read_text()), json.loads(run_list.read_text())
    if hashlib.sha256("\n".join(spec["task order"]).encode()).hexdigest() != ORDER_SHA:
        raise SystemExit("task order differs from the batch spec's sha")
    stage1 = spec["task order"][:spec["stage 1 (first tasks)"]]
    rows = d["rows"]
    checks = {
        "rows": len(rows),
        "positions 1-50, each with run 1 and run 2": sorted((r["position"], r["run_order"]) for r in rows) == [(p, k) for p in range(1, 51) for k in (1, 2)],
        "task at each position matches the spec's order": all(r["task"] == stage1[r["position"] - 1] for r in rows),
        "sample id names the task": all(r["sample_id"].split(":")[0] == r["task"] for r in rows),
        "distinct sample ids": len({r["sample_id"] for r in rows}) == len(rows),
        "distinct eval files": len({r["eval_sha256"] for r in rows}) == len(rows),
        "status success": sum(r["status"] == "success" for r in rows), "sample errors": sum(r["sample_errors"] for r in rows),
        "ended": {e: sum(r["ended"] == e for r in rows) for e in sorted({r["ended"] for r in rows})},
        "step-limit hits": sum(bool(r["step_limit_hit"]) for r in rows),
    }
    if not all(checks[k] for k in ("positions 1-50, each with run 1 and run 2", "task at each position matches the spec's order",
                                   "sample id names the task", "distinct sample ids", "distinct eval files")):
        raise SystemExit(f"run list doesn't match the batch spec: {checks}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    sheet = HERE / "results" / f"c_stage1_label_sheet_{stamp}.csv"
    with open(sheet, "x", newline="") as fh:
        w = csv.writer(fh); w.writerow(COLUMNS)
        for r in sorted(rows, key=lambda r: (r["position"], r["run_order"])):
            w.writerow([f"real-honest-oct11c|{r['key']}/{r['eval_file']}|{r['sample_id']}|1", "real-honest-oct11c", r["task"], r["run_order"], ""])
    out = {"git": head, "run list sha256": RUN_LIST_SHA, "task order sha256": ORDER_SHA, "checks": checks,
           "sheet": sheet.name, "sheet sha256": hashlib.sha256(sheet.read_bytes()).hexdigest(),
           "eval files": f"{S3}/{d['head']['run_folder']}/honest_b/<key>/<eval_file>",
           "dump": "empty until d9's extraction of this batch"}
    with open(dest := HERE / "results" / f"c_stage1_label_sheet_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
