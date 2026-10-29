"""Zoo C calibration list (AMENDMENT_C_CALIBRATION.md): the (run_id, step, action) items of the hand labels on the 50
stage-1 tasks' original overlap800 runs, for Aswin's panel. IDs only; it reads run_id, step and action from
screening_labels_v1.csv, nothing else. Local, $0.

  python3 experiments/2026-10-11_q2_new_real_runs/make_calibration_list.py <screening_labels_v1.csv> <recal_2026-10-10_225753.json> \\
      <c_stage1_label_sheet csv>
"""
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
SHA = {"hand labels v1": "edfeb7d43a104113453d5a768964b87b0ee8fc1bd9a6e286276e9b8fd58ae224", "recal": "c9ba6dfe3bd028027e162ae386dc5c4ec5343eff5825d63c4ed94534b2edb9e7",
       "label sheet": "5c236e2895a31f82cc6ee56982dd6f25b2fc3643ad86f9c94efde591da7a84ae"}


def calibration_runs(recal_p, sheet_p):
    rows = json.loads(Path(recal_p).read_text())["rows"]
    sim_tasks = {r["task"] for r in rows if r["world"] == 1}
    o800 = defaultdict(set)
    for r in rows:
        if r["world"] == 0 and "|overlap800_" in r["run"] and r["task"] in sim_tasks:
            o800[r["task"]].add(r["run"])
    with open(sheet_p, newline="") as fh:
        tasks = sorted({r["run_id"].split("|")[2].split(":")[0] for r in csv.DictReader(fh)})
    return sorted(x for t in tasks for x in o800.get(t, ())), tasks


def main(v1_p, recal_p, sheet_p):
    for p, k in ((v1_p, "hand labels v1"), (recal_p, "recal"), (sheet_p, "label sheet")):
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != SHA[k]:
            raise SystemExit(f"{k} differs from its pinned sha256")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    me = Path(__file__).resolve()
    if (subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode):
        raise SystemExit("the script differs from HEAD: commit it before running")
    runs, tasks = calibration_runs(recal_p, sheet_p)
    want = set(runs)
    with open(v1_p, newline="") as fh:
        items = [(r["run_id"], r["step"], r["action"]) for r in csv.DictReader(fh) if r["run_id"] in want]
    covered = sorted({i[0] for i in items})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(lst := HERE / "results" / f"c_calibration_items_{stamp}.csv", "x", newline="") as fh:
        w = csv.writer(fh); w.writerow(["run_id", "step", "action"]); w.writerows(items)
    out = {"git": head, "inputs sha256": SHA, "tasks": len(tasks), "calibration runs": len(runs), "runs with hand-label rows": len(covered),
           "items": len(items), "list": lst.name, "list sha256": hashlib.sha256(lst.read_bytes()).hexdigest(),
           "return file": "~/probe-shift-q2-margins/data/zoo_c/screening_labels_calibration_panel.csv (screening_labels_v1.csv's columns)"}
    with open(dest := HERE / "results" / f"c_calibration_items_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(*sys.argv[1:4])
