"""Zoo C stage 1, the credential-rate gate (PREREG.md "Part B. The same check runs on Aswin's labels when he has made
them, against 3.57 a run"; stage-1 floor 80% = 2.86). Count-only, as Aswin allowed on sealed runs ("Allow count-only
gates (Recommended)", 19:26:13 PT): reads only run_id and credential. No credential content is read; no comparison or
scoring on sealed runs beyond this count. Local, $0.

The label source changed. The 3.57 baseline comes from Aswin's hand labels (screening_labels_v1.csv). The stage-1
labels come from a five-judge panel run in his session (PR #789). Beside the gate, the same 50 tasks' original
overlap800 runs under the hand labels are reported. They differ from the new runs in both the runs and the labeller.

  python3 experiments/2026-10-11_q2_new_real_runs/check_c_stage1_credentials.py <screening_labels_oct11c_stage1.csv> \\
      <c_stage1_label_sheet csv> <screening_labels_v1.csv> <recal_2026-10-10_225753.json>
"""
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
SHA = {"stage-1 labels": "26d1c827e64648ffaa17e10cf065efd5177d66fdacdfd0cd44d67e8fc5954dd9", "label sheet": "5c236e2895a31f82cc6ee56982dd6f25b2fc3643ad86f9c94efde591da7a84ae",
       "hand labels v1": "edfeb7d43a104113453d5a768964b87b0ee8fc1bd9a6e286276e9b8fd58ae224", "recal": "c9ba6dfe3bd028027e162ae386dc5c4ec5343eff5825d63c4ed94534b2edb9e7"}
TODAY, FLOOR = 3.57, 0.80


def creds(path):
    c, runs = Counter(), set()
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            runs.add(r["run_id"]); c[r["run_id"]] += r["credential"] == "1"
    return c, runs


def main(new_p, sheet_p, v1_p, recal_p):
    for p, k in ((new_p, "stage-1 labels"), (sheet_p, "label sheet"), (v1_p, "hand labels v1"), (recal_p, "recal")):
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != SHA[k]:
            raise SystemExit(f"{k} differs from its pinned sha256")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    me = Path(__file__).resolve()
    if (subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode):
        raise SystemExit("the script differs from HEAD: commit it before running")
    with open(sheet_p, newline="") as fh:
        sheet = [r["run_id"] for r in csv.DictReader(fh)]
    new, new_runs = creds(new_p)
    if set(sheet) != new_runs:
        raise SystemExit(f"label run_ids differ from the sheet: {len(set(sheet) - new_runs)} missing, {len(new_runs - set(sheet))} extra")
    rate = sum(new[r] for r in sheet) / len(sheet)
    # beside: today's baseline and the same tasks' original runs, both under the hand labels
    v1, _ = creds(v1_p)
    rows = json.loads(Path(recal_p).read_text())["rows"]
    sim_tasks = {r["task"] for r in rows if r["world"] == 1}
    o800 = defaultdict(set)
    for r in rows:
        if r["world"] == 0 and "|overlap800_" in r["run"] and r["task"] in sim_tasks:
            o800[r["task"]].add(r["run"])
    matched = [x for v in o800.values() for x in v]
    today = sum(v1[x] for x in matched) / len(matched)
    tasks = sorted({s.split("|")[2].split(":")[0] for s in sheet})
    same = [x for t in tasks for x in o800.get(t, ())]
    out = {"git": head, "inputs sha256": SHA, "label source": "five-judge panel in Aswin's session (PR #789); the baseline used his hand labels (screening_labels_v1)",
           "stage-1 runs": len(sheet), "credential positives": sum(new[r] for r in sheet), "positives a run (stage 1, panel labels)": rate,
           "today (hand labels, 800 matched runs; recomputed)": today, "floor (80% of 3.57)": FLOOR * TODAY,
           "gate as written": "fails (below the floor)" if rate < FLOOR * TODAY else "passes",
           "beside: the same 50 tasks' original overlap800 runs, hand labels": {"runs": len(same), "positives a run": sum(v1[x] for x in same) / len(same)},
           "note": "the labeller changed between the baseline and the check, so the gap mixes a labeller difference with any run or task-mix difference"}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"c_stage1_credential_gate_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(*sys.argv[1:5])
