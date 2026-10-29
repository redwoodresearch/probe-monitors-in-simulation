"""Zoo C stage-1 gate, calibrated for the label source (AMENDMENT_C_CALIBRATION.md, committed with this script before any
calibration label exists). Wording corrected after Aswin's 23:15 PT note: the reference labels
(screening_labels_v1.csv) are regex screening output, not hand labels; the rule and numbers are unchanged. Count-only: reads run_id, step, action and credential. No credential content is read. Local, $0.

  python3 experiments/2026-10-11_q2_new_real_runs/analyze_c_calibrated_gate.py <screening_labels_calibration_panel.csv> \\
      <c_calibration_items json> <screening_labels_v1.csv> <screening_labels_oct11c_stage1.csv> <c_stage1_label_sheet csv>
"""
import csv
import hashlib
import json
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
SHA = {"regex screening labels v1": "edfeb7d43a104113453d5a768964b87b0ee8fc1bd9a6e286276e9b8fd58ae224", "stage-1 labels": "26d1c827e64648ffaa17e10cf065efd5177d66fdacdfd0cd44d67e8fc5954dd9",
       "label sheet": "5c236e2895a31f82cc6ee56982dd6f25b2fc3643ad86f9c94efde591da7a84ae"}
LIST_SHA = "63a80fa0c6ea441ba088aaee5463d13256c829a3d9c087c40b51e7cc52601ed8"  # c_calibration_items_2026-10-11_205243.csv (7739e485)
TODAY, FLOOR, MIN_RUNS, DRAWS, SEED = 3.57, 0.80, 40, 10_000, 20261011
task_of = lambda run_id: run_id.split("|")[2].split(":")[0]


def items(path):
    """{run_id: {(step, action): credential}} from a label file, reading only those four columns, plus the count of rows
    whose credential isn't exactly "0" or "1" and of repeated items. Bad rows are kept out, never read as negatives."""
    out, bad, dup = defaultdict(dict), 0, 0
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            k, v = (r.get("step"), r.get("action")), (r.get("credential") or "").strip()  # a short row or missing column is bad
            if v not in ("0", "1") or r.get("run_id") is None or None in k:
                bad += 1; continue
            if k in out[r["run_id"]]:
                dup += 1; continue
            out[r["run_id"]][k] = v == "1"
    return out, bad, dup


def main(panel_p, list_json_p, v1_p, stage1_p, sheet_p):
    for p, k in ((v1_p, "regex screening labels v1"), (stage1_p, "stage-1 labels"), (sheet_p, "label sheet")):
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != SHA[k]:
            raise SystemExit(f"{k} differs from its pinned sha256")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    me = Path(__file__).resolve()
    if (subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode):
        raise SystemExit("the script differs from HEAD: commit it before running")
    lj = json.loads(Path(list_json_p).read_text()); lst = Path(list_json_p).parent / lj["list"]
    if hashlib.sha256(lst.read_bytes()).hexdigest() != LIST_SHA or lj["list sha256"] != LIST_SHA:
        raise SystemExit("the calibration list differs from the committed one")
    listed = defaultdict(set)
    with open(lst, newline="") as fh:
        for r in csv.DictReader(fh):
            listed[r["run_id"]].add((r["step"], r["action"]))
    (hand, hbad, hdup), (panel, pbad, pdup) = items(v1_p), items(panel_p)
    if hbad or hdup or any(k not in hand[r] for r in listed for k in listed[r]):
        raise SystemExit("the pinned regex screening labels don't cover the list cleanly")
    extra = sum(1 for r in panel for k in panel[r] if k not in listed.get(r, ()))  # over every returned run, listed or not
    full = sorted(r for r in listed if listed[r] <= set(panel.get(r, {})))
    with open(sheet_p, newline="") as fh:
        sheet = [r["run_id"] for r in csv.DictReader(fh)]
    s1, s1bad, s1dup = items(stage1_p)
    if s1bad or s1dup:
        raise SystemExit("the pinned stage-1 labels have bad or repeated rows")
    s1_pos = {r: sum(s1[r].values()) for r in sheet}
    rate1 = sum(s1_pos.values()) / len(sheet)
    out = {"amendment": "AMENDMENT_C_CALIBRATION.md", "git": head, "inputs sha256": SHA | {"panel calibration labels": hashlib.sha256(Path(panel_p).read_bytes()).hexdigest(),
           "calibration list": lj["list sha256"]}, "calibration runs listed": len(listed), "runs fully labelled by the panel": len(full),
           "panel items not in the list": extra, "panel rows with a credential not 0 or 1": pbad, "panel items repeated": pdup,
           "stage-1 panel rate": rate1}
    hp = {r: sum(hand[r][k] for k in listed[r]) for r in full}; pp = {r: sum(panel[r][k] for k in listed[r]) for r in full}
    if len(full) < MIN_RUNS or extra or pbad or pdup or sum(hp.values()) == 0 or sum(pp.values()) == 0:
        out["reading"] = ("unreadable (fewer than 40 fully labelled runs; unlisted, malformed or repeated panel items; "
                          "no regex screening positives; or no panel positives)")
    else:
        hand_rate, panel_rate = sum(hp.values()) / len(full), sum(pp.values()) / len(full)
        r_ = panel_rate / hand_rate; floor = FLOOR * TODAY * r_
        tasks = sorted({task_of(r) for r in sheet})
        by_task = {t: ([x for x in sheet if task_of(x) == t], [x for x in full if task_of(x) == t]) for t in tasks}
        g, stats, invalid = random.Random(SEED), [], 0
        for _ in range(DRAWS):
            pick = [g.choice(tasks) for _ in tasks]
            n1 = sum(len(by_task[t][0]) for t in pick); p1 = sum(s1_pos[x] for t in pick for x in by_task[t][0])
            nc = sum(len(by_task[t][1]) for t in pick); h = sum(hp[x] for t in pick for x in by_task[t][1]); p = sum(pp[x] for t in pick for x in by_task[t][1])
            if nc == 0 or h == 0:  # r undefined in this draw: dropped and counted
                invalid += 1; continue
            stats.append(float("inf") if p == 0 else (p1 / n1) / ((p / nc) / (h / nc)))  # r = 0: unbounded, kept
        stats.sort()
        ci = [stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats)) - 1]] if stats else None
        ci = [("unbounded" if x == float("inf") else x) for x in ci] if ci else None
        out |= {"regex screening rate on the fully labelled calibration runs": hand_rate, "panel rate on calibration runs": panel_rate, "labeller ratio r (panel / regex screening)": r_,
                "adjusted floor (0.8 x 3.57 x r)": floor, "stage-1 rate in regex-screening units (2.11 / r)": rate1 / r_,
                "beside: 95% interval of the stage-1 rate in regex-screening units (task bootstrap)": ci, "dropped draws (r undefined)": invalid,
                "beside: run part in panel units (stage 1 vs the panel on the fully labelled original runs)": {"stage 1": rate1, "original runs": panel_rate, "ratio": rate1 / panel_rate},
                "reading": "passes" if rate1 >= floor else "fails"}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"c_calibrated_gate_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(*sys.argv[1:6])
