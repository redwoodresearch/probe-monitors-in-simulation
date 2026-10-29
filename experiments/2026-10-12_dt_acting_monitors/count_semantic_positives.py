"""Positives and unresolved rows per world for Aswin's PROVISIONAL semantic labels, before any fit (2026-10-13; $0).

Aswin, about 00:15 PT 2026-10-13, relayed by the orchestrator, answering whether to run a second fit on the semantic AI
labels: "6. yes". The orchestrator: "$0 first: validate the file (rows, IDs, columns). Then count, per world and per
arm, the unresolved rows and the positives among resolved rows, for both classes, against the size gate. Counts only,
through the held key." The labels are provisional (Aswin's hand-off: "A blank means unknown, never zero").

1. Validation: the semantic file (sha256 pinned) has the v3 sheet's 74,873 blind_ids in the same order, the columns
   blind_id, destructive_broad, transfer_semantic, destructive_unresolved, transfer_unresolved, labels 0/1 or blank,
   flags 0/1, and a blank label exactly where its flag is 1. The arm, task and impactful flag of each row come from
   count_label_positives.load (every hash and one-to-one join check, through the held key and the ID sheet).
2. Per class: a 1 is a positive; an unresolved row (blank) is excluded from both designs, read-only rows included
   (never counted as 0). Primary negatives are impactful 0s; secondary negatives are read-only 0s.
3. The decision rule, fixed before the counts (the orchestrator: "if the unresolved share differs between real and
   simulated by more than 2 points in either class, stop and report"): for each class, the unresolved share of
   real_attack_1's rows and of sim_attack's rows, each over all of that arm's rows (impactful and read-only) on the
   attack cohort (tasks with a row in all three attack arms); STOP if |real - sim| > 0.02 for either class. Shares for
   real_attack_2 and the honest arms are printed beside it, for information only.
4. The size gate per comparison, as count_label_positives.py (100 positives in 20 tasks per world), on resolved rows.
Prints and writes counts only.

  uv run --no-project --python 3.14 python -I -B count_semantic_positives.py --semantic <csv> --labels <q2 csv> \\
      --sheet <v3 csv> --ids <ID sheet> --key <held key> --out <json>
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import count_label_positives as C  # noqa: E402

SEM_SHA = "e9a6850b3ca8070385d9abbb4b2d270c36f9170988b6054ad5785553297157e9"
COLS = {"destructive": ("destructive_broad", "destructive_unresolved"), "transfer": ("transfer_semantic", "transfer_unresolved")}
ATTACK = ["real_attack_1", "sim_attack", "real_attack_2"]
MAX_SHARE_GAP = 0.02


def main(semantic: str, labels: str, sheet: str, ids: str, key: str, out: str):
    g = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout  # noqa: E731
    if g("status", "--porcelain", "--untracked-files=no") or not g("ls-files", "count_semantic_positives.py").strip():
        raise SystemExit("the script or another tracked file is uncommitted")
    git = g("rev-parse", "HEAD").strip()
    if hashlib.sha256(Path(semantic).read_bytes()).hexdigest() != SEM_SHA:
        raise SystemExit("semantic sha256 mismatch")
    base = C.load(labels, sheet, ids, key)  # arm, task, impactful per row, in the v3 order
    order = [r["blind_id"] for r in csv.DictReader(open(sheet, newline=""))]
    sem = list(csv.DictReader(open(semantic, newline="")))
    if list(sem[0]) != ["blind_id", "destructive_broad", "transfer_semantic", "destructive_unresolved", "transfer_unresolved"]:
        raise SystemExit("semantic file: wrong columns")
    if [r["blind_id"] for r in sem] != order:
        raise SystemExit("semantic file: blind_ids or order differ from the v3 sheet")
    rows = []
    for r, b in zip(sem, base):
        row = {"arm": b["arm"], "task": b["task"], "impactful": b["impactful"]}
        for cls, (lab, flag) in COLS.items():
            if r[flag] not in ("0", "1") or r[lab] not in ("0", "1", "") or (r[lab] == "") != (r[flag] == "1"):
                raise SystemExit("semantic file: a label or flag outside the rules")
            row[cls] = r[lab]
        rows.append(row)
    cohort = set.intersection(*({r["task"] for r in rows if r["arm"] == a} for a in ATTACK))
    report = {"git": git, "rows": len(rows), "provisional": True, "attack_cohort_tasks": len(cohort), "arms": {}, "decision": {}}
    for arm in C.ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        on = [r for r in rs if r["task"] in cohort] if arm in ATTACK else rs
        a = {"rows": len(rs), "impactful": sum(r["impactful"] for r in rs), "tasks": len({r["task"] for r in rs})}
        for cls in COLS:
            pos = [r for r in rs if r[cls] == "1"]
            a[cls] = {"unresolved": sum(r[cls] == "" for r in rs), "unresolved_impactful": sum(r[cls] == "" and r["impactful"] for r in rs),
                      "unresolved_share_on_cohort": round(sum(r[cls] == "" for r in on) / len(on), 6) if on else None,
                      "positives": len(pos), "positive_tasks": len({r["task"] for r in pos}), "read_only_positives": sum(not r["impactful"] for r in pos),
                      "primary_negatives": sum(r["impactful"] and r[cls] == "0" for r in rs), "secondary_negatives": sum(not r["impactful"] and r[cls] == "0" for r in rs)}
        report["arms"][arm] = a
    for cls in COLS:
        sr, ss = report["arms"]["real_attack_1"][cls]["unresolved_share_on_cohort"], report["arms"]["sim_attack"][cls]["unresolved_share_on_cohort"]
        report["decision"][cls] = {"real_attack_1": sr, "sim_attack": ss, "gap": round(sr - ss, 6), "stop": abs(sr - ss) > MAX_SHARE_GAP}
    report["decision"]["STOP"] = any(report["decision"][c]["stop"] for c in COLS)

    def role(r, cls, design):
        v = r[cls]
        if v == "1":
            return "pos"
        if v != "0":
            return None  # unresolved: excluded from every design
        return "neg" if r["impactful"] == (design == "primary") else None

    by = {arm: [r for r in rows if r["arm"] == arm] for arm in C.ARMS}
    sim_tasks = {r["task"] for r in by["sim_honest"]}
    comps = {"honest_real_vs_sim": (["real_honest_A", "sim_honest"], ["real_honest_A", "sim_honest"], sim_tasks),
             "honest_control_t01_t20": (["real_honest_A", "real_honest_B"], ["real_honest_A", "real_honest_B"], sim_tasks),
             "honest_control_all_tasks": (["real_honest_A", "real_honest_B"], ["real_honest_A", "real_honest_B"], None),
             "attack_real_vs_sim": (["real_attack_1", "sim_attack"], ATTACK, None),
             "attack_control": (["real_attack_1", "real_attack_2"], ATTACK, None)}
    report["comparisons"] = {}
    for name, (worlds, needed, restrict) in comps.items():
        for cls in COLS:
            for design in ("primary", "secondary"):
                present = set.intersection(*({r["task"] for r in by[w]} for w in needed))
                tasks = set.intersection(present, *({r["task"] for r in by[w] if role(r, cls, design)} for w in worlds))
                tasks = tasks & restrict if restrict is not None else tasks
                per = {}
                for w in worlds:
                    sub = [r for r in by[w] if r["task"] in tasks]
                    pos = [r for r in sub if role(r, cls, design) == "pos"]
                    per[w] = {"positives": len(pos), "positive_tasks": len({r["task"] for r in pos}),
                              "negatives": sum(role(r, cls, design) == "neg" for r in sub)}
                ok = all(v["positives"] >= C.GATE_POS and v["positive_tasks"] >= C.GATE_TASKS for v in per.values())
                report["comparisons"][f"{name}/{cls}/{design}"] = {"tasks": len(tasks), "worlds": per, "passes_gate": ok}
    report["written"] = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds")
    with open(out, "x") as fh:
        fh.write(json.dumps(report, indent=1) + "\n")
    print("PROVISIONAL semantic labels | attack cohort tasks", len(cohort))
    for arm, a in report["arms"].items():
        print(arm, {k: a[k] for k in ("rows", "impactful", "tasks")}, {c: {k: a[c][k] for k in ("unresolved", "unresolved_share_on_cohort", "positives", "positive_tasks")} for c in COLS})
    print("decision (unresolved share, real_attack_1 vs sim_attack, attack cohort; stop if gap > 0.02):", json.dumps(report["decision"]))
    for k, v in report["comparisons"].items():
        print(k, "tasks", v["tasks"], {w: (x["positives"], x["positive_tasks"], x["negatives"]) for w, x in v["worlds"].items()}, "PASS" if v["passes_gate"] else "below gate")


class Parser(argparse.ArgumentParser):
    def error(self, message):  # no echo: the arguments include the held key's path
        sys.stderr.write("failed: bad arguments\n")
        sys.exit(2)


if __name__ == "__main__":
    ap = Parser()
    for k in ("--semantic", "--labels", "--sheet", "--ids", "--key", "--out"):
        ap.add_argument(k, required=True)
    try:
        main(**vars(ap.parse_args()))
    except SystemExit:
        raise
    except BaseException as exc:  # no message or path: the key's location must not reach stderr
        print(f"failed: {type(exc).__name__} at line {traceback.extract_tb(exc.__traceback__)[-1].lineno}", file=sys.stderr)
        sys.exit(1)
