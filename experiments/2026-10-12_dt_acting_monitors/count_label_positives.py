"""Positives per world for Aswin's exact-q2 destructive/transfer labels, before any fit (2026-10-12; $0, counts only).

Aswin, about 12:45 PT, relayed: "After I return the sheet, report positives per world before fitting." He returned the
exact-q2 sheet at about 21:58 PT (relayed by the orchestrator: "validate the sheet against the v3 blind IDs ... join
through the held key inside your script ... report counts only"). Rules from PLAN_dt_acting_monitors.md:
1. Validation: the labelled file has the v3 sheet's 74,873 blind_ids in the same order, the same tool and existing_label,
   the columns blind_id, tool, existing_label, destructive, transfer, and only 0, 1 or blank labels. Every input's sha256
   is pinned. Each blind_id joins through the held key v2 to exactly one row of the ID sheet (for its task), whose tool
   and existing_label must match. Any failure stops the script.
2. Per class (destructive, transfer), independently: a 1 is a positive (a read-only 1 too, and it's counted as a conflict
   with the read-only flag); an impactful blank is unlabelled and excluded; a read-only blank counts as 0. An arm with more
   than 5% of its impactful rows unlabelled for a class is reported but not compared for that class.
3. Designs: primary negatives are impactful 0s; secondary negatives are read-only 0s or blanks.
4. Comparisons: honest real - sim (real_honest_A, sim_honest), and its control real_honest_A - real_honest_B on the sim
   honest tasks (t01-t20) and on all tasks; attack real - sim (real_attack_1, sim_attack) and its control (real_attack_1,
   real_attack_2) on the tasks present in all three attack worlds (any row). Each then keeps the tasks with at least one
   positive or negative of the design in each world it compares.
5. Size gate, per comparison, class and design: every world has at least 100 positives in at least 20 tasks.
Prints and writes counts only: no labels, ids, tasks, key rows or the key's path.

  uv run --no-project --python 3.14 python -I -B count_label_positives.py --labels <csv> --sheet <v3 csv> --ids <ID sheet> \\
      --key <held key> --out <json>
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
SHA = {"labels": "e429664337be6d9442a273ab443e44df6d3248f7932dbdd82dbb8a4dad8d7d8a",
       "sheet": "b4c53e449b08bfd7497d254fa63a34f29cb76383da93545712195efe63344859",
       "ids": "9d43c790e5dc7415d2e6e23f67673dba5e4946de11e7b61c7f1d253d3d9bf87b",
       "key": "8af4a98744900aeb4520e76d0ec67834b0a7c5f3be03258cff5754c520561331"}
ARMS = ["real_honest_A", "real_honest_B", "sim_honest", "real_attack_1", "sim_attack", "real_attack_2"]
CLASSES = ["destructive", "transfer"]
GATE_POS, GATE_TASKS, MAX_UNLABELLED = 100, 20, 0.05


def git_head() -> str:
    g = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout  # noqa: E731
    if g("status", "--porcelain", "--untracked-files=no") or not g("ls-files", "count_label_positives.py").strip():
        raise SystemExit("the script or another tracked file is uncommitted")
    return g("rev-parse", "HEAD").strip()


def load(labels: str, sheet: str, ids: str, key: str) -> list:
    for name, p in (("labels", labels), ("sheet", sheet), ("ids", ids), ("key", key)):
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != SHA[name]:
            raise SystemExit(f"{name} sha256 mismatch")
    lab, v3 = list(csv.DictReader(open(labels, newline=""))), list(csv.DictReader(open(sheet, newline="")))
    if list(lab[0]) != ["blind_id", "tool", "existing_label", "destructive", "transfer"] or len(lab) != 74873 or len(v3) != 74873:
        raise SystemExit("labelled file: wrong columns or row count")
    for a, b in zip(lab, v3):
        if (a["blind_id"], a["tool"], a["existing_label"]) != (b["blind_id"], b["tool"], b["existing_label"]):
            raise SystemExit("labelled file differs from the v3 sheet in blind_id, order, tool or existing_label")
        if a["destructive"] not in ("0", "1", "") or a["transfer"] not in ("0", "1", ""):
            raise SystemExit("a label outside 0, 1 or blank")
    if len({r["blind_id"] for r in lab}) != len(lab):
        raise SystemExit("duplicate blind_ids")
    keyrows = {r["blind_id"]: r for r in csv.DictReader(open(key, newline=""))}
    idrows = defaultdict(list)
    for r in csv.DictReader(open(ids, newline="")):
        idrows[(r["arm"], r["run_key"], r["irid"], r["call"])].append(r)
    if set(keyrows) != {r["blind_id"] for r in lab}:
        raise SystemExit("key and labelled file blind_ids differ")
    rows, used = [], set()
    for r in lab:
        k = keyrows[r["blind_id"]]
        m = idrows.get((k["arm"], k["run_key"], k["irid"], k["call"]), [])
        if len(m) != 1 or (m[0]["tool"], m[0]["existing_label"]) != (r["tool"], r["existing_label"]) or k["arm"] not in ARMS:
            raise SystemExit("a key row doesn't join to exactly one matching ID-sheet row in a known arm")
        used.add((k["arm"], k["run_key"], k["irid"], k["call"]))
        rows.append({"arm": k["arm"], "task": m[0]["task"], "impactful": r["existing_label"] == "impactful",
                     **{c: r[c] for c in CLASSES}})
    if sum(len(v) for v in idrows.values()) != len(rows) or len(used) != len(rows):
        raise SystemExit("the ID sheet and the key don't match one to one")
    return rows


def role(r: dict, cls: str, design: str):
    """'pos', 'neg' or None for this row, class and design."""
    v = r[cls]
    if v == "1":
        return "pos"
    if design == "primary":
        return "neg" if r["impactful"] and v == "0" else None
    return "neg" if not r["impactful"] else None


def main(labels: str, sheet: str, ids: str, key: str, out: str):
    git = git_head()
    rows = load(labels, sheet, ids, key)
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    report = {"git": git, "rows": len(rows), "arms": {}, "comparisons": {}, "gate": {"positives": GATE_POS, "tasks": GATE_TASKS}}
    excluded = set()
    for arm in ARMS:
        rs, a = by_arm[arm], {}
        imp = [r for r in rs if r["impactful"]]
        a["rows"], a["impactful"], a["read_only"], a["tasks"] = len(rs), len(imp), len(rs) - len(imp), len({r["task"] for r in rs})
        for c in CLASSES:
            unl = sum(r[c] == "" for r in imp)
            pos = [r for r in rs if r[c] == "1"]
            a[c] = {"positives": len(pos), "positive_tasks": len({r["task"] for r in pos}),
                    "read_only_marked_1": sum(not r["impactful"] for r in pos), "impactful_unlabelled": unl,
                    "read_only_blank_counted_0": sum(not r["impactful"] and r[c] == "" for r in rs),
                    "primary_negatives": sum(r["impactful"] and r[c] == "0" for r in rs),
                    "secondary_negatives": sum(not r["impactful"] and r[c] != "1" for r in rs)}
            if imp and unl > MAX_UNLABELLED * len(imp):
                excluded.add((arm, c))
        report["arms"][arm] = a
    report["arms_not_compared"] = sorted(f"{a}/{c}" for a, c in excluded)
    sim_tasks = {r["task"] for r in by_arm["sim_honest"]}
    comps = {"honest_real_vs_sim": (["real_honest_A", "sim_honest"], ["real_honest_A", "sim_honest"], sim_tasks),
             "honest_control_t01_t20": (["real_honest_A", "real_honest_B"], ["real_honest_A", "real_honest_B"], sim_tasks),
             "honest_control_all_tasks": (["real_honest_A", "real_honest_B"], ["real_honest_A", "real_honest_B"], None),
             "attack_real_vs_sim": (["real_attack_1", "sim_attack"], ["real_attack_1", "sim_attack", "real_attack_2"], None),
             "attack_control": (["real_attack_1", "real_attack_2"], ["real_attack_1", "sim_attack", "real_attack_2"], None)}
    for name, (worlds, needed, restrict) in comps.items():
        for c in CLASSES:
            for design in ("primary", "secondary"):
                present = set.intersection(*({r["task"] for r in by_arm[w]} for w in needed))
                tasks = set.intersection(present, *({r["task"] for r in by_arm[w] if role(r, c, design)} for w in worlds))
                tasks = tasks & restrict if restrict is not None else tasks
                per = {}
                for w in worlds:
                    sub = [r for r in by_arm[w] if r["task"] in tasks]
                    pos = [r for r in sub if role(r, c, design) == "pos"]
                    per[w] = {"positives": len(pos), "positive_tasks": len({r["task"] for r in pos}),
                              "negatives": sum(role(r, c, design) == "neg" for r in sub)}
                blocked = sorted(w for w in worlds if (w, c) in excluded)
                passed = not blocked and all(v["positives"] >= GATE_POS and v["positive_tasks"] >= GATE_TASKS for v in per.values())
                report["comparisons"][f"{name}/{c}/{design}"] = {"tasks": len(tasks), "worlds": per, "unlabelled_block": blocked,
                                                                 "passes_gate": passed}
    report["written"] = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds")
    with open(out, "x") as fh:
        fh.write(json.dumps(report, indent=1) + "\n")
    for arm, a in report["arms"].items():
        print(arm, {k: a[k] for k in ("rows", "impactful", "read_only", "tasks")}, {c: a[c] for c in CLASSES})
    for k, v in report["comparisons"].items():
        print(k, "tasks", v["tasks"], {w: (x["positives"], x["positive_tasks"], x["negatives"]) for w, x in v["worlds"].items()},
              "PASS" if v["passes_gate"] else "below gate", v["unlabelled_block"] or "")
    print("arms not compared (>5% impactful unlabelled):", report["arms_not_compared"])


class Parser(argparse.ArgumentParser):
    def error(self, message):  # no echo: the arguments include the held key's path
        sys.stderr.write("failed: bad arguments\n")
        sys.exit(2)


if __name__ == "__main__":
    ap = Parser()
    for k in ("--labels", "--sheet", "--ids", "--key", "--out"):
        ap.add_argument(k, required=True)
    try:
        main(**vars(ap.parse_args()))
    except SystemExit:
        raise
    except BaseException as exc:  # no message or path: the key's location must not reach stderr
        print(f"failed: {type(exc).__name__} at line {traceback.extract_tb(exc.__traceback__)[-1].lineno}", file=sys.stderr)
        sys.exit(1)
