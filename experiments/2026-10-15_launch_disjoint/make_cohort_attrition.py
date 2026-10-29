"""Cohort attrition by exclusion class, with keys (for the write-up): S_new 311 -> 272 and nm360 360 -> 312 (-> 244 without S4).

Sources (git by sha:path; keys and labels only, no score file):
- S_new: Stage A's pinned FINAL record (cohort_cells; pin in run_common455.py @ 214f74de4), with the run table and attempt
  labels its spec pins (final_spec_3_launch.json @ 3ce6ea55e). The cohort rule is "attempted in run1, run2 and sim".
- nm360: membership710.json @ 32abe4120 (cells, cohort, cohort_4r_sol_only, status_rows, batch_of_cell, cohort_without_S4).
- The frozen INPUTS_FINAL @ 32abe4120: the ledger's sim-honest cell exclusions (H_sim only, per its stage_a rule).
Every class list sums exactly to its drop (asserted). Writes results/cohort_attrition_<stamp PT>.{json,md} (create-only).

Usage (repo root, script committed first):  python3 -I experiments/2026-10-15_launch_disjoint/make_cohort_attrition.py
"""

import collections
import datetime
import hashlib
import json
import re
import subprocess
import zoneinfo
from pathlib import Path

HERE = Path(__file__).resolve().parent
FINAL = ("3ce6ea55e", "experiments/2026-10-14_xerr_scaleup_09/results/xerr_scaleup_final_2026-10-15_013255.json", "7dceb8bb25752aa3619cfde64add702fd4f7d13a1309a56dbc8880a6b27af9f5")
SPEC = ("3ce6ea55e", "experiments/2026-10-14_xerr_scaleup_09/specs/final_spec_3_launch.json")
STAGE_A_PIN = ("214f74de4", "experiments/2026-10-15_fpr_sweep_507_prep/run_common455.py")
PLUS311 = ("0caadbfb5", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json", "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b")
M710 = ("32abe4120", "experiments/2026-10-15_fpr_sweep_507_prep/config/membership710.json", "6516d15fab77d7f344497e72db10b60c5f709b35c3a1859a2c9e016d2fc6e4ad")
INPUTS_FINAL = ("32abe4120", "experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FINAL_2026-10-15_203213.json")
LEDGER_EXCLUDED = {"S_new": ["t361", "t389", "t411", "t413"], "S_newmains": ["nm107", "nm134", "nm150"]}


def git_bytes(sha, path):
    return subprocess.run(["git", "show", f"{sha}:{path}"], check=True, capture_output=True).stdout


def load(sha, path, want=None):
    b = git_bytes(sha, path)
    h = hashlib.sha256(b).hexdigest()
    if want and h != want:
        raise SystemExit(f"{path}: sha256 {h} != pinned {want}")
    return json.loads(b), {"commit": sha, "path": path, "sha256": h}


def main():
    pin = re.search(r'FINAL = \("([0-9a-f]+)", "([^"]+)",\s*"([0-9a-f]{64})"\)', git_bytes(*STAGE_A_PIN).decode())
    if not pin or pin.groups() != FINAL:
        raise SystemExit("STOP: FINAL differs from Stage A's pin")
    fin, s_fin = load(*FINAL)
    spec, s_spec = load(*SPEC)
    labels_ref, table_ref = spec["labels"][0], spec["run_table"][0]
    labels, s_lab = load(labels_ref["commit"], labels_ref["path"], labels_ref["sha256"])
    table, s_tab = load(table_ref["commit"], table_ref["path"], table_ref["sha256"])
    p311, s_p311 = load(*PLUS311)
    m710, s_m710 = load(*M710)
    inf, s_inf = load(*INPUTS_FINAL)

    # ---- S_new: 311 -> 272
    cells311 = [c["key"] for c in p311["cells"]]
    coh_new = sorted(c for c in fin["result"]["cohort"]["cohort_cells"] if c in set(cells311))
    in_table = {r["cell"] for r in table["cells"]}
    classes_new = collections.defaultdict(list)
    for d_ in table["dropped"]:
        classes_new[f"no run table row: dropped at {d_['stage']} ({d_['reason']})"].append(d_["cell"])
    lab = labels["labels"]
    for c in sorted(in_table):
        tup = [(arm, lab.get(f"{arm}|{c}", "no label")) for arm in ("run1", "run2", "sim")]
        bad = [f"{arm} {v}" for arm, v in tup if v != "attempted"]
        if bad:
            classes_new["not attempted in all three: " + " + ".join(bad)].append(c)
    missing_everywhere = sorted(set(cells311) - in_table - {d_["cell"] for d_ in table["dropped"]})
    if missing_everywhere:
        classes_new["in the +311 list but absent from the run table"].extend(missing_everywhere)
    computed = sorted(set(cells311) - {c for v in classes_new.values() for c in v})
    if computed != coh_new:
        raise SystemExit(f"S_new: classes do not reproduce the FINAL cohort ({len(computed)} vs {len(coh_new)})")
    drop_new = len(cells311) - len(coh_new)
    if sum(len(v) for v in classes_new.values()) != drop_new or (len(cells311), len(coh_new)) != (311, 272):
        raise SystemExit("S_new: classes do not sum to the drop")

    # ---- nm360: 360 -> 312 -> 244 (without S4)
    batch = m710["batch_of_cell"]
    cohort_nm = sorted(set(m710["cohort"]) | set(m710["cohort_4r_sol_only"]))
    status = m710["status_rows"]
    nm_cells = sorted(set(cohort_nm) | set(status))
    if len(nm_cells) != m710["counts"]["cells"] or set(cohort_nm) & set(status):
        raise SystemExit("nm360: cohort and status rows do not partition the 360 cells")
    classes_nm = collections.defaultdict(list)
    for c, why in sorted(status.items()):
        head = why.split(":", 1)[0].strip()
        if head.startswith("setup failure"):
            cls = why.split(";", 1)[0].strip()
        elif head == "S4r":
            cls = "S4r status row (" + why.split(":")[1].strip() + ")"
        else:
            cls = f"{head}: {why.split(':', 1)[1].strip()}"
        classes_nm[cls].append(c)
    by_head = collections.Counter(("S4r" if k.startswith("S4r") else k.split(":", 1)[0] if not k.startswith("setup failure") else k) for k, v in classes_nm.items() for _ in v)
    if dict(by_head) != m710["counts"]["status_by_reason"]:
        raise SystemExit(f"nm360: class heads {dict(by_head)} != membership710 status_by_reason")
    drop_nm = len(nm_cells) - len(cohort_nm)
    if sum(len(v) for v in classes_nm.values()) != drop_nm or (len(nm_cells), len(cohort_nm)) != (360, 312):
        raise SystemExit("nm360: classes do not sum to the drop")
    s4_out = collections.defaultdict(list)
    for c in cohort_nm:
        if batch.get(c) in ("S4", "S4r"):
            s4_out[f"batch {batch[c]} (S4 out of tonight's attack scope; pools whole)"].append(c)
    if sorted(set(cohort_nm) - {c for v in s4_out.values() for c in v}) != sorted(m710["cohort_without_S4"]):
        raise SystemExit("nm360: S4 removal does not reproduce cohort_without_S4")

    # ---- the frozen ledger's sim-honest cell exclusions (H_sim only)
    excl_new = {k.split("|", 1)[1] for k in inf["pools"]["stage_a"]["S_new"]["excluded_cells"]}
    label = inf["label"]["filled"]
    ledger = {}
    for s, ids in LEDGER_EXCLUDED.items():
        for i in ids:
            if s == "S_new" and i not in excl_new:
                raise SystemExit(f"{i}: not in INPUTS_FINAL stage_a S_new excluded_cells")
            if f"{s} {i}" not in label:
                raise SystemExit(f"{i}: not named in INPUTS_FINAL's ledger label")
            pool = cells311 if s == "S_new" else nm_cells
            cell = next(c for c in pool if c.startswith(i + "_"))
            coh = coh_new if s == "S_new" else cohort_nm
            ledger[cell] = {"set": s, "in_attack_cohort": cell in coh,
                            "in_cohort_without_S4": (cell in m710["cohort_without_S4"]) if s == "S_newmains" else None,
                            "applies_to": "H_sim (sim-honest) only, per INPUTS_FINAL pools.stage_a rule; the attack cohort count is unchanged"}

    stamp = datetime.datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    script_rel = "experiments/2026-10-15_launch_disjoint/make_cohort_attrition.py"
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    blob = subprocess.run(["git", "hash-object", script_rel], capture_output=True, text=True, check=True).stdout.strip()
    if subprocess.run(["git", "rev-parse", f"HEAD:{script_rel}"], capture_output=True, text=True).stdout.strip() != blob:
        raise SystemExit("commit the script before running it")
    out = {
        "label": "cohort attrition by exclusion class, keys only; no score read",
        "stamp (PT)": stamp, "generator": {"path": script_rel, "commit": head, "blob_sha1": blob},
        "sources": {"final": s_fin | {"stage_a_pin": f"{STAGE_A_PIN[0]}:{STAGE_A_PIN[1]}"}, "spec": s_spec, "attempt_labels": s_lab, "run_table": s_tab,
                    "plus311_cells": s_p311, "membership710": s_m710, "inputs_final": s_inf},
        "S_new": {"from": len(cells311), "to": len(coh_new), "drop": drop_new, "rule": "cohort = cells attempted in run1, run2 and sim (FINAL's S_new_attempted_all_three)",
                  "classes": {k: {"n": len(v), "cells": sorted(v)} for k, v in sorted(classes_new.items(), key=lambda kv: -len(kv[1]))}},
        "S_newmains": {"from": len(nm_cells), "to": len(cohort_nm), "drop": drop_nm, "rule": "membership710: cohort + cohort_4r_sol_only; the rest are status rows",
                       "classes": {k: {"n": len(v), "cells": sorted(v)} for k, v in sorted(classes_nm.items(), key=lambda kv: -len(kv[1]))},
                       "status_row_text": dict(sorted(status.items())),
                       "without_S4": {"from": len(cohort_nm), "to": len(m710["cohort_without_S4"]), "classes": {k: {"n": len(v), "cells": sorted(v)} for k, v in s4_out.items()}}},
        "ledger_sim_honest_exclusions": ledger,
    }
    base = HERE / "results" / f"cohort_attrition_{stamp}"
    with open(f"{base}.json", "x") as fh:
        json.dump(out, fh, indent=1)
    md = [f"# Cohort attrition ({stamp} PT; generator {head[:9]}; keys only, no score read)", ""]
    for s, v in (("S_new", out["S_new"]), ("nm360 (S_newmains)", out["S_newmains"])):
        md += [f"## {s}: {v['from']} -> {v['to']} (drop {v['drop']})", "", "| class | n | cells |", "|---|---:|---|"]
        md += [f"| {k} | {x['n']} | {', '.join(x['cells'])} |" for k, x in v["classes"].items()]
        md += [f"| **total** | **{v['drop']}** | |", ""]
    w = out["S_newmains"]["without_S4"]
    md += [f"## nm360 without S4 (tonight's attack scope): {w['from']} -> {w['to']}", "", "| class | n |", "|---|---:|"]
    md += [f"| {k} | {x['n']} |" for k, x in w["classes"].items()] + [""]
    md += ["## Frozen ledger's sim-honest cell exclusions (H_sim only; attack cohort unchanged)", "", "| cell | set | in attack cohort | in cohort without S4 |", "|---|---|---|---|"]
    md += [f"| {c} | {x['set']} | {x['in_attack_cohort']} | {'' if x['in_cohort_without_S4'] is None else x['in_cohort_without_S4']} |" for c, x in ledger.items()]
    Path(f"{base}.md").open("x").write("\n".join(md) + "\n")
    print(f"{base}.json")
    print("\n".join(md))


if __name__ == "__main__":
    main()
