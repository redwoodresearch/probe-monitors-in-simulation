"""Append-only NOTE explaining, by key, every gap in own_all_rs_L48's 3a (c) coverage assert (a fold-list main with no scored unit in its set):
the gap's cells, the evidence checked, and the reason class. Reads committed records only (git) plus famexp's S_paper493 messages on this
branch; no Modal; no score.

  python3 -I experiments/2026-10-16_own_all_rs/make_coverage_gap_note.py
"""

import hashlib
import json
import subprocess
from pathlib import Path

FAM = "experiments/2026-10-14_p2_family_expansion"
ASSERT = f"{FAM}/results/own_all_rs_coverage_assert_2026-10-16_012547.json"
FOLDS = "experiments/2026-10-16_own_all_rs/results/own_all_rs_folds_2026-10-16.json"
SPEC = ("dea0a4e79", None)  # unused placeholder
V5 = ("90c0c3148", "experiments/2026-10-15_launch_disjoint/results/launch_disjoint_eligibility_v5_2026-10-15_224056.json")
INPUTS = f"{FAM}/results/own_all_inputs_2026-10-15.json"
CELLS = {"S_old": ("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"),
         "S_new": ("0caadbfb5", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json"),
         "S_newmains": ("ef171c813", "experiments/2026-10-14_q3_new_mains_mode/config/new_mains_cells_v1.json")}
FACTS = ("d800edc6a", "experiments/2026-10-15_q3_sp493/results/sp493_facts_2026-10-15_165730.json")
SP_MSGS = [f"{FAM}/sp493_msgs/msg_sp493_{g}.json" for g in ("a", "b", "g1", "g2", "g4", "g5")]


def show(c, p):
    return subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True).stdout


def main():
    ab = show("HEAD", ASSERT)
    a = json.loads(ab)
    mo = {}
    for c, p in CELLS.values():
        mo |= {x["key"]: x["main"] for x in json.loads(show(c, p))["cells"]}
    mo |= {r["key"]: r["main"] for r in json.loads(show(*FACTS))["rows"] if not r["key"].startswith("t987_")}
    cs = json.loads(show("HEAD", FOLDS))["cell_set"]
    cells = {}
    for c, st in cs.items():
        cells.setdefault((st, mo[c]), []).append(c)
    inp = json.loads(show("HEAD", INPUTS))
    trained_cells = {e["cell"] for e in inp["training_epochs"]}
    one_world = {x.split(":")[0] for x in inp["universe"]["outside_one_world"]}
    v5 = json.loads(show(*V5))
    never = set(v5["s_new_cells_never_launched"]["cells"])
    sp_cells = set()
    for mp in SP_MSGS:
        sp_cells |= {e["cell"] for e in json.loads(Path(mp).read_bytes())["epochs"]}
    out = {}
    for st, v in a["sets"].items():
        for m, g in v["gaps"].items():
            cl = sorted(cells[(st, m)])
            ev = {"cells": cl, "routed_runs": g.get("runs", []), "in_own_all_training_epochs": sorted(set(cl) & trained_cells), "one_arm_main_in_v1": m in one_world}
            if st == "S_paper493":
                ev["in_any_released_S_paper493_message"] = sorted(set(cl) & sp_cells)
                cls = "S_paper493: the cell is in no S_paper493 set released in r1 (16 sets: a_main / a_run2 / b_main / g1 / g2 / g4 / g5); c_main / c_run2 / b_run2 / g3 are not routed yet" \
                    if not ev["in_any_released_S_paper493_message"] else g["reason"]
            elif st == "S_new" and set(cl) <= never:
                cls = "S_new: +311 cell never launched (eligibility v5 @ 90c0c3148 s_new_cells_never_launched)"
            elif g.get("runs"):
                cls = "runs routed and scored with 0 units in both readouts (zero-call / empty runs; S4 honest references with no policy call where applicable)"
            else:
                cls = "no routed run of this main in the set's scored arms or pools (outside the routed launches; e.g. an excluded or unlaunched cell)"
            out[f"{st}|{m}"] = {"reason_class": cls, "assert_reason": g["reason"], "evidence": ev}
    rec = {"label": "own_all_rs_L48 3a (c) coverage assert: gap NOTE (append-only; every gap explained by key); keys and counts only",
           "binds": {"path": ASSERT, "sha256": hashlib.sha256(ab).hexdigest()}, "gaps": out,
           "counts": {st: len(v["gaps"]) for st, v in a["sets"].items()}}
    p = Path(FAM) / "results" / "own_all_rs_coverage_assert_2026-10-16_012547_GAP_NOTE.json"
    with p.open("x") as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)
    import collections
    print(p, json.dumps(rec["counts"]), json.dumps(collections.Counter(v["reason_class"][:70] for v in out.values()), indent=0))


if __name__ == "__main__":
    main()
