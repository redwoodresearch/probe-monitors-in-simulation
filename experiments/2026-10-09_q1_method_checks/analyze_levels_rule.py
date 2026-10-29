"""Aswin's rule for running the C search at levels 1-6 (predictions_cgrid_2026-10-09.md, amendments of 2026-10-10 at
11:03 and 11:06 PT), applied mechanically to the levels 0 and 7 summary. Local, free.

Clause 1: best_layer_and_C or earliest_good_layer_and_C held-out AUROC differs from the published headline by > 0.02.
Clause 2: at some C in {0.1, 1, 10, 100}, T(C) > T(0.01) + 0.005 and H(C) > H(0.01) + 0.02, where T and H are the means
over the 8 layers of the table's train_auroc_mean and test.auroc (real vs simulated, scaled optimizer, real labels).
Gate: every per-layer reproduction difference (reproduction_C0.01, both levels, both comparisons, the script's own
check) within 0.002; if any fails, neither clause is applied. With a second file (run_repro_single_column.py's output,
the 11:16 PT amendment), its single-column refits replace the summary's joint refits at the pairs it covers. Comparisons are
rounded to 9 decimals first, so a difference of exactly 0.02 between 4-decimal AUROCs does not count as "more than".

  python3 experiments/2026-10-09_q1_method_checks/analyze_levels_rule.py <cgrid summary> [<repro_single_column file>]
"""

import json
import sys
from pathlib import Path

PUBLISHED = {"level7": 0.5408, "level0": 0.9706}  # Q1's headline held-out-environment run AUROC at C = 0.01
WEAKER, BASE, MOVE, T_RISE, GATE = [0.1, 1.0, 10.0, 100.0], 0.01, 0.02, 0.005, 0.002


def above(x: float, y: float) -> bool:
    return round(x - y, 9) > 0


def main(path: str, repro: str = ""):
    s = json.loads(Path(path).read_text())["levels"]
    diffs = {(lv, c, int(l)): v["diff"] for lv in PUBLISHED for c, r in s[lv].items() for l, v in r["reproduction_C0.01"].items()}
    for row in json.loads(Path(repro).read_text())["pairs"] if repro else []:
        key = (row["level"], row["comparison"], row["layer"])
        if key not in diffs:
            raise ValueError(f"{key} is not in the summary")
        if row["within_gate"] != (abs(row["diff"]) <= GATE):
            raise ValueError(f"{key}: the diagnostic's gate flag disagrees with its difference")
        diffs[key] = row["diff"]
    gates = {f"{lv} {c}": all(abs(d) <= GATE for (lv_, c_, _), d in diffs.items() if (lv_, c_) == (lv, c)) for lv in PUBLISHED for c in s[lv]}
    out, fire = {"reproduction_gates_passed": gates, "single_column_file": Path(repro).name if repro else None}, False
    if not all(gates.values()):  # nothing past the gate is computed or printed
        print(json.dumps(out | {"run_levels_1_to_6": None, "reason": "reproduction gate failed; the rule is not applied"}, indent=1))
        return
    for lv, pub in PUBLISHED.items():
        r = s[lv]["real_vs_sim"]
        held = {k: r[k]["test"]["auroc"] for k in ("best_layer_and_C", "earliest_good_layer_and_C")}
        c1 = {k: above(abs(v - pub), MOVE) for k, v in held.items()}
        by_c = {}
        for row in r["table"]:
            by_c.setdefault(float(row["C"]), []).append(row)
        if any(len(v) != 8 for v in by_c.values()):
            raise ValueError(f"{lv}: a C value without all 8 layers")
        T = {c: sum(x["train_auroc_mean"] for x in v) / 8 for c, v in by_c.items()}
        H = {c: sum(x["test"]["auroc"] for x in v) / 8 for c, v in by_c.items()}
        c2 = {c: above(T[c], T[BASE] + T_RISE) and above(H[c], H[BASE] + MOVE) for c in WEAKER}
        fired = any(c1.values()) or any(c2.values())
        fire |= fired
        out[lv] = {"published": pub, "clause1_held_out": held, "clause1_fired": c1,
                   "T": {str(c): round(T[c], 4) for c in sorted(T)}, "H": {str(c): round(H[c], 4) for c in sorted(H)},
                   "clause2_fired_at_C": {str(c): v for c, v in c2.items()}, "level_fires": fired}
    out["run_levels_1_to_6"] = fire
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
