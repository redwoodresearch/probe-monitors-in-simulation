"""The refreshed Q1 ladder (predictions_cgrid_2026-10-09.md, amendment of 14:20 PT on 2026-10-10): per scrub level, the
C search's earliest-good held-out-environment AUROC (the published headline's rule) against the published headline, with
its 95% interval, its selection-aware null, the e1-vs-e2 control through the same search, and best beside it. A level
enters only if its reproduction gate passed: the summary's joint refits, with the single-column refits in place at the
pairs they cover (all within 0.002). Local, free.

  python3 experiments/2026-10-09_q1_method_checks/analyze_ladder_refresh.py <summary 0/7> <diagnostic 0/7> <summary 1-6> <diagnostic 1-6>
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
PUBLISHED = {"level0": 0.9706, "level1": 0.9710, "level2": 0.9658, "level3": 0.9316, "level4": 0.8169, "level5": 0.8100,
             "level6": 0.7957, "level7": 0.5408}
GATE = 0.002


def main(s07: str, r07: str, s16: str, r16: str):
    rows = []
    for sp, rp in ((s07, r07), (s16, r16)):
        s = json.loads(Path(sp).read_text())["levels"]
        diffs = {(lv, c, int(l)): v["diff"] for lv in s for c, r in s[lv].items() for l, v in r["reproduction_C0.01"].items()}
        for row in json.loads(Path(rp).read_text())["pairs"]:
            diffs[(row["level"], row["comparison"], row["layer"])] = row["diff"]
        for lv in s:
            gate = all(abs(d) <= GATE for (lv_, _, _), d in diffs.items() if lv_ == lv)
            rs, rr = s[lv]["real_vs_sim"], s[lv]["null_e1_vs_e2"]
            eg, b, ee = rs["earliest_good_layer_and_C"], rs["best_layer_and_C"], rr["earliest_good_layer_and_C"]
            rows.append({"level": lv, "reproduction_gate_passed": gate, "published": PUBLISHED[lv],
                         "earliest_good": eg["test"], "delta": round(eg["test"]["auroc"] - PUBLISHED[lv], 4),
                         "null": {k: eg["null"][k] for k in ("null_p95", "null_max", "p")},
                         "C_chosen_by_fold": [c for _, c in eg["choice_by_fold"].values()],
                         "best_for_information": b["test"], "e1_vs_e2_earliest_good": ee["test"] | {"null_p": ee["null"]["p"]},
                         "summary_file": Path(sp).name, "diagnostic_file": Path(rp).name})
    out = {"rule": "earliest-good is the comparison to the published ladder; best is for information only", "levels": sorted(rows, key=lambda r: r["level"])}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with (HERE / "results" / f"ladder_refresh_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    for r in out["levels"]:
        e = r["earliest_good"]
        print(f"{r['level']}: gate {r['reproduction_gate_passed']} | published {r['published']} | earliest-good {e['auroc']} {e['ci95']} "
              f"(delta {r['delta']:+.4f}; null p95 {r['null']['null_p95']}, p {r['null']['p']}) | C {sorted(set(r['C_chosen_by_fold']))} | "
              f"e1 vs e2 {r['e1_vs_e2_earliest_good']['auroc']} {r['e1_vs_e2_earliest_good']['ci95']} p {r['e1_vs_e2_earliest_good']['null_p']}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
