"""Key lists for the interleaved run3+sim2 launch (spec Append 32): the 507's ESTIMATION and SELECTION sets and the
with-strategy (headline) cells, from pinned committed inputs only. No scores are read.

  python3 make_il_keylists.py <dir with xerr_v6_2026-10-14_161122.json, t196.json, scaleup_tasks_plus311_2026-10-14_125135.json,
                                strategy_bits_t196.json>
"""

import hashlib
import json
import sys
from pathlib import Path

PINS = {
    "xerr_v6_2026-10-14_161122.json": "e789144b62f71e0fb168dc3106656c2b5f29aadafe12df0f86dd1f74441d7645",
    "t196.json": "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb",
    "scaleup_tasks_plus311_2026-10-14_125135.json": "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b",
    "strategy_bits_t196.json": "9a5a4e7120037f929e85d1efd9c7cf3ba35756349d8135f15f4cfe1e8775cfc0",
}


def load(d, name):
    raw = (Path(d) / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PINS[name]:
        raise SystemExit(f"pin mismatch: {name}")
    return json.loads(raw)


def main(d):
    us = load(d, "xerr_v6_2026-10-14_161122.json")["result"]["universe_and_split"]
    t196 = {c["key"]: c for c in load(d, "t196.json")["cells"]}
    p311 = {c["key"]: c for c in load(d, "scaleup_tasks_plus311_2026-10-14_125135.json")["cells"]}
    sb = load(d, "strategy_bits_t196.json")["tasks"]
    if set(t196) & set(p311):
        raise SystemExit("T196 and +311 keys overlap")
    sel_tasks = sorted(us["eligibility_half"])
    sel_mains = sorted({t196[k]["main"] for k in sel_tasks})
    est_t196 = sorted(k for k in t196 if t196[k]["main"] not in set(sel_mains))
    not_universe = sorted(set(t196) - set(us["universe"]))
    est_311 = sorted(k for k in p311 if p311[k]["main"] not in set(sel_mains))
    dropped_311 = sorted(set(p311) - set(est_311))
    if any(bool(sb[k]["has_strategy"]) != (t196[k]["approach_chars"] > 0) for k in sb):
        raise SystemExit("strategy bits disagree with approach_chars on T196")

    def strat(k):
        return (t196[k] if k in t196 else p311[k])["approach_chars"] > 0

    est = est_t196 + est_311
    out = {
        "label": "interleaved launch key lists (spec Append 32); estimation excludes every selection-half MAIN (Amendment 7b)",
        "inputs_sha256": PINS,
        "selection_set": {
            "rule": "v6 eligibility half (random.Random(20261014) over the sorted 195-task universe; first floor(N/2))",
            "tasks": sel_tasks,
            "n_tasks": len(sel_tasks),
            "mains": sel_mains,
        },
        "estimation_set": {
            "rule": "all T196 and +311 cells whose MAIN is not a selection-half main",
            "t196_tasks": est_t196,
            "plus311_cells": est_311,
            "n": len(est),
            "t196_not_in_v6_universe_included_by_main_rule": [k for k in not_universe if k in est_t196],
            "plus311_dropped_main_in_selection_half": dropped_311,
        },
        "with_strategy_headline_cells": {
            "rule": "approach_chars > 0 (equals strategy_bits_t196 on T196; Append 18 strata)",
            "estimation": sorted(k for k in est if strat(k)),
            "n_estimation": sum(strat(k) for k in est),
            "estimation_without_strategy": sorted(k for k in est if not strat(k)),
        },
        "counts": {
            "t196": len(t196),
            "plus311": len(p311),
            "selection_tasks": len(sel_tasks),
            "estimation_t196": len(est_t196),
            "estimation_plus311": len(est_311),
        },
    }
    dst = Path(__file__).resolve().parent / "results" / "il_keylists_2026-10-14.json"
    dst.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out["counts"]), "with-strategy estimation", out["with_strategy_headline_cells"]["n_estimation"],
          "dropped311", len(dropped_311), "t196 not in universe", not_universe)


if __name__ == "__main__":
    main(sys.argv[1])
