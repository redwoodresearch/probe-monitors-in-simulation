"""Budget check before the first rlr3 (0d 04:08; 09 @ 7bd9d5e89): compare the usable HARD headroom from a fresh
il_headroom.py run against the hard committed rlr3 caps (cells x $14.61 + min(cells, 5) x $75 per host). If it fits, every
selected cell runs. If not, cells are taken in GLOBAL cell-ID order (sorted cell keys across hosts; nothing
outcome-based) while the per-host caps of the taken cells fit; the rest are recorded "not run (cap)". Retained cells keep
their original rlr order inside each host's task file. Aswin's $24.5k is never crossed.
  python3 rlr3_budget.py --headroom-json <fresh il_headroom json> <h> [<h> ...]  > results/il_rlr3_budget_<time>.json
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
PRICE, RESERVE = 14.6079, 75.0
cap = lambda n: round(n * PRICE + min(n, 5) * RESERVE) if n else 0  # noqa: E731


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headroom-json", required=True)
    ap.add_argument("hosts", nargs="+")
    a = ap.parse_args()
    hj = json.loads(Path(a.headroom_json).read_text())
    usable_hard = hj["usable_hard_usd"]
    sel = {}
    for h in a.hosts:  # in the original rlr execution order (the rlr task file), Codex r24
        s = set(json.loads((HERE / "results" / f"il_rlr3_selection_{h}.json").read_text())["selected"])
        sel[h] = [c["key"] for c in json.loads((HERE / "config" / f"tasks_ilrr_{h}.json").read_text())["cells"] if c["key"] in s]
        assert set(sel[h]) == s, h
    want = {h: cap(len(c)) for h, c in sel.items()}
    out = {"headroom_json": a.headroom_json, "usable_hard_usd": usable_hard, "order": "global cell-ID order (sorted keys across hosts)",
           "requested_caps_usd": want, "requested_total_usd": sum(want.values())}
    if sum(want.values()) <= usable_hard:
        out |= {"fits": True, "selected": sel, "held": {}, "caps_usd": want}
    else:
        taken = {h: [] for h in sel}
        for cell, h in sorted((c, h) for h, cs in sel.items() for c in cs):
            trial = {k: cap(len(v) + (k == h)) for k, v in taken.items()}
            if sum(trial.values()) <= usable_hard:
                taken[h].append(cell)
        kept = {h: [c for c in sel[h] if c in set(taken[h])] for h in sel}  # original rlr order kept
        out |= {"fits": False, "selected": kept, "held": {h: [c for c in sel[h] if c not in set(kept[h])] for h in sel},
                "held_status": "not run (cap)", "caps_usd": {h: cap(len(v)) for h, v in kept.items()}}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
