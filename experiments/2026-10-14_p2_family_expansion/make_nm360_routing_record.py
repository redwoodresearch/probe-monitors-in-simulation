"""09/4f: before any nm360 pass-2 write, show BY KEY how every new-main cell routes per probe. NO scores are read.
For each of the 360 new-main cells' mains: whether runs36 (own_a / adapted map) or b_asis's SA + SH maps hold it, and, for a
main in none, that it is in no fold model's training mains for any probe (own_a / adapted: T196 mains + runs36 mains; b_asis:
SA + SH mains), so fold -1 = unseen by all five is verified, not assumed.

  python make_nm360_routing_record.py --r36 <runs36.json>
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
R36_SHA = "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"
SIMH = ("573054a4d", "experiments/2026-10-06_q3_attack_detector/results/sim_honest_scores_2026-10-07_124717/runs.json", "f420ed2401dcc776cbbcfd414bbe9d6d1132a3bbd012b529a33aba7fa5ecd068")
NEWMAINS = ("ef171c813", "experiments/2026-10-14_q3_new_mains_mode/config/new_mains_cells_v1.json", "98bfb97e26e7ec580b08643a3a38a5e66a0f6b63333f4013aa57559383402e8e")
NM_BUILT = ("6feeee4b7", "experiments/2026-10-14_q3_new_mains_launch/config/nm_built.json", "93d9b538710b6bd930e7df01b0f5296d417550a0f0a35bcb6eb696166cf39a04")
T196 = ("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r36", required=True)
    a = ap.parse_args()
    show = lambda c, p: subprocess.run(["git", "-C", str(REPO), "show", f"{c}:{p}"], capture_output=True, check=True).stdout  # noqa: E731
    rb = Path(a.r36).read_bytes()
    raw = {"simh": show(*SIMH[:2]), "nm": show(*NEWMAINS[:2]), "nb": show(*NM_BUILT[:2]), "t196": show(*T196[:2])}
    want = {"simh": SIMH[2], "nm": NEWMAINS[2], "nb": NM_BUILT[2], "t196": T196[2]}
    if hashlib.sha256(rb).hexdigest() != R36_SHA or any(hashlib.sha256(raw[k]).hexdigest() != want[k] for k in want):
        raise SystemExit("a pinned input differs from its sha256")
    r36, simh = json.loads(rb), json.loads(raw["simh"])
    cells = [json.loads(raw[k])["cells"] for k in ("nm", "nb")]
    if any(len({c["key"] for c in cs}) != len(cs) for cs in cells) or {c["key"]: c["main"] for c in cells[0]} != {c["key"]: c["main"] for c in cells[1]}:
        raise SystemExit("the two new-main pins disagree, or a key repeats: stop")
    r36_fold = {r["main_task"]: r["fold"] for r in r36}
    b_fold = {r["main_task"]: r["fold"] for r in r36 if r["arm"] == "sim-attack"} | {r["main_task"]: r["fold"] for r in simh if r["arm"] == "sim-honest"}
    t196_mains = {c["main"] for c in json.loads(raw["t196"])["cells"]}
    own_train = t196_mains | set(r36_fold)  # every main any own_a / adapted fold model can have trained on (pass 2's superset)
    b_train = set(b_fold)  # every main any b_asis fold model trained on
    rows = {}
    for c in cells[0]:
        m = c["main"]
        rows[c["key"]] = {"main": m, "own_a_adapted": {"in_map": m in r36_fold, "fold": r36_fold.get(m, -1), "in_any_training_set": m in own_train - set(r36_fold)
                                                       if m in r36_fold else m in own_train},
                          "b_asis": {"in_map": m in b_fold, "fold": b_fold.get(m, -1), "in_any_training_set": False if m in b_fold else m in b_train}}
    unseen_ok = all((not r["own_a_adapted"]["in_map"] and not r["own_a_adapted"]["in_any_training_set"]) or r["own_a_adapted"]["in_map"] for r in rows.values()) and \
        all((not r["b_asis"]["in_map"] and not r["b_asis"]["in_any_training_set"]) or r["b_asis"]["in_map"] for r in rows.values())
    out = {"label": "by key; no scores read", "inputs": {"runs36": R36_SHA, "simh": list(SIMH), "new_mains": list(NEWMAINS), "nm_built": list(NM_BUILT), "t196": list(T196)},
           "n_cells": len(rows), "n_mains": len({r["main"] for r in rows.values()}),
           "own_a_adapted_mapped": sum(r["own_a_adapted"]["in_map"] for r in rows.values()), "b_asis_mapped": sum(r["b_asis"]["in_map"] for r in rows.values()),
           "verdict": "every unmapped main is in no fold model's training mains for any probe: fold -1 = unseen by all five, verified" if unseen_ok else "NOT VERIFIED: stop",
           "cells": rows}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    p = HERE / "results" / f"nm360_routing_{stamp}.json"
    with p.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({"path": str(p.relative_to(REPO)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), **{k: out[k] for k in ("n_cells", "n_mains", "own_a_adapted_mapped", "b_asis_mapped", "verdict")}}, indent=1))


if __name__ == "__main__":
    main()
