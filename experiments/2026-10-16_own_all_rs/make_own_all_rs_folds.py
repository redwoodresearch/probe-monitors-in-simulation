"""own_all_rs_L48 FOLD LIST (09 inclusion amendment 3 @ 4298ae33a §4; POST HOC; committed BEFORE any re-split fit).

Every main of all four sets gets exactly one fold, a superset of every set's cohort mains, so scoring has no fold -1 unit:
- T196 (196 cells), S_new (+311, 311), S_newmains (nm360, 360, S4 included): the committed cell lists (own_all v1's pins);
- S_paper493: the facts file @ d800edc6a rows, t987 OUT by name (09 READ_LOG @ 65e738c7a / 4e173ffb9).
Fold = the main's Sol #36 (r36) fold where r36 assigns one (as own_all v1, Append 40 am.1). Otherwise the environment-stratified seeded hash of
the seed string "own_all_rs_L48|2026-10-16|env-hash|v1" with own_all v1's algorithm (config/own_all_seed.json @ 4c466b443, rule field): for each
environment e (the cell key's suffix after the first '_'), its mains with no r36 fold are ordered by sha256(seed + '|' + main); fold =
(offset_e + rank) mod 5, offset_e = int(sha256(seed + '|env|' + e), 16) mod 5. Asserts: 0 mains without a fold; 0 mains in two folds; one env per
main; the facts file's r36_main_fold equals runs36's fold for every S_paper493 main. Keys, counts and shas only; no Modal; no score.

  python3 -I experiments/2026-10-16_own_all_rs/make_own_all_rs_folds.py <runs36.json> --out experiments/2026-10-16_own_all_rs/results/own_all_rs_folds_<date>.json
"""

import argparse
import collections
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = "own_all_rs_L48|2026-10-16|env-hash|v1"  # 09 inclusion amendment 3 §4, verbatim
RULE = ("4298ae33a", "experiments/2026-10-14_priority1_spec_09/RULE.md")
R36_SHA = "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"
CELLS = {"S_old": ("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb"),
         "S_new": ("0caadbfb5", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json", "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b"),
         "S_newmains": ("ef171c813", "experiments/2026-10-14_q3_new_mains_mode/config/new_mains_cells_v1.json", "98bfb97e26e7ec580b08643a3a38a5e66a0f6b63333f4013aa57559383402e8e")}
FACTS = ("d800edc6a", "experiments/2026-10-15_q3_sp493/results/sp493_facts_2026-10-15_165730.json", "07caa527db89ec51")  # sha256 prefix (4f's relay)
V1_INPUTS = ("807c52f78", "experiments/2026-10-14_p2_family_expansion/results/own_all_inputs_2026-10-15.json", "26f9d8d1fe08dd2ef430a1bef8af50f336da47234fe364e1a94879bcb54491d1")
OUT_BY_NAME = {"t987_model_registry": "no_usable_reference (refindex_L3 @ 627400b22); out of the S_paper493 attack cohort for every row (09 READ_LOG @ 65e738c7a / 4e173ffb9)"}


def show(c, p, want=None):
    b = subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True).stdout
    if want and not hashlib.sha256(b).hexdigest().startswith(want):
        raise SystemExit(f"{p} @ {c}: sha256 differs from its pin")
    return b


def sha_list(xs):
    return hashlib.sha256(json.dumps(sorted(xs)).encode()).hexdigest()


def env_of(cell):
    return cell.split("_", 1)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("r36")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rb = Path(a.r36).read_bytes()
    if hashlib.sha256(rb).hexdigest() != R36_SHA:
        raise SystemExit("runs36 sha256 differs from its pin")
    if SEED not in show(*RULE).decode():
        raise SystemExit("the seed string is not in 09's amendment 3 text")
    fom = {}
    for r in json.loads(rb):
        if fom.setdefault(r["main_task"], r["fold"]) != r["fold"]:
            raise SystemExit("r36: a main in two folds")
    cells = {}  # set -> {cell: main}
    env = {}
    for s, pin in CELLS.items():
        cells[s] = {c["key"]: c["main"] for c in json.loads(show(*pin))["cells"]}
    facts = json.loads(show(*FACTS))
    cells["S_paper493"] = {r["key"]: r["main"] for r in facts["rows"] if r["key"] not in OUT_BY_NAME}
    if len(facts["rows"]) - len(cells["S_paper493"]) != len(OUT_BY_NAME):
        raise SystemExit("t987 is not exactly one facts row")
    for s, cm in cells.items():
        for c, m in cm.items():
            if env.setdefault(m, env_of(c)) != env_of(c):
                raise SystemExit(f"main {m} in two environments")
    for r in facts["rows"]:
        if r["key"] in cells["S_paper493"]:
            if r["env"] != env_of(r["key"]):
                raise SystemExit(f"facts env differs from the key suffix: {r['key']}")
            if r["main_in_r36"] != (r["main"] in fom) or (r["main"] in fom and r["r36_main_fold"] != fom[r["main"]]):
                raise SystemExit(f"facts r36 fold differs from runs36 for {r['key']}")
    keys = [c for cm in cells.values() for c in cm]
    shared_keys = [k for k, n in collections.Counter(keys).items() if n > 1]
    if shared_keys:
        raise SystemExit(f"{len(shared_keys)} cell keys in two sets")
    mains = sorted(set().union(*(set(cm.values()) for cm in cells.values())))
    fold, basis = {}, {}
    for m in mains:
        if m in fom:
            fold[m], basis[m] = fom[m], "r36"
    by_env = collections.defaultdict(list)
    for m in mains:
        if m not in fold:
            by_env[env[m]].append(m)
    for e, ms in sorted(by_env.items()):
        off = int(hashlib.sha256(f"{SEED}|env|{e}".encode()).hexdigest(), 16) % 5
        for r_, m in enumerate(sorted(ms, key=lambda x: hashlib.sha256(f"{SEED}|{x}".encode()).hexdigest())):
            fold[m], basis[m] = (off + r_) % 5, "env-hash"
    unfolded = [m for m in mains if m not in fold]
    if unfolded or any(f not in range(5) for f in fold.values()):
        raise SystemExit(f"STOP: {len(unfolded)} mains without a fold")
    lists = {k: sorted(m for m in mains if fold[m] == k) for k in range(5)}
    in_two = [m for m in mains if sum(m in v for v in lists.values()) != 1]
    if in_two:
        raise SystemExit(f"STOP: {len(in_two)} mains not in exactly one fold")
    v1 = json.loads(show(*V1_INPUTS))
    v1f = {m: v[0] for m, v in v1["folds"]["by_main"].items()}
    same = sum(fold[m] == f for m, f in v1f.items())
    sizes = collections.defaultdict(lambda: [0] * 5)
    for s, cm in cells.items():
        for m in sorted(set(cm.values())):
            sizes[f"set {s}"][fold[m]] += 1
    for m in mains:
        sizes[f"env {env[m]}"][fold[m]] += 1
        sizes[f"basis {basis[m]}"][fold[m]] += 1
    out = {"label": "own_all_rs_L48 fold list (09 inclusion amendment 3 @ 4298ae33a §4; POST HOC; committed before any re-split fit); keys, counts and shas only",
           "rule": "Sol #36 (r36) fold where it assigns one; otherwise the environment-stratified seeded hash of the amendment's seed string (own_all v1's algorithm, config/own_all_seed.json @ 4c466b443)",
           "seed": SEED, "rule_source": {"commit": RULE[0], "path": RULE[1]},
           "pins": {"runs36_sha256": R36_SHA, "cells": CELLS, "sp493_facts": FACTS, "own_all_v1_inputs": V1_INPUTS, "out_by_name": OUT_BY_NAME},
           "sets": {s: {"cells": len(cm), "mains": len(set(cm.values()))} for s, cm in cells.items()},
           "shared_mains": {"S_paper493 & S_old": len(set(cells["S_paper493"].values()) & set(cells["S_old"].values())),
                            "S_paper493 & S_new": len(set(cells["S_paper493"].values()) & set(cells["S_new"].values())),
                            "S_paper493 & S_newmains": len(set(cells["S_paper493"].values()) & set(cells["S_newmains"].values()))},
           "mains": len(mains), "asserts": {"mains_without_fold": 0, "mains_not_in_exactly_one_fold": 0, "one_env_per_main": True, "facts_r36_equals_runs36": True,
                                            "cell_keys_in_two_sets": 0},
           "by_main": {m: [fold[m], basis[m]] for m in mains}, "cell_set": {c: s for s, cm in cells.items() for c in cm},
           "lists": lists, "lists_sha256": {k: sha_list(v) for k, v in lists.items()}, "sizes": dict(sizes),
           "vs_own_all_v1": {"v1_universe_mains": len(v1f), "same_fold": same, "changed_fold": len(v1f) - same,
                             "note": "r36 mains keep v1's fold; env-hash mains are re-drawn under the new seed over all four sets' mains"}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).open("x").write(json.dumps(out, indent=1))
    print(json.dumps({"mains": out["mains"], "sets": out["sets"], "shared": out["shared_mains"], "basis": dict(collections.Counter(basis.values())),
                      "lists_sha256": out["lists_sha256"], "vs_v1": {k: v for k, v in out["vs_own_all_v1"].items() if k != "note"}}, indent=1))
    for k, v in sorted(sizes.items()):
        print(f"  fold sizes {k:26s} {v}")


if __name__ == "__main__":
    main()
