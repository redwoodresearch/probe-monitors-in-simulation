"""09's held-out gate (@ c6023b805) for shared-pool own_frac rows, by key (4f's request, 2026-10-15): for each own_frac fraction (10, 25, 100) x fold
model (0-4), the nm360 new-main MAIN names (new-mains cells @ ef171c813) and the +311 MAIN names (q2 list @ 0caadbfb5) intersected with that model's
training MAIN names. Training mains are rebuilt as check_fold_training_counts.py does (pass 1's spec, sha-pinned; run_famexp._train_tasks / _group
@ b5a0586fa; mains from T196 @ 4d1eec572). The pinned training_mains_sha256.own_frac[f] (@ 5eedbfb0f) is recomputed under pass 2's definition
(T196 mains with Sol #36 fold != f, union Sol #36 mains with fold != f). 09's positive control runs in the SAME intersection code (inter()):
the T196 mains of folds g != f must intersect fold f's training list with the expected nonzero count (its full size: own_frac trains on T196 mains
only), and fold f's own held-out T196 mains must intersect it at 0. Three raw main ids per side (seeded sample, seed "heldout-overlap") are
printed so a format mismatch would show. Keys and counts only.

  python3 -I check_heldout_overlap.py <spec_2026-10-14_163027.json> <runs36.json> <repo> <out.json>
"""
import hashlib
import json
import math
import random
import subprocess
import sys

spec_p, r36_p, repo, out_p = sys.argv[1:5]


def show(c, p):
    return subprocess.run(["git", "-C", repo, "show", f"{c}:{p}"], capture_output=True, check=True).stdout


sb = open(spec_p, "rb").read()
assert hashlib.sha256(sb).hexdigest() == "5e2409a75edd6a92a1eae6c222e032911a17140bcefac80de28fdc4780b70b61"
spec = json.loads(sb)
rb = open(r36_p, "rb").read()
assert hashlib.sha256(rb).hexdigest() == "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"
r36 = json.loads(rb)
t196 = {c["key"]: c["main"] for c in json.loads(show("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"))["cells"]}
nm = {c["key"]: c["main"] for c in json.loads(show("ef171c813", "experiments/2026-10-14_q3_new_mains_mode/config/new_mains_cells_v1.json"))["cells"]}
q2 = {c["key"]: c["main"] for c in json.loads(show("0caadbfb5", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json"))["cells"]}
def inter(a, b):  # the ONE intersection used for every check below (nm360, +311, and the positive / negative controls)
    return sorted(set(a) & set(b))


def sample3(xs):
    return random.Random("heldout-overlap").sample(sorted(xs), 3)


pins = json.loads(show("5eedbfb0f", "experiments/2026-10-14_p2_family_expansion/results/simh_scaleup_pins_2026-10-15.json"))["25f_s2_pins"]["per_probe_routing_maps"]["training_mains_sha256"]["own_frac"]
fom = {}
for r in r36:
    assert fom.setdefault(r["main_task"], r["fold"]) == r["fold"]
nm_mains, q2_mains, t196_mains = set(nm.values()), set(q2.values()), set(t196.values())
t196_fold_of_main = {m: fom[m] for m in t196_mains}
q2_fold = {m: fom.get(m, -1) for m in q2_mains}
union = [sorted({t196[c] for c in t196 if fom[t196[c]] != f} | {r["main_task"] for r in r36 if r["fold"] != f}) for f in range(5)]
out = {"counts": {"nm360_cells": len(nm), "nm360_mains": len(nm_mains), "plus311_cells": len(q2), "plus311_mains": len(q2_mains),
                  "plus311_mains_mapped_r36": sum(f >= 0 for f in q2_fold.values()), "plus311_cells_mapped_r36": sum(fom.get(m, -1) >= 0 for m in q2.values()),
                  "nm360_mains_equal_any_t196_main": len(inter(nm_mains, t196_mains)), "nm360_mains_in_r36": len(inter(nm_mains, fom)),
                  "plus311_mains_equal_any_t196_main": len(inter(q2_mains, t196_mains))},
       "pins_own_frac": {"commit": "5eedbfb0f", "definition": "sha256(json.dumps(sorted(T196 mains with Sol #36 fold != f | Sol #36 mains with fold != f))), pass 2's training_manifests for every roster variant (not one fraction)",
                         "recomputed_equal": [hashlib.sha256(json.dumps(union[f]).encode()).hexdigest() == pins[f] for f in range(5)]},
       "fits": {}}
for ri, readout in enumerate(("R", "C")):
    keys = [k for k, (w, t) in spec["map"].items() if w in ("run1", "honest_a") and spec["n_rows"][k][ri] > 0]
    for f in range(5):
        t = sorted({spec["map"][k][1] for k in keys if spec["fot"][spec["map"][k][1]] != f})
        random.Random(f"famexp|own_a|f{f}").shuffle(t)
        for v in (1.0, 0.25, 0.1):
            tm = {t196[c] for c in t[: math.ceil(v * len(t))]}
            name = f"own_frac{int(round(v * 100))}_L*|{readout}|f{f}"
            other = [m for m in t196_mains if t196_fold_of_main[m] != f]
            held = [m for m in t196_mains if t196_fold_of_main[m] == f]
            out["fits"][name] = {"training_mains": len(tm), "training_mains_sha256": hashlib.sha256(json.dumps(sorted(tm)).encode()).hexdigest(),
                                 "subset_of_pinned_union": tm <= set(union[f]),
                                 "nm360_overlap": inter(nm_mains, tm), "plus311_overlap": len(inter(q2_mains, tm)),
                                 "plus311_in_fold_violations": [m for m in inter(q2_mains, tm) if q2_fold[m] == f],
                                 "positive_control_t196_other_folds": {"intersection": len(inter(other, tm)), "expected": len(tm), "pass": len(inter(other, tm)) == len(tm) > 0},
                                 "negative_control_t196_held_out_fold": {"intersection": len(inter(held, tm)), "expected": 0, "pass": len(inter(held, tm)) == 0}}
            if v == 1.0 and readout == "R":
                out["fits"][name]["raw_main_samples"] = {"training": sample3(tm), "nm360": sample3(nm_mains), "plus311": sample3(q2_mains), "t196_held_out_fold": sample3(held)}
u_q2 = [{"fold": f, "plus311_mains_in_union": len(inter(q2_mains, union[f])), "of_which_held_out_here": sum(q2_fold[m] == f for m in inter(q2_mains, union[f])),
         "plus311_mapped_mains_not_in_fold_f": sum(1 for m in q2_mains if q2_fold[m] >= 0 and q2_fold[m] != f), "nm360_mains_in_union": len(inter(nm_mains, union[f]))} for f in range(5)]
out["pinned_union_vs_plus311_nm360"] = u_q2
json.dump(out, open(out_p, "x"), indent=1)
print(json.dumps(out["counts"]), "| pins recomputed equal:", out["pins_own_frac"]["recomputed_equal"])
for k, v in out["fits"].items():
    print(f"{k:22s} train mains {v['training_mains']:3d} | subset of pinned union {v['subset_of_pinned_union']} | nm360 overlap {len(v['nm360_overlap'])} | +311 overlap {v['plus311_overlap']}"
          f" | +311 in-fold {len(v['plus311_in_fold_violations'])} | POS control {v['positive_control_t196_other_folds']['intersection']}/{v['positive_control_t196_other_folds']['expected']}"
          f" | NEG control {v['negative_control_t196_held_out_fold']['intersection']}")
    if "raw_main_samples" in v:
        print("   raw samples:", json.dumps(v["raw_main_samples"]))
for x in u_q2:
    print("union", x)
