"""own_all_L48 inputs (09's Append 40 + amendment 1 @ 01b2d13db; Aswin "ok own all 15"): the training universe, folds, per-fold main lists and
the training manifest, by key, with no Modal. Then the held-out gate on the new fold lists (in-fold STOP, positive and negative controls, all in
one intersection function) and a dry run of training runs / rows per fold.

  python3 -I make_own_all_inputs.py <spec_2026-10-14_163027.json> <runs36.json> <message.json>... --out <record.json>

- T196: pass 1's pinned spec (sha 5e2409a7…): its admitted run1 / honest_a epochs (volume q4own_evals; rows per epoch from spec n_rows).
- +311 and nm360: the committed pass-2 records' messages of the main roots (each message's sha256 must equal its record's message_sha256), run1 /
  honest_a epochs, minus the message's own setup-failure and unusable-honest keys and minus the S4 source map's orphan records (@ b68ab0a53).
  t309 / t325 / t389 train on their 25f SELECTED honest_a attempts (the ha25f messages), as amendment 1 §2 rules.
- Universe: mains with at least one admitted run1 AND one admitted honest_a epoch. Folds: Sol #36's (r36) where it assigns one; otherwise the
  committed environment-stratified seeded hash (config/own_all_seed.json @ 4c466b443). Fold model k trains on universe mains with fold != k.
Keys, counts and shas only.
"""

import argparse
import collections
import glob
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPEC_SHA, R36_SHA = "5e2409a75edd6a92a1eae6c222e032911a17140bcefac80de28fdc4780b70b61", "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"
SEED = ("4c466b443", "experiments/2026-10-14_p2_family_expansion/config/own_all_seed.json")
T196 = ("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb")
Q2 = ("0caadbfb5", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json", "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b")
NM = ("ef171c813", "experiments/2026-10-14_q3_new_mains_mode/config/new_mains_cells_v1.json", "98bfb97e26e7ec580b08643a3a38a5e66a0f6b63333f4013aa57559383402e8e")
S4MAP = ("b68ab0a53", "experiments/2026-10-14_q3_overnight_chain/results/s4_source_map/s4_source_map_4r_2026-10-15.json")
HA25F = {"t309_slack_clone": "69c964700620933d0fbfa771fc83068d208a3e20c48dfad47d443a2ec88f5028", "t325_model_registry": "e1cdf8d1594bac78b6e64377557bf63d008f2af0248912f21c7bd60c523b7247",
         "t389_model_registry": "d160ffdc956ffc89180043b3575b002d4b0f1cc3f7600082f38decc81f28dae0"}  # 09's 25f H_A notes
VOL = {"T196": "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", "scaleup": "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"}
ROWS_PER_RUN_EST = 35.1  # T196's observed R rows per run1 / honest_a epoch (13437 / 383): the dry-run estimate for scale-up rows (exact counts come from the rows job)


def show(c, p, want=None):
    b = subprocess.run(["git", "-C", str(REPO), "show", f"{c}:{p}"], capture_output=True, check=True).stdout
    if want and hashlib.sha256(b).hexdigest() != want:
        raise SystemExit(f"{p} @ {c}: sha256 differs from its pin")
    return b


def sha_list(xs):
    return hashlib.sha256(json.dumps(sorted(xs)).encode()).hexdigest()


def inter(a, b):  # the ONE intersection used by every gate check below (in-fold, positive and negative controls)
    return sorted(set(a) & set(b))


def env_of(cell):
    return cell.split("_", 1)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("r36")
    ap.add_argument("messages", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sb, rb = Path(a.spec).read_bytes(), Path(a.r36).read_bytes()
    if hashlib.sha256(sb).hexdigest() != SPEC_SHA or hashlib.sha256(rb).hexdigest() != R36_SHA:
        raise SystemExit("spec or runs36 sha256 differs from its pin")
    spec, r36 = json.loads(sb), json.loads(rb)
    seed_b = show(*SEED)
    seed = json.loads(seed_b)["seed"]
    fom = {}
    for r in r36:
        if fom.setdefault(r["main_task"], r["fold"]) != r["fold"]:
            raise SystemExit("r36: a main in two folds")
    main_of = {c["key"]: c["main"] for c in json.loads(show(*T196))["cells"]}
    pool_of = dict.fromkeys(main_of, "T196")
    for pin, pool in ((Q2, "S_new"), (NM, "S_newmains")):
        for c in json.loads(show(*pin))["cells"]:
            if c["key"] in main_of:
                raise SystemExit(f"cell {c['key']} in two pools")
            main_of[c["key"]], pool_of[c["key"]] = c["main"], pool
    if len(set(main_of.values())) != len(main_of):
        raise SystemExit("a main with two cells across T196 / +311 / nm360: the per-main fold rule needs one cell per main here")
    s4 = json.loads(show(*S4MAP))
    orphans = {(x["root"], x["world"], x["cell"], x["record"]) for x in s4["outside_source_map"]}

    ep, excl = [], collections.defaultdict(list)  # training epochs (run1 / honest_a) by key
    for k, (w, cell) in spec["map"].items():  # T196: pass 1's admitted epochs
        if w in ("run1", "honest_a"):
            ep.append({"pool": "T196", "key": k, "world": w, "cell": cell, "run": spec["run_of"][k], "arm": spec["arm_of"][k], "volume": VOL["T196"],
                       "n_rows_spec": spec["n_rows"][k], "source": f"spec {SPEC_SHA[:8]}"})
    recs = {}
    for p in glob.glob(str(HERE / "results" / "pass2_*.json")):
        if "TEST" in p or "note" in p or "postcheck" in p:
            continue
        recs.setdefault(json.load(open(p))["message_sha256"], []).append(os.path.relpath(p, REPO))
    msgs_used = []
    for mp in a.messages:
        mb = Path(mp).read_bytes()
        h = hashlib.sha256(mb).hexdigest()
        if h not in recs:
            raise SystemExit(f"{mp}: no committed pass-2 record names this message sha256")
        m = json.loads(mb)
        bad = {k for p_ in m["provenance"] for k in ((p_.get("setup_failures") or {}).get("keys") or []) + ((p_.get("unusable_honest") or {}).get("keys") or [])}
        ha = any("_ha25f_" in e["folder"] for e in m["epochs"])
        msgs_used.append({"message": os.path.basename(mp), "sha256": h, "records": sorted(recs[h]), "d9_arm": m["d9_arm"]})
        for e in m["epochs"]:
            if e["world"] not in ("run1", "honest_a"):
                continue
            root = e["folder"].split("__")[0]
            if ha and (e["cell"] not in HA25F or e.get("eval_sha256") != HA25F[e["cell"]]):
                raise SystemExit(f"{mp}: an ha25f epoch that is not 09's selected attempt: {e['cell']}")
            if e["key"] in bad:
                excl["setup failure / unusable honest (message provenance)"].append(f"{e['world']}|{e['cell']}")
                continue
            if (root, e["world"], e["cell"], e["key"].removesuffix("#e0")) in orphans:
                excl["S4 orphan (outside the S4 source map @ b68ab0a53)"].append(f"{e['world']}|{e['cell']}")
                continue
            ep.append({"pool": pool_of.get(e["cell"], "?"), "key": e["key"], "world": e["world"], "cell": e["cell"], "run": e["run"], "arm": e["arm"],
                       "staged": e["staged"], "staged_sha256": e["staged_sha256"], "folder": e["folder"], "volume": m["volume"], "source": f"message {h[:12]}",
                       "ha25f_selected": ha})
    if any(e["pool"] == "?" for e in ep):
        raise SystemExit("a training epoch whose cell is in no pool")
    dup = [k for k, n in collections.Counter((e["world"], e["cell"]) for e in ep).items() if n > 1]
    if dup:
        raise SystemExit(f"{len(dup)} world|cell keys with two training epochs, e.g. {dup[:3]}")
    if {e["cell"] for e in ep if e["world"] == "honest_a" and e.get("ha25f_selected")} != set(HA25F):
        raise SystemExit("the three 25f selected honest_a attempts are not all present")

    have = collections.defaultdict(set)
    for e in ep:
        have[main_of[e["cell"]]].add(e["world"])
    universe = sorted(m_ for m_, w in have.items() if w == {"run1", "honest_a"})
    one_world = sorted(f"{m_}: {sorted(w)}" for m_, w in have.items() if w != {"run1", "honest_a"})
    cell_of_main = {m_: c for c, m_ in main_of.items()}
    fold, basis = {}, {}
    for m_ in universe:
        if m_ in fom:
            fold[m_], basis[m_] = fom[m_], "r36"
    by_env = collections.defaultdict(list)
    for m_ in universe:
        if m_ not in fold:
            by_env[env_of(cell_of_main[m_])].append(m_)
    for e_, ms in sorted(by_env.items()):
        off = int(hashlib.sha256(f"{seed}|env|{e_}".encode()).hexdigest(), 16) % 5
        for r_, m_ in enumerate(sorted(ms, key=lambda x: hashlib.sha256(f"{seed}|{x}".encode()).hexdigest())):
            fold[m_], basis[m_] = (off + r_) % 5, "env-hash"
    lists = {k: sorted(m_ for m_ in universe if fold[m_] == k) for k in range(5)}
    train = {k: sorted(m_ for m_ in universe if fold[m_] != k) for k in range(5)}

    # held-out gate on the new lists: in-fold STOP, positive control (each main is in the other four models' lists), negative control
    in_fold = [m_ for m_ in universe if inter([m_], train[fold[m_]])]
    pos_fail = [m_ for m_ in universe if any(not inter([m_], train[g]) for g in range(5) if g != fold[m_])]
    neg = {k: len(inter(lists[k], train[k])) for k in range(5)}
    pos_counts = {k: len(inter([m_ for g in range(5) if g != k for m_ in lists[g]], train[k])) for k in range(5)}
    rnd = random.Random("own_all-gate")
    gate = {"in_fold_violations": in_fold, "positive_control_failures": pos_fail, "positive_control_counts": pos_counts,
            "positive_control_expected": {k: len(train[k]) for k in range(5)}, "negative_control_counts": neg,
            "pass": not in_fold and not pos_fail and all(v == 0 for v in neg.values()) and all(pos_counts[k] == len(train[k]) > 0 for k in range(5)),
            "raw_samples": {"universe": rnd.sample(universe, 3), **{f"train_{k}": rnd.sample(train[k], 3) for k in range(5)}}}

    run_fold = collections.Counter()
    rows_fold = {"T196_exact": collections.Counter(), "T196_exact_C": collections.Counter(), "scaleup_estimate": collections.Counter()}
    for e in ep:
        m_ = main_of[e["cell"]]
        if m_ not in fold:
            continue
        for k in range(5):
            if fold[m_] != k:
                run_fold[(k, e["pool"])] += 1
                if e["pool"] == "T196":
                    rows_fold["T196_exact"][k] += e["n_rows_spec"][0]
                    rows_fold["T196_exact_C"][k] += e["n_rows_spec"][1]
                else:
                    rows_fold["scaleup_estimate"][k] += ROWS_PER_RUN_EST
    sizes = collections.defaultdict(lambda: [0] * 5)
    for m_ in universe:
        c = cell_of_main[m_]
        sizes[f"pool {pool_of[c]}"][fold[m_]] += 1
        sizes[f"env {env_of(c)}"][fold[m_]] += 1
        sizes[f"basis {basis[m_]}"][fold[m_]] += 1
    out = {"label": "own_all_L48 inputs: universe, folds, per-fold main lists, training manifest and the held-out gate on the new fold lists; keys, counts and shas only (09 Append 40 + amendment 1 @ 01b2d13db)",
           "seed": {"commit": SEED[0], "path": SEED[1], "sha256": hashlib.sha256(seed_b).hexdigest(), "seed": seed},
           "pins": {"spec_sha256": SPEC_SHA, "runs36_sha256": R36_SHA, "t196": T196, "plus311": Q2, "nm360": NM, "s4_source_map": S4MAP, "ha25f_selected": HA25F},
           "messages": msgs_used,
           "universe": {"rule": "mains with >= 1 admitted run1 AND >= 1 admitted honest_a epoch (T196: pass 1's spec; +311 / nm360: the main-root messages)", "n": len(universe),
                        "by_pool": dict(collections.Counter(pool_of[cell_of_main[m_]] for m_ in universe)), "outside_one_world": one_world},
           "folds": {"rule": "Sol #36 (r36) fold where it assigns one; otherwise the environment-stratified seeded hash of the committed seed",
                     "by_main": {m_: [fold[m_], basis[m_]] for m_ in universe},
                     "lists": lists, "lists_sha256": {k: sha_list(v) for k, v in lists.items()},
                     "training_mains_sha256": {k: sha_list(v) for k, v in train.items()}, "sizes": dict(sizes)},
           "exclusions_by_key": {k: sorted(v) for k, v in excl.items()},
           "training_epochs": ep,
           "counts": {"training_epochs": len(ep), "by_pool_world": {f"{p}|{w}": n for (p, w), n in sorted(collections.Counter((e["pool"], e["world"]) for e in ep).items())}},
           "gate": gate,
           "dry_run": {"training_runs_per_fold_model": {k: {p: run_fold[(k, p)] for p in ("T196", "S_new", "S_newmains")} for k in range(5)},
                       "training_rows_per_fold_model_R": {k: {"T196_exact": rows_fold["T196_exact"][k], "scaleup_estimate": round(rows_fold["scaleup_estimate"][k])} for k in range(5)},
                       "training_rows_per_fold_model_C_T196_exact": dict(rows_fold["T196_exact_C"]),
                       "note": f"scale-up rows are an ESTIMATE ({ROWS_PER_RUN_EST} R rows per run, T196's observed rate); the rows job counts them exactly before the fit"}}
    Path(a.out).open("x").write(json.dumps(out, indent=1))
    print(json.dumps({"universe": out["universe"]["n"], "by_pool": out["universe"]["by_pool"], "outside_one_world": len(one_world), "gate_pass": gate["pass"],
                      "lists_sha256": out["folds"]["lists_sha256"], "exclusions": {k: len(v) for k, v in excl.items()}, "counts": out["counts"]}, indent=1))
    for k, v in out["dry_run"]["training_runs_per_fold_model"].items():
        print(f"fold model {k}: training runs {v} | R rows: T196 {out['dry_run']['training_rows_per_fold_model_R'][k]['T196_exact']} exact + scale-up ~{out['dry_run']['training_rows_per_fold_model_R'][k]['scaleup_estimate']}"
              f" | C rows T196 {rows_fold['T196_exact_C'][k]}")
    for k, v in sorted(sizes.items()):
        print(f"  fold sizes {k:28s} {v}")


if __name__ == "__main__":
    main()
